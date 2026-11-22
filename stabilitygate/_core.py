"""Shared estimators for the stability screen.

Every quantity here matches the definitions in the method paper's
reproducibility package, so that :func:`stabilitygate.screen`
reproduces the paper's four-model verdicts and :func:`stabilitygate.power`
reproduces its operating-characteristic tables.

The estimators, in the paper's terms:

* ``lam``            reliability lambda, the share of observed between-cell
                     variance that is not binomial sampling noise. It is the
                     Spearman attenuation ratio, the factor by which an observed
                     between-run correlation is depressed relative to the
                     correlation between the underlying propensities.
* ``components``     the two drift signals. A signed run effect ``z_run`` on the
                     log-odds scale (Woolf inverse-variance-weighted mean log
                     odds ratio, asymptotically Mantel-Haenszel), and a
                     ``Q`` heterogeneity statistic (Cochran's Q) for cell-by-run
                     interaction after the common shift is removed.
* ``bootstrap_null`` critical values and p-values drawn from the design's OWN
                     perfectly-stable null, so the screen is calibrated at every
                     baseline without tuning.

All boundary-safe logits use the Haldane 0.5 continuity correction, so a cell at
0/20 or 20/20 still yields a finite log odds and a finite variance.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "as_rates",
    "haldane_logit",
    "components",
    "statistics",
    "statistics_batch",
    "component_alpha",
    "lam",
    "lambda_interval",
    "pearson_r",
    "spearman_rho",
    "drift_scales",
    "null_p_values",
    "bootstrap_null_decomposition",
    "registered_gate_false_exclusion",
    "level_critical_values",
    "reject_level",
    "resolve_outcome",
    "ALPHA_SPLITS",
    "RUN_SIDES",
    "COMBINES",
    "PRECEDENCES",
    "LEVEL_STATISTICS",
    "HETEROGENEITY_STATISTICS",
    "NULLS",
    "TIE_TOLERANCE",
]

# --------------------------------------------------------------------------- #
# The admissible values of the screen's configuration parameters
# --------------------------------------------------------------------------- #
#: How ``alpha`` is divided between the two component tests.
ALPHA_SPLITS = ("bonferroni", "sidak", "none")
#: Sidedness of the level (run-effect) test.
RUN_SIDES = ("two", "greater", "less")
#: How the two component tests are combined into one exclusion decision.
COMBINES = ("or", "and")
#: Whether the run-effect test may override the information floor.
PRECEDENCES = ("test_first", "resolution_first")
#: The level statistic.
LEVEL_STATISTICS = ("woolf_z", "pooled_logit")
#: The heterogeneity statistic.
HETEROGENEITY_STATISTICS = ("cochran_q", "pearson_chi2")
#: The null the component tests are calibrated against.
NULLS = ("bootstrap", "oracle")
#: Relative tolerance under which a null statistic counts as tied with the
#: observed one when a bootstrap p-value is counted.
TIE_TOLERANCE = 1e-9


def component_alpha(alpha: float, alpha_split: str = "bonferroni") -> float:
    """The level each component test runs at, given the union target ``alpha``.

    The screen excludes when either component test rejects, so two tests each run
    at ``alpha`` give a union near ``2 * alpha``. ``alpha_split`` says how the
    union budget is divided.

    * ``"bonferroni"`` (default) each component at ``alpha / 2``.
    * ``"sidak"``      each component at ``1 - (1 - alpha) ** 0.5``, which is the
      exact split for two independent tests and is marginally less conservative.
    * ``"none"``       each component at ``alpha``, so the union sits near
      ``2 * alpha``.
    """
    if alpha_split not in ALPHA_SPLITS:
        raise ValueError(f"alpha_split must be one of {ALPHA_SPLITS}")
    if alpha_split == "bonferroni":
        return alpha / 2.0
    if alpha_split == "sidak":
        return 1.0 - math.sqrt(1.0 - alpha)
    return alpha


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
def as_rates(
    values: np.ndarray | list, n_calls: int, input_kind: str = "auto"
) -> np.ndarray:
    """Return per-cell rates as a float array, validating the range.

    Parameters
    ----------
    values : per-cell rates or success counts.
    n_calls : calls per cell (used to convert counts).
    input_kind : how ``values`` are interpreted.

        * ``"rate"``  values are proportions in ``[0, 1]``.
        * ``"count"`` values are integer success counts in ``[0, n_calls]`` and
          are divided through by ``n_calls``.
        * ``"auto"``  (default) values are treated as rates unless at least one
          entry exceeds 1, in which case the whole vector is read as counts.
          Because a vector of only 0s and 1s is ambiguous, pass
          ``input_kind="count"`` explicitly for 0/1 count data.

    Non-finite values (NaN, inf) are rejected.
    """
    if input_kind not in {"auto", "rate", "count"}:
        raise ValueError("input_kind must be 'auto', 'rate', or 'count'")
    a = np.asarray(values, dtype=float)
    if a.ndim != 1:
        raise ValueError("expected a one-dimensional array of per-cell values")
    if a.size == 0:
        raise ValueError("no cells supplied")
    if not np.all(np.isfinite(a)):
        raise ValueError("per-cell values must be finite (no NaN or inf)")

    treat_as_count = input_kind == "count" or (
        input_kind == "auto" and np.any(a > 1.0)
    )
    if treat_as_count:
        if not np.all(np.mod(a, 1) == 0):
            raise ValueError(
                "success counts must be whole numbers in [0, n_calls]"
            )
        if np.any(a > n_calls) or np.any(a < 0):
            raise ValueError("success counts must lie in [0, n_calls]")
        a = a / n_calls
    if np.any(a < 0) or np.any(a > 1):
        raise ValueError("rates must lie in [0, 1]")
    return a


# --------------------------------------------------------------------------- #
# Boundary-safe log-odds machinery
# --------------------------------------------------------------------------- #
def haldane_logit(k: np.ndarray, n: int) -> np.ndarray:
    """Log odds of ``k`` successes out of ``n`` with a 0.5 continuity correction."""
    k = np.asarray(k, dtype=float)
    return np.log((k + 0.5) / (n - k + 0.5))


def components(a: np.ndarray, b: np.ndarray, n: int) -> tuple[float, float]:
    """Return the signed run-effect z and the interaction Q for one run pair.

    Parameters
    ----------
    a, b : arrays of per-cell RATES for the main run and the retest run.
    n    : calls per cell.

    Returns
    -------
    (z_run, Q_interaction)
        ``z_run`` is the Woolf inverse-variance-weighted mean per-cell log odds
        ratio, scaled to a standard-normal test of a common shift. ``Q`` is
        Cochran's Q, the weighted heterogeneity of the per-cell shifts about
        that common shift.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("main and retest arrays must have the same shape")
    ka, kb = a * n, b * n
    d = haldane_logit(kb, n) - haldane_logit(ka, n)
    v = (
        1.0 / (ka + 0.5)
        + 1.0 / (n - ka + 0.5)
        + 1.0 / (kb + 0.5)
        + 1.0 / (n - kb + 0.5)
    )
    w = 1.0 / v
    sw = float(np.sum(w))
    dbar = float(np.sum(w * d) / sw)
    z_run = dbar * math.sqrt(sw)
    q_int = float(np.sum(w * (d - dbar) ** 2))
    return z_run, q_int


def _components_batch(A: np.ndarray, B: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised :func:`components` over the rows of ``A`` and ``B`` (R x K)."""
    ka, kb = A * n, B * n
    d = np.log((kb + 0.5) / (n - kb + 0.5)) - np.log((ka + 0.5) / (n - ka + 0.5))
    v = (
        1.0 / (ka + 0.5)
        + 1.0 / (n - ka + 0.5)
        + 1.0 / (kb + 0.5)
        + 1.0 / (n - kb + 0.5)
    )
    w = 1.0 / v
    sw = w.sum(axis=1)
    dbar = (w * d).sum(axis=1) / sw
    z = dbar * np.sqrt(sw)
    q = (w * (d - dbar[:, None]) ** 2).sum(axis=1)
    return z, q


# --------------------------------------------------------------------------- #
# Alternative statistics, as used by the published operating-characteristics run
# --------------------------------------------------------------------------- #
def _pooled_logit_batch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Mean log-odds difference between the two runs, on the pooled rate.

    The alternative level statistic. Each run's per-cell rates are averaged first
    and the difference is taken on the logit of those two averages, with the
    averages clipped at ``1e-9`` rather than continuity corrected. It is the
    statistic of the simulation that produced ``OC_2026_08_28``, and unlike
    ``woolf_z`` it is not scaled to a standard normal, so its critical values
    live on the log-odds scale and saturate near 20.7 in degenerate cells.
    """
    ma = np.clip(A.mean(axis=1), 1e-9, 1 - 1e-9)
    mb = np.clip(B.mean(axis=1), 1e-9, 1 - 1e-9)
    return np.log(mb / (1 - mb)) - np.log(ma / (1 - ma))


def _pearson_chi2_batch(A: np.ndarray, B: np.ndarray, n: int) -> np.ndarray:
    """Pearson chi-square on the per-cell differences between the two runs.

    The alternative heterogeneity statistic, ``sum_i (a_i - b_i)^2 / (p_i (1 - p_i)
    2 / n)`` with ``p_i`` the per-cell mean of the two runs. Unlike Cochran's Q it
    does not first remove a common shift, so it carries some of the run effect as
    well as the interaction.
    """
    phat = (A + B) / 2.0
    var = phat * (1 - phat) * (2.0 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(var > 0, (A - B) ** 2 / var, 0.0)
    return t.sum(axis=1)


def statistics_batch(
    A: np.ndarray,
    B: np.ndarray,
    n: int,
    level_statistic: str = "woolf_z",
    heterogeneity_statistic: str = "cochran_q",
) -> tuple[np.ndarray, np.ndarray]:
    """Level and heterogeneity statistics over the rows of ``A`` and ``B`` (R x K).

    ``level_statistic`` is ``"woolf_z"`` (default, the package's inverse-variance
    weighted mean per-cell log odds ratio scaled to a z) or ``"pooled_logit"``.
    ``heterogeneity_statistic`` is ``"cochran_q"`` (default) or ``"pearson_chi2"``.
    """
    if level_statistic not in LEVEL_STATISTICS:
        raise ValueError(f"level_statistic must be one of {LEVEL_STATISTICS}")
    if heterogeneity_statistic not in HETEROGENEITY_STATISTICS:
        raise ValueError(
            f"heterogeneity_statistic must be one of {HETEROGENEITY_STATISTICS}"
        )
    z = q = None
    if level_statistic == "woolf_z" or heterogeneity_statistic == "cochran_q":
        z, q = _components_batch(A, B, n)
    lev = z if level_statistic == "woolf_z" else _pooled_logit_batch(A, B)
    het = q if heterogeneity_statistic == "cochran_q" else _pearson_chi2_batch(A, B, n)
    return lev, het


def statistics(
    a: np.ndarray,
    b: np.ndarray,
    n: int,
    level_statistic: str = "woolf_z",
    heterogeneity_statistic: str = "cochran_q",
) -> tuple[float, float]:
    """Scalar :func:`statistics_batch` for one run pair.

    With the defaults this returns exactly what :func:`components` returns.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("main and retest arrays must have the same shape")
    lev, het = statistics_batch(
        a[None, :], b[None, :], n, level_statistic, heterogeneity_statistic
    )
    return float(lev[0]), float(het[0])


def reject_level(lev: np.ndarray, crit: np.ndarray, run_sided: str) -> np.ndarray:
    """Whether the level test rejects, given its critical value(s) ``crit``.

    ``crit`` broadcasts against ``lev``, so it may be one shared critical value or
    one per replicate. Pair it with :func:`level_critical_values`, which produces
    the critical value in the tail ``run_sided`` names.
    """
    if run_sided == "two":
        return np.abs(lev) > crit
    if run_sided == "greater":
        return lev > crit
    if run_sided == "less":
        return lev < crit
    raise ValueError(f"run_sided must be one of {RUN_SIDES}")


def level_critical_values(
    lev_null: np.ndarray, alpha_component: float, run_sided: str
) -> np.ndarray | float:
    """The critical value(s) of the level test in a null sample ``lev_null``.

    Two-sided tests take the ``1 - alpha_component`` quantile of ``|lev_null|``;
    ``"greater"`` takes the upper quantile of ``lev_null`` and ``"less"`` the
    lower one. ``lev_null`` may be one dimensional (one calibration shared by
    every replicate) or two dimensional (one calibration per replicate, over
    axis 1).
    """
    if run_sided not in RUN_SIDES:
        raise ValueError(f"run_sided must be one of {RUN_SIDES}")
    axis = None if lev_null.ndim == 1 else 1
    if run_sided == "two":
        return np.percentile(np.abs(lev_null), 100 * (1 - alpha_component), axis=axis)
    if run_sided == "greater":
        return np.percentile(lev_null, 100 * (1 - alpha_component), axis=axis)
    return np.percentile(lev_null, 100 * alpha_component, axis=axis)


# --------------------------------------------------------------------------- #
# Reliability lambda
# --------------------------------------------------------------------------- #
def lam(a: np.ndarray, n: int) -> float:
    """Reliability lambda: 1 - (mean binomial sampling variance) / (observed variance).

    Bounded to ``[0, 1]``. Returns ``0.0`` when the observed between-cell
    variance is zero (a fully degenerate design carries no reliable signal).
    """
    a = np.asarray(a, dtype=float)
    obs = float(a.var(ddof=1)) if a.size > 1 else 0.0
    if obs <= 0.0:
        return 0.0
    samp = float(np.mean(a * (1.0 - a) / n))
    return float(max(0.0, 1.0 - samp / obs))


def lambda_interval(
    a: np.ndarray, n: int, rng: np.random.Generator, n_boot: int = 10_000
) -> tuple[float, float, float]:
    """Point estimate and nonparametric percentile bootstrap 95% CI for lambda.

    The bootstrap resamples the ``K`` cells with replacement. At small ``K`` the
    interval is wide, which is the point: the screen exposes rather than hides
    the imprecision of a single-run reliability estimate.
    """
    a = np.asarray(a, dtype=float)
    k = a.size
    point = lam(a, n)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, k, k)
        boot[i] = lam(a[idx], n)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Reference associations (numpy-only, no scipy dependency)
# --------------------------------------------------------------------------- #
def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, or ``nan`` if either run is constant.

    ``nan`` mirrors ``safe_cor`` in the registered pipeline: an undefined
    correlation is what the original gate must contend with near a ceiling.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, or ``nan`` if either run is constant."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    ra = _average_rank(a)
    rb = _average_rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def _average_rank(x: np.ndarray) -> np.ndarray:
    """Ranks with ties resolved by their average (matches scipy's default)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)
    # average tied ranks
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    return sums[inv] / counts[inv]


def drift_scales(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> dict[str, float]:
    """Report mean drift on the probability and log-odds scales, plus the odds ratio."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pa, pb = float(a.mean()), float(b.mean())

    def _logit(p: float) -> float:
        p = min(max(p, eps), 1 - eps)
        return math.log(p / (1 - p))

    d_logit = _logit(pb) - _logit(pa)
    return {
        "main_rate": pa,
        "retest_rate": pb,
        "drift_probability": pb - pa,
        "drift_logit": d_logit,
        "odds_ratio": math.exp(d_logit),
        "mean_abs_drift": float(np.mean(np.abs(a - b))),
        "max_abs_drift": float(np.max(np.abs(a - b))),
    }


# --------------------------------------------------------------------------- #
# Design-own bootstrap null for the decomposition
# --------------------------------------------------------------------------- #
def null_p_values(
    z_obs: float,
    q_obs: float,
    z_null: np.ndarray,
    q_null: np.ndarray,
    run_sided: str = "two",
) -> tuple[float, float]:
    """Monte Carlo p-values of the two observed statistics in a null sample.

    This is the one rejection rule of the bootstrap null. :func:`stabilitygate.screen`
    applies it to observed data and :func:`stabilitygate.power` applies it to
    every simulated replicate, so the operating characteristics ``power`` reports
    are those of the rule ``screen`` runs.

    Counts are discrete, so a null draw can reproduce the observed statistic
    exactly. Such a tie is "at least as extreme" and must be counted, but the
    observed and the null statistics are summed in a different order, so a true
    tie can differ in its last bits. A small tolerance keeps ties counted.

    Both p-values use add-one smoothing, ``(1 + count) / (B + 1)``, so a p-value
    is never exactly zero and the test is valid at any number of null draws.
    """
    if run_sided not in RUN_SIDES:
        raise ValueError(f"run_sided must be one of {RUN_SIDES}")
    n_boot = z_null.size
    tol_z = TIE_TOLERANCE * max(1.0, abs(z_obs))
    tol_q = TIE_TOLERANCE * max(1.0, abs(q_obs))
    if run_sided == "two":
        as_extreme = np.abs(z_null) >= abs(z_obs) - tol_z
    elif run_sided == "greater":
        as_extreme = z_null >= z_obs - tol_z
    else:
        as_extreme = z_null <= z_obs + tol_z
    p_run = float((1 + np.sum(as_extreme)) / (n_boot + 1))
    p_int = float((1 + np.sum(q_null >= q_obs - tol_q)) / (n_boot + 1))
    return p_run, p_int


def bootstrap_null_decomposition(
    a: np.ndarray,
    b: np.ndarray,
    n: int,
    rng: np.random.Generator,
    n_boot: int = 20_000,
    run_sided: str = "two",
    level_statistic: str = "woolf_z",
    heterogeneity_statistic: str = "cochran_q",
) -> dict[str, float]:
    """p-values for the run effect and interaction under the design's stable null.

    The null propensity of each cell is the pooled rate across the two runs.
    Both simulated runs are drawn from that same propensity, so any departure of
    the statistics from their null is pure binomial sampling noise: a model
    cannot be more stable than this. The returned p-values are the tail
    probabilities of the observed level and heterogeneity statistics in that null.

    ``run_sided`` selects the tail of the level test. ``"two"`` (default) compares
    absolute values, ``"greater"`` counts null draws at or above the observed
    statistic, and ``"less"`` counts null draws at or below it. Heterogeneity is
    always upper one-sided, because only excess dispersion is evidence of drift.
    """
    if run_sided not in RUN_SIDES:
        raise ValueError(f"run_sided must be one of {RUN_SIDES}")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    z_obs, q_obs = statistics(a, b, n, level_statistic, heterogeneity_statistic)
    pooled = (a + b) / 2.0
    X = rng.binomial(n, pooled, size=(n_boot, a.size)) / n
    Y = rng.binomial(n, pooled, size=(n_boot, a.size)) / n
    z_null, q_null = statistics_batch(
        X, Y, n, level_statistic, heterogeneity_statistic
    )
    p_run, p_int = null_p_values(z_obs, q_obs, z_null, q_null, run_sided)
    return {
        "z_run": z_obs,
        "Q_interaction": q_obs,
        "p_run": p_run,
        "p_interaction": p_int,
    }


# --------------------------------------------------------------------------- #
# Combining the two component tests with the information floor
# --------------------------------------------------------------------------- #
def resolve_outcome(
    run_detected,
    int_detected,
    clears_floor,
    combine: str = "or",
    precedence: str = "resolution_first",
):
    """Turn two test decisions and a floor decision into the three outcomes.

    Works on scalars or on equal-shaped boolean arrays, and returns a triple of
    boolean arrays ``(fail, indeterminate, no_drift_detected)`` that partition
    the input.

    ``combine`` says when the pair of tests rejects. ``"or"`` (default) rejects
    when either component does, ``"and"`` only when both do.

    ``precedence`` says what happens to a case that rejects but sits below the
    information floor. ``"resolution_first"`` (default) evaluates the floor
    first, so indeterminate dominates and no test can rule on an unresolvable
    design. That is the rule of the simulation that produced ``OC_2026_08_28``.
    ``"test_first"`` evaluates the level test before the floor, so a detected run
    effect excludes a model whose design could not otherwise support a ruling.
    At a saturated baseline the choice moves false exclusion by a factor of
    two or more.

    Under ``combine="and"`` no single test is decisive, so the level test cannot
    override the floor on its own and the two precedence settings coincide.
    """
    if combine not in COMBINES:
        raise ValueError(f"combine must be one of {COMBINES}")
    if precedence not in PRECEDENCES:
        raise ValueError(f"precedence must be one of {PRECEDENCES}")
    run_detected = np.asarray(run_detected, dtype=bool)
    int_detected = np.asarray(int_detected, dtype=bool)
    clears_floor = np.asarray(clears_floor, dtype=bool)

    rejected = (
        run_detected | int_detected if combine == "or" else run_detected & int_detected
    )
    if precedence == "test_first" and combine == "or":
        overrides_floor = run_detected
    else:
        overrides_floor = np.zeros_like(run_detected)

    fail = overrides_floor | (clears_floor & rejected)
    indeterminate = (~overrides_floor) & (~clears_floor)
    no_drift = (~fail) & (~indeterminate)
    return fail, indeterminate, no_drift


# --------------------------------------------------------------------------- #
# Registered-gate false exclusion (for the reference comparison)
# --------------------------------------------------------------------------- #
def registered_gate_false_exclusion(
    a: np.ndarray,
    b: np.ndarray,
    n: int,
    rng: np.random.Generator,
    threshold: float = 0.80,
    n_boot: int = 20_000,
    convention: str = "exclude",
) -> float:
    """P(a correlation gate at ``threshold`` fails a perfectly stable model).

    ``convention`` governs a bootstrap draw whose correlation is undefined
    because a simulated run is constant:

    * ``"exclude"``  undefined counts as a failure (the registered behaviour).
    * ``"discard"``  undefined draws are dropped from the denominator.
    * ``"pass"``     undefined counts as a pass (most generous to the gate).
    """
    if convention not in {"exclude", "discard", "pass"}:
        raise ValueError("convention must be 'exclude', 'discard', or 'pass'")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pooled = (a + b) / 2.0
    rs = np.empty(n_boot)
    for i in range(n_boot):
        x = rng.binomial(n, pooled) / n
        y = rng.binomial(n, pooled) / n
        rs[i] = pearson_r(x, y)
    finite = np.isfinite(rs)
    if convention == "exclude":
        return float((~(finite & (rs > threshold))).mean())
    if convention == "pass":
        return float((finite & (rs <= threshold)).mean())
    defined = rs[finite]
    return float((defined <= threshold).mean()) if defined.size else float("nan")
