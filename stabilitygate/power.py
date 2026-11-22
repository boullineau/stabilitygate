"""Design-time operating characteristics and design recommendation.

Before collecting any data, an analyst can ask what the screen will do on the
design they intend to run. :func:`power` maps the operating characteristics of a
planned design across a set of drift scenarios. :func:`recommend_design`
searches a grid of designs for the smallest one that meets a stated
false-exclusion budget and a stated drift-detection power, which automates the
short calibration simulation the paper recommends running at the planned design.

The simulation model matches the paper: per-cell log-odds propensities are drawn
Normal(logit(baseline), sigma), responses are Binomial(n, expit(propensity)),
and each replicate's critical values come from its own perfectly-stable null
(per-replicate calibration), so the exercise is conditioned on the design an
analyst would actually face.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, asdict
from typing import Any, Optional, Sequence

import numpy as np

from . import _core

__all__ = ["PowerRow", "DesignRecommendation", "power", "recommend_design"]

DEFAULT_MODES = ("stable", "shift_0.5", "shift_1.0", "scramble", "idio_0.5")


def _cell_rng(seed: int, *coords: object) -> np.random.Generator:
    """A generator that is a pure function of ``seed`` and the given coordinates.

    Each (baseline, mode, design) cell draws from its own independent stream, so
    operating characteristics do not depend on the order in which cells are
    evaluated. Float coordinates are quantised to micro-units and strings are
    hashed with a stable CRC32 so the key is reproducible across processes.
    """
    key: list[int] = []
    for c in coords:
        if isinstance(c, str):
            key.append(int(zlib.crc32(c.encode()) & 0xFFFFFFFF))
        elif isinstance(c, float):
            # hash a rounded string form so negative floats yield a valid
            # (non-negative) spawn_key entry without collisions
            key.append(int(zlib.crc32(repr(round(c, 6)).encode()) & 0xFFFFFFFF))
        else:
            key.append(int(c) & 0xFFFFFFFF)
    return np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=tuple(key)))


def _calibration_rng(
    seed: int, baseline: float, sigma: float, K: int, n_calls: int
) -> np.random.Generator:
    """The stream behind the oracle null.

    Deliberately a function of the design alone and not of the drift mode, so one
    configuration's critical values are shared by every alternative evaluated
    against it.
    """
    return _cell_rng(seed, "oracle-calibration", float(baseline), float(sigma), K, n_calls)


def _validate_design(K: int, n_calls: int) -> None:
    if not (isinstance(K, (int, np.integer)) and K >= 3):
        raise ValueError("K must be an integer of at least 3 cells")
    if not (isinstance(n_calls, (int, np.integer)) and n_calls > 0):
        raise ValueError("n_calls must be a positive integer")


@dataclass
class PowerRow:
    """Operating characteristics of one design under one drift scenario."""

    baseline: float
    mode: str
    drift_logit: float
    final_pass: float
    final_fail: float
    final_indeterminate: float
    pct_lambda_ge_floor: float
    role: str  # "false_exclusion" for the stable row, "power" for a drift row
    K_cells: int
    n_per_cell: int
    sigma_logit: float
    lambda_floor: float
    alpha: float
    n_rep: int
    # the decision rule that produced this row. power() always sets every one of
    # these fields. The defaults exist only so the fields can carry one.
    alpha_component: float = 0.025
    alpha_split: str = "bonferroni"
    run_sided: str = "two"
    combine: str = "or"
    precedence: str = "resolution_first"
    level_statistic: str = "woolf_z"
    heterogeneity_statistic: str = "cochran_q"
    null: str = "bootstrap"
    lambda_rule: str = "point"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DesignRecommendation:
    """The smallest evaluated design meeting the false-exclusion and power targets."""

    recommended: bool
    K_cells: int | None
    n_per_cell: int | None
    total_calls: int | None
    false_exclusion: float | None
    power: float | None
    indeterminate_under_stable: float | None
    target_false_exclusion: float
    target_power: float
    baseline: float
    drift_logit_that_matters: float
    lambda_floor: float
    message: str
    evaluated: Optional[list[dict]] = None  # every candidate, for transparency

    def __str__(self) -> str:
        if not self.recommended:
            return f"No candidate met the targets. {self.message}"
        return (
            f"Recommended design: K = {self.K_cells} cells, n = {self.n_per_cell} "
            f"calls/cell ({self.total_calls} calls per run).\n"
            f"  false-exclusion of a stable model : {self.false_exclusion:.1%} "
            f"(budget {self.target_false_exclusion:.1%})\n"
            f"  power vs a {self.drift_logit_that_matters:+.2f} log-odds shift : "
            f"{self.power:.1%} (target {self.target_power:.1%})\n"
            f"  abstention (indeterminate) on a stable model : "
            f"{self.indeterminate_under_stable:.1%}"
        )


def _simulate_second_run(
    theta: np.ndarray, mode: str, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Return the retest propensities for a drift ``mode`` and its nominal logit shift."""
    if mode == "stable":
        return theta, 0.0
    if mode.startswith("shift"):
        s = float(mode.split("_")[1])
        return theta + s, s
    if mode == "scramble":
        idx = np.argsort(rng.random(theta.shape), axis=1)
        return np.take_along_axis(theta, idx, axis=1), 0.0
    if mode.startswith("idio"):
        s = float(mode.split("_")[1])
        return theta + rng.normal(0.0, s, theta.shape), 0.0
    raise ValueError(f"unknown drift mode: {mode!r}")


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _lambda_rows(A: np.ndarray, n: int) -> np.ndarray:
    """Reliability lambda for every row of ``A``."""
    obs = A.var(axis=1, ddof=1)
    samp = (A * (1 - A) / n).mean(axis=1)
    return np.where(obs > 0, np.maximum(0.0, 1 - samp / np.where(obs > 0, obs, 1)), 0.0)


def _lambda_lower_rows(
    A: np.ndarray, n: int, rng: np.random.Generator, n_boot_lambda: int
) -> np.ndarray:
    """Lower bound of lambda's 95% percentile bootstrap interval, per row of ``A``.

    Resamples the ``K`` cells with replacement, matching
    :func:`stabilitygate._core.lambda_interval`. Replicates are processed in
    chunks so the intermediate array stays small at large ``n_rep``.
    """
    n_rep, K = A.shape
    out = np.empty(n_rep)
    chunk = max(1, int(2_000_000 // max(1, n_boot_lambda * K)))
    for start in range(0, n_rep, chunk):
        block = A[start : start + chunk]
        idx = rng.integers(0, K, size=(block.shape[0], n_boot_lambda, K))
        resampled = np.take_along_axis(block[:, None, :], idx, axis=2)
        out[start : start + block.shape[0]] = np.percentile(
            _lambda_rows(resampled.reshape(-1, K), n).reshape(block.shape[0], -1),
            2.5,
            axis=1,
        )
    return out


def _oracle_critical_values(
    baseline: float,
    K: int,
    n: int,
    sigma: float,
    alpha_component: float,
    run_sided: str,
    level_statistic: str,
    heterogeneity_statistic: str,
    n_cal: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Critical values from the design's own stable null at the TRUE parameters.

    The oracle null draws fresh propensities from ``Normal(logit(baseline),
    sigma)`` rather than bootstrapping the observed pooled cell rates, so it
    knows the design the way only a simulation can. It is a single pair of
    critical values shared by every replicate and every drift mode of one
    ``(baseline, sigma, K, n)`` configuration, which is what the run behind
    ``presets.OC_2026_08_28`` did.
    """
    mu = math.log(baseline / (1.0 - baseline))
    theta = rng.normal(mu, sigma, (n_cal, K))
    p = _expit(theta)
    A = rng.binomial(n, p) / n
    B = rng.binomial(n, p) / n
    lev_null, het_null = _core.statistics_batch(
        A, B, n, level_statistic, heterogeneity_statistic
    )
    lev_crit = float(
        _core.level_critical_values(lev_null, alpha_component, run_sided)
    )
    het_crit = float(np.percentile(het_null, 100 * (1 - alpha_component)))
    return lev_crit, het_crit


def _operating_point(
    baseline: float,
    mode: str,
    K: int,
    n: int,
    sigma: float,
    lambda_floor: float,
    alpha: float,
    n_rep: int,
    n_boot: int,
    rng: np.random.Generator,
    *,
    alpha_split: str = "bonferroni",
    run_sided: str = "two",
    combine: str = "or",
    precedence: str = "resolution_first",
    level_statistic: str = "woolf_z",
    heterogeneity_statistic: str = "cochran_q",
    null: str = "bootstrap",
    lambda_rule: str = "point",
    n_boot_lambda: int = 1_000,
    calibration_rng: Optional[np.random.Generator] = None,
) -> dict[str, float]:
    """Monte Carlo outcome rates for one (design, scenario).

    ``rng`` drives the replicates and, under ``null="bootstrap"``, their
    per-replicate calibration. ``calibration_rng`` drives the oracle calibration
    and must be a pure function of ``(baseline, sigma, K, n)`` alone, so that the
    critical values one configuration uses do not depend on which drift mode is
    being evaluated.
    """
    alpha_component = _core.component_alpha(alpha, alpha_split)
    if null not in _core.NULLS:
        raise ValueError(f"null must be one of {_core.NULLS}")
    if lambda_rule not in {"point", "lower"}:
        raise ValueError("lambda_rule must be 'point' or 'lower'")

    mu = math.log(baseline / (1.0 - baseline))
    theta = rng.normal(mu, sigma, (n_rep, K))
    theta2, drift_logit = _simulate_second_run(theta, mode, rng)

    A = rng.binomial(n, _expit(theta)) / n
    B = rng.binomial(n, _expit(theta2)) / n
    z, q = _core.statistics_batch(A, B, n, level_statistic, heterogeneity_statistic)

    if null == "oracle":
        if calibration_rng is None:
            raise ValueError("null='oracle' requires a calibration_rng")
        z_crit_scalar, q_crit_scalar = _oracle_critical_values(
            baseline, K, n, sigma, alpha_component, run_sided,
            level_statistic, heterogeneity_statistic, n_boot, calibration_rng,
        )
        run_sig = _core.reject_level(z, z_crit_scalar, run_sided)
        int_sig = q > q_crit_scalar
    else:
        # per-replicate calibration: each replicate is tested against its own
        # pooled-rate stable null with the rule screen() applies to observed
        # data, an add-one Monte Carlo p-value compared with the component level.
        pooled = (A + B) / 2.0
        run_sig = np.zeros(n_rep, dtype=bool)
        int_sig = np.zeros(n_rep, dtype=bool)
        # A replicate below the information floor is indeterminate whatever the
        # tests say, unless the level test is allowed to rule first. Its null
        # sample is then never used, so it is not drawn.
        tests_can_rule_below_floor = precedence == "test_first" and combine == "or"
        if lambda_rule == "point" and not tests_can_rule_below_floor:
            needed = np.flatnonzero(_lambda_rows(A, n) >= lambda_floor)
        else:
            needed = np.arange(n_rep)
        for i in needed:
            p = pooled[i]
            Xb = rng.binomial(n, p, size=(n_boot, K)) / n
            Yb = rng.binomial(n, p, size=(n_boot, K)) / n
            zb, qb = _core.statistics_batch(
                Xb, Yb, n, level_statistic, heterogeneity_statistic
            )
            p_run, p_int = _core.null_p_values(z[i], q[i], zb, qb, run_sided)
            run_sig[i] = p_run < alpha_component
            int_sig[i] = p_int < alpha_component

    # reliability per replicate
    lam = _lambda_rows(A, n)
    if lambda_rule == "lower":
        lam_decides = _lambda_lower_rows(A, n, rng, n_boot_lambda)
    else:
        lam_decides = lam

    lambda_ok = lam_decides >= lambda_floor

    final_fail, final_indeterminate, final_pass = _core.resolve_outcome(
        run_sig, int_sig, lambda_ok, combine, precedence
    )

    return {
        "final_pass": float(final_pass.mean()),
        "final_fail": float(final_fail.mean()),
        "final_indeterminate": float(final_indeterminate.mean()),
        "pct_lambda_ge_floor": float(lambda_ok.mean()),
        "drift_logit": float(drift_logit),
    }


def power(
    K: int,
    n_calls: int,
    *,
    baseline: float | Sequence[float] = 0.90,
    sigma_logit: float = 0.50,
    modes: Sequence[str] = DEFAULT_MODES,
    lambda_floor: float = 0.50,
    alpha: float = 0.05,
    alpha_split: str = "bonferroni",
    run_sided: str = "two",
    combine: str = "or",
    precedence: str = "resolution_first",
    level_statistic: str = "woolf_z",
    heterogeneity_statistic: str = "cochran_q",
    null: str = "bootstrap",
    lambda_rule: str = "point",
    n_rep: int = 1500,
    n_boot: int = 300,
    n_boot_lambda: int = 1_000,
    seed: int = 20260101,
) -> list[PowerRow]:
    """Map the operating characteristics of a planned design.

    Parameters
    ----------
    K, n_calls : cells and calls per cell in the planned design.
    baseline : expected response rate, or an iterable of baselines to sweep.
    sigma_logit : expected between-cell spread of propensities, in log-odds.
    modes : drift scenarios. ``"stable"`` gives the false-exclusion and
        abstention rates; ``"shift_x"`` a uniform x-log-odds shift; ``"scramble"``
        a rank permutation; ``"idio_x"`` independent per-cell noise of scale x.
    lambda_floor, alpha : screen settings to evaluate.
    alpha_split, run_sided, combine, precedence, level_statistic,
    heterogeneity_statistic, lambda_rule : the screen's decision rule, with the
        same meanings as in :func:`stabilitygate.screen`.
    null : ``"bootstrap"`` (default) calibrates each replicate against a
        parametric bootstrap from its own observed pooled rates, which is what an
        analyst holding data can do. ``"oracle"`` calibrates once per
        ``(baseline, sigma, K, n)`` configuration by drawing from the true
        parameters, which is available only in simulation and is what the run
        behind ``presets.OC_2026_08_28`` did. The oracle critical values are
        shared across drift modes, so they do not depend on the alternative being
        evaluated.
    n_rep, n_boot : Monte Carlo replicates and null draws. Under
        ``null="bootstrap"`` ``n_boot`` is the per-replicate bootstrap size;
        under ``null="oracle"`` it is the size of the one shared calibration
        sample. Larger is more precise and slower.
    n_boot_lambda : bootstrap draws behind ``lambda_rule="lower"``. Ignored under
        the default ``lambda_rule="point"``.
    seed : base seed; all randomness derives from it.

    Returns
    -------
    list[PowerRow]
        One row per (baseline, mode). For the ``"stable"`` mode, ``final_fail``
        is the false-exclusion rate and ``final_indeterminate`` the abstention
        rate. For a drift mode, ``final_fail`` is the detection power.

    Notes
    -----
    Under ``null="bootstrap"`` each replicate is decided by the rule
    :func:`screen` applies to observed data, an add-one-smoothed Monte Carlo
    p-value compared with the component level. The only difference is the number
    of null draws, ``n_boot`` here against 20,000 in ``screen``. With add-one
    smoothing the test is valid at any ``n_boot`` and slightly conservative at a
    small one. At the default 300 each component test runs at 7 / 301 = .023
    where ``screen`` runs it at .025. Under ``null="oracle"`` replicates are
    decided against the shared empirical critical values.
    """
    _validate_design(K, n_calls)
    baselines = [baseline] if isinstance(baseline, (int, float)) else list(baseline)
    rows: list[PowerRow] = []
    rule = dict(
        alpha_split=alpha_split,
        run_sided=run_sided,
        combine=combine,
        precedence=precedence,
        level_statistic=level_statistic,
        heterogeneity_statistic=heterogeneity_statistic,
        null=null,
        lambda_rule=lambda_rule,
        n_boot_lambda=n_boot_lambda,
    )
    for pb in baselines:
        if not 0.0 < pb < 1.0:
            raise ValueError("baseline must lie strictly in (0, 1)")
        for mode in modes:
            rng = _cell_rng(seed, float(pb), mode, K, n_calls)
            op = _operating_point(
                pb, mode, K, n_calls, sigma_logit, lambda_floor, alpha, n_rep, n_boot, rng,
                calibration_rng=_calibration_rng(seed, pb, sigma_logit, K, n_calls),
                **rule,
            )
            role = "false_exclusion" if mode == "stable" else "power"
            rows.append(
                PowerRow(
                    baseline=float(pb),
                    mode=mode,
                    drift_logit=op["drift_logit"],
                    final_pass=round(op["final_pass"], 4),
                    final_fail=round(op["final_fail"], 4),
                    final_indeterminate=round(op["final_indeterminate"], 4),
                    pct_lambda_ge_floor=round(op["pct_lambda_ge_floor"], 4),
                    role=role,
                    K_cells=K,
                    n_per_cell=n_calls,
                    sigma_logit=sigma_logit,
                    lambda_floor=lambda_floor,
                    alpha=alpha,
                    n_rep=n_rep,
                    alpha_component=_core.component_alpha(alpha, alpha_split),
                    alpha_split=alpha_split,
                    run_sided=run_sided,
                    combine=combine,
                    precedence=precedence,
                    level_statistic=level_statistic,
                    heterogeneity_statistic=heterogeneity_statistic,
                    null=null,
                    lambda_rule=lambda_rule,
                )
            )
    return rows


def recommend_design(
    *,
    baseline: float,
    drift_logit_that_matters: float,
    sigma_logit: float = 0.50,
    target_false_exclusion: float = 0.05,
    target_power: float = 0.80,
    lambda_floor: float = 0.50,
    alpha: float = 0.05,
    alpha_split: str = "bonferroni",
    run_sided: str = "two",
    combine: str = "or",
    precedence: str = "resolution_first",
    level_statistic: str = "woolf_z",
    heterogeneity_statistic: str = "cochran_q",
    null: str = "bootstrap",
    lambda_rule: str = "point",
    candidate_K: Sequence[int] = (10, 14, 20, 28, 40),
    candidate_n: Sequence[int] = (20, 30, 50, 80),
    n_rep: int = 1200,
    n_boot: int = 250,
    n_boot_lambda: int = 1_000,
    seed: int = 20260101,
) -> DesignRecommendation:
    """Search designs for the smallest that meets a false-exclusion and power target.

    For each candidate ``(K, n)`` the routine estimates two operating points at
    the expected ``baseline``: the false-exclusion rate on a stable model, and
    the detection power against a uniform shift of ``drift_logit_that_matters``.
    It returns the design with the fewest total calls per run
    (``K * n``) whose false-exclusion is at or below ``target_false_exclusion``
    and whose power is at or above ``target_power``.

    This is the "run a short simulation at your planned design" step, automated:
    it turns the screen from a template into a concrete design decision.
    """
    if not 0.0 < baseline < 1.0:
        raise ValueError("baseline must lie strictly in (0, 1)")
    for K in candidate_K:
        _validate_design(K, min(candidate_n))
    for n in candidate_n:
        _validate_design(min(candidate_K), n)
    evaluated: list[dict] = []
    feasible: list[dict] = []
    rule = dict(
        alpha_split=alpha_split,
        run_sided=run_sided,
        combine=combine,
        precedence=precedence,
        level_statistic=level_statistic,
        heterogeneity_statistic=heterogeneity_statistic,
        null=null,
        lambda_rule=lambda_rule,
        n_boot_lambda=n_boot_lambda,
    )

    shift_mode = f"shift_{drift_logit_that_matters}"
    designs = sorted(
        ((K, n) for K in candidate_K for n in candidate_n),
        key=lambda kn: kn[0] * kn[1],
    )
    for K, n in designs:
        # a fresh generator each time, so both calls see the same critical values
        stable = _operating_point(
            baseline, "stable", K, n, sigma_logit, lambda_floor, alpha, n_rep, n_boot,
            _cell_rng(seed, float(baseline), "stable", K, n),
            calibration_rng=_calibration_rng(seed, baseline, sigma_logit, K, n), **rule,
        )
        drift = _operating_point(
            baseline, shift_mode, K, n, sigma_logit, lambda_floor, alpha, n_rep, n_boot,
            _cell_rng(seed, float(baseline), shift_mode, K, n),
            calibration_rng=_calibration_rng(seed, baseline, sigma_logit, K, n), **rule,
        )
        rec = {
            "K_cells": K,
            "n_per_cell": n,
            "total_calls": K * n,
            "false_exclusion": round(stable["final_fail"], 4),
            "indeterminate_under_stable": round(stable["final_indeterminate"], 4),
            "power": round(drift["final_fail"], 4),
            "meets_false_exclusion": stable["final_fail"] <= target_false_exclusion,
            "meets_power": drift["final_fail"] >= target_power,
        }
        evaluated.append(rec)
        if rec["meets_false_exclusion"] and rec["meets_power"]:
            feasible.append(rec)

    if feasible:
        best = min(feasible, key=lambda r: r["total_calls"])
        return DesignRecommendation(
            recommended=True,
            K_cells=best["K_cells"],
            n_per_cell=best["n_per_cell"],
            total_calls=best["total_calls"],
            false_exclusion=best["false_exclusion"],
            power=best["power"],
            indeterminate_under_stable=best["indeterminate_under_stable"],
            target_false_exclusion=target_false_exclusion,
            target_power=target_power,
            baseline=baseline,
            drift_logit_that_matters=drift_logit_that_matters,
            lambda_floor=lambda_floor,
            message="A design in the candidate grid met both targets.",
            evaluated=evaluated,
        )

    # nothing feasible: report the closest so the caller can widen the grid
    closest = max(
        evaluated,
        key=lambda r: (r["meets_false_exclusion"], r["power"]),
    )
    return DesignRecommendation(
        recommended=False,
        K_cells=None,
        n_per_cell=None,
        total_calls=None,
        false_exclusion=None,
        power=None,
        indeterminate_under_stable=None,
        target_false_exclusion=target_false_exclusion,
        target_power=target_power,
        baseline=baseline,
        drift_logit_that_matters=drift_logit_that_matters,
        lambda_floor=lambda_floor,
        message=(
            "No candidate met both targets. Widen candidate_K / candidate_n, "
            "relax the targets, or reconsider whether a shift this small is "
            f"resolvable near baseline {baseline:.2f}. Closest was "
            f"K={closest['K_cells']}, n={closest['n_per_cell']} with "
            f"false-exclusion {closest['false_exclusion']:.1%} and power "
            f"{closest['power']:.1%}."
        ),
        evaluated=evaluated,
    )
