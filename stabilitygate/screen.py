"""The three-outcome stability screen applied to observed data."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from . import _core

__all__ = [
    "ScreenResult",
    "screen",
    "screen_dataframe",
    "NO_DRIFT_DETECTED",
    "FAIL",
    "INDETERMINATE",
    "VERDICTS",
]

#: The admitting verdict. It is deliberately not called PASS. The screen
#: calibrates its two component tests against the design's own no-drift null, so
#: this verdict is a failure to reject no drift, not a rejection of consequential
#: drift. Those are different criteria with different error-control properties,
#: and the weaker one is the one implemented here. See README section "What this
#: is not".
NO_DRIFT_DETECTED = "no_drift_detected"
#: Drift was detected, by the level test or the heterogeneity test or both.
FAIL = "FAIL"
#: The design carried too little reliable between-cell information to rule.
INDETERMINATE = "INDETERMINATE"
#: Every verdict the screen can return.
VERDICTS = (NO_DRIFT_DETECTED, FAIL, INDETERMINATE)


@dataclass
class ScreenResult:
    """The verdict of the stability screen and every diagnostic behind it.

    Attributes
    ----------
    verdict : one of ``"no_drift_detected"``, ``"FAIL"``, ``"INDETERMINATE"``.
        ``"no_drift_detected"`` is not called PASS on purpose, because it
        certifies a failure to reject a no-drift null rather than a rejection of
        consequential drift.
    reason : a short human-readable explanation of the verdict.
    lambda_point, lambda_ci_lo, lambda_ci_hi : reliability and its bootstrap 95% CI.
    clears_floor : whether the design met the information floor under the chosen rule.
    z_run, p_run : signed run-effect statistic and its bootstrap p-value. The
        statistic is the Woolf z unless ``level_statistic`` says otherwise.
    Q_interaction, p_interaction : cell-by-run heterogeneity statistic and its
        p-value. The statistic is Cochran's Q unless ``heterogeneity_statistic``
        says otherwise.
    alpha_component : the level each component test actually ran at, after
        ``alpha_split`` divided ``alpha`` between them.
    pearson_r, spearman_rho : reference associations (NOT used by the verdict).
    registered_gate_verdict : what a bare ``pearson_r > gate_threshold`` gate would
        return. This one keeps the labels ``"PASS"`` and ``"FAIL"`` because it
        describes the incumbent gate's own behaviour, not this screen's verdict.
    p_fail_if_stable : optional. P(that bare gate fails a perfectly stable model),
        the false-exclusion rate of the correlation gate on this design.
    run_detected_below_floor : ``True`` when the run-effect test rejected on a
        design below the information floor. Under ``precedence="resolution_first"``
        the verdict is then INDETERMINATE, and this flag records that
        ``precedence="test_first"`` would have returned FAIL on the same data.
    """

    verdict: str
    reason: str
    lambda_point: float
    lambda_ci_lo: float
    lambda_ci_hi: float
    clears_floor: bool
    z_run: float
    p_run: float
    Q_interaction: float
    p_interaction: float
    pearson_r: float
    spearman_rho: float
    drift_probability: float
    drift_logit: float
    odds_ratio: float
    mean_abs_drift: float
    max_abs_drift: float
    registered_gate_verdict: str
    K_cells: int
    n_per_cell: int
    lambda_floor: float
    alpha: float
    lambda_rule: str
    seed: int
    # the decision rule that produced this result. screen() always sets every
    # one of these fields. The defaults exist only so the fields can carry one.
    alpha_component: float = 0.025
    alpha_split: str = "bonferroni"
    run_sided: str = "two"
    combine: str = "or"
    precedence: str = "resolution_first"
    level_statistic: str = "woolf_z"
    heterogeneity_statistic: str = "cochran_q"
    null: str = "bootstrap"
    p_fail_if_stable: float | None = None
    run_detected_below_floor: bool = False
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Flat dictionary of every field, convenient for CSV or JSON export."""
        d = asdict(self)
        settings = d.pop("settings", {}) or {}
        # keep the settings as flat columns so an exported result is complete
        for key in ("gate_threshold", "n_boot"):
            if key in settings:
                d[key] = settings[key]
        return d

    def _fmt_p(self, p: float) -> str:
        n_boot = self.settings.get("n_boot", 20_000)
        floor = 1.0 / (n_boot + 1)
        if p <= floor:
            return f"<{floor:.1e}"
        return f"{p:.4f}"

    def __str__(self) -> str:
        ci = f"[{self.lambda_ci_lo:.2f}, {self.lambda_ci_hi:.2f}]"
        lines = [
            f"Stability screen verdict: {self.verdict}",
            f"  reason              : {self.reason}",
            f"  reliability lambda  : {self.lambda_point:.3f}  95% CI {ci}  "
            f"(floor {self.lambda_floor:.2f}, rule '{self.lambda_rule}')",
            f"  run effect  {self.level_statistic:<11}: {self.z_run:+.2f}  "
            f"(p = {self._fmt_p(self.p_run)}, {self.run_sided}-sided)",
            f"  interaction {self.heterogeneity_statistic:<11}: {self.Q_interaction:.1f}   "
            f"(p = {self._fmt_p(self.p_interaction)})",
            f"  decision rule       : each test at alpha = {self.alpha_component:.4f} "
            f"('{self.alpha_split}' split of {self.alpha:.3f}), combined with "
            f"'{self.combine}', precedence '{self.precedence}'",
            f"  drift               : {self.drift_probability:+.3f} on probability, "
            f"{self.drift_logit:+.2f} on log-odds (OR {self.odds_ratio:.2f})",
            f"  reference Pearson r : {self.pearson_r:+.3f}  "
            f"-> a bare r>{self.settings.get('gate_threshold', 0.80):.2f} gate would say "
            f"{self.registered_gate_verdict}",
        ]
        if self.p_fail_if_stable is not None:
            lines.append(
                f"  that bare gate would falsely exclude a perfectly stable model "
                f"{self.p_fail_if_stable * 100:.1f}% of the time"
            )
        if self.run_detected_below_floor:
            lines.append(
                "  flag                : the run-effect test rejected on a design "
                "below the information floor"
            )
        if self.verdict == NO_DRIFT_DETECTED:
            lines.append(
                "  note                : 'no_drift_detected' is a failure to reject "
                "a no-drift null, not a rejection of consequential drift"
            )
        return "\n".join(lines)


def screen(
    main_rates: np.ndarray | list,
    retest_rates: np.ndarray | list,
    n_calls: int,
    *,
    input_kind: str = "auto",
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
    gate_threshold: float = 0.80,
    n_boot: int = 20_000,
    n_boot_lambda: int = 10_000,
    compute_gate_false_exclusion: bool = False,
    seed: int = 20260101,
) -> ScreenResult:
    """Run the three-outcome stability screen on one model's matched cells.

    Under the defaults the screen asks three questions in a fixed order and
    returns the first decisive answer:

    1. Did the design carry enough reliable between-cell information? If
       reliability lambda is below the floor, return **INDETERMINATE**. The
       design cannot resolve the question, so the screen declines to rule rather
       than making a claim about the model. If the run-effect test rejected on
       such a design, the result says so in ``run_detected_below_floor`` and in
       ``reason``. Set ``precedence="test_first"`` to let a detected run effect
       return FAIL below the floor instead.
    2. Did the model's overall propensity shift? If the signed run effect is
       detected, return **FAIL**. This is the drift a rank-order correlation can
       miss.
    3. Did particular cells move differently once the common shift is removed?
       If the interaction is detected, return **FAIL**. Otherwise return
       **no_drift_detected**.

    The admitting verdict is called ``"no_drift_detected"`` rather than PASS
    because both component tests are calibrated against the design's own
    no-drift null. Admission is therefore a failure to reject no drift, which is
    a weaker statement than a rejection of consequential drift. This screen is
    not an equivalence test.

    Parameters
    ----------
    main_rates, retest_rates : per-cell response rates (or success counts) for
        the two runs, one entry per matched cell.
    n_calls : calls per cell in each run.
    lambda_floor : the preregistered information floor. ``0.50`` is illustrative,
        not a universal constant; choose it with :func:`stabilitygate.power` at
        your planned design.
    alpha : the union detection level across the two component tests.
    alpha_split : how ``alpha`` is divided between them. ``"bonferroni"``
        (default) runs each at ``alpha / 2``, so the union of two positively
        correlated tests lands near ``alpha``. ``"sidak"`` runs each at
        ``1 - (1 - alpha) ** 0.5``. ``"none"`` runs each at the full ``alpha``,
        which puts the union near ``2 * alpha``.
    run_sided : the tail of the level test. ``"two"`` (default) detects a shift
        in either direction. ``"greater"`` and ``"less"`` detect only an increase
        or only a decrease in the retest run, and are the right choice when the
        direction that would matter is fixed in advance.
    combine : how the two component tests combine into one exclusion.
        ``"or"`` (default) excludes when either rejects. ``"and"`` excludes only
        when both do, which lowers false exclusion and lowers power.
    precedence : what happens to a case that rejects but sits below the
        information floor. ``"resolution_first"`` (default) evaluates the floor
        first, so INDETERMINATE dominates. It is the rule used by the simulation
        behind ``stabilitygate.presets.OC_2026_08_28``. ``"test_first"``
        evaluates the level test before the floor, so a detected run effect
        excludes the model anyway. The choice between them moves false exclusion
        by a factor of two or more at a saturated baseline.
    level_statistic : ``"woolf_z"`` (default) for the inverse-variance weighted
        mean per-cell log odds ratio scaled to a z, or ``"pooled_logit"`` for the
        difference of the logits of the two runs' mean rates.
    heterogeneity_statistic : ``"cochran_q"`` (default) for weighted
        heterogeneity about the common shift, or ``"pearson_chi2"`` for a Pearson
        chi-square on the raw per-cell differences.
    null : the null the component tests are calibrated against. Only
        ``"bootstrap"`` is available here. The ``"oracle"`` null of
        :func:`stabilitygate.power` needs the true design parameters, which an
        analyst holding data does not have.
    lambda_rule : how the floor decision handles the sampling uncertainty in
        lambda itself. ``"point"`` compares the point estimate to the floor
        (matches the paper's headline). ``"lower"`` compares the lower bound of
        lambda's bootstrap CI to the floor, so a design is credited with enough
        information only when even the pessimistic reliability estimate clears
        the floor. ``"lower"`` returns INDETERMINATE more often and is the
        conservative choice.
    gate_threshold : the correlation threshold of the bare gate used only for the
        reference comparison.
    compute_gate_false_exclusion : if ``True``, also estimate the false-exclusion
        rate of the bare correlation gate on this design (an extra bootstrap).
    seed : base seed. All randomness is derived deterministically from it, so a
        given call is fully reproducible and independent of call order.

    Returns
    -------
    ScreenResult
    """
    if lambda_rule not in {"point", "lower"}:
        raise ValueError("lambda_rule must be 'point' or 'lower'")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly in (0, 1)")
    if not 0.0 <= lambda_floor <= 1.0:
        raise ValueError("lambda_floor must lie in [0, 1]")
    if not (isinstance(n_calls, (int, np.integer)) and n_calls > 0):
        raise ValueError("n_calls must be a positive integer")
    if run_sided not in _core.RUN_SIDES:
        raise ValueError(f"run_sided must be one of {_core.RUN_SIDES}")
    if combine not in _core.COMBINES:
        raise ValueError(f"combine must be one of {_core.COMBINES}")
    if precedence not in _core.PRECEDENCES:
        raise ValueError(f"precedence must be one of {_core.PRECEDENCES}")
    if level_statistic not in _core.LEVEL_STATISTICS:
        raise ValueError(f"level_statistic must be one of {_core.LEVEL_STATISTICS}")
    if heterogeneity_statistic not in _core.HETEROGENEITY_STATISTICS:
        raise ValueError(
            f"heterogeneity_statistic must be one of {_core.HETEROGENEITY_STATISTICS}"
        )
    if null not in _core.NULLS:
        raise ValueError(f"null must be one of {_core.NULLS}")
    if null == "oracle":
        raise ValueError(
            "null='oracle' draws from the true (baseline, sigma, K, n) of a "
            "simulated design and is available only in stabilitygate.power(). "
            "On observed data use null='bootstrap'."
        )
    # raises on an unknown alpha_split
    alpha_component = _core.component_alpha(alpha, alpha_split)

    a = _core.as_rates(main_rates, n_calls, input_kind)
    b = _core.as_rates(retest_rates, n_calls, input_kind)
    if a.shape != b.shape:
        raise ValueError("main_rates and retest_rates must have the same length")
    k = a.size
    if k < 3:
        raise ValueError("need at least 3 matched cells to run the screen")

    # Independent, order-stable child streams for each Monte Carlo step.
    ss = np.random.SeedSequence(seed)
    rng_lambda, rng_null, rng_gate = (np.random.default_rng(s) for s in ss.spawn(3))

    lam_point, lam_lo, lam_hi = _core.lambda_interval(a, n_calls, rng_lambda, n_boot_lambda)
    dec = _core.bootstrap_null_decomposition(
        a,
        b,
        n_calls,
        rng_null,
        n_boot,
        run_sided=run_sided,
        level_statistic=level_statistic,
        heterogeneity_statistic=heterogeneity_statistic,
    )
    drift = _core.drift_scales(a, b)
    r = _core.pearson_r(a, b)
    rho = _core.spearman_rho(a, b)

    clears_floor = (lam_lo if lambda_rule == "lower" else lam_point) >= lambda_floor
    run_detected = dec["p_run"] < alpha_component
    int_detected = dec["p_interaction"] < alpha_component

    p_floor = 1.0 / (n_boot + 1)

    def _pfmt(p: float) -> str:
        return f"<{p_floor:.1e}" if p <= p_floor else f"{p:.4f}"

    fail, indeterminate, _ = _core.resolve_outcome(
        run_detected, int_detected, clears_floor, combine, precedence
    )
    lam_used = lam_lo if lambda_rule == "lower" else lam_point
    lam_label = "lambda lower CI" if lambda_rule == "lower" else "lambda"
    run_text = (
        f"run effect {level_statistic} = {dec['z_run']:+.2f}, "
        f"p = {_pfmt(dec['p_run'])}"
    )
    int_text = (
        f"interaction {heterogeneity_statistic} = {dec['Q_interaction']:.1f}, "
        f"p = {_pfmt(dec['p_interaction'])}"
    )

    if bool(fail):
        verdict = FAIL
        # every component named here rejected, so "and" reads correctly whether
        # one of them or both of them did
        detected = [t for t, hit in ((run_text, run_detected), (int_text, int_detected)) if hit]
        reason = (
            f"drift detected at alpha = {alpha_component:.4f} per test "
            f"({' and '.join(detected)})"
        )
        if not clears_floor:
            reason += (
                f", and precedence '{precedence}' lets the level test rule although "
                f"{lam_label} = {lam_used:.2f} is below the floor {lambda_floor:.2f}"
            )
    elif bool(indeterminate):
        verdict = INDETERMINATE
        reason = (
            f"design carries too little reliable between-cell information "
            f"({lam_label} = {lam_used:.2f} < floor {lambda_floor:.2f})"
        )
        if precedence == "test_first" and combine == "or":
            reason += " and no run effect was detected"
        elif run_detected:
            reason += (
                f". The run-effect test rejected ({run_text}), and precedence "
                f"'{precedence}' does not let it rule below the floor"
            )
    else:
        verdict = NO_DRIFT_DETECTED
        reason = (
            f"no drift detected at alpha = {alpha_component:.4f} per test and "
            f"{lam_label} = {lam_used:.2f} clears the floor {lambda_floor:.2f}. "
            "This is a failure to reject no drift, not a rejection of "
            "consequential drift"
        )

    reg_verdict = (
        "PASS" if (np.isfinite(r) and r > gate_threshold) else "FAIL"
    )

    p_fail = None
    if compute_gate_false_exclusion:
        p_fail = _core.registered_gate_false_exclusion(
            a, b, n_calls, rng_gate, threshold=gate_threshold, n_boot=n_boot
        )

    return ScreenResult(
        verdict=verdict,
        reason=reason,
        lambda_point=lam_point,
        lambda_ci_lo=lam_lo,
        lambda_ci_hi=lam_hi,
        clears_floor=clears_floor,
        z_run=dec["z_run"],
        p_run=dec["p_run"],
        Q_interaction=dec["Q_interaction"],
        p_interaction=dec["p_interaction"],
        pearson_r=r,
        spearman_rho=rho,
        drift_probability=drift["drift_probability"],
        drift_logit=drift["drift_logit"],
        odds_ratio=drift["odds_ratio"],
        mean_abs_drift=drift["mean_abs_drift"],
        max_abs_drift=drift["max_abs_drift"],
        registered_gate_verdict=reg_verdict,
        K_cells=k,
        n_per_cell=int(n_calls),
        lambda_floor=lambda_floor,
        alpha=alpha,
        lambda_rule=lambda_rule,
        seed=seed,
        alpha_component=alpha_component,
        alpha_split=alpha_split,
        run_sided=run_sided,
        combine=combine,
        precedence=precedence,
        level_statistic=level_statistic,
        heterogeneity_statistic=heterogeneity_statistic,
        null=null,
        p_fail_if_stable=p_fail,
        run_detected_below_floor=bool(run_detected and not clears_floor),
        settings={"gate_threshold": gate_threshold, "n_boot": n_boot},
    )


def screen_dataframe(
    rows: list[dict],
    *,
    group_key: str = "model",
    main_col: str = "main_rate",
    retest_col: str = "stability_rate",
    n_col: str = "main_n",
    **kwargs: Any,
) -> dict[str, ScreenResult]:
    """Run :func:`screen` once per group in a list-of-dicts table.

    Convenience for the common case of a long-format CSV with one row per cell
    and a column identifying the model. Returns a mapping from group value to
    :class:`ScreenResult`.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row[group_key]), []).append(row)

    out: dict[str, ScreenResult] = {}
    for key, grp in groups.items():
        main = [float(r[main_col]) for r in grp]
        retest = [float(r[retest_col]) for r in grp]
        n_vals = {int(float(r[n_col])) for r in grp}
        if len(n_vals) != 1:
            raise ValueError(
                f"group '{key}' has non-constant {n_col}: {sorted(n_vals)}; "
                "the screen assumes equal calls per cell"
            )
        n = n_vals.pop()
        # if a separate retest-n column is present, it must match: the screen
        # assumes the two runs share the per-cell call budget.
        retest_n_col = "stability_n"
        if grp and retest_n_col in grp[0]:
            retest_ns = {int(float(r[retest_n_col])) for r in grp}
            if retest_ns != {n}:
                raise ValueError(
                    f"group '{key}' has {retest_n_col} {sorted(retest_ns)} that does "
                    f"not match {n_col} = {n}; the screen assumes equal calls per "
                    "cell across both runs"
                )
        out[key] = screen(main, retest, n, **kwargs)
    return out
