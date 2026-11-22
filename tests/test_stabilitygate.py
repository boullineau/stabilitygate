"""Test suite for stabilitygate.

Runs with pytest. The headline test reproduces the four-model verdicts of
the method paper's worked case. The remainder pin down the decision logic,
determinism, input handling, and the design-calibration behaviour.
"""

from __future__ import annotations

import csv
import math
from importlib import resources

import numpy as np
import pytest

from stabilitygate import (
    NO_DRIFT_DETECTED,
    VERDICTS,
    ScreenResult,
    power,
    recommend_design,
    screen,
    screen_dataframe,
)
from stabilitygate import _core


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def four_model_rows() -> list[dict]:
    with resources.files("stabilitygate.data").joinpath(
        "four_model_stability.csv"
    ).open("r") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def four_model_results(four_model_rows) -> dict[str, ScreenResult]:
    return screen_dataframe(four_model_rows, compute_gate_false_exclusion=True)


# --------------------------------------------------------------------------- #
# Headline reproduction
# --------------------------------------------------------------------------- #
EXPECTED_VERDICT = {
    "claude-opus-4-7": "FAIL",
    "gemini-3.1-pro": NO_DRIFT_DETECTED,
    "deepseek-v4-flash": NO_DRIFT_DETECTED,
    "gpt-5.5": "INDETERMINATE",
}
EXPECTED_LAMBDA = {
    "claude-opus-4-7": 0.98,
    "gemini-3.1-pro": 0.94,
    "deepseek-v4-flash": 0.82,
    "gpt-5.5": 0.29,
}
# the bare correlation gate keeps the labels PASS and FAIL: they describe the
# incumbent gate's own behaviour, not this screen's verdict.
EXPECTED_GATE = {
    "claude-opus-4-7": "PASS",   # bare gate passes the model that drifted
    "gemini-3.1-pro": "PASS",
    "deepseek-v4-flash": "PASS",
    "gpt-5.5": "FAIL",           # bare gate fails the model that barely moved
}


def test_four_model_verdicts(four_model_results):
    for model, expected in EXPECTED_VERDICT.items():
        assert four_model_results[model].verdict == expected, model


def test_four_model_lambda(four_model_results):
    for model, expected in EXPECTED_LAMBDA.items():
        assert four_model_results[model].lambda_point == pytest.approx(expected, abs=0.01), model


def test_bare_gate_inversion(four_model_results):
    # the paper's central point: the bare gate passes Claude and fails GPT-5.5,
    # exactly opposite to the screen.
    for model, expected in EXPECTED_GATE.items():
        assert four_model_results[model].registered_gate_verdict == expected, model
    assert four_model_results["gpt-5.5"].p_fail_if_stable > 0.5
    assert four_model_results["claude-opus-4-7"].p_fail_if_stable < 0.05


def test_claude_fails_on_run_effect(four_model_results):
    r = four_model_results["claude-opus-4-7"]
    assert r.verdict == "FAIL"
    assert r.p_run < 0.05
    assert "run effect" in r.reason


def test_pinned_statistics(four_model_results):
    # pin the actual statistics, not only the verdicts, so a sign flip or a
    # broken estimator that preserved verdicts would still be caught.
    claude = four_model_results["claude-opus-4-7"]
    gpt = four_model_results["gpt-5.5"]
    assert claude.z_run == pytest.approx(-3.48, abs=0.02)
    assert claude.Q_interaction == pytest.approx(14.6, abs=0.1)
    assert claude.drift_logit == pytest.approx(-0.54, abs=0.02)
    assert gpt.z_run == pytest.approx(0.55, abs=0.02)
    assert gpt.Q_interaction == pytest.approx(2.29, abs=0.1)
    # add-one smoothing: a bootstrap p-value is never exactly zero
    assert claude.p_run > 0.0


def test_gpt_indeterminate_not_fail(four_model_results):
    r = four_model_results["gpt-5.5"]
    assert r.verdict == "INDETERMINATE"
    assert not r.clears_floor
    assert r.p_run >= 0.05  # no run effect detected


# --------------------------------------------------------------------------- #
# Determinism and order independence
# --------------------------------------------------------------------------- #
def test_determinism():
    a = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4]
    b = [0.15, 0.35, 0.45, 0.72, 0.88, 0.25, 0.38]
    r1 = screen(a, b, 20, seed=123)
    r2 = screen(a, b, 20, seed=123)
    assert r1.to_dict() == r2.to_dict()


def test_seed_changes_pvalues_not_verdict():
    """The verdict is stable across seeds on a clear-cut case, the p-values are not.

    This is a statistical assertion, not a deterministic one. A bootstrap p-value
    moves with the seed, and a case whose p-value sits near the component level
    would flip. The fixture below is not such a case, so 20 seeds are run and at
    most one disagreement is tolerated, which keeps the test honest about what it
    is asserting rather than presenting a probabilistic result as a certainty.
    """
    a = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4]
    b = [0.15, 0.35, 0.45, 0.72, 0.88, 0.25, 0.38]
    seeds = range(1, 21)
    results = [screen(a, b, 20, seed=s) for s in seeds]
    verdicts = [r.verdict for r in results]
    majority = max(set(verdicts), key=verdicts.count)
    disagreements = sum(1 for v in verdicts if v != majority)
    assert disagreements <= 1, dict(zip(seeds, verdicts))
    # and the p-values do move, so the test is not passing because nothing varies
    assert len({round(r.p_run, 6) for r in results}) > 1


def test_order_independence(four_model_rows):
    grouped = screen_dataframe(four_model_rows)
    gpt_rows = [r for r in four_model_rows if r["model"] == "gpt-5.5"]
    alone = screen(
        [float(r["main_rate"]) for r in gpt_rows],
        [float(r["stability_rate"]) for r in gpt_rows],
        20,
    )
    assert alone.verdict == grouped["gpt-5.5"].verdict
    assert alone.lambda_point == pytest.approx(grouped["gpt-5.5"].lambda_point)


# --------------------------------------------------------------------------- #
# Decision-logic unit cases
# --------------------------------------------------------------------------- #
def test_uniform_shift_fails_on_run_effect():
    rng = np.random.default_rng(0)
    theta = rng.normal(0.0, 0.8, 14)  # baseline 0.5, good spread
    n = 40
    a = rng.binomial(n, 1 / (1 + np.exp(-theta))) / n
    b = rng.binomial(n, 1 / (1 + np.exp(-(theta + 1.0))) ) / n  # +1 logit shift
    r = screen(a, b, n)
    assert r.verdict == "FAIL"
    assert r.p_run < 0.05


def test_near_ceiling_stable_is_indeterminate():
    # all cells jammed near 1.0: no reliable between-cell variance
    a = np.array([1.0, 1.0, 0.95, 1.0, 1.0, 0.95, 1.0, 1.0, 1.0, 0.95, 1.0, 1.0, 1.0, 1.0])
    b = a.copy()
    r = screen(a, b, 20)
    assert r.verdict == "INDETERMINATE"
    assert not r.clears_floor


def test_well_spread_stable_returns_no_drift_detected():
    rng = np.random.default_rng(7)
    theta = rng.normal(0.0, 1.0, 20)  # wide spread -> high lambda
    n = 80
    p = 1 / (1 + np.exp(-theta))
    a = rng.binomial(n, p) / n
    b = rng.binomial(n, p) / n  # same propensity, truly stable
    r = screen(a, b, n)
    assert r.verdict == NO_DRIFT_DETECTED
    assert r.clears_floor


def test_lambda_rule_lower_is_more_conservative(four_model_rows):
    deepseek = [r for r in four_model_rows if r["model"] == "deepseek-v4-flash"]
    a = [float(r["main_rate"]) for r in deepseek]
    b = [float(r["stability_rate"]) for r in deepseek]
    point = screen(a, b, 20, lambda_rule="point")
    lower = screen(a, b, 20, lambda_rule="lower")
    # DeepSeek's lambda point (~0.82) clears the floor; its lower CI (~0.05) does not
    assert point.verdict == NO_DRIFT_DETECTED
    assert lower.verdict == "INDETERMINATE"


# --------------------------------------------------------------------------- #
# Input handling and validation
# --------------------------------------------------------------------------- #
def test_counts_and_rates_are_equivalent():
    counts_a = [2, 6, 10, 14, 18, 4, 8]
    counts_b = [3, 7, 9, 15, 17, 5, 7]
    n = 20
    rates_a = [c / n for c in counts_a]
    rates_b = [c / n for c in counts_b]
    r_counts = screen(counts_a, counts_b, n, seed=42)
    r_rates = screen(rates_a, rates_b, n, seed=42)
    assert r_counts.to_dict() == r_rates.to_dict()


def test_zero_one_counts_need_explicit_kind():
    # a 0/1 count vector is ambiguous: 'auto' reads it as rates (0.0/1.0), while
    # 'count' reads it as k/n (0.00/0.05). The interpretations must differ, and
    # the ambiguous 'auto' default must equal the explicit 'rate' reading.
    counts_a = [0, 1, 0, 1, 1, 0, 1]
    counts_b = [1, 1, 0, 1, 0, 0, 1]
    n = 20
    as_rate = screen(counts_a, counts_b, n, input_kind="rate", seed=1)
    as_count = screen(counts_a, counts_b, n, input_kind="count", seed=1)
    # as rates the cells are maximally spread (0/1) so reliability is high;
    # as tiny counts the cells barely differ, so reliability collapses
    assert as_count.lambda_point < as_rate.lambda_point
    assert as_count.verdict == "INDETERMINATE"
    assert as_rate.verdict != "INDETERMINATE"
    as_auto = screen(counts_a, counts_b, n, input_kind="auto", seed=1)
    assert as_auto.to_dict() == as_rate.to_dict()


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        screen([0.1, 0.2], [0.1, 0.2], 20)  # too few cells
    with pytest.raises(ValueError):
        screen([0.1, 0.2, 1.4], [0.1, 0.2, 0.3], 20)  # non-integer > 1
    with pytest.raises(ValueError):
        screen([0.1, 0.2, 0.3], [0.1, 0.2], 20)  # length mismatch
    with pytest.raises(ValueError):
        screen([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], 20, alpha=1.5)
    with pytest.raises(ValueError):
        screen([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], 0)  # non-positive n_calls
    with pytest.raises(ValueError):
        screen([0.1, float("nan"), 0.3], [0.1, 0.2, 0.3], 20)  # non-finite
    with pytest.raises(ValueError):
        screen([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], 20, input_kind="bogus")


def test_k_equals_three_runs():
    r = screen([0.1, 0.5, 0.9], [0.12, 0.48, 0.88], 40)
    assert r.verdict in VERDICTS


def test_fully_degenerate_is_indeterminate():
    # every cell identical: zero between-cell variance, lambda = 0
    a = np.full(10, 0.5)
    r = screen(a, a.copy(), 20)
    assert r.verdict == "INDETERMINATE"
    assert r.lambda_point == 0.0


def test_screen_dataframe_rejects_uneven_n():
    rows = [
        {"model": "m", "main_rate": 0.5, "stability_rate": 0.5, "main_n": 20},
        {"model": "m", "main_rate": 0.4, "stability_rate": 0.4, "main_n": 30},
        {"model": "m", "main_rate": 0.6, "stability_rate": 0.6, "main_n": 20},
    ]
    with pytest.raises(ValueError):
        screen_dataframe(rows)


# --------------------------------------------------------------------------- #
# Estimator helpers
# --------------------------------------------------------------------------- #
def test_pearson_matches_numpy():
    a = np.array([0.1, 0.4, 0.5, 0.9, 0.3])
    b = np.array([0.2, 0.35, 0.55, 0.8, 0.25])
    assert _core.pearson_r(a, b) == pytest.approx(np.corrcoef(a, b)[0, 1])


def test_pearson_nan_on_constant():
    assert math.isnan(_core.pearson_r(np.ones(5), np.arange(5.0)))


def test_spearman_matches_scipy_if_available():
    a = np.array([0.1, 0.4, 0.5, 0.9, 0.3, 0.6])
    b = np.array([0.2, 0.35, 0.55, 0.8, 0.25, 0.7])
    try:
        from scipy import stats
    except ImportError:
        pytest.skip("scipy not installed")
    assert _core.spearman_rho(a, b) == pytest.approx(stats.spearmanr(a, b).statistic, abs=1e-9)


def test_lambda_bounds():
    # a perfectly level design has zero reliable variance
    assert _core.lam(np.full(10, 0.5), 20) == 0.0
    # a widely spread design has high reliability
    a = np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.5, 0.3, 0.7, 0.1])
    assert 0.0 <= _core.lam(a, 200) <= 1.0
    assert _core.lam(a, 200) > 0.8


# --------------------------------------------------------------------------- #
# Design calibration
# --------------------------------------------------------------------------- #
def test_power_shapes_and_bounds():
    rows = power(K=14, n_calls=20, baseline=[0.5, 0.99], n_rep=300, n_boot=100)
    assert len(rows) == 2 * 5  # 2 baselines x 5 default modes
    for r in rows:
        for v in (r.final_pass, r.final_fail, r.final_indeterminate):
            assert 0.0 <= v <= 1.0
        # the three outcomes partition the replicates; equal to 1 up to 4-dp rounding
        assert r.final_pass + r.final_fail + r.final_indeterminate == pytest.approx(1.0, abs=1e-3)


def test_power_validates_design():
    with pytest.raises(ValueError):
        power(K=1, n_calls=20, n_rep=50, n_boot=20)  # K below 3
    with pytest.raises(ValueError):
        power(K=14, n_calls=0, n_rep=50, n_boot=20)  # non-positive n
    with pytest.raises(ValueError):
        recommend_design(baseline=0.9, drift_logit_that_matters=0.5,
                         candidate_K=(2,), candidate_n=(20,), n_rep=50, n_boot=20)


def test_power_near_ceiling_mostly_indeterminate():
    rows = power(K=14, n_calls=20, baseline=0.99, modes=("stable",), n_rep=400, n_boot=120)
    stable = rows[0]
    assert stable.final_indeterminate > 0.6  # design can rarely rule near a ceiling


def test_power_large_shift_detected_with_spread():
    rows = power(K=14, n_calls=20, baseline=0.5, sigma_logit=1.0,
                 modes=("shift_1.0",), n_rep=400, n_boot=120)
    row = rows[0]
    # a full-logit shift is easy to catch mid-range once the design is resolved
    assert row.final_fail / (row.final_fail + row.final_pass) > 0.95
    assert row.final_fail > 0.8


def test_power_is_order_independent():
    # the operating characteristics of a (baseline, mode) cell must not depend
    # on the order in which cells are requested
    forward = power(K=14, n_calls=20, baseline=[0.5, 0.9],
                    modes=("stable", "shift_1.0"), n_rep=300, n_boot=100)
    reversed_ = power(K=14, n_calls=20, baseline=[0.9, 0.5],
                      modes=("shift_1.0", "stable"), n_rep=300, n_boot=100)
    key = lambda r: (r.baseline, r.mode)
    fwd = {key(r): r.final_fail for r in forward}
    rev = {key(r): r.final_fail for r in reversed_}
    for k in fwd:
        assert fwd[k] == rev[k], k


def test_recommend_design_is_order_independent():
    a = recommend_design(baseline=0.7, drift_logit_that_matters=0.5,
                         target_false_exclusion=0.12, target_power=0.7,
                         candidate_K=(10, 14), candidate_n=(20, 40),
                         n_rep=300, n_boot=100)
    b = recommend_design(baseline=0.7, drift_logit_that_matters=0.5,
                         target_false_exclusion=0.12, target_power=0.7,
                         candidate_K=(14, 10), candidate_n=(40, 20),
                         n_rep=300, n_boot=100)
    assert (a.K_cells, a.n_per_cell, a.false_exclusion, a.power) == (
        b.K_cells, b.n_per_cell, b.false_exclusion, b.power
    )


def test_recommend_design_feasible_and_infeasible():
    ok = recommend_design(
        baseline=0.70, drift_logit_that_matters=0.5,
        target_false_exclusion=0.12, target_power=0.70,
        candidate_K=(10, 14, 20), candidate_n=(20, 40),
        n_rep=300, n_boot=100,
    )
    assert ok.recommended
    assert ok.total_calls == ok.K_cells * ok.n_per_cell
    # an impossible target near a ceiling: tiny shift, tiny budget
    no = recommend_design(
        baseline=0.99, drift_logit_that_matters=0.1,
        target_false_exclusion=0.01, target_power=0.99,
        candidate_K=(10, 14), candidate_n=(20,),
        n_rep=300, n_boot=100,
    )
    assert not no.recommended
    assert "closest" in no.message.lower() or "no candidate" in no.message.lower()


# --------------------------------------------------------------------------- #
# Ties in the bootstrap p-values
# --------------------------------------------------------------------------- #
def test_null_p_values_count_a_tie_that_differs_in_its_last_bits():
    from stabilitygate import _core

    z_null = np.array([1.0 - 1e-13, -1.0 + 1e-13, 0.5, 0.2])
    q_null = np.array([3.0 + 1e-13, 3.0 - 1e-13, 1.0, 0.5])
    p_run, p_int = _core.null_p_values(1.0, 3.0, z_null, q_null, "two")
    # two null draws tie with each observed statistic and both are counted
    assert p_run == pytest.approx(3 / 5)
    assert p_int == pytest.approx(3 / 5)
    assert _core.null_p_values(1.0, 3.0, z_null, q_null, "greater")[0] == pytest.approx(2 / 5)
    assert _core.null_p_values(-1.0, 3.0, z_null, q_null, "less")[0] == pytest.approx(2 / 5)


def test_swapping_the_runs_leaves_the_two_sided_p_values_unchanged():
    # a small near-ceiling design, where many null draws tie with the data
    a = [20, 20, 19, 20, 20, 18]
    b = [20, 19, 20, 20, 20, 20]
    forward = screen(a, b, 20, input_kind="count", n_boot=4000, n_boot_lambda=200)
    backward = screen(b, a, 20, input_kind="count", n_boot=4000, n_boot_lambda=200)
    assert forward.z_run == pytest.approx(-backward.z_run)
    assert forward.p_run == backward.p_run
    assert forward.p_interaction == backward.p_interaction


def test_power_uses_the_rule_screen_uses():
    """Under the bootstrap null a replicate is decided by an add-one p-value."""
    rows = power(K=14, n_calls=20, baseline=0.5, sigma_logit=1.0, modes=("stable",),
                 n_rep=600, n_boot=199)
    # 199 null draws put each component test at 4 / 200 = .020, so the union of
    # the two sits at or below .04 up to Monte Carlo error
    assert rows[0].final_fail < 0.075
