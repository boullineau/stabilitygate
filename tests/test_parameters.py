"""The decision-rule parameters.

The package and the simulation that produced the published operating
characteristics can resolve sidedness, the per-component level, the combination
rule and the order of the floor and the tests differently, and the difference
is a factor of two or more at a saturated baseline. These four parameters expose
every one of those choices.

Each parameter gets a test that a non-default setting changes at least one
verdict, on a fixture chosen so the flip is not marginal. The p-values quoted in
the comments were measured at the default ``n_boot = 20_000`` and ``seed =
20260101`` and are deterministic under them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stabilitygate import NO_DRIFT_DETECTED, power, screen
from stabilitygate import _core

N = 20

# lambda .88, p_run .0378, p_interaction .68. The level test rejects at
# alpha = .05 and does not reject at alpha / 2.
ALPHA_SPLIT_MAIN = [0.20, 0.10, 0.35, 0.05, 0.55, 0.10, 0.75,
                    0.20, 0.75, 0.25, 0.45, 0.80, 0.70, 0.30]
ALPHA_SPLIT_RETEST = [0.45, 0.00, 0.60, 0.15, 0.50, 0.10, 0.80,
                      0.35, 0.60, 0.30, 0.60, 0.95, 0.75, 0.40]

# lambda .77, z_run +1.91, two-sided p_run .0358 against one-sided .0167. The
# retest run sits above the main run, so only the "greater" tail rejects.
RUN_SIDED_MAIN = [0.65, 0.35, 0.60, 0.50, 0.55, 0.85, 0.65,
                  0.95, 0.85, 0.20, 0.75, 0.70, 0.45, 0.75]
RUN_SIDED_RETEST = [0.50, 0.55, 0.70, 0.75, 0.55, 0.85, 0.80,
                    0.80, 0.85, 0.45, 0.75, 0.75, 0.65, 0.90]

# lambda .86, p_run .0001, p_interaction .55. Exactly one component rejects.
COMBINE_MAIN = [0.35, 0.80, 0.80, 0.85, 0.65, 0.65, 0.70,
                0.75, 0.15, 0.55, 0.90, 1.00, 0.95, 0.90]
COMBINE_RETEST = [0.55, 0.75, 1.00, 1.00, 0.95, 0.85, 0.75,
                  0.95, 0.45, 0.70, 1.00, 1.00, 1.00, 0.95]

# lambda .00, far below the floor, with p_run .0001. The design cannot resolve
# anything, yet the level test rejects.
PRECEDENCE_MAIN = [0.70, 0.85, 0.70, 0.85, 0.85, 0.85, 0.90,
                   0.85, 0.75, 0.95, 0.75, 0.85, 0.95, 0.75]
PRECEDENCE_RETEST = [0.90, 0.90, 0.95, 0.90, 1.00, 0.95, 0.95,
                     0.90, 0.95, 0.85, 0.95, 0.95, 1.00, 0.95]


# --------------------------------------------------------------------------- #
# One verdict flip per parameter
# --------------------------------------------------------------------------- #
def test_alpha_split_changes_a_verdict():
    default = screen(ALPHA_SPLIT_MAIN, ALPHA_SPLIT_RETEST, N)
    none = screen(ALPHA_SPLIT_MAIN, ALPHA_SPLIT_RETEST, N, alpha_split="none")
    assert default.alpha_component == pytest.approx(0.025)
    assert none.alpha_component == pytest.approx(0.05)
    # the same p_run, read against two different component levels
    assert default.p_run == pytest.approx(none.p_run)
    assert 0.025 < default.p_run < 0.05
    assert default.verdict == NO_DRIFT_DETECTED
    assert none.verdict == "FAIL"


def test_run_sided_changes_a_verdict():
    default = screen(RUN_SIDED_MAIN, RUN_SIDED_RETEST, N)
    greater = screen(RUN_SIDED_MAIN, RUN_SIDED_RETEST, N, run_sided="greater")
    less = screen(RUN_SIDED_MAIN, RUN_SIDED_RETEST, N, run_sided="less")
    assert default.z_run > 0
    # a one-sided test spends the whole budget in one tail, so it roughly halves
    # the p-value in the direction the shift actually went and inverts it in the
    # other
    assert greater.p_run < default.p_run < less.p_run
    assert default.verdict == NO_DRIFT_DETECTED
    assert greater.verdict == "FAIL"
    assert less.verdict == NO_DRIFT_DETECTED


def test_combine_changes_a_verdict():
    default = screen(COMBINE_MAIN, COMBINE_RETEST, N)
    both = screen(COMBINE_MAIN, COMBINE_RETEST, N, combine="and")
    # exactly one of the two components rejects
    assert default.p_run < default.alpha_component
    assert default.p_interaction > default.alpha_component
    assert default.verdict == "FAIL"
    assert both.verdict == NO_DRIFT_DETECTED


def test_precedence_changes_a_verdict():
    default = screen(PRECEDENCE_MAIN, PRECEDENCE_RETEST, N)
    test_first = screen(
        PRECEDENCE_MAIN, PRECEDENCE_RETEST, N, precedence="test_first"
    )
    assert default.precedence == "resolution_first"
    assert not default.clears_floor
    assert default.p_run < default.alpha_component
    # the default does not let the level test rule on a design below the floor
    assert default.verdict == "INDETERMINATE"
    # and it says that the level test rejected
    assert default.run_detected_below_floor
    assert "run-effect test rejected" in default.reason
    # test_first lets it rule
    assert test_first.verdict == "FAIL"
    assert test_first.run_detected_below_floor


def test_run_detected_below_floor_is_false_above_the_floor():
    result = screen(COMBINE_MAIN, COMBINE_RETEST, N)
    assert result.clears_floor
    assert not result.run_detected_below_floor


# --------------------------------------------------------------------------- #
# What the settings mean
# --------------------------------------------------------------------------- #
def test_alpha_split_component_levels():
    assert _core.component_alpha(0.05, "bonferroni") == pytest.approx(0.025)
    assert _core.component_alpha(0.05, "sidak") == pytest.approx(1 - math.sqrt(0.95))
    assert _core.component_alpha(0.05, "none") == pytest.approx(0.05)
    # Sidak sits between the other two, and much nearer Bonferroni
    assert 0.025 < _core.component_alpha(0.05, "sidak") < 0.05
    assert _core.component_alpha(0.05, "sidak") == pytest.approx(0.0253, abs=1e-4)


def test_alpha_split_none_raises_union_false_exclusion():
    """Running each component at the full alpha roughly doubles false exclusion."""
    split = power(K=14, n_calls=20, baseline=0.90, sigma_logit=0.50,
                  modes=("stable",), n_rep=600, n_boot=200)[0]
    unsplit = power(K=14, n_calls=20, baseline=0.90, sigma_logit=0.50,
                    modes=("stable",), alpha_split="none", n_rep=600, n_boot=200)[0]
    assert unsplit.final_fail > split.final_fail * 1.4


def test_precedence_resolution_first_lowers_false_exclusion_in_power():
    """The default lowers false exclusion against test_first, measured in power()."""
    test_first = power(K=14, n_calls=20, baseline=0.90, sigma_logit=0.50,
                       modes=("stable",), precedence="test_first",
                       n_rep=600, n_boot=200)[0]
    resolution_first = power(K=14, n_calls=20, baseline=0.90, sigma_logit=0.50,
                             modes=("stable",), precedence="resolution_first",
                             n_rep=600, n_boot=200)[0]
    assert resolution_first.final_fail < test_first.final_fail
    assert resolution_first.final_indeterminate > test_first.final_indeterminate


def test_combine_and_lowers_exclusion_in_power():
    either = power(K=14, n_calls=20, baseline=0.50, sigma_logit=1.00,
                   modes=("stable",), n_rep=600, n_boot=200)[0]
    both = power(K=14, n_calls=20, baseline=0.50, sigma_logit=1.00,
                 modes=("stable",), combine="and", n_rep=600, n_boot=200)[0]
    assert both.final_fail < either.final_fail


def test_combine_and_makes_precedence_inert():
    """Under "and" no single test is decisive, so nothing overrides the floor."""
    a, b = PRECEDENCE_MAIN, PRECEDENCE_RETEST
    first = screen(a, b, N, combine="and", precedence="test_first")
    second = screen(a, b, N, combine="and", precedence="resolution_first")
    assert first.verdict == second.verdict == "INDETERMINATE"


# --------------------------------------------------------------------------- #
# The alternative statistics and the oracle null
# --------------------------------------------------------------------------- #
def test_pooled_logit_matches_the_standalone_definition():
    a = np.array([0.1, 0.4, 0.9, 0.5, 0.2, 0.7, 0.3])
    b = np.array([0.2, 0.35, 0.85, 0.6, 0.25, 0.75, 0.4])
    ma, mb = a.mean(), b.mean()
    expected = math.log(mb / (1 - mb)) - math.log(ma / (1 - ma))
    lev, _ = _core.statistics(a, b, 20, level_statistic="pooled_logit")
    assert lev == pytest.approx(expected)


def test_pearson_chi2_matches_the_standalone_definition():
    a = np.array([0.1, 0.4, 0.9, 0.5, 0.2, 0.7, 0.3])
    b = np.array([0.2, 0.35, 0.85, 0.6, 0.25, 0.75, 0.4])
    n = 20
    phat = (a + b) / 2.0
    expected = float((((a - b) ** 2) / (phat * (1 - phat) * (2.0 / n))).sum())
    _, het = _core.statistics(a, b, n, heterogeneity_statistic="pearson_chi2")
    assert het == pytest.approx(expected)


def test_default_statistics_are_woolf_and_cochran():
    a = np.array([0.1, 0.4, 0.9, 0.5, 0.2, 0.7, 0.3])
    b = np.array([0.2, 0.35, 0.85, 0.6, 0.25, 0.75, 0.4])
    assert _core.statistics(a, b, 20) == pytest.approx(_core.components(a, b, 20))


def test_alternative_statistics_reach_the_result():
    r = screen(COMBINE_MAIN, COMBINE_RETEST, N,
               level_statistic="pooled_logit",
               heterogeneity_statistic="pearson_chi2")
    assert r.level_statistic == "pooled_logit"
    assert r.heterogeneity_statistic == "pearson_chi2"
    expected_lev, expected_het = _core.statistics(
        np.array(COMBINE_MAIN), np.array(COMBINE_RETEST), N,
        "pooled_logit", "pearson_chi2",
    )
    assert r.z_run == pytest.approx(expected_lev)
    assert r.Q_interaction == pytest.approx(expected_het)


def test_oracle_null_is_rejected_by_screen():
    with pytest.raises(ValueError, match="oracle"):
        screen(COMBINE_MAIN, COMBINE_RETEST, N, null="oracle")


def test_oracle_null_shares_critical_values_across_modes():
    """One configuration's oracle critical values do not depend on the alternative."""
    rows = power(K=14, n_calls=20, baseline=0.90, sigma_logit=0.50,
                 modes=("stable", "shift_1.0"), null="oracle",
                 n_rep=400, n_boot=1000)
    forward = power(K=14, n_calls=20, baseline=0.90, sigma_logit=0.50,
                    modes=("shift_1.0", "stable"), null="oracle",
                    n_rep=400, n_boot=1000)
    got = {r.mode: r.final_fail for r in rows}
    rev = {r.mode: r.final_fail for r in forward}
    assert got == rev


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha_split": "holm"},
        {"run_sided": "upper"},
        {"combine": "xor"},
        {"precedence": "floor_first"},
        {"level_statistic": "welch"},
        {"heterogeneity_statistic": "levene"},
        {"null": "permutation"},
    ],
)
def test_invalid_values_raise(kwargs):
    with pytest.raises(ValueError):
        screen([0.1, 0.5, 0.9], [0.12, 0.48, 0.88], 40, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha_split": "holm"},
        {"run_sided": "upper"},
        {"combine": "xor"},
        {"precedence": "floor_first"},
        {"null": "permutation"},
    ],
)
def test_power_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        power(K=14, n_calls=20, modes=("stable",), n_rep=40, n_boot=20, **kwargs)


def test_settings_are_reported_on_the_result():
    r = screen([0.1, 0.5, 0.9, 0.3, 0.7], [0.12, 0.48, 0.88, 0.35, 0.65], 40,
               alpha_split="sidak", run_sided="greater", combine="and",
               precedence="resolution_first")
    d = r.to_dict()
    assert d["alpha_split"] == "sidak"
    assert d["run_sided"] == "greater"
    assert d["combine"] == "and"
    assert d["precedence"] == "resolution_first"
    assert d["alpha_component"] == pytest.approx(1 - math.sqrt(0.95))
    # and they are visible to a human reading the formatted result
    text = str(r)
    assert "resolution_first" in text and "sidak" in text


def test_run_sided_has_no_power_in_the_wrong_tail():
    """The one-sided settings reach power() and the oracle null, not only screen()."""
    common = dict(K=14, n_calls=50, baseline=0.50, sigma_logit=1.00,
                  modes=("shift_-1.0",), n_rep=400, n_boot=200)
    two = power(run_sided="two", **common)[0]
    less = power(run_sided="less", **common)[0]
    greater = power(run_sided="greater", **common)[0]
    assert two.final_fail > 0.9
    assert less.final_fail > 0.9
    # a test looking only for an increase cannot see a decrease
    assert greater.final_fail < 0.15


def test_lambda_rule_lower_abstains_more_in_power():
    """power() honours both floor rules, as screen() does."""
    common = dict(K=14, n_calls=20, baseline=0.70, sigma_logit=1.00,
                  modes=("stable",), n_rep=300, n_boot=150)
    point = power(lambda_rule="point", **common)[0]
    lower = power(lambda_rule="lower", n_boot_lambda=400, **common)[0]
    assert lower.pct_lambda_ge_floor < point.pct_lambda_ge_floor
    assert lower.final_indeterminate > point.final_indeterminate
