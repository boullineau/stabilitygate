"""Every number the paper and the README publish, pinned.

This file pins each number that appears in the paper's simulation results and in
the README's operating-characteristics section.

Every test here reads a reference output from ``repro/expected/`` rather than
recomputing, so the fast suite stays fast. Two slow variants recompute, one
through the reproduction harness and one through the verification script, and
are marked ``slow``.

The arithmetic behind each number is the same arithmetic
``repro/check_paper_numbers.py`` applies, and the two are kept in step by
``test_every_claim_has_a_test``, which fails if a claim is added to
``paper_numbers.json`` without a test here.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ROOT, load_reference, reference_dir

MODES = ("stable", "shift_1.0", "shift_-1.0")


# --------------------------------------------------------------------------- #
# Helpers, matching repro/check_paper_numbers.py exactly
# --------------------------------------------------------------------------- #
def _key(row: dict) -> tuple:
    return (row["baseline"], row["sigma_logit"], row["K_cells"],
            row["n_per_cell"], row["mode"])


def _rows(design: list[dict], mode: str) -> list[dict]:
    return [r for r in design if r["mode"] == mode]


def _joint_index(joint: list[dict]) -> dict:
    return {_key(r): r for r in joint}


def _resolvable_excluded(j: dict) -> tuple[int, int]:
    g = lambda c: int(j[c])  # noqa: E731
    resolvable = (g("n_admit_gatepass") + g("n_admit_gatefail")
                  + g("n_exclude_gatepass") + g("n_exclude_gatefail"))
    return resolvable, g("n_exclude_gatepass") + g("n_exclude_gatefail")


def _gate_failed(j: dict) -> int:
    return int(j["n_admit_gatefail"]) + int(j["n_exclude_gatefail"])


def _conditional(row: dict, numerator: str) -> float:
    admit, exclude = float(row["proposed_admit"]), float(row["proposed_exclude"])
    denom = admit + exclude
    if denom <= 0:
        return float("nan")
    return (exclude if numerator == "exclude" else admit) / denom


def _well_resolved(design: list[dict]) -> set:
    return {_key(r)[:4] for r in _rows(design, "stable")
            if float(r["proposed_indeterminate"]) < 0.10}


#: The paper's own yardstick, Appendix A. Monte Carlo standard error on any
#: reported rate is at most .008 at 4,000 replicates and .004 at 20,000.
MC_4000 = 0.008
MC_20000 = 0.004


# --------------------------------------------------------------------------- #
# README section "Operating characteristics"
# --------------------------------------------------------------------------- #
def test_pooled_conditional_size(published_reference, extension_design, extension_joint):
    """.0501 on 645,061 resolvable draws.

    The draw count is exact, taken from the joint integers. Recomputing the
    denominator from marginals rounded to four decimal places gives 645,063.
    """
    joint = _joint_index(extension_joint)
    resolvable = excluded = 0
    for row in _rows(extension_design, "stable"):
        res, exc = _resolvable_excluded(joint[_key(row)])
        resolvable += res
        excluded += exc
    assert resolvable == 645_061
    assert round(excluded / resolvable, 4) == 0.0501


def test_max_conditional_size(published_reference, extension_design):
    """.2936, and 41 of 252 configurations over .10."""
    values = [v for v in (_conditional(r, "exclude")
                          for r in _rows(extension_design, "stable")) if v == v]
    assert len(values) == 252
    assert max(values) == pytest.approx(0.2936, abs=MC_4000)
    assert sum(1 for v in values if v > 0.10) == 41


def test_size_by_resolution(extension_design, extension_joint):
    """.048 on the 124 well-resolved configurations and .056 on the other 128."""
    well = _well_resolved(extension_design)
    joint = _joint_index(extension_joint)
    pooled = {}
    for label, want in (("well", True), ("poor", False)):
        rows = [r for r in _rows(extension_design, "stable")
                if (_key(r)[:4] in well) is want]
        resolvable = excluded = 0
        for row in rows:
            res, exc = _resolvable_excluded(joint[_key(row)])
            resolvable += res
            excluded += exc
        pooled[label] = (len(rows), excluded / resolvable)
    assert pooled["well"][0] == 124
    assert pooled["poor"][0] == 128
    assert pooled["well"][1] == pytest.approx(0.048, abs=MC_4000)
    assert pooled["poor"][1] == pytest.approx(0.056, abs=MC_4000)
    # the aggregate is carried by the well-resolved half
    assert pooled["well"][1] < pooled["poor"][1]


def test_false_admission_boundaries(published_reference, extension_design):
    """50 configurations and .9194 conditional, 19 and .4098 unconditional."""
    rows = _rows(extension_design, "shift_1.0")
    conditional = [v for v in (_conditional(r, "admit") for r in rows) if v == v]
    unconditional = [float(r["proposed_admit"]) for r in rows]
    assert sum(1 for v in conditional if v > 0.05) == 50
    assert max(conditional) == pytest.approx(0.9194, abs=MC_4000)
    assert sum(1 for v in unconditional if v > 0.05) == 19
    assert max(unconditional) == pytest.approx(0.4098, abs=MC_4000)


def test_indeterminate_share(extension_design):
    """Median .1082, maximum .9988, under stability."""
    values = [float(r["proposed_indeterminate"])
              for r in _rows(extension_design, "stable")]
    assert statistics.median(values) == pytest.approx(0.1082, abs=MC_4000)
    assert max(values) == pytest.approx(0.9988, abs=MC_4000)


def test_power_well_resolved(extension_design):
    """Median 1.000 proposed against .1618 incumbent, at .0487 against .0583."""
    well = _well_resolved(extension_design)
    shifted = [r for r in _rows(extension_design, "shift_1.0") if _key(r)[:4] in well]
    stable = [r for r in _rows(extension_design, "stable") if _key(r)[:4] in well]
    assert len(shifted) == len(stable) == 124
    assert statistics.median([_conditional(r, "exclude") for r in shifted]) == pytest.approx(
        1.000, abs=MC_4000)
    assert statistics.median([1 - float(r["pearson_pass"]) for r in shifted]) == pytest.approx(
        0.1618, abs=MC_4000)
    assert statistics.median([_conditional(r, "exclude") for r in stable]) == pytest.approx(
        0.0487, abs=MC_4000)
    assert statistics.median([1 - float(r["pearson_pass"]) for r in stable]) == pytest.approx(
        0.0583, abs=MC_4000)


def test_incumbent_medians_full_grid(extension_design):
    """.8570 against a positive shift and .7127 against a negative one, all 252."""
    for mode, want in (("shift_1.0", 0.8570), ("shift_-1.0", 0.7127)):
        values = [1 - float(r["pearson_pass"]) for r in _rows(extension_design, mode)]
        assert len(values) == 252
        assert statistics.median(values) == pytest.approx(want, abs=MC_4000)


def test_direction_symmetry(extension_design):
    """Median conditional power 1.000 in both directions, on the well-resolved half."""
    well = _well_resolved(extension_design)
    for mode in ("shift_1.0", "shift_-1.0"):
        rows = [r for r in _rows(extension_design, mode) if _key(r)[:4] in well]
        values = [v for v in (_conditional(r, "exclude") for r in rows) if v == v]
        assert statistics.median(values) == pytest.approx(1.000, abs=MC_4000)


# --------------------------------------------------------------------------- #
# The joint counts, and what they recover
# --------------------------------------------------------------------------- #
def test_joint_counts_partition_and_derive_the_marginals(extension_design, extension_joint):
    """Row sums equal n_rep, and every published marginal follows to 4 places.

    This is the guarantee that adding the joint counts did not move the two
    files whose numbers are already published.
    """
    design = {_key(r): r for r in extension_design}
    for j in extension_joint:
        n = int(j["n_rep"])
        g = lambda c: int(j[c])  # noqa: E731
        six = (g("n_admit_gatepass") + g("n_admit_gatefail")
               + g("n_exclude_gatepass") + g("n_exclude_gatefail")
               + g("n_indet_gatepass") + g("n_indet_gatefail"))
        assert six == n, _key(j)
        reference = design[_key(j)]
        derived = {
            "pearson_pass": (g("n_admit_gatepass") + g("n_exclude_gatepass")
                             + g("n_indet_gatepass")) / n,
            "pearson_undefined": (g("n_undef_admit") + g("n_undef_exclude")
                                  + g("n_undef_indet")) / n,
            "proposed_admit": (g("n_admit_gatepass") + g("n_admit_gatefail")) / n,
            "proposed_exclude": (g("n_exclude_gatepass") + g("n_exclude_gatefail")) / n,
            "proposed_indeterminate": (g("n_indet_gatepass") + g("n_indet_gatefail")) / n,
        }
        for column, value in derived.items():
            assert round(value, 4) == pytest.approx(float(reference[column]), abs=1e-9), (
                _key(j), column
            )


def test_incumbent_conditional_false_exclusion(extension_design, extension_joint):
    """.348, the like-for-like comparison against the screen's .0501.

    This figure is not recoverable from the marginals, whose closest proxy is
    .3611. It is recoverable from the joint counts, and this test is what keeps
    it so.
    """
    joint = _joint_index(extension_joint)
    resolvable = failed = 0
    for row in _rows(extension_design, "stable"):
        j = joint[_key(row)]
        res, _ = _resolvable_excluded(j)
        resolvable += res
        failed += _gate_failed(j)
    assert failed / resolvable == pytest.approx(0.348, abs=MC_4000)


# --------------------------------------------------------------------------- #
# The paper's own tables
# --------------------------------------------------------------------------- #
TABLE_4 = {"pearson": 0.763, "spearman": 0.954, "ccc": 0.816,
           "absdrift": 0.287, "chi2": 0.050, "logit": 0.035}


def test_paper_pearson_median(simulation_grid):
    """.763 for the registered rule, from simulation_grid.csv, to within .008."""
    stable = [r for r in simulation_grid if r["mode"] == "stable"]
    assert len(stable) == 252
    assert len(simulation_grid) == 1260
    values = [float(r["err_pearson"]) for r in stable]
    assert statistics.median(values) == pytest.approx(0.763, abs=MC_4000)


@pytest.mark.parametrize("screen,want", sorted(TABLE_4.items()))
def test_paper_table4_medians(simulation_grid, screen, want):
    """All six medians of Table 4, All six screens under no drift."""
    stable = [r for r in simulation_grid if r["mode"] == "stable"]
    values = [float(r["err_" + screen]) for r in stable]
    assert statistics.median(values) == pytest.approx(want, abs=MC_4000)


def test_paper_table4_ranges(simulation_grid):
    """The calibrated screen's .038 to .064, and the log-odds rule's .796."""
    stable = [r for r in simulation_grid if r["mode"] == "stable"]
    chi2 = [float(r["err_chi2"]) for r in stable]
    assert min(chi2) == pytest.approx(0.038, abs=MC_4000)
    assert max(chi2) == pytest.approx(0.064, abs=MC_4000)
    assert max(float(r["err_logit"]) for r in stable) == pytest.approx(0.796, abs=MC_4000)
    # only the calibrated screen sits near nominal across the whole grid
    assert max(chi2) - min(chi2) < 0.05


#: Table 1, one row per baseline. Registered exclusion, correlation power,
#: unsigned power, calibrated correlation threshold.
TABLE_1 = {
    0.50: (0.945, 0.066, 0.971, 0.151),
    0.70: (0.956, 0.090, 0.881, 0.083),
    0.85: (0.975, 0.102, 0.541, -0.068),
    0.90: (0.986, 0.096, 0.316, -0.158),
    0.95: (0.993, 0.082, 0.085, -0.275),
    0.98: (0.993, 0.131, 0.018, -0.351),
    0.99: (0.988, 0.352, 0.013, -0.286),
}


def test_paper_table1_values(fig3_values):
    """All 28 values of Table 1, from fig3_values.json."""
    for baseline, row in TABLE_1.items():
        index = fig3_values["bl"].index(baseline)
        for series, want in zip(("reg_fe", "pw_p", "pw_c", "thr"), row):
            got = fig3_values[series][index]
            assert got == pytest.approx(want, abs=MC_20000), (baseline, series)


def test_paper_negative_threshold_region(fig3_values):
    """Above a baseline of .80 the calibrated correlation threshold is negative."""
    for baseline, threshold in zip(fig3_values["bl"], fig3_values["thr"]):
        assert (threshold < 0) == (baseline > 0.80), baseline


def test_appendix_c_scramble_series(fig3_values):
    """.692 against .632 at p0 = .50, the one comparison the paper does not lead with.

    The published values came from a run whose stream was never written out, so
    the harness recomputes them as an independent replicate. The tolerance is
    that of a difference between two 20,000-replicate runs. What the paper
    actually claims is the ordering, and that is asserted separately and exactly.
    """
    if "pw_p_scramble" not in fig3_values:
        pytest.skip("this reference output predates the scramble series")
    index = fig3_values["bl"].index(0.50)
    tolerance = 2 * MC_20000 * 2 ** 0.5
    assert fig3_values["pw_p_scramble"][index] == pytest.approx(0.692, abs=tolerance)
    assert fig3_values["pw_c_scramble"][index] == pytest.approx(0.632, abs=tolerance)
    # the claim itself, which no tolerance is needed to state
    assert fig3_values["pw_p_scramble"][index] > fig3_values["pw_c_scramble"][index]
    # and it converges by p0 = .90
    high = fig3_values["bl"].index(0.90)
    assert abs(fig3_values["pw_p_scramble"][high]
               - fig3_values["pw_c_scramble"][high]) < 0.05


# --------------------------------------------------------------------------- #
# The worked case, from the package rather than from a file
# --------------------------------------------------------------------------- #
def test_paper_worked_case_headline_numbers():
    """lambda = .29, r = -.135, and 83.9 per cent, for the excluded model."""
    import csv
    from importlib import resources
    from stabilitygate import screen_dataframe
    with resources.files("stabilitygate.data").joinpath(
        "four_model_stability.csv"
    ).open("r") as f:
        rows = list(csv.DictReader(f))
    results = screen_dataframe(rows, compute_gate_false_exclusion=True)
    gpt = results["gpt-5.5"]
    claude = results["claude-opus-4-7"]
    assert gpt.lambda_point == pytest.approx(0.29, abs=0.005)
    assert gpt.pearson_r == pytest.approx(-0.135, abs=0.001)
    assert gpt.mean_abs_drift == pytest.approx(0.018, abs=0.001)
    assert gpt.p_fail_if_stable == pytest.approx(0.839, abs=MC_4000)
    assert claude.mean_abs_drift == pytest.approx(0.129, abs=0.001)
    assert claude.max_abs_drift == pytest.approx(0.55, abs=0.001)
    assert claude.pearson_r == pytest.approx(0.886, abs=0.001)
    # the inversion itself
    assert gpt.registered_gate_verdict == "FAIL"
    assert claude.registered_gate_verdict == "PASS"
    gpt_rows = [r for r in rows if r["model"] == "gpt-5.5"]
    identical = sum(1 for r in gpt_rows
                    if float(r["main_rate"]) == float(r["stability_rate"]))
    assert identical == 10


# --------------------------------------------------------------------------- #
# Panel loss, which needs the corpus
# --------------------------------------------------------------------------- #
def test_panel_loss_counts():
    """8,433 tie events, 3,843 one-way flips, 2,811 exposed items.

    Read from the deposited per-annotator counts. The corpus itself is
    distributed by its own authors and is not redistributed here, so this test
    skips when neither the counts nor the corpus is present.
    """
    rows = load_reference("mdagreement_removal_counts.csv")
    assert len(rows) == 819
    assert sum(int(r["items_labelled"]) for r in rows) == 49_070
    assert sum(int(r["tie_events_created"]) for r in rows) == 8_433
    assert sum(int(r["pass_to_fail_flips"]) for r in rows) == 3_843
    # a three-two item loses its majority under exactly three of its five
    # removals, so the exposed-item count follows from the event count
    assert 8_433 % 3 == 0
    assert 8_433 // 3 == 2_811


# --------------------------------------------------------------------------- #
# The two files stay in step
# --------------------------------------------------------------------------- #
def test_every_claim_has_a_test():
    """Every claim in paper_numbers.json is covered by a test in this file.

    The mapping is by the claim's source and subject rather than one test per
    claim, so this asserts that no published number has been added to the
    verification without a test pinning it here.
    """
    claims_path = ROOT / "repro" / "paper_numbers.json"
    if not claims_path.exists():
        pytest.skip("the reproduction harness is not present")
    claims = json.loads(claims_path.read_text())["claims"]
    recipes = {c["recipe"] for c in claims}
    covered = {
        "worked_case", "grid_stat", "grid_count", "fig3",
        "pooled_conditional_size", "conditional_size", "size_by_resolution",
        "false_admission", "indeterminate", "well_resolved",
        "incumbent_full_grid", "direction_symmetry",
        "incumbent_pooled_conditional", "joint_rows_partition", "panel",
    }
    assert recipes <= covered, f"uncovered recipes: {sorted(recipes - covered)}"
    assert len(claims) >= 80


# --------------------------------------------------------------------------- #
# Slow variants that recompute rather than read
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_verification_script_reports_no_failures(published_reference):
    """``repro/check_paper_numbers.py`` exits 0 against the reference outputs."""
    script = ROOT / "repro" / "check_paper_numbers.py"
    if not script.exists():
        pytest.skip("the reproduction harness is not present")
    completed = subprocess.run(
        [sys.executable, str(script), "--results", str(reference_dir())],
        capture_output=True, text=True, env=dict(os.environ, NO_COLOR="1"),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "0 FAIL" in completed.stdout


@pytest.mark.slow
def test_realised_design_recomputes():
    """Stage 3 rewrites the five published series of Table 1 unchanged."""
    script = ROOT / "repro" / "run_realised.py"
    if not script.exists():
        pytest.skip("the reproduction harness is not present")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, SG_RESULTS=tmp)
        env.pop("SG_QUICK", None)
        completed = subprocess.run([sys.executable, str(script)],
                                   env=env, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        values = json.loads((Path(tmp) / "fig3_values.json").read_text())
    for baseline, row in TABLE_1.items():
        index = values["bl"].index(baseline)
        for series, want in zip(("reg_fe", "pw_p", "pw_c", "thr"), row):
            assert values[series][index] == pytest.approx(want, abs=MC_20000)
