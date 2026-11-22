"""The `OC_2026_08_28` reproduction preset.

`results/extension_design.csv` and `results/extension_realised.csv` were
produced on 28 August 2026 by a standalone simulation, not by this package. The
published operating characteristics are those files. The released package must
be able to reach them from a documented configuration.

``test_oc_preset_reproduces_design_csv`` and its realised-design sibling assert
that the preset lands on the published numbers to within Monte Carlo error.

``test_test_first_differs_from_preset`` records what the other precedence rule
does. The defaults share the preset's decision rule, resolution first, and
``precedence="test_first"`` excludes a stable model more than twice as often at
a saturated baseline.
"""

from __future__ import annotations

import filecmp
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ROOT, reference_dir, load_reference
from stabilitygate import power
from stabilitygate.presets import OC_2026_08_28, PRESETS

#: Replicates per configuration in the published run. Matching it is what makes
#: the comparison a Monte Carlo comparison rather than a coarser one.
N_REP = 4000
#: Draws in the published run's per-configuration calibration sample.
N_CAL = 4000

MODES = ("stable", "shift_1.0", "shift_-1.0")
BASELINES = (0.50, 0.70, 0.85, 0.90, 0.95, 0.98, 0.99)
SIGMAS = (0.25, 0.50, 1.00, 1.50)

#: The three outcome columns, paired with the ``PowerRow`` field each one holds.
OUTCOME_COLUMNS = (
    ("proposed_admit", "final_pass"),
    ("proposed_exclude", "final_fail"),
    ("proposed_indeterminate", "final_indeterminate"),
)

#: Standard errors of the difference between two independent runs of ``N_REP``
#: replicates, at two standard errors, with the binomial standard error inflated
#: to allow for the critical values both runs estimate from a finite calibration
#: sample. The inflation factor was measured at 1.8 over twelve seeds at six grid
#: cells, so 2.0 is the measured figure rounded up. ``2 * sqrt(2) * 2 = 5.657``.
#: The absolute floor covers configurations whose published rate is 0 or 1, where
#: the binomial standard error vanishes but a handful of draws can still differ.
TOLERANCE_SE_MULTIPLE = 5.657
TOLERANCE_FLOOR = 0.005


def monte_carlo_tolerance(p: float, n_rep: int = N_REP) -> float:
    """Tolerance on a published rate ``p``, on the basis documented above."""
    return TOLERANCE_SE_MULTIPLE * math.sqrt(max(p * (1 - p), 0.0) / n_rep) + TOLERANCE_FLOOR


def _load_reference(name: str) -> dict:
    """The published CSV, keyed by configuration and mode.

    ``conftest.reference_dir`` resolves where it lives, so the fixture can be
    relocated or generated without touching this file.
    """
    rows = {}
    for row in load_reference(name):
        key = (
            float(row["baseline"]),
            float(row["sigma_logit"]),
            int(row["K_cells"]),
            int(row["n_per_cell"]),
            row["mode"],
        )
        rows[key] = row
    assert rows, f"{name} is empty"
    return rows


def _compare(reference: dict, baseline: float, sigma: float, K: int, n: int) -> list[str]:
    """Run the preset at one configuration and return one message per mismatch."""
    rows = power(
        K=K, n_calls=n, baseline=baseline, sigma_logit=sigma, modes=MODES,
        n_rep=N_REP, n_boot=N_CAL, **OC_2026_08_28
    )
    failures = []
    for row in rows:
        published = reference[(baseline, sigma, K, n, row.mode)]
        for column, field in OUTCOME_COLUMNS:
            expected = float(published[column])
            got = getattr(row, field)
            tol = monte_carlo_tolerance(expected)
            if abs(got - expected) > tol:
                failures.append(
                    f"baseline {baseline} sigma {sigma} K {K} n {n} {row.mode} "
                    f"{column}: published {expected:.4f}, preset {got:.4f}, "
                    f"difference {abs(got - expected):.4f} over tolerance {tol:.4f}"
                )
    return failures


# --------------------------------------------------------------------------- #
# The preset is what its documentation says it is
# --------------------------------------------------------------------------- #
def test_preset_contents():
    assert PRESETS["OC_2026_08_28"] is OC_2026_08_28
    assert OC_2026_08_28 == {
        "lambda_floor": 0.50,
        "lambda_rule": "point",
        "alpha": 0.05,
        "alpha_split": "bonferroni",
        "run_sided": "two",
        "combine": "or",
        "precedence": "resolution_first",
        "level_statistic": "pooled_logit",
        "heterogeneity_statistic": "pearson_chi2",
        "null": "oracle",
        "seed": 20260101,
    }


# --------------------------------------------------------------------------- #
# The proof step
# --------------------------------------------------------------------------- #
#: Eight configurations spanning the full design grid: both ends of the baseline
#: range, all four spreads, all three cell counts and all three call counts, and
#: both the well-resolved and the near-ceiling regimes.
SAMPLED_DESIGN_CELLS = (
    (0.50, 0.25, 14, 20),
    (0.50, 1.00, 14, 50),
    (0.70, 1.50, 56, 200),
    (0.85, 0.50, 28, 50),
    (0.90, 0.50, 14, 20),
    (0.95, 0.25, 28, 20),
    (0.98, 1.50, 56, 20),
    (0.99, 0.50, 14, 200),
)


def test_oc_preset_reproduces_design_csv():
    """The preset reproduces the published design grid to within Monte Carlo error.

    Eight of the 252 configurations, three drift modes each, three outcome rates
    each: 72 published numbers. Over the whole 252-configuration grid the largest
    observed difference was 0.78 of the tolerance applied here.
    """
    reference = _load_reference("extension_design.csv")
    failures = []
    for baseline, sigma, K, n in SAMPLED_DESIGN_CELLS:
        failures += _compare(reference, baseline, sigma, K, n)
    assert not failures, "\n".join(failures)


def test_oc_preset_reproduces_realised_csv():
    """The same, over every configuration of the realised design K = 14, n = 20."""
    reference = _load_reference("extension_realised.csv")
    failures = []
    for baseline in BASELINES:
        for sigma in SIGMAS:
            failures += _compare(reference, baseline, sigma, 14, 20)
    assert not failures, "\n".join(failures)


# --------------------------------------------------------------------------- #
# The permanent record of the divergence
# --------------------------------------------------------------------------- #
def test_test_first_differs_from_preset():
    """Under ``precedence="test_first"`` the package departs from the published numbers.

    At baseline .90, sigma .50, K = 14, n = 20 the published run excludes a
    stable model on .0182 of draws and detects a positive one-logit shift on
    .2067, abstaining on .7682 of the shifted draws because the design sits below
    the information floor. ``precedence="test_first"`` evaluates the level test
    before the floor, so it excludes and detects more than twice as often and
    abstains far less.
    """
    reference = _load_reference("extension_design.csv")
    stable = reference[(0.90, 0.50, 14, 20, "stable")]
    shifted = reference[(0.90, 0.50, 14, 20, "shift_1.0")]
    published_false_exclusion = float(stable["proposed_exclude"])
    published_power = float(shifted["proposed_exclude"])
    published_abstention = float(shifted["proposed_indeterminate"])

    default = {
        row.mode: row
        for row in power(
            K=14, n_calls=20, baseline=0.90, sigma_logit=0.50,
            modes=("stable", "shift_1.0"), precedence="test_first",
            n_rep=1200, n_boot=300,
        )
    }

    # measured at between 2.3 and 2.9 times the published rate over three seeds
    assert default["stable"].final_fail > 2.0 * published_false_exclusion
    # measured at about 3.5 times
    assert default["shift_1.0"].final_fail > 3.0 * published_power
    # measured at about 0.3 times
    assert default["shift_1.0"].final_indeterminate < 0.5 * published_abstention

    # and the two shifted rates sit an order of magnitude outside the tolerance
    # the reproduction test allows, so the divergence is not a Monte Carlo
    # artefact.
    for got, expected in (
        (default["shift_1.0"].final_fail, published_power),
        (default["shift_1.0"].final_indeterminate, published_abstention),
    ):
        assert abs(got - expected) > 3.0 * monte_carlo_tolerance(expected, n_rep=1200)


def test_default_precedence_is_close_to_the_published_abstention():
    """The default rule abstains about as often as the published run does."""
    reference = _load_reference("extension_design.csv")
    published = float(reference[(0.90, 0.50, 14, 20, "shift_1.0")]["proposed_indeterminate"])
    common = dict(K=14, n_calls=20, baseline=0.90, sigma_logit=0.50,
                  modes=("shift_1.0",), n_rep=1200, n_boot=300)
    default = power(**common)[0]
    test_first = power(precedence="test_first", **common)[0]
    assert default.precedence == "resolution_first"
    assert abs(default.final_indeterminate - published) < abs(
        test_first.final_indeterminate - published
    )
    assert default.final_indeterminate > 0.6


@pytest.mark.parametrize("mode", MODES)
def test_preset_outcomes_partition(mode):
    row = power(K=14, n_calls=20, baseline=0.90, sigma_logit=0.50, modes=(mode,),
                n_rep=500, n_boot=500, **OC_2026_08_28)[0]
    total = row.final_pass + row.final_fail + row.final_indeterminate
    assert total == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# The byte-for-byte variant, slow
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_oc_preset_reproduces_bitwise():
    """The harness script rewrites both published CSVs byte for byte.

    The two tests above compare the PACKAGE under the preset against the
    published numbers, and the comparison is necessarily to within Monte Carlo
    error. ``power()`` draws each configuration from its own independent stream
    so that operating characteristics do not depend on evaluation order, while
    the published run drew every configuration from one sequential stream. The
    two agree in distribution and cannot agree draw for draw.

    What can agree draw for draw is ``repro/run_extension_oc.py``, which is that
    same sequential run promoted into the harness. This test runs it into a
    scratch directory and compares the bytes, which is the strongest available
    statement that nothing in the published record has moved.

    It takes about twenty seconds, so it is marked slow.
    """
    script = ROOT / "repro" / "run_extension_oc.py"
    if not script.exists():
        pytest.skip("the reproduction harness is not present")
    reference = reference_dir()
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, SG_RESULTS=tmp)
        env.pop("SG_QUICK", None)
        completed = subprocess.run(
            [sys.executable, str(script)], env=env, capture_output=True, text=True
        )
        assert completed.returncode == 0, completed.stderr
        for name in ("extension_design.csv", "extension_realised.csv"):
            assert filecmp.cmp(Path(tmp) / name, reference / name, shallow=False), (
                f"{name} is not byte-for-byte identical to {reference / name}"
            )
