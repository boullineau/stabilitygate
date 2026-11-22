"""Named, frozen configurations of the screen.

A preset is a plain dictionary of keyword arguments. Splat it into
:func:`stabilitygate.power`, :func:`stabilitygate.recommend_design` or
:func:`stabilitygate.screen` to run that configuration.

The package's own defaults are deliberately not a preset. They are what a user
holding data should run, and they may change with the package version. A preset
is a historical record and does not change.
"""

from __future__ import annotations

__all__ = ["OC_2026_08_28", "PRESETS"]


#: The configuration that produced the published operating characteristics.
#:
#: ``results/extension_design.csv`` and ``results/extension_realised.csv`` were
#: produced on 28 August 2026 by a standalone simulation, not by this package.
#: That simulation used the same decision rule as the package's defaults,
#: resolution first with each component test at ``alpha / 2``, and different
#: statistics and a different null. This preset exists so the published run is
#: reachable from released code.
#:
#: Three of its settings are available only in simulation or only for this
#: historical record.
#:
#: * ``level_statistic="pooled_logit"`` and
#:   ``heterogeneity_statistic="pearson_chi2"`` are the statistics that run used.
#:   The defaults are the Woolf z and Cochran's Q.
#: * ``null="oracle"`` draws the calibration sample from the true
#:   ``(baseline, sigma, K, n)`` rather than bootstrapping the observed pooled
#:   cell rates. It is accepted only by :func:`stabilitygate.power` and
#:   :func:`stabilitygate.recommend_design`, because an analyst with data in hand
#:   does not know the truth. :func:`stabilitygate.screen` rejects it.
#:
#: ``precedence="resolution_first"`` is shared with the defaults. Under it
#: INDETERMINATE dominates, so no test rules on a design below the information
#: floor. Under ``precedence="test_first"`` the package excludes a stable model
#: more than twice as often at a saturated baseline.
OC_2026_08_28 = dict(
    lambda_floor=0.50,
    lambda_rule="point",
    alpha=0.05,
    alpha_split="bonferroni",
    run_sided="two",
    combine="or",
    precedence="resolution_first",
    level_statistic="pooled_logit",
    heterogeneity_statistic="pearson_chi2",
    null="oracle",
    seed=20260101,
)

#: Every preset, by name.
PRESETS = {"OC_2026_08_28": OC_2026_08_28}
