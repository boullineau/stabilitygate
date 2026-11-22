# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-22

First public release. A research prototype that accompanies *The Screen Decides
the Verdict: Operating Characteristics of Model-Exclusion Rules in LLM
Evaluation*.

### Added

- `screen` runs the three-outcome stability screen on observed main-run and
  retest cell rates and returns `no_drift_detected`, `FAIL` or `INDETERMINATE`,
  with every diagnostic behind the verdict. It reproduces the four-model
  verdicts of the paper's worked case.
- `power` maps the operating characteristics of a planned design across drift
  scenarios. `recommend_design` searches candidate designs for the smallest that
  meets a false-exclusion budget and a drift-detection power target.
- Reliability `lambda` with a bootstrap interval, a signed Woolf run effect and a
  Cochran's Q interaction, each calibrated by the design's own no-drift
  parametric bootstrap.
- Four parameters that expose the decision rule, `alpha_split`, `run_sided`,
  `combine` and `precedence`, on `screen`, `power`, `recommend_design` and the
  matching command-line subcommands. Each is covered by a test asserting that a
  non-default setting changes at least one verdict.
- Two parameters that select the statistics, `level_statistic` and
  `heterogeneity_statistic`, and `null="oracle"` on `power` and
  `recommend_design`.
- `stabilitygate.presets.OC_2026_08_28`, the configuration under which `power`
  reproduces the paper's operating characteristics to within Monte Carlo error.
- `ScreenResult.run_detected_below_floor`, which records that the run-effect
  test rejected on a design below the information floor.
- A command-line interface with `screen`, `power`, `recommend` and `demo`.
- `repro/`, a one-command reproduction of the paper's numbers, and
  `repro/check_paper_numbers.py`, which recomputes 83 published numbers and
  exits non-zero if one moves.
- `repro/run_default_grid.py` and `repro/default_grid.csv`, the paper's 252
  simulated designs measured again under the package defaults.
- The verdict labels are exported as `NO_DRIFT_DETECTED`, `FAIL`,
  `INDETERMINATE` and `VERDICTS`.

### Decision rule

- The admitting verdict is `no_drift_detected`, not PASS. Both component tests
  are calibrated against the design's own no-drift null, so the verdict records
  a failure to reject no drift and not a rejection of consequential drift.
- `precedence="resolution_first"` is the default. A design below the information
  floor returns INDETERMINATE whatever the tests say. This is the decision order
  of the paper's reference screen. The package's test statistics and
  calibration differ from that reference, and the README reports both.
- `alpha_split="bonferroni"` is the default, so each component test runs at
  `alpha / 2` and the union sits near `alpha`.
- Bootstrap p-values count a null draw that ties with the observed statistic.
  A small tolerance keeps exact ties counted when the two are summed in a
  different order.

### Not in this release

- A power criterion for the indeterminate outcome, in place of the floor on
  `lambda`.
- An equivalence test with a stated drift margin.
- The paper's stored reference outputs. `repro/` regenerates them from the
  scripts and the seed.
