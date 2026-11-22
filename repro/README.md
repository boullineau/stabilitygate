# Reproduction harness

One command reproduces every number in the paper and in README section
"Operating characteristics".

```bash
pip install -e .
bash repro/reproduce.sh
```

Run it from the repository root, in an environment built from
`repro/requirements-repro.txt`, which pins numpy 2.2.6. The final line reads
`0 FAIL` when everything reproduced. numpy 2.5 changed how large binomials are
drawn. From that release the rates agree within Monte Carlo error and a few
exact counts differ, which the final stage reports as failures. Measured wall clock on the authors' machine is in the root README
under "Faithfulness and reproducibility".

Two environment variables change what happens.

| Variable | Effect |
|---|---|
| `SG_QUICK=1` | Every stage runs at reduced replicate counts into `results/quick/`, leaving the full outputs alone. Reduced replicates cannot match published rates, so stage 7 verifies against `expected/` instead, generating it locally first via `_ensure_expected.py` if it is not already there (see "`expected/`" below). Completes in well under a minute plus a one-time generation cost. |
| `SG_MDAGREEMENT=<path>` | Enables stage 5. Point it at a local copy of the MD-Agreement corpus, either the released zip, the extracted directory, or `MD-Agreement_dataset.tsv`. With it unset stage 5 explains itself and exits 0, and stage 7 marks its rows SKIPPED rather than FAILED. |

## The seven stages

| Stage | Script | Writes | Backs |
|---|---|---|---|
| 1 | `python -m stabilitygate demo` | console table | the four-model worked case, lambda = .29, r = -.135, 83.9% |
| 2 | `run_grid.py` | `results/simulation_grid.csv`, 1,260 rows | Figure 2, Table 4, the section 4 grid claims |
| 3 | `run_realised.py` | `results/fig3_values.json` | Table 1, Figure 3, the negative-threshold region, Appendix C |
| 4 | `run_extension_oc.py` | `results/extension_design.csv`, `extension_realised.csv`, `extension_joint_counts.csv` | README section "Operating characteristics", the negative-shift arm |
| 5 | `run_panel.py` | `results/mdagreement_removal_counts.csv`, `mdagreement_summary.json` | section 5 and Appendix F, the 8,433 and 3,843 counts |
| 6 | `make_figures.py` | `figures/fig1_inversion.pdf`, `fig2_size.pdf`, `fig3_regime.pdf` | all three figures |
| 7 | `check_paper_numbers.py` | console PASS or FAIL table, exit code | the verification below |

Every stage is deterministic under the seed 20260101, which is fixed inside each
script. Stages 2 and 4 rewrite their deposited outputs byte for byte.

## Verification

`check_paper_numbers.py` reads `paper_numbers.json`, recomputes each entry from
the freshly written outputs, and prints one row per claim.

```bash
python repro/check_paper_numbers.py            # after a full run
python repro/check_paper_numbers.py --verbose  # with the source and recipe of each claim
```

It exits 0 when nothing failed and 1 otherwise, so continuous integration can
gate on it. Each entry names a recipe rather than carrying code, and the recipes
live in a registry in the script, so nothing in the JSON is evaluated and a
reader can follow any claim from the number in the paper to the arithmetic
behind it.

Tolerances are Monte Carlo, at two standard errors on the reported quantity, not
an arbitrary epsilon. The paper's Appendix A states the yardstick, Monte Carlo
standard error on any reported rate is at most .008 at 4,000 replicates and .004
at 20,000. Integer counts carry a tolerance of zero.

## Two further scripts

These are not stages of `reproduce.sh`. They measure the package itself and back
figures in the root README.

| Script | Writes | Backs |
|---|---|---|
| `run_default_grid.py` | `default_grid.csv`, 756 rows of outcome counts | README section "The same grid under the defaults" |
| `run_dependence_check.py` | console lines | README section "Scope", the figures for dependent calls |

`run_default_grid.py` refuses to start a full run while `default_grid.csv`
exists, so the shipped file is never reused or overwritten by accident.
`SG_QUICK=1` writes a fresh `results/quick/default_grid.csv`. Every run checks
its own output and exits with an error if a row is missing or inconsistent.

## `expected/`

Reference outputs for a reader to diff against, and for the fast test suite to
read rather than recomputing per test.

`expected/` is not redistributed. It is not in this repository's history and
not in the packaged sdist or wheel (`MANIFEST.in` prunes it and `.gitignore`
keeps it out of any commit).

What ships instead is the scripts and the seed. The first time anything needs
`expected/` — the fast test suite, or `SG_QUICK=1 bash reproduce.sh` — the
resolver in `tests/conftest.py` calls `_ensure_expected.py`, which runs
`run_grid.py`, `run_realised.py`, and `run_extension_oc.py` once (about a
minute) into `expected/` and caches the result there for the rest of that
checkout. Because every stage is deterministic under the fixed seed, the
generated files are the published numbers when numpy 2.2.6 is installed. Delete the directory to force
regeneration.

`expected/fig3_values.json` carries seven series: five reproduce the deposited
file to the last digit, and two are the scramble comparison of Appendix C,
which the original computed and never wrote out — Appendix C is therefore an
independent replicate of that comparison, not a bit-for-bit reproduction (see
the root README's "Faithfulness and reproducibility" section).

## Provenance

Each script names its origin in its own docstring. In summary, stage 2 comes
from the reproduction package of 11 August 2026, stages 3 and 6 from the figure
scripts of 27 August 2026, stage 4 from the standalone simulation of 28 August
2026, and stage 5 from the quorum analysis of 27 August 2026. Three things
changed in promotion and each is stated in the script that changed.

1. Stage 4 adds `extension_joint_counts.csv`, nine integers per configuration and
   mode giving the joint contingency of the incumbent gate and the proposed
   screen on the same draws. The two existing CSVs carry only marginals, from
   which the like-for-like comparison of the two instruments is not
   reconstructable. It is recoverable from the counts, and it is .348 against the
   screen's .0501.
2. Stage 3 adds the two scramble series, in a second pass that runs after the
   first has finished so the original five series do not move.
3. Stage 5 takes the corpus as an argument and uses only the standard library,
   so the harness keeps numpy and matplotlib as its only dependencies.

## What is not here

The MD-Agreement corpus, which is distributed by its own authors. `run_panel.py`
takes a path to a locally obtained copy and skips with a clear message when it is
absent.
