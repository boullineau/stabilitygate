#!/usr/bin/env bash
#
# One command reproducing every number in the paper and in README section
# "Operating characteristics".
#
#   bash repro/reproduce.sh
#
# Run from the repository root with the package installed (pip install -e .).
# It takes no arguments and reads two environment variables.
#
#   SG_QUICK=1                  run every stage at reduced replicate counts as a
#                               smoke test, in a scratch directory, leaving the
#                               full outputs in place. Reduced replicates cannot
#                               match published rates, so stage 7 verifies
#                               against repro/expected instead. repro/expected
#                               is not redistributed, so this generates it
#                               locally first if it is not already there. Completes in a minute or two.
#   SG_MDAGREEMENT=/path        enable stage 5 by pointing at a local copy of the
#                               MD-Agreement corpus, either the released zip, the
#                               extracted directory, or MD-Agreement_dataset.tsv.
#                               With it unset stage 5 explains itself, exits 0,
#                               and stage 7 marks its rows SKIPPED.
#
# Exits 0 when every stage succeeded and stage 7 reported no failures.

set -euo pipefail

REPRO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$REPRO")"
PYTHON="${SG_PYTHON:-python3}"
cd "$ROOT"

QUICK="${SG_QUICK:-0}"
if [ "$QUICK" = "1" ]; then
    export SG_RESULTS="$ROOT/results/quick"
    LABEL="smoke run, reduced replicates"
else
    export SG_RESULTS="${SG_RESULTS:-$ROOT/results}"
    LABEL="full run"
fi
export SG_FIGURES="${SG_FIGURES:-$ROOT/figures}"
mkdir -p "$SG_RESULTS" "$SG_FIGURES"

STARTED=$(date +%s)
STAGE=0

rule() { printf '%s\n' "------------------------------------------------------------------------"; }

stage() {
    STAGE=$((STAGE + 1))
    printf '\n'
    rule
    printf 'Stage %d of 7   %s\n' "$STAGE" "$1"
    rule
}

printf 'stabilitygate reproduction (%s)\n' "$LABEL"
printf 'results  %s\n' "$SG_RESULTS"
printf 'figures  %s\n' "$SG_FIGURES"
printf 'python   %s\n' "$($PYTHON -c 'import sys;print(".".join(str(v) for v in sys.version_info[:3]))')"
printf 'numpy    %s\n' "$($PYTHON -c 'import numpy;print(numpy.__version__)')"

stage "the four-model worked case"
$PYTHON -m stabilitygate demo

stage "the design grid, six screens under four failure modes"
$PYTHON "$REPRO/run_grid.py"

stage "the size-matched comparison at the realised design"
$PYTHON "$REPRO/run_realised.py"

stage "operating characteristics of the proposed screen, both directions"
$PYTHON "$REPRO/run_extension_oc.py"

stage "panel loss in a published five-annotator corpus"
$PYTHON "$REPRO/run_panel.py"

stage "figures"
$PYTHON "$REPRO/make_figures.py"

stage "verification"
if [ "$QUICK" = "1" ]; then
    printf 'Reduced replicates cannot match published rates, so this verifies the\n'
    printf 'reference outputs instead and treats the smoke run as a stage check.\n'
    printf 'repro/expected/ is not redistributed, so a fresh checkout generates it\n'
    printf 'here, once, from the scripts and seed above.\n\n'
    $PYTHON "$REPRO/_ensure_expected.py" >/dev/null
    $PYTHON "$REPRO/check_paper_numbers.py" --results "$REPRO/expected" --extra "$SG_RESULTS"
else
    $PYTHON "$REPRO/check_paper_numbers.py"
fi

ELAPSED=$(($(date +%s) - STARTED))
printf '\n'
rule
printf 'Reproduction complete in %dm %02ds (%s).\n' $((ELAPSED / 60)) $((ELAPSED % 60)) "$LABEL"
rule
