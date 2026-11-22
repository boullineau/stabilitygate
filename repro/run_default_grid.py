#!/usr/bin/env python3
"""Operating characteristics of the screen under the package defaults.

The published operating characteristics were measured under the configuration
``stabilitygate.presets.OC_2026_08_28``, whose critical values come from the true
design parameters. An analyst holding data cannot use that null. This script
measures the same 252 configurations under the default decision rule and
statistics of :func:`stabilitygate.power`. Each replicate is tested against a
bootstrap of its own pooled cell rates with the rejection rule
:func:`stabilitygate.screen` applies to observed data, an add-one Monte Carlo
p-value compared with ``alpha / 2``. The script writes one row of outcome counts
per configuration and drift mode to ``repro/default_grid.csv``.

Each replicate uses 499 null draws where ``screen`` uses 20,000, so each
component test runs at 12 / 500 = .024 here against .025 there.

Each configuration draws from its own stream, so the output does not depend on
the order of evaluation.

    python repro/run_default_grid.py            # all 252 configurations
    python repro/run_default_grid.py 0 2        # worker 0 of 2, resumable
    SG_QUICK=1 python repro/run_default_grid.py # reduced replicates, smoke test

A full run refuses to start if its output file already exists, so the shipped
``repro/default_grid.csv`` is never reused or overwritten by accident. Set
``SG_DEFAULT_GRID`` to another path, or delete the file, to run again. A quick
run always writes a fresh ``results/quick/default_grid.csv`` and leaves the
shipped file alone. Only a worker run resumes, from its own part file. Every run
ends by checking that it wrote every expected row and that each row's counts sum
to its replicates, and exits with an error if not.
"""
from __future__ import annotations

import csv
import itertools
import os
import sys
from pathlib import Path

from stabilitygate import power

BASELINES = [0.50, 0.70, 0.85, 0.90, 0.95, 0.98, 0.99]
SIGMAS = [0.25, 0.50, 1.00, 1.50]
KS = [14, 28, 56]
NS = [20, 50, 200]
MODES = ("stable", "shift_1.0", "shift_-1.0")
SEED = 20260101
QUICK = os.environ.get("SG_QUICK") == "1"
N_REP = 200 if QUICK else 2000
N_BOOT = 99 if QUICK else 499

FIELDS = ["baseline", "sigma_logit", "K_cells", "n_per_cell", "mode", "n_rep",
          "n_no_drift_detected", "n_fail", "n_indeterminate", "n_boot", "seed"]


def check(out: Path, expected_configurations: int) -> None:
    """Fail unless ``out`` holds every expected row and consistent counts."""
    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    wanted = expected_configurations * len(MODES)
    if len(rows) != wanted:
        sys.exit(f"{out} holds {len(rows)} rows, expected {wanted}")
    for r in rows:
        total = int(r["n_no_drift_detected"]) + int(r["n_fail"]) + int(r["n_indeterminate"])
        if total != int(r["n_rep"]) or int(r["n_rep"]) != N_REP or int(r["n_boot"]) != N_BOOT:
            sys.exit(f"{out} has an inconsistent row: {r}")
    print(f"wrote and checked {len(rows)} rows in {out}")


def main() -> None:
    worker, workers = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (0, 1)
    root = Path(__file__).resolve().parent
    if "SG_DEFAULT_GRID" in os.environ:
        out = Path(os.environ["SG_DEFAULT_GRID"])
    elif QUICK:
        out = root.parent / "results" / "quick" / "default_grid.csv"
    else:
        out = root / "default_grid.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if workers > 1:
        out = out.with_name(f"{out.stem}.part{worker}.csv")
        if out.exists():
            with out.open() as fh:
                done = {(r["baseline"], r["sigma_logit"], r["K_cells"], r["n_per_cell"])
                        for r in csv.DictReader(fh)}
    elif QUICK:
        out.unlink(missing_ok=True)
    elif out.exists():
        sys.exit(f"{out} already exists. Delete it or set SG_DEFAULT_GRID to another "
                 "path to run the grid again. Nothing was computed.")
    grid = list(itertools.product(BASELINES, SIGMAS, KS, NS))
    mine = [g for index, g in enumerate(grid) if index % workers == worker]
    with out.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if not done:
            writer.writeheader()
        for baseline, sigma, K, n in mine:
            if (f"{baseline:.2f}", f"{sigma:.2f}", str(K), str(n)) in done:
                continue
            rows = power(K=K, n_calls=n, baseline=baseline, sigma_logit=sigma,
                         modes=MODES, n_rep=N_REP, n_boot=N_BOOT, seed=SEED)
            for row in rows:
                writer.writerow({
                    "baseline": f"{baseline:.2f}", "sigma_logit": f"{sigma:.2f}",
                    "K_cells": K, "n_per_cell": n, "mode": row.mode,
                    "n_rep": N_REP,
                    "n_no_drift_detected": round(row.final_pass * N_REP),
                    "n_fail": round(row.final_fail * N_REP),
                    "n_indeterminate": round(row.final_indeterminate * N_REP),
                    "n_boot": N_BOOT, "seed": SEED,
                })
            fh.flush()
    check(out, len(mine))


if __name__ == "__main__":
    main()
