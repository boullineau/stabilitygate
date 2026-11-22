#!/usr/bin/env python3
"""Stage 2. False exclusion and power of six screens across the design grid.

Promoted into this harness from the reproduction package of 11 August 2026. The
data-generating process, the six screens, the seed and the grid are unchanged.
One thing is renamed. The published grid, the 1,260-row file the paper cites, is
the DESIGN grid that crosses K and n, and the script that produced it wrote that
grid second, after the realised-design pass had already advanced the random
stream. Both passes therefore run here, in that order, and the design pass is
written as ``simulation_grid.csv`` under the name the paper uses. The realised
pass is written as ``simulation_realised_design.csv`` so nothing is discarded.

DATA-GENERATING PROCESS
K prompt-by-condition cells. Main-run propensities on the logit scale,
``theta_k ~ N(logit(pbar), sigma^2)``. Stability-run propensities depend on the
failure mode.

    stable    theta' = theta                  no drift at all
    shift_d   theta' = theta + d              uniform provider drift, rank order
                                              fully preserved
    scramble  theta' = permute(theta)         rank instability, marginals held
    idio_d    theta' = theta + N(0, d^2)      cell-specific drift

Observed rates are binomial at the design's calls per cell.

THE SIX SCREENS
    pearson   Pearson r > .80             the incumbent, this study's registered rule
    spearman  Spearman rho > .80          does a rank statistic rescue it
    ccc       Lin's concordance > .80     preempts "just use an agreement statistic"
    absdrift  mean|d| <= .05 and max|d| <= .15
    logit     |mean logit drift| <= .25
    chi2      noise-calibrated standardised drift, critical value refit per design
              from the stable null at the 95th percentile

READING THE OUTPUT
PASS means the screen says "stable, retain this model".
    mode = stable, a FAIL is an error, so false exclusion is 1 - pass.
    mode = drift, a PASS is an error, so false retention is pass.
A usable screen needs false exclusion at or below .05 AND power against real drift.

Read-only on all inputs. Deterministic under the registered seed.
"""
from __future__ import annotations

import csv
import itertools
import math
import os
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("SG_RESULTS", Path(__file__).resolve().parent.parent / "results"))
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260101
QUICK = os.environ.get("SG_QUICK") == "1"
N_REP = 400 if QUICK else 4000
N_CAL = 400 if QUICK else 4000

TAU_R = 0.80          # the registered gate, prereg Section 3.3 requires r > 0.80
ABS_MEAN, ABS_MAX = 0.05, 0.15
LOGIT_TAU = 0.25
LAMBDA_FLOOR = 0.50

BASELINES = [0.50, 0.70, 0.85, 0.90, 0.95, 0.98, 0.99]
SIGMAS = [0.25, 0.50, 1.00, 1.50]
MODES = ["stable", "shift_0.5", "shift_1.0", "scramble", "idio_0.5"]

K_REAL, N_REAL = 14, 20          # the realised design
DESIGN_KS = [14, 28, 56]
DESIGN_NS = [20, 50, 200]
if QUICK:
    DESIGN_KS, DESIGN_NS = [14, 28], [20, 50]

GATES = ["pearson", "spearman", "ccc", "absdrift", "logit", "chi2"]


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def draw(pbar, sigma, K, n, mode, reps, rng):
    """Vectorised. Returns (a, b), each (reps, K) observed cell rates."""
    mu = math.log(pbar / (1 - pbar))
    theta = rng.normal(mu, sigma, (reps, K))

    if mode == "stable":
        theta2 = theta
    elif mode.startswith("shift"):
        theta2 = theta + float(mode.split("_")[1])
    elif mode == "scramble":
        idx = np.argsort(rng.random((reps, K)), axis=1)
        theta2 = np.take_along_axis(theta, idx, axis=1)
    elif mode.startswith("idio"):
        theta2 = theta + rng.normal(0, float(mode.split("_")[1]), (reps, K))
    else:
        raise ValueError(mode)

    a = rng.binomial(n, expit(theta)) / n
    b = rng.binomial(n, expit(theta2)) / n
    return a, b


def rowwise_corr(a, b):
    """Pearson r per row, 0 where a row has no variance, so the gate cannot pass."""
    az = a - a.mean(1, keepdims=True)
    bz = b - b.mean(1, keepdims=True)
    num = (az * bz).sum(1)
    den = np.sqrt((az**2).sum(1) * (bz**2).sum(1))
    out = np.zeros(len(a))
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def rank_rows(x, rng):
    """Average-rank transform per row, ties broken consistently."""
    order = np.argsort(x + rng.random(x.shape) * 1e-12, axis=1)
    ranks = np.empty_like(order, dtype=float)
    np.put_along_axis(ranks, order, np.arange(x.shape[1], dtype=float), axis=1)
    return ranks


def rowwise_ccc(a, b):
    ma, mb = a.mean(1), b.mean(1)
    va, vb = a.var(1), b.var(1)
    cov = ((a - ma[:, None]) * (b - mb[:, None])).mean(1)
    den = va + vb + (ma - mb) ** 2
    out = np.zeros(len(a))
    ok = den > 0
    out[ok] = 2 * cov[ok] / den[ok]
    return out


def chi2_T(a, b, n):
    phat = (a + b) / 2.0
    var = phat * (1 - phat) * (2.0 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(var > 0, (a - b) ** 2 / var, 0.0)
    return t.sum(1)


def lam_rows(a, n):
    obs = a.var(1, ddof=1)
    samp = (a * (1 - a) / n).mean(1)
    out = np.zeros(len(a))
    ok = obs > 0
    out[ok] = np.maximum(0.0, 1 - samp[ok] / obs[ok])
    return out


def evaluate(a, b, n, chi2_crit, rng):
    d = np.abs(a - b)
    ra, rb = rank_rows(a, rng), rank_rows(b, rng)
    ma = np.clip(a.mean(1), 1e-9, 1 - 1e-9)
    mb = np.clip(b.mean(1), 1e-9, 1 - 1e-9)
    dlogit = np.abs(np.log(mb / (1 - mb)) - np.log(ma / (1 - ma)))
    return {
        "pearson": rowwise_corr(a, b) > TAU_R,
        "spearman": rowwise_corr(ra, rb) > TAU_R,
        "ccc": rowwise_ccc(a, b) > TAU_R,
        "absdrift": (d.mean(1) <= ABS_MEAN) & (d.max(1) <= ABS_MAX),
        "logit": dlogit <= LOGIT_TAU,
        "chi2": chi2_T(a, b, n) <= chi2_crit,
        "lambda_ok": lam_rows(a, n) >= LAMBDA_FLOOR,
    }


def calibrate_chi2(pbar, sigma, K, n, rng):
    """95th percentile of T under PERFECT stability, so nominal .05 false exclusion."""
    a, b = draw(pbar, sigma, K, n, "stable", N_CAL, rng)
    return float(np.percentile(chi2_T(a, b, n), 95))


def run_grid(ks, ns, filename, rng):
    rows = []
    for pbar, sigma, K, n in itertools.product(BASELINES, SIGMAS, ks, ns):
        crit = calibrate_chi2(pbar, sigma, K, n, rng)
        for mode in MODES:
            a, b = draw(pbar, sigma, K, n, mode, N_REP, rng)
            res = evaluate(a, b, n, crit, rng)
            row = {
                "baseline": pbar, "sigma_logit": sigma, "K_cells": K,
                "n_per_cell": n, "mode": mode, "n_rep": N_REP,
                "pct_lambda_ge_.5": round(float(res["lambda_ok"].mean()), 4),
                "chi2_crit": round(crit, 2),
            }
            for g in GATES:
                pr = float(res[g].mean())
                row[f"pass_{g}"] = round(pr, 4)
                row[f"err_{g}"] = round(1 - pr if mode == "stable" else pr, 4)
            rows.append(row)
    with (OUT / filename).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def main() -> None:
    rng = np.random.default_rng(SEED)

    print("Stage 2, realised design (K = 14 cells, n = 20 calls per cell)")
    realised = run_grid([K_REAL], [N_REAL], "simulation_realised_design.csv", rng)

    print("Stage 2, design grid (K and n varying). This is the published grid.")
    design = run_grid(DESIGN_KS, DESIGN_NS, "simulation_grid.csv", rng)

    stable = [r for r in design if r["mode"] == "stable"]
    print()
    print(f"{'screen':<24}{'median':>9}{'min':>9}{'max':>9}")
    print("-" * 51)
    order = sorted(
        GATES, key=lambda g: -float(np.median([r["err_" + g] for r in stable]))
    )
    for g in order:
        v = np.array([r["err_" + g] for r in stable])
        print(f"{g:<24}{np.median(v):>9.3f}{v.min():>9.3f}{v.max():>9.3f}")
    print()
    print(f"Wrote {OUT / 'simulation_grid.csv'} ({len(design)} rows, "
          f"{len(stable)} stable configurations)")
    print(f"Wrote {OUT / 'simulation_realised_design.csv'} ({len(realised)} rows)")


if __name__ == "__main__":
    main()
