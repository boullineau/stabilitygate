#!/usr/bin/env python3
"""Stage 4. Operating characteristics of the proposed screen, both shift directions.

Promoted into this harness from the standalone simulation of 28 August 2026. The
statistics, the calibration, the seed and the grid are unchanged, so this file
rewrites ``extension_design.csv`` and ``extension_realised.csv`` byte for byte.
One output is new, the joint counts described at the end of this docstring.

ARM 1, NEGATIVE SHIFT.
The body of the paper simulates only a positive one-logit shift and concedes in
Limitations that near a ceiling an equal negative shift is easier to detect.
This runs the negative arm on the same grid so the concession becomes a number.

ARM 2, OPERATING CHARACTERISTICS OF THE PROPOSED SCREEN.
Section 6 proposes a three-outcome screen without reporting how it behaves.
This implements it and reports false exclusion, power and the indeterminate
share, so the replacement is held to the standard applied to the incumbent.

The proposed screen, as specified in Section 6.
  1. Compute lambda on the main run. If lambda < LAMBDA_FLOOR the design cannot
     resolve the target drift and the screen returns INDETERMINATE.
  2. Otherwise apply a signed level test on the mean log-odds difference and a
     condition-by-run heterogeneity test, the same chi-square statistic the
     incumbent grid already uses.
  3. ADMIT only if neither test rejects. EXCLUDE if either rejects.

CALIBRATION, AND WHY EACH COMPONENT RUNS AT ALPHA/2
Both critical values are taken at the 100*(1 - ALPHA/2) percentile of that
design's own stable null, so each component test runs at 2.5 per cent rather
than 5. Because the screen excludes when EITHER test rejects, the union of two
positively correlated 2.5 per cent tests lands near 5 per cent overall. This is
a Bonferroni split deliberately retained. Sidak would give 0.0253 per component.
Running each component at a true 5 per cent would give a pooled union near 9.8
per cent, roughly double nominal. The realised pooled conditional size over the
design grid is 0.0501.

Note that chi_crit here will NOT match chi2_crit in the deposited
simulation_grid.csv, which calibrates that statistic alone at the 95th
percentile because it is applied as a single test there.

THIS CONFIGURATION IS REACHABLE FROM THE PACKAGE
Everything this file computes is also reachable through
``stabilitygate.power`` under ``stabilitygate.presets.OC_2026_08_28``. The
package's own defaults share this decision rule and use different statistics and
a bootstrap null, so they do not reproduce these files draw for draw.
``repro/run_default_grid.py`` measures the same grid under the defaults.

DEPARTURES FROM SECTION 6, STATED RATHER THAN HIDDEN
(a) Section 6 triggers indeterminate on a power criterion via recommend_design.
    A lambda floor of 0.50 is substituted here as a computable proxy.
(b) Section 6 frames the rule as an equivalence-testing move, where admission
    would require rejecting the presence of consequential drift. This
    implementation calibrates against the stable null instead, so admission is
    failure to reject no drift. That is a weaker criterion and it is the reason
    the screen still admits a shifted model in a minority of poorly resolved
    configurations.
(c) Sidedness, level and the OR combination are choices Section 6 does not fix.
(d) Section 6 specifies lambda AND its bootstrap interval. This uses a point
    estimate against a hard cut, which would change the indeterminate rate.

THE JOINT COUNTS, NEW IN THIS HARNESS
The two CSVs above carry only per-cell marginal means. Any figure needing the
JOINT distribution of the incumbent gate and the proposed screen on the same
draws is not reconstructable from them. ``extension_joint_counts.csv`` adds nine
integers per configuration and mode. The six screen-by-gate cells partition the
replicates, and the three undefined-correlation columns are a subset of the
gate-fail cells, because an undefined correlation resolves to a gate failure
under the registered convention. Every marginal the two existing files carry is
derivable from these counts, so those files do not move.

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
TAU_R = 0.80
LAMBDA_FLOOR = 0.50
ALPHA = 0.05          # union target, each component runs at ALPHA/2, see header

BASELINES = [0.50, 0.70, 0.85, 0.90, 0.95, 0.98, 0.99]
SIGMAS = [0.25, 0.50, 1.00, 1.50]
DESIGN_KS = [14, 28, 56]
DESIGN_NS = [20, 50, 200]
K_REAL, N_REAL = 14, 20

MODES = ["stable", "shift_1.0", "shift_-1.0"]

JOINT_COLUMNS = [
    "n_admit_gatepass", "n_admit_gatefail",
    "n_exclude_gatepass", "n_exclude_gatefail",
    "n_indet_gatepass", "n_indet_gatefail",
    "n_undef_admit", "n_undef_exclude", "n_undef_indet",
]

expit = lambda x: 1.0 / (1.0 + np.exp(-x))


def draw(pbar, sigma, K, n, mode, reps, rng):
    mu = math.log(pbar / (1 - pbar))
    theta = rng.normal(mu, sigma, (reps, K))
    if mode == "stable":
        theta2 = theta
    elif mode.startswith("shift"):
        theta2 = theta + float(mode.split("_")[1])
    else:
        raise ValueError(mode)
    a = rng.binomial(n, expit(theta)) / n
    b = rng.binomial(n, expit(theta2)) / n
    return a, b


def rowwise_corr(a, b):
    az = a - a.mean(1, keepdims=True); bz = b - b.mean(1, keepdims=True)
    num = (az * bz).sum(1); den = np.sqrt((az**2).sum(1) * (bz**2).sum(1))
    out = np.zeros(len(a)); ok = den > 0; out[ok] = num[ok] / den[ok]
    return out, ~ok            # also return the undefined mask


def chi2_T(a, b, n):
    phat = (a + b) / 2.0
    var = phat * (1 - phat) * (2.0 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(var > 0, (a - b) ** 2 / var, 0.0)
    return t.sum(1)


def lam_rows(a, n):
    obs = a.var(1, ddof=1); samp = (a * (1 - a) / n).mean(1)
    out = np.zeros(len(a)); ok = obs > 0
    out[ok] = np.maximum(0.0, 1 - samp[ok] / obs[ok])
    return out


def signed_level(a, b):
    """Mean log-odds difference. The averaged rate is clipped at 1e-9 rather
    than Haldane-Anscombe corrected, so the statistic saturates near 20.7 in
    degenerate cells. Three grid cells at p0=.99, K=14, n=20 inherit critical
    values of 16.489, 16.489 and 16.198 from this, which the statistic exceeds
    only rarely, so the level test fires on 1.6 to 11.3 per cent of draws there
    rather than never. Rerunning those cells with a pooled-count
    Haldane-Anscombe correction moves admit rates by at most 0.003, inside
    Monte Carlo noise."""
    ma = np.clip(a.mean(1), 1e-9, 1 - 1e-9); mb = np.clip(b.mean(1), 1e-9, 1 - 1e-9)
    return np.log(mb / (1 - mb)) - np.log(ma / (1 - ma))


def calibrate(pbar, sigma, K, n, rng):
    """Critical values from THIS design's own stable null. Two sided for level."""
    a, b = draw(pbar, sigma, K, n, "stable", N_CAL, rng)
    lev = np.abs(signed_level(a, b))
    return (float(np.percentile(lev, 100 * (1 - ALPHA / 2))),
            float(np.percentile(chi2_T(a, b, n), 100 * (1 - ALPHA / 2))))


def proposed_screen(a, b, n, lev_crit, chi_crit):
    lam = lam_rows(a, n)
    indet = lam < LAMBDA_FLOOR
    rej = (np.abs(signed_level(a, b)) > lev_crit) | (chi2_T(a, b, n) > chi_crit)
    admit = (~indet) & (~rej)
    exclude = (~indet) & rej
    return admit, exclude, indet


def joint_counts(admit, exclude, indet, gate_pass, undef):
    """The nine joint-count integers, from per-draw vectors already computed."""
    return {
        "n_admit_gatepass": int((admit & gate_pass).sum()),
        "n_admit_gatefail": int((admit & ~gate_pass).sum()),
        "n_exclude_gatepass": int((exclude & gate_pass).sum()),
        "n_exclude_gatefail": int((exclude & ~gate_pass).sum()),
        "n_indet_gatepass": int((indet & gate_pass).sum()),
        "n_indet_gatefail": int((indet & ~gate_pass).sum()),
        "n_undef_admit": int((undef & admit).sum()),
        "n_undef_exclude": int((undef & exclude).sum()),
        "n_undef_indet": int((undef & indet).sum()),
    }


def run(ks, ns, tag, rng, joint_rows):
    rows = []
    for pbar, sigma, K, n in itertools.product(BASELINES, SIGMAS, ks, ns):
        lev_crit, chi_crit = calibrate(pbar, sigma, K, n, rng)
        for mode in MODES:
            a, b = draw(pbar, sigma, K, n, mode, N_REP, rng)
            r, undef = rowwise_corr(a, b)
            pearson_pass = (r > TAU_R)
            adm, exc, ind = proposed_screen(a, b, n, lev_crit, chi_crit)
            keys = {
                "baseline": pbar, "sigma_logit": sigma, "K_cells": K,
                "n_per_cell": n, "mode": mode, "n_rep": N_REP,
            }
            rows.append({
                **keys,
                "pearson_pass": round(float(pearson_pass.mean()), 4),
                "pearson_undefined": round(float(undef.mean()), 4),
                "proposed_admit": round(float(adm.mean()), 4),
                "proposed_exclude": round(float(exc.mean()), 4),
                "proposed_indeterminate": round(float(ind.mean()), 4),
                "lev_crit": round(lev_crit, 4), "chi_crit": round(chi_crit, 2),
            })
            joint_rows.append({
                **keys, "arm": tag,
                **joint_counts(adm, exc, ind, pearson_pass, undef),
            })
    with (OUT / f"extension_{tag}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return rows


def main():
    rng = np.random.default_rng(SEED)
    joint_rows: list[dict] = []
    realised = run([K_REAL], [N_REAL], "realised", rng, joint_rows)
    full = run(DESIGN_KS, DESIGN_NS, "design", rng, joint_rows)
    with (OUT / "extension_joint_counts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(joint_rows[0].keys()))
        w.writeheader(); w.writerows(joint_rows)
    print(f"wrote {len(realised)} realised rows and {len(full)} design rows")
    print(f"wrote {len(joint_rows)} joint-count rows to extension_joint_counts.csv")


if __name__ == "__main__":
    main()
