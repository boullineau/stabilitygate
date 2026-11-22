#!/usr/bin/env python3
"""Stage 3. Size-matched comparison at the realised design.

Promoted into this harness from the figure script of 27 August 2026. It backs
Table 1, Figure 3, the negative-threshold region and the two daggered cells.

The realised design is K = 14 conditions, n = 20 calls, sigma = .5 logits, at
20,000 replicates per cell. For each baseline the script

  1. draws 20,000 stable replicates and records the registered r > .80 rule's
     false-exclusion rate under the registered undefined-value convention, which
     resolves an undefined correlation to exclusion,
  2. calibrates both screens on those same stable draws, taking the correlation
     threshold at the 5th percentile of defined correlations and the unsigned
     statistic's at the 95th percentile, so both sit at 5 per cent size among
     draws with a defined correlation,
  3. draws 20,000 replicates carrying a positive one-logit shift and reports
     each screen's power at that matched size.

THE SCRAMBLE SERIES, NEW IN THIS HARNESS
Appendix C reports that the correlation has somewhat more power than the
unsigned statistic against a pure rank permutation at low baselines, .692
against .632 at p0 = .50. That comparison was computed but never written out, so
nothing in the deposited outputs covered it. A second pass adds it.

The second pass runs after the first has finished rather than inside it, so it
does not advance the random stream the first pass reads. ``fig3_values.json``
therefore keeps its five original series unchanged to the last digit and gains
two more.

Read-only on all inputs. Deterministic under the registered seed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("SG_RESULTS", Path(__file__).resolve().parent.parent / "results"))
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260101
QUICK = os.environ.get("SG_QUICK") == "1"
K, n, SIG = 14, 20, 0.5
N = 2000 if QUICK else 20000

BASELINES = [0.50, 0.70, 0.85, 0.90, 0.95, 0.98, 0.99]


def draw(p0, sig, mode, reps, rng):
    lo = np.log(p0 / (1 - p0))
    th = rng.normal(lo, sig, size=(reps, K))
    a = rng.binomial(n, 1 / (1 + np.exp(-th))) / n
    th2 = th.copy() if mode == "stable" else th + 1.0
    b = rng.binomial(n, 1 / (1 + np.exp(-th2))) / n
    return a, b


def draw_scramble(p0, sig, reps, rng):
    """A pure rank permutation. The level is held and only the profile moves."""
    lo = np.log(p0 / (1 - p0))
    th = rng.normal(lo, sig, size=(reps, K))
    a = rng.binomial(n, 1 / (1 + np.exp(-th))) / n
    idx = np.argsort(rng.random((reps, K)), axis=1)
    th2 = np.take_along_axis(th, idx, axis=1)
    b = rng.binomial(n, 1 / (1 + np.exp(-th2))) / n
    return a, b


def pear(a, b):
    az = a - a.mean(1, keepdims=True)
    bz = b - b.mean(1, keepdims=True)
    d = np.sqrt((az**2).sum(1) * (bz**2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (az * bz).sum(1) / d


def chi2T(a, b):
    pb = (a + b) / 2
    v = 2 * pb * (1 - pb) / n
    v = np.where(v <= 0, np.nan, v)
    return np.nansum((a - b) ** 2 / v, axis=1)


def main() -> None:
    rng = np.random.default_rng(SEED)

    reg_fe, pw_p, pw_c, thr = [], [], [], []
    chi_crit = []
    for p0 in BASELINES:
        a, b = draw(p0, SIG, "stable", N, rng)
        r0 = pear(a, b)
        t0 = chi2T(a, b)
        reg_fe.append(float(np.where(np.isnan(r0), True, r0 <= 0.80).mean()))
        rt = float(np.nanpercentile(r0, 5))
        ct = float(np.nanpercentile(t0, 95))
        thr.append(rt)
        chi_crit.append(ct)
        aa, bb = draw(p0, SIG, "shift_1.0", N, rng)
        rr = pear(aa, bb)
        tt = chi2T(aa, bb)
        pw_p.append(1 - float(np.where(np.isnan(rr), False, rr > rt).mean()))
        pw_c.append(1 - float((tt <= ct).mean()))

    # second pass, after the first has finished, so the stream above is untouched
    pw_p_scramble, pw_c_scramble = [], []
    for i, p0 in enumerate(BASELINES):
        aa, bb = draw_scramble(p0, SIG, N, rng)
        rr = pear(aa, bb)
        tt = chi2T(aa, bb)
        pw_p_scramble.append(1 - float(np.where(np.isnan(rr), False, rr > thr[i]).mean()))
        pw_c_scramble.append(1 - float((tt <= chi_crit[i]).mean()))

    values = {
        "bl": BASELINES,
        "reg_fe": reg_fe,
        "pw_p": pw_p,
        "pw_c": pw_c,
        "thr": thr,
        "pw_p_scramble": pw_p_scramble,
        "pw_c_scramble": pw_c_scramble,
    }
    with (OUT / "fig3_values.json").open("w") as f:
        json.dump(values, f, indent=1)

    head = (f"{'baseline':>9}{'reg r>.80':>11}{'power r':>9}{'power U':>9}"
            f"{'r threshold':>13}{'scramble r':>12}{'scramble U':>12}")
    print(head)
    print("-" * len(head))
    for i, p0 in enumerate(BASELINES):
        print(f"{p0:>9.2f}{reg_fe[i]:>11.3f}{pw_p[i]:>9.3f}{pw_c[i]:>9.3f}"
              f"{thr[i]:>+13.3f}{pw_p_scramble[i]:>12.3f}{pw_c_scramble[i]:>12.3f}")
    print()
    print(f"Wrote {OUT / 'fig3_values.json'} ({N} replicates per cell)")


if __name__ == "__main__":
    main()
