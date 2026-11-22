#!/usr/bin/env python3
"""False exclusion of a stable model when calls within a cell are dependent.

The screen assumes independent calls within a cell. This script draws each
cell's calls from a beta-binomial with a chosen intraclass correlation, keeps
the model stable across the two runs, and reports how often the screen returns
FAIL under the package defaults. It prints one line per design and writes
nothing. README section "Scope" quotes the first three lines. Each replicate
is decided by the rule ``screen`` applies, with 499 null draws.

    python repro/run_dependence_check.py
"""
import math

import numpy as np

from stabilitygate import _core
def run(K, n, baseline, sigma, rho, n_rep=2000, n_boot=499, seed=20260101):
    rng = np.random.default_rng([seed, K, n, int(baseline * 1000), int(sigma * 100), int(rho * 1000)])
    mu = math.log(baseline / (1 - baseline))
    p = 1 / (1 + np.exp(-rng.normal(mu, sigma, (n_rep, K))))

    def draw():
        if rho == 0:
            return rng.binomial(n, p) / n
        return rng.binomial(n, rng.beta(p * (1 - rho) / rho, (1 - p) * (1 - rho) / rho)) / n

    A, B = draw(), draw()
    z, q = _core.statistics_batch(A, B, n)
    level = _core.component_alpha(0.05, "bonferroni")
    pooled = (A + B) / 2
    run_hit = np.zeros(n_rep, dtype=bool)
    int_hit = np.zeros(n_rep, dtype=bool)
    for i in range(n_rep):
        X = rng.binomial(n, pooled[i], size=(n_boot, K)) / n
        Y = rng.binomial(n, pooled[i], size=(n_boot, K)) / n
        zb, qb = _core.statistics_batch(X, Y, n)
        p_run, p_int = _core.null_p_values(z[i], q[i], zb, qb, "two")
        run_hit[i], int_hit[i] = p_run < level, p_int < level
    lam = np.array([_core.lam(A[i], n) for i in range(n_rep)])
    fail, ind, _ = _core.resolve_outcome(run_hit, int_hit, lam >= 0.5)
    return dict(K=K, n=n, baseline=baseline, sigma=sigma, icc=rho,
                n_fail=int(fail.sum()), n_indeterminate=int(ind.sum()), n_rep=n_rep,
                fail=round(float(fail.mean()), 4))


for (K, n, b, s) in [(14, 20, 0.5, 1.0), (14, 20, 0.7, 1.0), (28, 50, 0.5, 1.0)]:
    for rho in (0.0, 0.05, 0.10):
        print(run(K, n, b, s, rho), flush=True)
