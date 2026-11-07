"""Pre-registered confirmation run of 7 November 2026, Python part (analysis/confirm/prereg-confirm.json).

Experiment S: skill-less searches from the nine families (analysis/v2/nzpy.py), T = 504, N = 20.
Experiment L: skill-less searches of N independent, unit-variance Gaussian AR(1) strategies with phi = 0.02, 0.05, 0.1.
On every search: arch's SPA at the merge commit of pull request 882 (studentize=True, its default; stationary
bootstrap, 1,000 resamples) at mean block length 8 and at its default int(sqrt(T)) = 22, and this repository's
fixed-variance SPA test (spa_c, Hansen's definition) and Reality Check (rc), on their own bootstrap, at 8 and 22.

Seeds: SeedSequence([20261108, experiment (21 = S, 22 = L), cell, chunk]); no earlier run used 20261108 as its first
entropy word (seeds.py checks this). Jobs are dispatched round by round (chunk 0 of every cell, then chunk 1, ...).
Stopping rule (pre-registered): if the run reaches DEADLINE_MIN minutes, it stops and keeps only the rounds that
finished for every cell, so the number of searches is set by time and is the same in every cell.
Usage: python analysis/confirm/run_confirm.py [workers] [rounds] [base] > analysis/confirm/results-py.json
The optional rounds and base arguments are for smoke tests only (the registered run uses 30 and 20261108).
"""
import inspect
import json
import os
import sys
import time
from multiprocessing import Pool

import arch
import numpy as np
from arch.bootstrap import SPA

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v2"))
sys.path.insert(0, HERE)
import nzpy  # noqa: E402

MERGE_COMMIT = "bbfa879a9575ce126178264ac385417f4647fcc1"
BASE = 20261108
CHUNK, ROUNDS, REPS = 100, 30, 1000
N, K = 504, 20
DEFAULT_BLOCK = int(np.sqrt(N))
LEVELS = (0.01, 0.05, 0.1)
PHIS = (0.02, 0.05, 0.1)
DEADLINE_MIN = 85
CELLS = [("S", f) for f in nzpy.FAMILIES] + [("L", phi) for phi in PHIS]
EXPERIMENT_NO = {"S": 21, "L": 22}


def ar1_search(phi, rng, n=N, k=K):
    """k independent stationary Gaussian AR(1) series, coefficient phi, unit variance, mean zero (nzpy's ar1 at any phi)."""
    z = rng.standard_normal((n + 1, k))
    x = np.empty((n, k))
    prev = z[0]
    s = np.sqrt(1 - phi * phi)
    for t in range(n):
        prev = phi * prev + s * z[t + 1]
        x[t] = prev
    return x


def run_chunk(job):
    cell, chunk, base = job
    exp, what = CELLS[cell]
    rng = np.random.default_rng(np.random.SeedSequence([base, EXPERIMENT_NO[exp], cell, chunk]))
    counts = {}
    for _ in range(CHUNK):
        x = nzpy.draw_search(what, rng, n=N, k=K) if exp == "S" else ar1_search(what, rng)
        p = {}
        for label, block in (("b8", 8), ("b22", None)):
            seed = int(rng.integers(2 ** 31 - 1))
            spa = SPA(np.zeros(N), -x, block_size=block, reps=REPS, bootstrap="stationary",
                      seed=np.random.default_rng(seed))
            spa.compute()
            p[f"arch882_{label}_consistent"] = float(spa.pvalues["consistent"])
            p[f"arch882_{label}_upper"] = float(spa.pvalues["upper"])
            ours = nzpy.joint_tests(x, block or DEFAULT_BLOCK, REPS, rng, {"spa_c", "rc"})
            p[f"fixed_spa_c_{label}"] = ours["spa_c"] / REPS
            p[f"rc_{label}"] = ours["rc"] / REPS
        for name, v in p.items():
            row = counts.setdefault(name, [0] * len(LEVELS))
            for i, a in enumerate(LEVELS):
                row[i] += int(v <= a)
    return cell, chunk, counts


def main():
    src = inspect.getsource(sys.modules[SPA.__module__])
    if "_KernelVariance" not in src or "Studentize using the bootstrap sample" not in src:
        raise SystemExit("this arch build does not contain the PR 882 change")
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else ROUNDS
    base = int(sys.argv[3]) if len(sys.argv) > 3 else BASE
    if base == BASE:
        import seeds  # noqa: F401  (raises if the registered seeds overlap an earlier run's)
    jobs = [(cell, r, base) for r in range(rounds) for cell in range(len(CELLS))]
    t0 = time.time()
    done = {}
    stopped = False
    sys.stderr.write(f"confirm (python): START {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} jobs={len(jobs)} workers={workers}\n")
    pool = Pool(workers)
    try:
        for i, (cell, chunk, counts) in enumerate(pool.imap_unordered(run_chunk, jobs), 1):
            done[(cell, chunk)] = counts
            el = time.time() - t0
            if i % 12 == 0:
                sys.stderr.write(f"confirm (python): {i}/{len(jobs)} chunks, {el / 60:.1f} min\n")
            if el > DEADLINE_MIN * 60:
                stopped = True
                break
    finally:
        pool.terminate()
        pool.join()
    complete = 0
    while complete < rounds and all((c, complete) in done for c in range(len(CELLS))):
        complete += 1
    out = {"schema": "canli.null-zoo.confirm.results-py", "registered": True,
           "prereg": "analysis/confirm/prereg-confirm.json", "base_seed": base, "arch": arch.__version__,
           "arch_merge_commit": MERGE_COMMIT, "numpy": np.__version__, "chunk": CHUNK, "rounds_planned": rounds,
           "rounds_complete": complete, "stopped_at_deadline": stopped, "deadline_minutes": DEADLINE_MIN,
           "searches_per_cell": complete * CHUNK, "bootstrap_reps": REPS, "n": N, "k": K,
           "default_block": DEFAULT_BLOCK, "levels": list(LEVELS), "minutes": round((time.time() - t0) / 60, 1),
           "cells": {}}
    for c, (exp, what) in enumerate(CELLS):
        tot = {}
        for r in range(complete):
            for name, row in done[(c, r)].items():
                tot[name] = [a + b for a, b in zip(tot.get(name, [0] * len(LEVELS)), row)]
        label = f"S.{what}" if exp == "S" else f"L.phi{what:g}"
        out["cells"][label] = {"searches": complete * CHUNK, "rejections": tot}
    sys.stdout.write(json.dumps(out, sort_keys=True, indent=1) + "\n")
    sys.stderr.write(f"confirm (python): DONE in {(time.time() - t0) / 60:.1f} min, {complete} complete rounds\n")


if __name__ == "__main__":
    main()
