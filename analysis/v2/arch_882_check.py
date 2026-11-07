"""Unregistered check (added 2026-11-07): arch's SPA after pull request 882 was merged.

After issue 879 (opened 7 October 2026) reported that the fixed-variance studentization merged in PR 871 was
oversized, arch's maintainer merged PR 882 the same day (merge commit bbfa879a9575ce126178264ac385417f4647fcc1,
7 October 2026, 15:25 UTC). With studentize=True, the default, it divides each bootstrap resample by its own
standard error, the square root of the bootstrap-implied kernel long-run variance computed from the resample
(for the stationary bootstrap, the Politis-Romano weights). The change is scheduled for arch 9.0, unreleased on
7 November 2026 (latest release 8.0.0).

This is the first external implementation of a studentized data-snooping test run in the study. The script runs
arch at that merge commit on 2,000 new null searches per family in all nine families, at mean block length 8 and at
arch's default (int(sqrt(n)) = 22), beside this repository's fixed-variance test (spa_c, Hansen's definition) and
Reality Check on the same searches. It was not pre-registered; it reports whatever it finds.
Usage: python analysis/v2/arch_882_check.py [workers] > analysis/v2/arch-882-check.json
An optional second argument sets the number of chunks of 100 searches per family (smoke tests only).
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nzpy  # noqa: E402

MERGE_COMMIT = "bbfa879a9575ce126178264ac385417f4647fcc1"
BASE = 20261107
FAMILIES = nzpy.FAMILIES
SEARCHES = 2000
CHUNK = 100
REPS = 1000
N, K = 504, 20
DEFAULT_BLOCK = int(np.sqrt(N))


def run_chunk(job):
    f, chunk = job
    rng = np.random.default_rng(np.random.SeedSequence([BASE, 11, f, chunk]))
    rows = []
    for _ in range(CHUNK):
        x = nzpy.draw_search(FAMILIES[f], rng, n=N, k=K)
        row = {}
        for label, block in (("b8", 8), ("default", None)):
            seed = int(rng.integers(2 ** 31 - 1))
            spa = SPA(np.zeros(N), -x, block_size=block, reps=REPS, bootstrap="stationary",
                      seed=np.random.default_rng(seed))
            spa.compute()
            row[f"arch_{label}_consistent"] = float(spa.pvalues["consistent"])
            row[f"arch_{label}_upper"] = float(spa.pvalues["upper"])
            ours = nzpy.joint_tests(x, block or DEFAULT_BLOCK, REPS, rng, {"spa_c", "rc"})
            row[f"ours_{label}_spa_c"] = ours["spa_c"] / REPS
            row[f"ours_{label}_rc"] = ours["rc"] / REPS
        rows.append(row)
    return f, chunk, rows


def main():
    src = inspect.getsource(sys.modules[SPA.__module__])
    if "_KernelVariance" not in src or "Studentize using the bootstrap sample" not in src:
        raise SystemExit("this arch build does not contain the PR 882 change")
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    chunks = int(sys.argv[2]) if len(sys.argv) > 2 else SEARCHES // CHUNK
    jobs = [(f, c) for f in range(len(FAMILIES)) for c in range(chunks)]
    t0 = time.time()
    sys.stderr.write(f"arch 882 check: START jobs={len(jobs)} workers={workers}\n")
    with Pool(workers) as pool:
        res = sorted(pool.map(run_chunk, jobs), key=lambda r: (r[0], r[1]))
    out = {"schema": "canli.null-zoo.v2.arch-882-check", "registered": False, "arch": arch.__version__,
           "arch_merge_commit": MERGE_COMMIT, "numpy": np.__version__, "base_seed": BASE,
           "searches_per_family": chunks * CHUNK, "bootstrap_reps": REPS, "n": N, "k": K,
           "default_block": DEFAULT_BLOCK, "families": {}}
    for f, fam in enumerate(FAMILIES):
        rows = [row for (ff, _, rr) in res if ff == f for row in rr]
        out["families"][fam] = {"searches": len(rows),
                                "rejections_at_5pct": {k: int(sum(r[k] <= 0.05 for r in rows)) for k in rows[0]}}
    sys.stdout.write(json.dumps(out, sort_keys=True, indent=1) + "\n")
    sys.stderr.write(f"arch 882 check: DONE in {(time.time() - t0) / 60:.1f} min\n")


if __name__ == "__main__":
    main()
