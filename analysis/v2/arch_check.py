"""Null Zoo v2 experiment D: arch 8.0.0's SPA on Null Zoo searches, as pre-registered in
analysis/v2/prereg-v2.json.

For 2,000 null searches in each of three families it runs arch.bootstrap.SPA with studentize=True and
studentize=False from the same seed, at block 8 and at arch's default block (int(sqrt(n)) = 22), and
records (i) whether the two flags ever give different p-values and (ii) the rejection rates of the
consistent and upper p-values, beside this repository's own implementation on the same searches.
Usage: python analysis/v2/arch_check.py [workers] > analysis/v2/arch-check-v2.json
An optional second argument sets the number of chunks of 100 searches per family (smoke tests only).
"""
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

BASE = 20261009
FAMILIES = ("iid_normal", "skew_negative", "ar1")
SEARCHES = 2000
CHUNK = 100
REPS = 1000
N, K = 504, 20
DEFAULT_BLOCK = int(np.sqrt(N))   # arch's default block_size


def run_chunk(job):
    f, chunk = job
    rng = np.random.default_rng(np.random.SeedSequence([BASE, 4, f, chunk]))
    rows = []
    for _ in range(CHUNK):
        x = nzpy.draw_search(FAMILIES[f], rng, n=N, k=K)
        row = {}
        for label, block in (("b8", 8), ("default", None)):
            seed = int(rng.integers(2 ** 31 - 1))
            p = {}
            for stud in (True, False):
                spa = SPA(np.zeros(N), -x, block_size=block, reps=REPS, bootstrap="stationary",
                          studentize=stud, seed=np.random.default_rng(seed))
                spa.compute()
                p[stud] = (float(spa.pvalues["consistent"]), float(spa.pvalues["upper"]))
            row[f"arch_{label}_consistent"] = p[True][0]
            row[f"arch_{label}_upper"] = p[True][1]
            row[f"arch_{label}_flag_changes_p"] = p[True] != p[False]
            ours = nzpy.joint_tests(x, block or DEFAULT_BLOCK, REPS, rng, {"spa_c_unstud", "rc", "spa_c"})
            row[f"ours_{label}_spa_c_unstud"] = ours["spa_c_unstud"] / REPS
            row[f"ours_{label}_rc"] = ours["rc"] / REPS
            row[f"ours_{label}_spa_c"] = ours["spa_c"] / REPS
        rows.append(row)
    return f, chunk, rows


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    chunks = int(sys.argv[2]) if len(sys.argv) > 2 else SEARCHES // CHUNK
    jobs = [(f, c) for f in range(len(FAMILIES)) for c in range(chunks)]
    t0 = time.time()
    sys.stderr.write(f"null-zoo v2 arch check: START jobs={len(jobs)} workers={workers}\n")
    with Pool(workers) as pool:
        res = sorted(pool.map(run_chunk, jobs), key=lambda r: (r[0], r[1]))
    out = {"schema": "canli.null-zoo.v2.arch-check", "prereg": "analysis/v2/prereg-v2.json", "arch": arch.__version__,
           "numpy": np.__version__, "base_seed": BASE, "searches_per_family": SEARCHES, "bootstrap_reps": REPS,
           "n": N, "k": K, "default_block": DEFAULT_BLOCK, "families": {}}
    for f, fam in enumerate(FAMILIES):
        rows = [row for (ff, _, rr) in res if ff == f for row in rr]
        keys = [k for k in rows[0] if not k.endswith("flag_changes_p")]
        fam_out = {"searches": len(rows),
                   "rejections_at_5pct": {k: int(sum(r[k] <= 0.05 for r in rows)) for k in keys},
                   "searches_where_studentize_flag_changes_p": {lab: int(sum(r[f"arch_{lab}_flag_changes_p"] for r in rows)) for lab in ("b8", "default")},
                   "mean_abs_p_difference_arch_vs_ours": {
                       lab: {"consistent": float(np.mean([abs(r[f"arch_{lab}_consistent"] - r[f"ours_{lab}_spa_c_unstud"]) for r in rows])),
                             "upper": float(np.mean([abs(r[f"arch_{lab}_upper"] - r[f"ours_{lab}_rc"]) for r in rows]))}
                       for lab in ("b8", "default")}}
        out["families"][fam] = fam_out
    out["seconds"] = round(time.time() - t0, 1)
    sys.stdout.write(json.dumps(out, sort_keys=True, indent=1) + "\n")
    sys.stderr.write(f"null-zoo v2 arch check: DONE in {(time.time() - t0) / 60:.1f} min\n")


if __name__ == "__main__":
    main()
