"""EXPLORATORY (not pre-registered): why prediction P9 failed.

In the poor-alternatives null of experiment C, every consistent SPA variant that floors its statistic
at zero (spa_c, spa_c_sd, spa_c_boot_t) rejected about 7.4% of searches, while the unfloored
spa_c_unstud rejected 4.8%. Conjecture, formed after seeing the result: when the observed statistic
max(0, max_k t_k) is zero and every strategy falls below the consistent threshold, every recentred
resampled statistic is near or below zero, so the share of resampled statistics strictly above zero
is tiny and p = P*(T* > T) rejects. Counting ties (p = P*(T* >= T)) gives p = 1 whenever T = 0.

This script re-draws the same design on a fresh seed (20261010), never used by the governed runs, and
reports, for the bootstrap-t and fixed-variance variants, the rejection rate with each convention and
how often the observed statistic is zero with every strategy below the threshold.
Usage: python analysis/v2/explore_floor.py [workers] > analysis/v2/explore-floor-v2.json
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nzpy  # noqa: E402

SEED = 20261010
SEARCHES = 10000
CHUNK = 250
REPS = 1000
N, K, BLOCK = 504, 20, 8
SHARPE = [0.0] + [-3.0] * 19


def chunk(c):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, c]))
    out = []
    for _ in range(CHUNK):
        x = nzpy.draw_search("iid_normal", rng, n=N, k=K, sharpe=SHARPE)
        n, k = x.shape
        m = x.mean(0)
        w2 = nzpy.pr_variance(x, BLOCK)
        w = np.sqrt(w2)
        counts = nzpy.stationary_counts(n, BLOCK, REPS, rng)
        ms = counts @ x / n
        root = np.sqrt(n)
        thr = -np.sqrt(w2 / n * 2 * np.log(np.log(n)))
        below = m < thr
        centre = np.where(below, 0.0, m)
        row = {"all_below": bool(below.all())}
        # fixed-variance studentization (Hansen) and bootstrap-t, each with > and with >=
        T = max(0.0, float(np.max(root * m / w)))
        Ts = np.maximum(0.0, np.max(root * (ms - centre) / w, axis=1))
        sd = x.std(0, ddof=1)
        sq = counts @ (x * x) / n
        sd_star = np.sqrt(np.maximum(sq - ms * ms, 1e-300) * n / (n - 1))
        Tt = max(0.0, float(np.max(root * m / sd)))
        Tts = np.maximum(0.0, np.max(root * (ms - centre) / sd_star, axis=1))
        row["T_zero"] = T == 0.0
        row["spa_c_gt"] = float(np.mean(Ts > T))
        row["spa_c_ge"] = float(np.mean(Ts >= T))
        row["boot_t_gt"] = float(np.mean(Tts > Tt))
        row["boot_t_ge"] = float(np.mean(Tts >= Tt))
        out.append(row)
    return out


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with Pool(workers) as p:
        rows = [r for ch in p.map(chunk, range(SEARCHES // CHUNK)) for r in ch]
    n = len(rows)
    rej = lambda key, cond=lambda r: True: sum(1 for r in rows if cond(r) and r[key] <= 0.05) / n  # noqa: E731
    out = {
        "schema": "canli.null-zoo.v2.explore-floor", "exploratory": True, "seed": SEED, "searches": n,
        "design": "experiment C poor_alternatives null: iid normal, n 504, k 20, block 8, strategy 0 Sharpe 0, 19 at -3",
        "share_T_zero": sum(r["T_zero"] for r in rows) / n,
        "share_T_zero_and_all_below_threshold": sum(r["T_zero"] and r["all_below"] for r in rows) / n,
        "size": {
            "spa_c_strict": rej("spa_c_gt"), "spa_c_ties_count": rej("spa_c_ge"),
            "spa_c_boot_t_strict": rej("boot_t_gt"), "spa_c_boot_t_ties_count": rej("boot_t_ge"),
        },
        "size_from_searches_with_T_zero": {
            "spa_c_strict": rej("spa_c_gt", lambda r: r["T_zero"]), "spa_c_boot_t_strict": rej("boot_t_gt", lambda r: r["T_zero"]),
        },
    }
    sys.stdout.write(json.dumps(out, sort_keys=True, indent=1) + "\n")


if __name__ == "__main__":
    main()
