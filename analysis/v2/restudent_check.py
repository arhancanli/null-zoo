"""Unregistered check (2026-10-07), prompted by the arch maintainer's comment on issue 879: which
per-resample studentizer keeps the SPA test's size, the resample's sample standard deviation (the
bootstrap-t of arch PR 881) or Hansen's closed-form Politis-Romano long-run variance recomputed in every
resample? Both are compared with the fixed-variance test (arch after PR 871) on the same null searches.
Usage: python analysis/v2/restudent_check.py [workers] > analysis/v2/restudent-check.json
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nzpy  # noqa: E402

BASE = 20261008
FAMILIES = ("iid_normal", "skew_negative", "ar1")
SEARCHES, CHUNK, REPS, N, K = 2000, 100, 1000, 504, 20
BLOCKS = (8, 22)


def pr_weights(n, block):
    p = 1.0 / block
    i = np.arange(1, n)
    return (1 - i / n) * (1 - p) ** i + (i / n) * (1 - p) ** (n - i)


def pr_var_many(xs, kappa):
    """Politis-Romano variance of each column for a stack of samples, xs shape (reps, n, k)."""
    n = xs.shape[1]
    d = xs - xs.mean(axis=1, keepdims=True)
    f = np.fft.rfft(d, n=2 * n, axis=1)
    acov = np.fft.irfft(f * np.conj(f), n=2 * n, axis=1)[:, :n, :] / n
    return acov[:, 0, :] + 2 * np.einsum("i,rik->rk", kappa, acov[:, 1:, :])


def run_chunk(job):
    f, chunk = job
    rng = np.random.default_rng(np.random.SeedSequence([BASE, 9, f, chunk]))
    counts = {f"{v}_b{b}": 0 for v in ("fixed", "boot_t_sd", "boot_t_pr") for b in BLOCKS}
    for _ in range(CHUNK):
        x = nzpy.draw_search(FAMILIES[f], rng, n=N, k=K)
        m = x.mean(0)
        root = np.sqrt(N)
        for b in BLOCKS:
            kappa = pr_weights(N, b)
            idx = nzpy.stationary_indices(N, b, REPS, rng)          # (reps, n)
            xs = x[idx]                                                 # (reps, n, k)
            ms = xs.mean(axis=1)
            w2 = nzpy.pr_variance(x, b)
            thr = -np.sqrt(w2 / N * 2 * np.log(np.log(N)))
            centre = np.where(m >= thr, m, 0.0)
            # Fixed: Politis-Romano variance of the sample, held fixed (Hansen 2005; arch after PR 871).
            T = max(0.0, float(np.max(root * m / np.sqrt(w2))))
            Ts = np.maximum(0.0, np.max(root * (ms - centre) / np.sqrt(w2), axis=1))
            counts[f"fixed_b{b}"] += int(np.mean(Ts >= T) <= 0.05)
            # Bootstrap-t with the sample standard deviation (arch PR 881).
            sd = x.std(0, ddof=1)
            T = max(0.0, float(np.max(root * m / sd)))
            sds = xs.std(axis=1, ddof=1)
            Ts = np.maximum(0.0, np.max(root * (ms - centre) / sds, axis=1))
            counts[f"boot_t_sd_b{b}"] += int(np.mean(Ts >= T) <= 0.05)
            # Bootstrap-t with Hansen's closed-form Politis-Romano variance recomputed in each resample.
            T = max(0.0, float(np.max(root * m / np.sqrt(w2))))
            w2s = np.maximum(pr_var_many(xs, kappa), 1e-300)
            Ts = np.maximum(0.0, np.max(root * (ms - centre) / np.sqrt(w2s), axis=1))
            counts[f"boot_t_pr_b{b}"] += int(np.mean(Ts >= T) <= 0.05)
    return f, counts


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    chunks = int(sys.argv[2]) if len(sys.argv) > 2 else SEARCHES // CHUNK
    jobs = [(f, c) for f in range(len(FAMILIES)) for c in range(chunks)]
    t0 = time.time()
    with Pool(workers) as pool:
        res = pool.map(run_chunk, jobs)
    out = {"schema": "canli.null-zoo.v2.restudent-check", "registered": False, "base_seed": BASE, "searches_per_family": chunks * CHUNK,
           "reps": REPS, "n": N, "k": K, "blocks": list(BLOCKS), "ties_counted": True, "families": {}}
    for f, fam in enumerate(FAMILIES):
        tot = {}
        for ff, c in res:
            if ff == f:
                for key, v in c.items():
                    tot[key] = tot.get(key, 0) + v
        out["families"][fam] = {key: v / (chunks * CHUNK) for key, v in sorted(tot.items())}
    out["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
