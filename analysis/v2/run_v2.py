"""Null Zoo v2 governed run: experiments A (mechanism, nine families), B (sample size and number of
strategies) and C (heterogeneous volatility and poor alternatives), as pre-registered in
analysis/v2/prereg-v2.json, which was committed before this script was first run.

Usage (from the repository root):
  python analysis/v2/run_v2.py [workers] > analysis/v2/results-v2.json 2> analysis/v2/run-v2.log
An optional second argument scales every cell's number of searches (smoke tests only; the governed run
uses the default of 1). A scaled run's chunks are prefixes of the full run's, with the same seeds.
Every cell is split into chunks of 250 searches with seeds SeedSequence([BASE, experiment, cell,
chunk]); chunks are merged in order, so the output is identical for any number of workers.
"""
import json
import math
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nzpy  # noqa: E402

BASE = 20261009
CHUNK = 250
REPS = 1000
MAIN = ("spa_c", "spa_u", "rc", "spa_c_unstud", "spa_c_sd", "spa_c_boot_t")
B22 = ("spa_c", "rc", "spa_c_unstud")
SIZE_VARIANTS = ("spa_c", "rc", "spa_c_unstud", "spa_c_sd", "spa_c_boot_t")


def block_for(n):
    return max(1, round(n ** (1 / 3)))


def cells():
    out = []
    # A: the nine v1 families at the v1 design, null and a true annualized Sharpe of 2.
    for family in nzpy.FAMILIES:
        for skill in (0, 2):
            out.append(dict(experiment="A", family=family, n=504, k=20, skill=skill, design="v1",
                            sharpe=None if skill == 0 else [2.0] + [0.0] * 19, vol=None,
                            variants=MAIN, extra_block=22, extra_variants=B22, sidak=False, searches=10000))
    # B: sample size and number of strategies, null only.
    for family in ("iid_normal", "skew_negative"):
        for n, k in ((252, 20), (504, 20), (1260, 20), (2520, 20), (504, 5), (504, 100)):
            out.append(dict(experiment="B", family=family, n=n, k=k, skill=0, design=f"n{n}_k{k}",
                            sharpe=None, vol=None, variants=SIZE_VARIANTS, extra_block=None,
                            extra_variants=(), sidak=False, searches=5000))
    # C: where studentizing and consistent recentring should help (i.i.d. normal returns).
    het = [0.5 * 4 ** (j / 19) for j in range(20)]
    for design, vol, base in (("heterogeneous_volatility", het, [0.0] * 20), ("poor_alternatives", None, [0.0] + [-3.0] * 19)):
        for skill in (0, 2):
            sharpe = list(base)
            sharpe[0] = float(skill)
            out.append(dict(experiment="C", family="iid_normal", n=504, k=20, skill=skill, design=design,
                            sharpe=sharpe if any(sharpe) else None, vol=vol, variants=MAIN, extra_block=None,
                            extra_variants=(), sidak=True, searches=10000))
    for i, c in enumerate(out):
        c["cell"] = i
        c["block"] = block_for(c["n"])
    return out


EXPERIMENT_NO = {"A": 1, "B": 2, "C": 3}


def run_chunk(job):
    c, chunk, count = job
    rng = np.random.default_rng(np.random.SeedSequence([BASE, EXPERIMENT_NO[c["experiment"]], c["cell"], chunk]))
    hist = {name: Counter() for name in c["variants"]}
    for name in c["extra_variants"]:
        hist[f"{name}_b{c['extra_block']}"] = Counter()
    sidak = Counter()
    best_mean_is_0 = best_t_is_0 = 0
    for _ in range(count):
        x = nzpy.draw_search(c["family"], rng, n=c["n"], k=c["k"], sharpe=c["sharpe"], vol=c["vol"])
        m = x.mean(0)
        t = m / x.std(0, ddof=1)
        best_mean_is_0 += int(np.argmax(m) == 0)
        best_t_is_0 += int(np.argmax(t) == 0)
        for name, h in nzpy.joint_tests(x, c["block"], REPS, rng, set(c["variants"])).items():
            hist[name][h] += 1
        if c["extra_block"]:
            for name, h in nzpy.joint_tests(x, c["extra_block"], REPS, rng, set(c["extra_variants"])).items():
                hist[f"{name}_b{c['extra_block']}"][h] += 1
        if c["sidak"]:
            sidak[math.ceil(nzpy.sidak_t(x) * 1e6)] += 1
    return c["cell"], chunk, {k: dict(v) for k, v in hist.items()}, dict(sidak), best_mean_is_0, best_t_is_0, count


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, (os.cpu_count() or 2) - 2)
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    cs = cells()
    for c in cs:
        c["searches"] = max(1, round(c["searches"] * scale))
    jobs = [(c, j, min(CHUNK, c["searches"] - j * CHUNK)) for c in cs for j in range(math.ceil(c["searches"] / CHUNK))]
    # Longest jobs first keeps the workers busy to the end; the merge below restores chunk order.
    jobs.sort(key=lambda jb: -(jb[0]["n"] * jb[0]["k"] * (2 if jb[0]["extra_block"] else 1)))
    t0 = time.time()
    sys.stderr.write(f"null-zoo v2: START {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} cells={len(cs)} jobs={len(jobs)} workers={workers}\n")
    results = {}
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_chunk, jobs), 1):
            results[(r[0], r[1])] = r
            if i % 20 == 0 or i == len(jobs):
                s = time.time() - t0
                sys.stderr.write(f"null-zoo v2: {i}/{len(jobs)} chunks, {s:.0f} s, about {s / i * (len(jobs) - i) / 60:.1f} min left\n")
                sys.stderr.flush()
    out_cells = []
    for c in cs:
        chunks = sorted((key, r) for key, r in results.items() if key[0] == c["cell"])
        hist, sidak = {}, Counter()
        bm = bt = total = 0
        for _, (_, _, h, sd, b_m, b_t, count) in chunks:
            for name, counts in h.items():
                acc = hist.setdefault(name, Counter())
                for k, v in counts.items():
                    acc[int(k)] += v
            for k, v in sd.items():
                sidak[int(k)] += v
            bm += b_m
            bt += b_t
            total += count
        cell = {key: c[key] for key in ("cell", "experiment", "design", "family", "n", "k", "skill", "block", "extra_block", "sharpe", "vol", "searches")}
        cell["searches"] = total
        cell["best_mean_is_strategy_0"] = bm
        cell["best_t_is_strategy_0"] = bt
        cell["exceedances"] = {name: sorted([k, v] for k, v in counts.items()) for name, counts in sorted(hist.items())}
        if c["sidak"]:
            cell["sidak_t_p_micro"] = sorted([k, v] for k, v in sidak.items())
        out_cells.append(cell)
    doc = {
        "schema": "canli.null-zoo.v2.run",
        "prereg": "analysis/v2/prereg-v2.json",
        "generator": "numpy PCG64 via SeedSequence([base, experiment, cell, chunk])",
        "base_seed": BASE, "chunk": CHUNK, "bootstrap_reps": REPS, "scale": scale,
        "numpy": np.__version__,
        "encoding": {
            "exceedances": "[h, count] pairs: h of the bootstrap_reps resampled statistics exceeded the observed one, so p = h / bootstrap_reps",
            "sidak_t_p_micro": "[ceil(p * 1e6), count] pairs for the Sidak-adjusted Student-t p-value",
        },
        "cells": out_cells,
    }
    sys.stdout.write(json.dumps(doc, sort_keys=True) + "\n")
    sys.stderr.write(f"null-zoo v2: DONE {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} in {(time.time() - t0) / 60:.1f} min\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"null-zoo v2: FAILED {e!r}\n")
        raise
