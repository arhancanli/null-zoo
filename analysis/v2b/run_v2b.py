"""Null Zoo v2b governed run: experiments E (number of strategies and block length), F (re-studentizing
by a block variance, nine families) and C2 (where the skilled strategy sits in the volatility ranking;
poor alternatives with ties counted), as pre-registered in analysis/v2b/prereg-v2b.json, which was
committed before this script was first run at full size.

Usage (from the repository root):
  python analysis/v2b/run_v2b.py [workers] [scale] > analysis/v2b/results-v2b.json 2> analysis/v2b/run-v2b.log
`scale` multiplies every cell's number of searches (smoke tests only; the governed run uses 1). Every
cell is split into chunks of 250 searches seeded by SeedSequence([BASE, experiment, cell, chunk]) and
merged in order, so the output does not depend on the number of workers.
"""
import json
import math
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "v2"))
import nzpy  # noqa: E402

BASE = 20261011
CHUNK = 250
REPS = 1000
EXPERIMENT_NO = {"E": 5, "F": 6, "C2": 7}


def cells():
    out = []
    # E: number of strategies and block length, i.i.d. normal null, T = 504.
    for k in (1, 2, 5, 10, 20, 50, 100):
        for block in (1, 8, 22):
            out.append(dict(experiment="E", design=f"k{k}_b{block}", family="iid_normal", n=504, k=k, block=block,
                            skill=0, sharpe=None, vol=None, sidak=False, searches=5000))
    # F: the nine families at the v1 design, block 8; null 10,000 searches, Sharpe 2 5,000 searches.
    for family in nzpy.FAMILIES:
        for skill, searches in ((0, 10000), (2, 5000)):
            out.append(dict(experiment="F", design="v1", family=family, n=504, k=20, block=8, skill=skill,
                            sharpe=None if skill == 0 else [2.0] + [0.0] * 19, vol=None, sidak=False, searches=searches))
    # C2: unequal volatilities with the skilled strategy at the lowest, median and highest volatility, and
    # poor alternatives, all with ties counted; i.i.d. normal returns, block 8, 10,000 searches per cell.
    het = [0.5 * 4 ** (j / 19) for j in range(20)]
    out.append(dict(experiment="C2", design="het_null", family="iid_normal", n=504, k=20, block=8, skill=0,
                    sharpe=None, vol=het, sidak=True, searches=10000))
    for where, j in (("low", 0), ("median", 10), ("high", 19)):
        sharpe = [0.0] * 20
        sharpe[j] = 2.0
        out.append(dict(experiment="C2", design=f"het_skill_{where}", family="iid_normal", n=504, k=20, block=8, skill=2,
                        skilled_index=j, sharpe=sharpe, vol=het, sidak=True, searches=10000))
    for skill in (0, 2):
        sharpe = [float(skill)] + [-3.0] * 19
        out.append(dict(experiment="C2", design=f"poor_{'null' if skill == 0 else 'skill'}", family="iid_normal", n=504, k=20,
                        block=8, skill=skill, sharpe=sharpe if any(sharpe) else None, vol=None, sidak=True, searches=10000))
    for i, c in enumerate(out):
        c["cell"] = i
    return out


def run_chunk(job):
    c, chunk, count = job
    rng = np.random.default_rng(np.random.SeedSequence([BASE, EXPERIMENT_NO[c["experiment"]], c["cell"], chunk]))
    hist = {}
    sidak = Counter()
    best_mean_skilled = best_t_skilled = 0
    skilled = c.get("skilled_index", 0)
    for _ in range(count):
        x = nzpy.draw_search(c["family"], rng, n=c["n"], k=c["k"], sharpe=c["sharpe"], vol=c["vol"])
        m = x.mean(0)
        best_mean_skilled += int(np.argmax(m) == skilled)
        best_t_skilled += int(np.argmax(m / x.std(0, ddof=1)) == skilled)
        for name, h in nzpy.joint_tests_v2b(x, c["block"], REPS, rng).items():
            hist.setdefault(name, Counter())[h] += 1
        if c["sidak"]:
            sidak[math.ceil(nzpy.sidak_t(x) * 1e6)] += 1
    return c["cell"], chunk, {k: dict(v) for k, v in hist.items()}, dict(sidak), best_mean_skilled, best_t_skilled, count


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, (os.cpu_count() or 2) - 2)
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    cs = cells()
    for c in cs:
        c["searches"] = max(1, round(c["searches"] * scale))
    jobs = [(c, j, min(CHUNK, c["searches"] - j * CHUNK)) for c in cs for j in range(math.ceil(c["searches"] / CHUNK))]
    jobs.sort(key=lambda jb: -(jb[0]["k"] * (3 if jb[0]["block"] == 8 else 1)))
    t0 = time.time()
    sys.stderr.write(f"null-zoo v2b: START {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} cells={len(cs)} jobs={len(jobs)} workers={workers}\n")
    results = {}
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_chunk, jobs), 1):
            results[(r[0], r[1])] = r
            if i % 20 == 0 or i == len(jobs):
                s = time.time() - t0
                sys.stderr.write(f"null-zoo v2b: {i}/{len(jobs)} chunks, {s:.0f} s, about {s / i * (len(jobs) - i) / 60:.1f} min left\n")
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
        cell = {key: c.get(key) for key in ("cell", "experiment", "design", "family", "n", "k", "block", "skill", "skilled_index", "sharpe", "vol")}
        cell["searches"] = total
        cell["best_mean_is_skilled"] = bm
        cell["best_t_is_skilled"] = bt
        cell["exceedances"] = {name: sorted([k, v] for k, v in counts.items()) for name, counts in sorted(hist.items())}
        if c["sidak"]:
            cell["sidak_t_p_micro"] = sorted([k, v] for k, v in sidak.items())
        out_cells.append(cell)
    doc = {
        "schema": "canli.null-zoo.v2b.run", "prereg": "analysis/v2b/prereg-v2b.json",
        "generator": "numpy PCG64 via SeedSequence([base, experiment, cell, chunk])",
        "base_seed": BASE, "chunk": CHUNK, "bootstrap_reps": REPS, "scale": scale, "numpy": np.__version__,
        "encoding": {"exceedances": "[h, count] pairs; p = h / bootstrap_reps", "sidak_t_p_micro": "[ceil(p * 1e6), count] pairs"},
        "cells": out_cells,
    }
    sys.stdout.write(json.dumps(doc, sort_keys=True) + "\n")
    sys.stderr.write(f"null-zoo v2b: DONE {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} in {(time.time() - t0) / 60:.1f} min\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"null-zoo v2b: FAILED {e!r}\n")
        raise
