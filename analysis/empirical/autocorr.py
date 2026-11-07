"""Unregistered diagnostic (2026-11-07): how large is the lag-one autocorrelation of real strategy returns?

Proposition 2 and the autocorrelation-levels run show that best-of-N tests that ignore autocorrelation are
oversized already at phi = 0.05. This script measures the lag-one sample autocorrelation of the series the
empirical illustration tests: each of the 20 moving-average rules' daily return minus buy-and-hold, in each
two-year window and over the full sample, and of the market's own daily return. It reads the same recorded data
file as run_empirical.py and runs no test.
Usage (from the repository root): python analysis/empirical/autocorr.py > analysis/empirical/autocorr.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_empirical as emp  # noqa: E402


def rho1(x):
    x = x - x.mean()
    den = float(x @ x)
    return float(x[1:] @ x[:-1] / den) if den > 0 else None


def main():
    dates, r, rf, manifest = emp.load()
    d, day = emp.excess_returns(r, rf)
    n_windows = d.shape[0] // emp.WINDOW
    per_window = []
    for w in range(n_windows):
        a, b = w * emp.WINDOW, (w + 1) * emp.WINDOW
        per_window.append([rho1(d[a:b, j]) for j in range(d.shape[1])])
    flat = np.array([x for row in per_window for x in row if x is not None])
    best = []
    for w in range(n_windows):
        a, b = w * emp.WINDOW, (w + 1) * emp.WINDOW
        x = d[a:b]
        sd = x.std(axis=0, ddof=1)
        t = np.where(sd > 0, x.mean(axis=0) / np.where(sd > 0, sd, 1), -np.inf)
        best.append(per_window[w][int(np.argmax(t))])
    best = np.array([x for x in best if x is not None])
    q = lambda v, p: round(float(np.quantile(v, p)), 4)  # noqa: E731
    doc = {
        "schema": "canli.null-zoo.empirical-autocorr", "registered": False, "data": manifest,
        "window": emp.WINDOW, "windows": n_windows, "rules": list(emp.LENGTHS),
        "rule_windows": int(flat.size),
        "rho1_rule_windows": {"median": q(flat, 0.5), "q25": q(flat, 0.25), "q75": q(flat, 0.75),
                              "share_ge_0.05": round(float(np.mean(flat >= 0.05)), 4),
                              "share_ge_0.1": round(float(np.mean(flat >= 0.1)), 4),
                              "share_le_-0.05": round(float(np.mean(flat <= -0.05)), 4)},
        "rho1_best_rule_per_window": {"median": q(best, 0.5), "q25": q(best, 0.25), "q75": q(best, 0.75),
                                      "share_ge_0.05": round(float(np.mean(best >= 0.05)), 4)},
        "rho1_full_sample_by_rule": {str(L): round(rho1(d[:, j]), 4) for j, L in enumerate(emp.LENGTHS)},
        "rho1_market_full_sample": round(rho1(r[day[0]:day[-1] + 1]), 4),
        "full_sample": f"{dates[day[0]]}-{dates[day[-1]]}",
        "numpy": np.__version__,
    }
    sys.stdout.write(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
