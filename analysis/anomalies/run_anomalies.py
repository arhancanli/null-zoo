"""Null Zoo anomalies illustration, as pre-registered in analysis/anomalies/prereg-anomalies.json.

Usage: python analysis/anomalies/run_anomalies.py <PredictorLSretWide.csv> > analysis/anomalies/anomalies-results.json
"""
import csv
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v2"))
import nzpy  # noqa: E402

SHA256 = "79aa0896d79924e1efa83f9afe51a3b4ce94137adfd2c5c3fd2d5d350532ee18"
BASE = 20261007
REPS = 1999
ALPHA = 0.05
STARTS = list(range(1975, 2016, 5))


def load(path):
    raw = open(path, "rb").read()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise SystemExit("data file hash does not match the pre-registration")
    rows = list(csv.reader(raw.decode("utf-8").splitlines()))
    names = rows[0][1:]
    dates = [r[0] for r in rows[1:]]
    vals = np.array([[np.nan if x in ("", "NA") else float(x) for x in r[1:]] for r in rows[1:]])
    return names, dates, vals


def stepwise(t_obs, t_star):
    """Romano-Wolf stepwise: t_obs (k,), t_star (reps, k) recentred resampled statistics."""
    active = np.ones(t_obs.shape[0], dtype=bool)
    superior = np.zeros_like(active)
    while active.any():
        crit = np.quantile(t_star[:, active].max(axis=1), 1 - ALPHA, method="higher")
        new = active & (t_obs > crit)
        if not new.any():
            break
        superior |= new
        active &= ~new
    return superior


def first_step_p(t_obs, t_star):
    return float(np.mean(t_star.max(axis=1) >= t_obs.max()))


def run_window(x, w):
    n, k = x.shape
    root = np.sqrt(n)
    m = x.mean(0)
    out = {}
    for label, q in (("q5", 5), ("q10", 10)):
        rng = np.random.default_rng(np.random.SeedSequence([BASE, 31, w, q]))
        counts = nzpy.stationary_counts(n, q, REPS, rng)
        ms = counts @ x / n
        wq = np.sqrt(nzpy.pr_variance(x, q))
        t_f, ts_f = root * m / wq, root * (ms - m) / wq
        name = "fixed" if q == 5 else "fixed_q10"
        out[name] = (t_f, ts_f)
        if q == 5:
            sd = x.std(0, ddof=1)
            sq = counts @ (x * x) / n
            sd_star = np.sqrt(np.maximum(sq - ms * ms, 1e-300) * n / (n - 1))
            out["boot_t"] = (root * m / sd, root * (ms - m) / sd_star)
            out["rc"] = (root * m, root * (ms - m))
    return {name: {"superior": int(stepwise(*pair).sum()), "first_step_p": first_step_p(*pair),
                   "superior_names_idx": [int(i) for i in np.flatnonzero(stepwise(*pair))]}
            for name, pair in out.items()}


def main():
    names, dates, vals = load(sys.argv[1])
    res = {"schema": "canli.null-zoo.anomalies.results", "prereg": "analysis/anomalies/prereg-anomalies.json",
           "data_sha256": SHA256, "numpy": np.__version__, "reps": REPS, "windows": []}
    for w, start in enumerate(STARTS):
        sel = [i for i, d in enumerate(dates) if start <= int(d[:4]) <= start + 9]
        block = vals[sel]
        keep = [j for j in range(block.shape[1]) if not np.isnan(block[:, j]).any()]
        x = block[:, keep]
        r = run_window(x, w)
        for v in r.values():
            v["superior_names"] = [names[keep[i]] for i in v.pop("superior_names_idx")]
        res["windows"].append({"start": f"{start}-01", "end": f"{start + 9}-12", "months": len(sel),
                               "predictors": len(keep), "tests": r})
        sys.stderr.write(f"{start}: k={len(keep)} " + " ".join(f"{t}={v['superior']}" for t, v in r.items()) + "\n")
    sys.stdout.write(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
