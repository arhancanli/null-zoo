"""Null Zoo v2b, part G: an empirical illustration, as pre-registered in analysis/v2b/prereg-v2b.json.

A search of 20 moving-average timing rules on the US stock market, evaluated against buy-and-hold in
every non-overlapping two-year window since the rules can first be computed, with the data-snooping
tests of the paper. Data: the Fama/French 3 Factors [Daily] file of the Kenneth R. French Data Library
(market return = Mkt-RF + RF). The file is downloaded, not committed; its SHA-256 is recorded.

Usage (from the repository root):
  python analysis/empirical/run_empirical.py fetch            # download the data, record its hash
  python analysis/empirical/run_empirical.py run [workers]    # > analysis/empirical/empirical-results.json
"""
import csv
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "v2"))
import nzpy  # noqa: E402

URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
DATA = os.path.join(HERE, "data", "F-F_Research_Data_Factors_daily_CSV.zip")
MANIFEST = os.path.join(HERE, "data-manifest.json")
LENGTHS = tuple(range(10, 201, 10))      # 20 moving-average lengths, in trading days
WINDOW = 504
REPS = 1000
SEED = 20261012


def fetch():
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "null-zoo-research/2.0 (academic replication)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        blob = r.read()
    with open(DATA, "wb") as f:
        f.write(blob)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = z.namelist()[0]
        text = z.read(name)
    manifest = {"source": URL, "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "zip_sha256": hashlib.sha256(blob).hexdigest(), "csv_name": name, "csv_sha256": hashlib.sha256(text).hexdigest(),
                "note": "Kenneth R. French Data Library; the file is updated over time, so a later download may differ."}
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")
    print(json.dumps(manifest, indent=1))


def load():
    with open(DATA, "rb") as f:
        blob = f.read()
    manifest = json.load(open(MANIFEST))
    if hashlib.sha256(blob).hexdigest() != manifest["zip_sha256"]:
        raise SystemExit("data file does not match data-manifest.json; run fetch again or restore the recorded file")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    dates, mkt, rf = [], [], []
    for row in csv.reader(io.StringIO(text)):
        if len(row) >= 5 and row[0].strip().isdigit() and len(row[0].strip()) == 8:
            dates.append(row[0].strip())
            mkt.append(float(row[1]) / 100 + float(row[4]) / 100)
            rf.append(float(row[4]) / 100)
    return np.array(dates), np.array(mkt), np.array(rf), manifest


def excess_returns(r, rf):
    """d[t, j]: rule j's return on day t minus buy-and-hold, for days with every signal defined.

    Rule j is long the market on day t+1 when the price index closes above its L_j-day moving average on
    day t, and holds the risk-free asset otherwise. The first usable day is max(L) + 1.
    """
    price = np.cumprod(1 + r)
    csum = np.r_[0.0, np.cumsum(price)]
    first = max(LENGTHS)                       # signals at day index t need t >= L - 1
    t = np.arange(first - 1, len(r) - 1)       # signal days; positions held on t + 1
    d = np.empty((len(t), len(LENGTHS)))
    for j, L in enumerate(LENGTHS):
        ma = (csum[t + 1] - csum[t + 1 - L]) / L
        long = price[t] > ma
        d[:, j] = np.where(long, r[t + 1], rf[t + 1]) - r[t + 1]
    return d, t + 1


def evaluate(job):
    label, d, block, seed = job
    keep = d.std(axis=0, ddof=1) > 0
    x = d[:, keep]
    out = {"label": label, "periods": int(d.shape[0]), "strategies_used": int(keep.sum()), "block": int(block)}
    if x.shape[1] == 0:
        out["p"] = None
        return out
    rng = np.random.default_rng(np.random.SeedSequence([SEED, seed]))
    p = {k: v / REPS for k, v in nzpy.joint_tests_v2b(x, block, REPS, rng).items()}
    p22 = nzpy.joint_tests_v2b(x, 22, REPS, rng)
    p["spa_c_ge_b22"] = p22["spa_c_ge"] / REPS
    p["spa_c_unstud_b22"] = p22["spa_c_unstud"] / REPS
    p["sidak_t"] = float(nzpy.sidak_t(x))
    out["p"] = p
    return out


def run(workers):
    dates, r, rf, manifest = load()
    d, day = excess_returns(r, rf)
    jobs = []
    n_windows = d.shape[0] // WINDOW
    for w in range(n_windows):
        a, b = w * WINDOW, (w + 1) * WINDOW
        jobs.append((f"{dates[day[a]]}-{dates[day[b - 1]]}", d[a:b], 8, w))
    jobs.append((f"{dates[day[0]]}-{dates[day[-1]]} (full sample)", d, max(1, round(d.shape[0] ** (1 / 3))), 9999))
    with Pool(workers) as pool:
        res = pool.map(evaluate, jobs)
    windows = [x for x in res if "full sample" not in x["label"]]
    full = [x for x in res if "full sample" in x["label"]][0]
    tests = sorted(windows[0]["p"].keys())
    valid = [w for w in windows if w["p"] is not None]
    reject = {t: sum(1 for w in valid if w["p"][t] <= 0.05) for t in tests}
    pairs = {}
    for a in tests:
        for b in tests:
            if a < b:
                pairs[f"{a}|{b}"] = {"only_first": sum(1 for w in valid if w["p"][a] <= 0.05 < w["p"][b]),
                                     "only_second": sum(1 for w in valid if w["p"][b] <= 0.05 < w["p"][a])}
    doc = {"schema": "canli.null-zoo.v2b.empirical", "prereg": "analysis/v2b/prereg-v2b.json", "data": manifest,
           "rules": {"type": "moving-average timing, long the market above the average and in the risk-free asset below it",
                     "lengths": list(LENGTHS), "benchmark": "buy-and-hold market"},
           "window": WINDOW, "bootstrap_reps": REPS, "seed": SEED, "numpy": np.__version__,
           "windows": windows, "full_sample": full, "summary": {"windows": len(valid), "rejections_at_5pct": reject, "disagreements": pairs}}
    sys.stdout.write(json.dumps(doc, sort_keys=True, indent=1) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        fetch()
    elif len(sys.argv) > 1 and sys.argv[1] == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    else:
        raise SystemExit(__doc__)
