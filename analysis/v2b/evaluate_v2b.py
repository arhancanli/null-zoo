"""Scores the Null Zoo v2b run against analysis/v2b/prereg-v2b.json and summarizes experiments E, F, C2
and the empirical illustration G. Writes analysis/v2b/evaluation-v2b.json.

Usage (from the repository root): python analysis/v2b/evaluate_v2b.py [results] [empirical] [output]
"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEVEL = 0.05


def P(path):
    return os.path.join(ROOT, path)


def load(path):
    with open(P(path)) as f:
        return json.load(f)


def cumulative(pairs, searches, reps):
    acc = [0] * (reps + 1)
    for h, count in pairs:
        acc[h] += count
    out, run = [], 0
    for v in acc:
        run += v
        out.append(run / searches)
    return out


def rate(cell, test, reps):
    if test == "sidak_t":
        return sum(c for p, c in cell["sidak_t_p_micro"] if p <= LEVEL * 1e6) / cell["searches"]
    return cumulative(cell["exceedances"][test], cell["searches"], reps)[math.floor(LEVEL * reps + 1e-9)]


def size_adjusted(null_cell, skill_cell, test, reps):
    if test == "sidak_t":
        nulls = sorted(p for p, c in null_cell["sidak_t_p_micro"] for _ in range(c))
        allowed = math.floor(LEVEL * null_cell["searches"] + 1e-9)
        cut = nulls[allowed] - 1 if allowed < len(nulls) else 10 ** 6
        return sum(c for p, c in skill_cell["sidak_t_p_micro"] if p <= cut) / skill_cell["searches"]
    cn = cumulative(null_cell["exceedances"][test], null_cell["searches"], reps)
    cs = cumulative(skill_cell["exceedances"][test], skill_cell["searches"], reps)
    edge = -1
    for h in range(reps + 1):
        if cn[h] <= LEVEL:
            edge = h
        else:
            break
    return 0.0 if edge < 0 else cs[edge]


def main():
    args = sys.argv[1:] + [None] * 3
    run = load(args[0] or "analysis/v2b/results-v2b.json")
    emp = load(args[1] or "analysis/empirical/empirical-results.json")
    prereg = load("analysis/v2b/prereg-v2b.json")
    reps = run["bootstrap_reps"]
    cells = run["cells"]

    def find(**kw):
        hits = [c for c in cells if all(c.get(k) == v for k, v in kw.items())]
        assert len(hits) == 1, kw
        return hits[0]

    E = {}
    for c in (c for c in cells if c["experiment"] == "E"):
        E.setdefault(f"k{c['k']}", {})[f"b{c['block']}"] = {"searches": c["searches"], "size": {t: rate(c, t, reps) for t in sorted(c["exceedances"])}}
    F = {}
    for c in (c for c in cells if c["experiment"] == "F" and c["skill"] == 0):
        skill = find(experiment="F", family=c["family"], skill=2)
        tests = sorted(c["exceedances"])
        F[c["family"]] = {"searches_null": c["searches"], "searches_skill": skill["searches"],
                          "size": {t: rate(c, t, reps) for t in tests},
                          "power": {t: rate(skill, t, reps) for t in tests},
                          "size_adjusted_power": {t: size_adjusted(c, skill, t, reps) for t in tests}}
    C2 = {}
    het_null = find(experiment="C2", design="het_null")
    tests = sorted(het_null["exceedances"]) + ["sidak_t"]
    C2["het_null"] = {"searches": het_null["searches"], "size": {t: rate(het_null, t, reps) for t in tests}}
    for where in ("low", "median", "high"):
        s = find(experiment="C2", design=f"het_skill_{where}")
        C2[f"het_skill_{where}"] = {"searches": s["searches"], "power": {t: rate(s, t, reps) for t in tests},
                                    "size_adjusted_power": {t: size_adjusted(het_null, s, t, reps) for t in tests},
                                    "best_mean_is_skilled": s["best_mean_is_skilled"] / s["searches"],
                                    "best_t_is_skilled": s["best_t_is_skilled"] / s["searches"]}
    pn, ps = find(experiment="C2", design="poor_null"), find(experiment="C2", design="poor_skill")
    C2["poor"] = {"searches": pn["searches"], "size": {t: rate(pn, t, reps) for t in tests},
                  "power": {t: rate(ps, t, reps) for t in tests},
                  "size_adjusted_power": {t: size_adjusted(pn, ps, t, reps) for t in tests}}

    p = {}
    v = E["k20"]["b8"]["size"]["spa_c_ge"]
    p["P10"] = {"held": v >= 0.065, "detail": {"spa_c_ge": v}}
    a, b = E["k1"]["b22"]["size"]["rc"], E["k1"]["b1"]["size"]["rc"]
    p["P11"] = {"held": a >= b + 0.01, "detail": {"rc_k1_b22": a, "rc_k1_b1": b}}
    a, b = E["k1"]["b8"]["size"]["rc"], E["k100"]["b8"]["size"]["rc"]
    p["P12"] = {"held": a >= b + 0.005, "detail": {"rc_k1_b8": a, "rc_k100_b8": b}}
    nonar = {f: F[f]["size"]["spa_c_nb_ge"] for f in F if f != "ar1"}
    p["P13"] = {"held": all(x <= 0.065 for x in nonar.values()), "detail": nonar}
    a, b = F["ar1"]["size"]["spa_c_nb_ge"], F["ar1"]["size"]["spa_c_boot_t_ge"]
    p["P14"] = {"held": a <= b - 0.005, "detail": {"spa_c_nb_ge": a, "spa_c_boot_t_ge": b}}
    hi = C2["het_skill_high"]["power"]
    p["P15"] = {"held": hi["rc"] >= hi["spa_c_boot_t_ge"], "detail": {"rc": hi["rc"], "spa_c_boot_t_ge": hi["spa_c_boot_t_ge"]}}
    poor = C2["poor"]
    p["P16"] = {"held": poor["size"]["spa_c_boot_t_ge"] <= 0.065 and poor["power"]["spa_c_boot_t_ge"] >= poor["power"]["rc"] + 0.20,
                "detail": {"size_spa_c_boot_t_ge": poor["size"]["spa_c_boot_t_ge"], "power_spa_c_boot_t_ge": poor["power"]["spa_c_boot_t_ge"], "power_rc": poor["power"]["rc"]}}
    for item in prereg["predictions"]:
        p[item["id"]]["statement"] = item["statement"]

    G = {"windows": emp["summary"]["windows"], "rejections_at_5pct": emp["summary"]["rejections_at_5pct"],
         "disagreements": emp["summary"]["disagreements"], "full_sample": emp["full_sample"],
         "first_window": emp["windows"][0]["label"], "last_window": emp["windows"][-1]["label"],
         "data_csv_sha256": emp["data"]["csv_sha256"], "downloaded_utc": emp["data"]["downloaded_utc"]}
    out = {"schema": "canli.null-zoo.v2b.evaluation", "level": LEVEL, "bootstrap_reps": reps,
           "from": ["analysis/v2b/results-v2b.json", "analysis/empirical/empirical-results.json", "analysis/v2b/prereg-v2b.json"],
           "E": E, "F": F, "C2": C2, "G": G, "predictions": p}
    with open(P(args[2] or "analysis/v2b/evaluation-v2b.json"), "w") as f:
        json.dump(out, f, sort_keys=True, indent=1)
        f.write("\n")
    for k in sorted(p, key=lambda s: int(s[1:])):
        print(k, "held" if p[k]["held"] else "DID NOT HOLD")


if __name__ == "__main__":
    main()
