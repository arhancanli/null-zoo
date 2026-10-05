"""Scores the Null Zoo v2 run against its pre-registration and derives every v2 number the paper uses.

Reads analysis/v2/results-v2.json, analysis/v2/arch-check-v2.json, analysis/v2/prereg-v2.json and the
v1b evaluation (src/config/research/null-zoo-v1b-evaluation.json); writes analysis/v2/evaluation-v2.json.
Usage (from the repository root): python analysis/v2/evaluate_v2.py [results] [arch check] [output]
Nothing here is typed by hand: every rate comes from the stored exceedance histograms.
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
LEVEL = 0.05
NON_AR = ("iid_normal", "student_t4", "skew_negative", "skew_positive", "garch", "regimes",
          "correlated_trials", "block_cluster")


def load(path):
    with open(P(path)) as f:
        return json.load(f)


def cumulative(pairs, searches, reps):
    """Share of searches with p <= h / reps, for h = 0..reps."""
    acc = [0] * (reps + 1)
    for h, count in pairs:
        acc[h] += count
    out, run = [], 0
    for v in acc:
        run += v
        out.append(run / searches)
    return out


def rate(cell, test, reps, level=LEVEL):
    if test == "sidak_t":
        return sum(c for p, c in cell["sidak_t_p_micro"] if p <= level * 1e6) / cell["searches"]
    return cumulative(cell["exceedances"][test], cell["searches"], reps)[math.floor(level * reps + 1e-9)]


def size_adjusted(null_cell, skill_cell, test, reps):
    if test == "sidak_t":
        nulls = sorted(p for p, c in null_cell["sidak_t_p_micro"] for _ in range(c))
        allowed = math.floor(LEVEL * null_cell["searches"] + 1e-9)
        # the largest cutoff with at most `allowed` null p-values at or below it
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


def se(p, n):
    return math.sqrt(p * (1 - p) / n)


def main():
    import sys
    args = sys.argv[1:] + [None] * 3
    run = load(args[0] or "analysis/v2/results-v2.json")
    arch = load(args[1] or "analysis/v2/arch-check-v2.json")
    prereg = load("analysis/v2/prereg-v2.json")
    v1b = load("src/config/research/null-zoo-v1b-evaluation.json")
    reps = run["bootstrap_reps"]
    cells = run["cells"]

    def find(**kw):
        hits = [c for c in cells if all(c[k] == v for k, v in kw.items())]
        assert len(hits) == 1, kw
        return hits[0]

    # Experiment A
    a = {}
    for fam in ("iid_normal", "student_t4", "skew_negative", "skew_positive", "garch", "ar1", "regimes", "correlated_trials", "block_cluster"):
        null, skill = find(experiment="A", family=fam, skill=0), find(experiment="A", family=fam, skill=2)
        tests = sorted(null["exceedances"])
        a[fam] = {
            "searches": null["searches"],
            "size": {t: rate(null, t, reps) for t in tests},
            "power": {t: rate(skill, t, reps) for t in tests},
            "size_adjusted_power": {t: size_adjusted(null, skill, t, reps) for t in tests},
            "best_t_is_skilled": skill["best_t_is_strategy_0"] / skill["searches"],
            "best_mean_is_skilled": skill["best_mean_is_strategy_0"] / skill["searches"],
        }
    # Experiment B
    b = {}
    for c in (c for c in cells if c["experiment"] == "B"):
        b.setdefault(c["family"], {})[c["design"]] = {"n": c["n"], "k": c["k"], "block": c["block"], "searches": c["searches"],
                                                      "size": {t: rate(c, t, reps) for t in sorted(c["exceedances"])}}
    # Experiment C
    cc = {}
    for design in ("heterogeneous_volatility", "poor_alternatives"):
        null, skill = find(experiment="C", design=design, skill=0), find(experiment="C", design=design, skill=2)
        tests = sorted(null["exceedances"]) + ["sidak_t"]
        cc[design] = {
            "searches": null["searches"],
            "size": {t: rate(null, t, reps) for t in tests},
            "power": {t: rate(skill, t, reps) for t in tests},
            "size_adjusted_power": {t: size_adjusted(null, skill, t, reps) for t in tests},
            "best_t_is_skilled": skill["best_t_is_strategy_0"] / skill["searches"],
            "best_mean_is_skilled": skill["best_mean_is_strategy_0"] / skill["searches"],
        }
    # Experiment D
    d = {}
    for fam, f in arch["families"].items():
        n = f["searches"]
        d[fam] = {"searches": n, "flag_changes_p": f["searches_where_studentize_flag_changes_p"],
                  "size": {k: v / n for k, v in f["rejections_at_5pct"].items()},
                  "mean_abs_p_difference_arch_vs_ours": f["mean_abs_p_difference_arch_vs_ours"]}

    # Predictions, scored exactly as worded in the pre-registration.
    v1b_size = v1b["table"]
    n_v1b = v1b["run"]["reps_per_cell"]
    p = {}
    rows = []
    for fam in a:
        for ours, theirs in (("spa_c", "spa_consistent"), ("rc", "reality_check")):
            x, y = a[fam]["size"][ours], v1b_size[theirs]["size"][fam]
            bound = 3 * math.sqrt(se(x, a[fam]["searches"]) ** 2 + se(y, n_v1b) ** 2)
            rows.append({"family": fam, "test": ours, "v2": x, "v1b": y, "difference": x - y, "bound": bound, "within": abs(x - y) <= bound})
    p["P1"] = {"held": all(r["within"] for r in rows), "detail": rows}
    p2 = {fam: a[fam]["size"]["spa_c_sd"] for fam in ("iid_normal", "student_t4", "skew_positive", "garch", "regimes", "correlated_trials", "block_cluster")}
    p["P2"] = {"held": all(v <= 0.06 for v in p2.values()), "detail": p2}
    p3 = {"spa_c_sd": a["skew_negative"]["size"]["spa_c_sd"], "spa_c_boot_t": a["skew_negative"]["size"]["spa_c_boot_t"]}
    p["P3"] = {"held": p3["spa_c_sd"] > 0.08 and p3["spa_c_boot_t"] <= 0.065, "detail": p3}
    p4 = {fam: a[fam]["size"]["spa_c_boot_t"] for fam in NON_AR}
    p["P4"] = {"held": all(v <= 0.065 for v in p4.values()), "detail": p4}
    p5 = {fam: a[fam]["size"]["spa_c_unstud"] for fam in NON_AR}
    flags = {fam: d[fam]["flag_changes_p"] for fam in d}
    p["P5"] = {"held": all(v <= 0.065 for v in p5.values()) and all(v == 0 for f in flags.values() for v in f.values()),
               "detail": {"spa_c_unstud_size": p5, "arch_searches_where_flag_changes_p": flags}}
    p6 = {"spa_c_b22": a["iid_normal"]["size"]["spa_c_b22"], "spa_c": a["iid_normal"]["size"]["spa_c"]}
    p["P6"] = {"held": p6["spa_c_b22"] >= p6["spa_c"] + 0.01, "detail": p6}
    bi = b["iid_normal"]
    p7 = {"n252": bi["n252_k20"]["size"]["spa_c"], "n2520": bi["n2520_k20"]["size"]["spa_c"],
          "k5": bi["n504_k5"]["size"]["spa_c"], "k100": bi["n504_k100"]["size"]["spa_c"]}
    p["P7"] = {"held": p7["n2520"] <= p7["n252"] - 0.01 and p7["k100"] >= p7["k5"] + 0.01, "detail": p7}
    het, poor = cc["heterogeneous_volatility"], cc["poor_alternatives"]
    p8 = {"het_size_adjusted_power": {t: het["size_adjusted_power"][t] for t in ("spa_c_boot_t", "rc")},
          "poor_power": {t: poor["power"][t] for t in ("spa_c_boot_t", "rc")}}
    p["P8"] = {"held": p8["het_size_adjusted_power"]["spa_c_boot_t"] >= p8["het_size_adjusted_power"]["rc"] + 0.05
               and p8["poor_power"]["spa_c_boot_t"] >= p8["poor_power"]["rc"] + 0.05, "detail": p8}
    p9 = {"rc": poor["size"]["rc"], "spa_c_boot_t": poor["size"]["spa_c_boot_t"]}
    p["P9"] = {"held": p9["rc"] < 0.03 and p9["spa_c_boot_t"] <= 0.065, "detail": p9}
    for item in prereg["predictions"]:
        p[item["id"]]["statement"] = item["statement"]

    out = {
        "schema": "canli.null-zoo.v2.evaluation",
        "from": ["analysis/v2/results-v2.json", "analysis/v2/arch-check-v2.json", "analysis/v2/prereg-v2.json",
                 "src/config/research/null-zoo-v1b-evaluation.json"],
        "level": LEVEL, "bootstrap_reps": reps,
        "monte_carlo_se_at_5pct": {"10000": se(0.05, 10000), "5000": se(0.05, 5000), "2000": se(0.05, 2000)},
        "A": a, "B": b, "C": cc, "D": d, "predictions": p,
    }
    with open(P(args[2] or "analysis/v2/evaluation-v2.json"), "w") as f:
        json.dump(out, f, sort_keys=True, indent=1)
        f.write("\n")
    for k in sorted(p):
        print(k, "held" if p[k]["held"] else "DID NOT HOLD")


if __name__ == "__main__":
    main()
