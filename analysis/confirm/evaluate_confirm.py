"""Scores the hypotheses of analysis/confirm/prereg-confirm.json from the result files, without hand input.

Reads results-py.json, results-js.json and closed-form.json in this folder; writes evaluation-confirm.json.
Every check is reported with its value and bar, whether it holds or not. A hypothesis holds when all its checks do.
Usage: python analysis/confirm/evaluate_confirm.py
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FAMILIES = ("iid_normal", "student_t4", "skew_negative", "skew_positive", "garch", "ar1",
            "regimes", "correlated_trials", "block_cluster")
UNCORRELATED = ("iid_normal", "student_t4", "skew_negative", "skew_positive", "garch", "regimes")
PHIS = ("0.02", "0.05", "0.1")
Z99 = 2.5758293035489


def load(name):
    return json.load(open(os.path.join(HERE, name)))


PY, JS, CF = load("results-py.json"), load("results-js.json"), load("closed-form.json")


def py(cell, test):
    c = PY["cells"][cell]
    return c["rejections"][test][1] / c["searches"], c["searches"]


def js(exp, cell, test):
    c = JS["experiments"][str(exp)]["cells"][cell]
    return c["rejections"][test][1] / c["searches"], c["searches"]


def se(p, n):
    return math.sqrt(p * (1 - p) / n)


def cf(rho, phi, test):
    return CF["size"][f"rho{rho:g}_phi{float(phi):g}"][test]["0.05"]


checks = {}


def check(h, label, value, bar, ok):
    checks.setdefault(h, []).append({"check": label, "value": round(value, 6), "bar": bar, "holds": bool(ok)})


def main():
    for f in FAMILIES:
        v, _ = py(f"S.{f}", "arch882_b8_consistent")
        check("H1", f"arch882 block 8, {f}", v, "<= 0.065", v <= 0.065)
        v, _ = py(f"S.{f}", "arch882_b22_consistent")
        check("H2", f"arch882 block 22, {f}", v, "<= 0.070", v <= 0.070)
        d = py(f"S.{f}", "fixed_spa_c_b22")[0] - v
        check("H3", f"fixed-variance minus arch882, block 22, {f}", d, ">= 0.020", d >= 0.020)
        c, r = js(1, f"S.{f}", "corrected.spa_consistent")[0], js(1, f"S.{f}", "corrected.reality_check")[0]
        if f in UNCORRELATED:
            check("H4", f"corrected SPA, {f}", c, "<= 0.065", c <= 0.065)
        check("H5", f"|corrected SPA - Reality Check|, {f}", abs(c - r), "<= 0.010", abs(c - r) <= 0.010)
    v = js(1, "S.skew_negative", "v1.spa_consistent")[0]
    check("H6", "v1 SPA, skew_negative", v, ">= 0.10", v >= 0.10)
    v = js(1, "S.skew_negative", "corrected.spa_consistent")[0]
    check("H6", "corrected SPA, skew_negative", v, "<= 0.065", v <= 0.065)
    v = py("S.skew_negative", "fixed_spa_c_b8")[0]
    check("H6", "fixed-variance spa_c block 8, skew_negative", v, ">= 0.10", v >= 0.10)
    v = py("S.skew_negative", "arch882_b8_consistent")[0]
    check("H6", "arch882 block 8, skew_negative", v, "<= 0.065", v <= 0.065)
    for cell in JS["experiments"]["1"]["cells"]:
        d = abs(js(1, cell, "corrected.stepm_any")[0] - js(1, cell, "corrected.spa_consistent")[0])
        check("H7", f"|corrected StepM - corrected SPA|, {cell}", d, "<= 0.005", d <= 0.005)
    for phi in PHIS:
        cell = f"L.phi{phi}"
        for t in ("luck_trials", "haircut_bonferroni_one_sided"):
            v, n = js(2, cell, t)
            edge = 0.05 + Z99 * se(0.05, n)
            check("H8", f"{t}, phi {phi}", v, f"> {edge:.5f}", v > edge)
            pi = cf(0, phi, t)
            z = (v - pi) / se(pi, n)
            check("H9", f"{t} minus Proposition 2 ({pi:.5f}) in standard errors, phi {phi}", z, "|z| <= 3", abs(z) <= 3)
        for t in ("luck_trials_lo", "haircut_bonferroni_lo_one_sided"):
            v, n = js(2, cell, t)
            lo, hi = 0.05 - Z99 * se(0.05, n), 0.05 + Z99 * se(0.05, n)
            check("H10", f"{t}, phi {phi}", v, f"in [{lo:.5f}, {hi:.5f}]", lo <= v <= hi)
        b8 = {"arch882 block 8": py(cell, "arch882_b8_consistent")[0], "Reality Check block 8": py(cell, "rc_b8")[0],
              "corrected SPA": js(1, cell, "corrected.spa_consistent")[0], "corrected StepM": js(1, cell, "corrected.stepm_any")[0]}
        for name, v in b8.items():
            check("H11", f"{name}, phi {phi}", v, "<= 0.065", v <= 0.065)
        v = py(cell, "arch882_b22_consistent")[0]
        check("H11", f"arch882 block 22, phi {phi}", v, "<= 0.070", v <= 0.070)
        v = py(cell, "fixed_spa_c_b22")[0]
        check("H12", f"fixed-variance spa_c block 22, phi {phi}", v, ">= 0.09", v >= 0.09)
        v = js(1, cell, "v1.spa_consistent")[0]
        check("H12", f"v1 SPA (block 8), phi {phi}", v, ">= 0.065", v >= 0.065)
        if phi != "0.02":
            hc = js(2, cell, "haircut_bonferroni_one_sided")[0]
            for name, v in b8.items():
                check("H13", f"haircut ({hc:.4f}) minus {name}, phi {phi}", hc - v, ">= 0.010", hc - v >= 0.010)
    for cell, spec in JS["experiments"]["3"]["cells"].items():
        for t in ("luck_trials", "haircut_bonferroni_one_sided"):
            v, n = js(3, cell, t)
            pi = cf(spec["rho"], spec["phi"], t)
            z = (v - pi) / se(pi, n)
            check("H14", f"{t} minus the equicorrelated closed form ({pi:.5f}) in standard errors, {cell}", z, "|z| <= 3", abs(z) <= 3)
            if spec["rho"] == 0.6:
                pi0 = cf(0, spec["phi"], t)
                gap = (pi0 - v) / se(v, n)
                check("H15", f"independent closed form ({pi0:.5f}) minus simulated {t}, in standard errors, {cell}", gap, "> 3", gap > 3)
    prereg = load("prereg-confirm.json")
    out = {"schema": "canli.null-zoo.confirm.evaluation", "prereg": "analysis/confirm/prereg-confirm.json",
           "searches": {"python_joint": PY["searches_per_cell"],
                        "js_joint": JS["experiments"]["1"]["searches_per_cell"],
                        "js_single_L": JS["experiments"]["2"]["searches_per_cell"],
                        "js_single_C": JS["experiments"]["3"]["searches_per_cell"]},
           "stopped_at_deadline": {"python": PY["stopped_at_deadline"], "js": JS["stopped_at_deadline"]},
           "hypotheses": {}}
    for h in prereg["hypotheses"]:
        cs = checks[h["id"]]
        out["hypotheses"][h["id"]] = {"statement": h["statement"], "holds": all(c["holds"] for c in cs),
                                      "checks_held": sum(c["holds"] for c in cs), "checks": cs}
    out["held"] = [k for k, v in out["hypotheses"].items() if v["holds"]]
    out["failed"] = [k for k, v in out["hypotheses"].items() if not v["holds"]]
    with open(os.path.join(HERE, "evaluation-confirm.json"), "w") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    for k, v in out["hypotheses"].items():
        print(k, "HOLDS" if v["holds"] else "FAILS", f"{v['checks_held']}/{len(v['checks'])}")


if __name__ == "__main__":
    main()
