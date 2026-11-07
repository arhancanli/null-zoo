"""Scores the pre-registered predictions of analysis/anomalies/prereg-anomalies.json."""
import json
import sys

res = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "analysis/anomalies/anomalies-results.json"))
W = res["windows"]
tot = {t: sum(w["tests"][t]["superior"] for w in W) for t in ("fixed", "fixed_q10", "boot_t", "rc")}
pred = {
    "A1": {"held": tot["fixed"] > tot["boot_t"], "values": [tot["fixed"], tot["boot_t"]]},
    "A2": {"held": tot["fixed_q10"] >= tot["fixed"], "values": [tot["fixed_q10"], tot["fixed"]]},
    "A3": {"held": sum(w["tests"]["fixed"]["superior"] >= w["tests"]["boot_t"]["superior"] for w in W) >= 5,
           "values": sum(w["tests"]["fixed"]["superior"] >= w["tests"]["boot_t"]["superior"] for w in W)},
}
json.dump({"totals": tot, "predictions": pred}, sys.stdout, indent=1)
sys.stdout.write("\n")
