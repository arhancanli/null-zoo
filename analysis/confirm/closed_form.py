"""Closed-form sizes for the 2026-11-08 confirmation run (analysis/confirm/prereg-confirm.json).

Proposition 2 (independent strategies) and its extension to equicorrelated strategies (Appendix A of the paper).
A best-of-N test rejects when the largest t-statistic exceeds a cutoff c chosen for i.i.d. returns. Under a
Gaussian AR(1) with coefficient phi each t-statistic is approximately sqrt(lambda) times a Student t_{T-1},
lambda = (1 + phi) / (1 - phi), so one strategy exceeds c with probability u = F_bar_{T-1}(c / sqrt(lambda)).
  independent strategies:      size = 1 - (1 - u)^N
  equicorrelated, corr. rho:   size = 1 - int Phi((c' - sqrt(rho) z) / sqrt(1 - rho))^N phi(z) dz,  c' = Phi_bar^{-1}(u)
The second reduces to the first at rho = 0. The values are predictions: this script draws nothing at random, and
its output is committed with the pre-registration, before any search of the confirmation run is drawn.
Usage: python analysis/confirm/closed_form.py > analysis/confirm/closed-form.json
"""
import json
import os
import sys

import numpy as np
from scipy import integrate, stats

T, N = 504, 20
LEVELS = (0.01, 0.05, 0.1)
PHIS = (0.0, 0.02, 0.05, 0.1, 0.2)
RHOS = (0.0, 0.3, 0.6)
HERE = os.path.dirname(os.path.abspath(__file__))


def cutoffs(alpha, n=N):
    """Each test's cutoff on the i.i.d. Student t scale (one-sided haircut: Bonferroni; LET: Sidak)."""
    return {"haircut_bonferroni_one_sided": float(stats.t.isf(alpha / n, T - 1)),
            "luck_trials": float(stats.t.isf(-np.expm1(np.log1p(-alpha) / n), T - 1))}


def size_s(c, s, rho, n=N):
    """The size when each t-statistic is s times a Student t_{T-1} variable (s = sqrt(lambda))."""
    u = float(stats.t.sf(c / s, T - 1))
    if rho == 0:
        return float(-np.expm1(n * np.log1p(-u)))
    cp = float(stats.norm.isf(u))
    a, b = np.sqrt(rho), np.sqrt(1 - rho)
    f = lambda z: np.exp(n * stats.norm.logcdf((cp - a * z) / b)) * stats.norm.pdf(z)
    return float(1 - integrate.quad(f, -12, 12, epsabs=1e-13, epsrel=1e-12, limit=400)[0])


def size(c, phi, rho, n=N):
    """Proposition 2 (rho = 0) and its equicorrelated extension, at AR(1) coefficient phi."""
    return size_s(c, np.sqrt((1 + phi) / (1 - phi)), rho, n)


def elasticity(c, rho, n=N, h=1e-5):
    """d ln size / d ln sqrt(lambda) at phi = 0, by a central difference in s = sqrt(lambda)."""
    return (np.log(size_s(c, 1 + h, rho, n)) - np.log(size_s(c, 1 - h, rho, n))) / (np.log(1 + h) - np.log(1 - h))


key = lambda rho, phi: f"rho{rho:g}_phi{phi:g}"


def main():
    out = {"schema": "canli.null-zoo.confirm.closed-form", "T": T, "N": N,
           "formula": {"independent": "1 - (1 - u)^N, u = F_bar_{T-1}(c / sqrt((1 + phi) / (1 - phi)))",
                       "equicorrelated": "1 - int Phi((c' - sqrt(rho) z) / sqrt(1 - rho))^N phi(z) dz, c' = Phi_bar^{-1}(u)"},
           "integration": "scipy.integrate.quad on [-12, 12]", "scipy": __import__("scipy").__version__,
           "cutoffs": {str(a): cutoffs(a) for a in LEVELS}, "size": {}}
    for rho in RHOS:
        for phi in PHIS:
            out["size"][key(rho, phi)] = {
                "rho": rho, "phi": phi,
                **{name: {str(a): round(size(cutoffs(a)[name], phi, rho), 8) for a in LEVELS}
                   for name in ("haircut_bonferroni_one_sided", "luck_trials")}}
    # Proportional sensitivity to the standard error at phi = 0, 5% level, LET cutoff.
    out["elasticity_5pct_luck_trials"] = {f"rho{rho:g}": round(float(elasticity(cutoffs(0.05)["luck_trials"], rho)), 4)
                                          for rho in (0.0, 0.3, 0.6, 0.9)}
    out["elasticity_5pct_single_test"] = round(float(elasticity(cutoffs(0.05, 1)["luck_trials"], 0.0, n=1)), 4)
    # Check: at rho = 0 this must reproduce the closed form in the committed ar1-levels file (JavaScript t tail).
    ar1 = json.load(open(os.path.join(HERE, "../../src/config/research/null-zoo-v1-ar1-levels.json")))
    worst = 0.0
    for lev in ar1["levels"]:
        for name, by in lev["closed_form"].items():
            for a, v in by.items():
                worst = max(worst, abs(v - out["size"][key(0, lev["phi"])][name][a]))
    if worst > 2e-6:
        raise SystemExit(f"rho = 0 closed form differs from ar1-levels by {worst}")
    out["check_against_ar1_levels_max_abs_diff"] = worst
    sys.stdout.write(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
