// The Null Zoo is an instrument: before any validator is scored on it, each family must have the
// properties it claims (mean zero, unit variance, the stated shape), and a run must be reproducible.
import assert from "node:assert/strict";
import test from "node:test";

import { FAMILIES, drawSearch, rng } from "./families.mjs";
import { DESIGN, scoreCell } from "./run.mjs";
import { moments } from "./validators.mjs";

function pooled(family, { trials = 4, observations = 50000, skill = 0 } = {}) {
  const search = drawSearch(family, { trials, observations, periodsPerYear: 252, skill }, rng(7));
  return search;
}
const lag1 = (xs) => { const m = moments(xs).mean; let c = 0, v = 0; for (let i = 1; i < xs.length; i++) c += (xs[i] - m) * (xs[i - 1] - m); for (const x of xs) v += (x - m) ** 2; return c / v; };
const corr = (a, b) => { const ma = moments(a).mean, mb = moments(b).mean; let c = 0, va = 0, vb = 0; for (let i = 0; i < a.length; i++) { c += (a[i] - ma) * (b[i] - mb); va += (a[i] - ma) ** 2; vb += (b[i] - mb) ** 2; } return c / Math.sqrt(va * vb); };

test("every family has mean zero and unit variance, so every trial's true Sharpe is zero", () => {
  for (const family of FAMILIES) {
    for (const xs of pooled(family)) {
      const m = moments(xs);
      const variance = xs.reduce((a, x) => a + (x - m.mean) ** 2, 0) / xs.length;
      assert.ok(Math.abs(m.mean) < 0.03, `${family} mean ${m.mean}`);
      assert.ok(Math.abs(variance - 1) < 0.12, `${family} variance ${variance}`);
    }
  }
});

test("each family has the shape it claims", () => {
  const shape = (family) => moments(pooled(family)[0]);
  assert.ok(shape("skew_negative").skew < -1, "negative skew");
  assert.ok(shape("skew_positive").skew > 1, "positive skew");
  assert.ok(shape("student_t4").kurtosis > 5, "fat tails");
  assert.ok(Math.abs(shape("iid_normal").kurtosis - 3) < 0.15, "normal kurtosis");
  assert.ok(shape("garch").kurtosis > 3.3, "GARCH clustering fattens tails");
  const [a] = pooled("ar1");
  assert.ok(Math.abs(lag1(a) - 0.2) < 0.02, `AR(1) coefficient ${lag1(a)}`);
  assert.ok(Math.abs(lag1(pooled("iid_normal")[0])) < 0.02, "iid has no autocorrelation");
  const sq = (xs) => Float64Array.from(xs, (x) => x * x);
  assert.ok(lag1(sq(pooled("garch")[0])) > 0.1, "GARCH volatility clusters");
  assert.ok(lag1(sq(pooled("regimes")[0])) > 0.1, "regimes cluster volatility");
  const c = pooled("correlated_trials");
  assert.ok(Math.abs(corr(c[0], c[1]) - 0.5) < 0.03, "trials correlate at 0.5");
  assert.ok(Math.abs(corr(pooled("iid_normal")[0], pooled("iid_normal")[1])) < 0.03, "independent trials do not");
});

test("skill gives trial 0, and only trial 0, the stated annualized Sharpe", () => {
  for (const family of ["iid_normal", "correlated_trials", "skew_negative"]) {
    const [skilled, other] = pooled(family, { trials: 2, observations: 252 * 400, skill: 2 });
    assert.ok(Math.abs(moments(skilled).sharpe * Math.sqrt(252) - 2) < 0.2, `${family} skilled`);
    assert.ok(Math.abs(moments(other).sharpe * Math.sqrt(252)) < 0.2, `${family} other`);
  }
});

test("a cell is reproducible from its seed, and different seeds differ", () => {
  const a = scoreCell("skew_negative", { reps: 30, skill: 0, seed: 11 });
  assert.deepEqual(scoreCell("skew_negative", { reps: 30, skill: 0, seed: 11 }), a);
  const x = drawSearch("iid_normal", DESIGN, rng(1))[0][0];
  assert.notEqual(drawSearch("iid_normal", DESIGN, rng(2))[0][0], x);
});
