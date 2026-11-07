// The autocorrelation-levels study must draw the autocorrelation it claims, its closed form must be exact where it can
// be, and its committed results must agree with v1 where the two overlap and with the closed form everywhere.
// Small settings keep this fast; the committed file was written with 10,000 searches per level.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { xoshiro } from "../xoshiro.mjs";
import { AR1_DESIGN, AR1_TESTS, closedFormSize, cutoffs, drawAr1Search, runLevel, study } from "./ar1-levels.mjs";

const read = (rel) => JSON.parse(readFileSync(new URL(`../../../../${rel}`, import.meta.url), "utf8"));
const RESULTS = read("config/research/null-zoo-v1-ar1-levels.json");
const V1 = read("config/research/null-zoo-v1-evaluation.json");

test("drawAr1Search: lag-one autocorrelation phi and unit variance", () => {
  for (const phi of [0.05, 0.2]) {
    const [xs] = drawAr1Search(phi, { trials: 1, observations: 400000 }, xoshiro(7));
    let m = 0;
    for (const x of xs) m += x;
    m /= xs.length;
    let v = 0, c = 0;
    for (let i = 0; i < xs.length; i++) { const d = xs[i] - m; v += d * d; if (i > 0) c += d * (xs[i - 1] - m); }
    assert.ok(Math.abs(c / v - phi) < 0.005, `lag one ${c / v} for ${phi}`);
    assert.ok(Math.abs(v / xs.length - 1) < 0.02, `variance ${v / xs.length}`);
  }
});

test("closed form: exact without autocorrelation", () => {
  for (const level of AR1_DESIGN.levels) {
    const cut = cutoffs(level, AR1_DESIGN);
    assert.ok(Math.abs(closedFormSize(cut.luck_trials, 0, AR1_DESIGN) - level) < 1e-9, "Sidak over 20 is exact");
    const bonferroni = -Math.expm1(AR1_DESIGN.trials * Math.log1p(-level / AR1_DESIGN.trials));
    assert.ok(Math.abs(closedFormSize(cut.haircut_bonferroni_one_sided, 0, AR1_DESIGN) - bonferroni) < 1e-9, "Bonferroni over 20");
  }
  const cut = cutoffs(0.05, AR1_DESIGN).haircut_bonferroni_one_sided;
  assert.ok(closedFormSize(cut, 0.05, AR1_DESIGN) > closedFormSize(cut, 0, AR1_DESIGN) * 1.4, "a small autocorrelation lifts the size by much more than the standard error");
});

test("a level is reproducible from its seed", () => {
  const a = runLevel(0.1, 40, 11);
  assert.deepEqual(a, runLevel(0.1, 40, 11));
  assert.deepEqual(Object.keys(a), Object.keys(AR1_TESTS));
});

test("committed results: the script's layout, seeds and closed form", () => {
  const small = study(20);
  assert.deepEqual(Object.keys(small), Object.keys(RESULTS));
  assert.deepEqual(small.design, RESULTS.design);
  assert.deepEqual(small.levels.map((l) => [l.phi, l.seed, l.closed_form]), RESULTS.levels.map((l) => [l.phi, l.seed, l.closed_form]));
  assert.equal(RESULTS.reps_per_level, 10000);
});

test("committed results: phi 0 and 0.2 agree with v1's iid_normal and ar1 cells within four Monte Carlo standard errors", () => {
  for (const [phi, family] of [[0, "iid_normal"], [0.2, "ar1"]]) {
    const row = RESULTS.levels.find((l) => l.phi === phi);
    for (const name of Object.keys(AR1_TESTS)) {
      const ours = row.size[name]["0.05"];
      const theirs = V1.table[name].size[family];
      const se = Math.sqrt((ours * (1 - ours)) / RESULTS.reps_per_level + (theirs * (1 - theirs)) / V1.run.reps_per_cell);
      assert.ok(Math.abs(ours - theirs) < 4 * se, `${name} at ${phi}: ${ours} against v1's ${theirs}`);
    }
  }
});

test("committed results: the plain tests follow the closed form, and Lo's correction holds the level", () => {
  for (const row of RESULTS.levels) {
    for (const name of ["haircut_bonferroni_one_sided", "luck_trials"]) {
      const gap = row.size[name]["0.05"] - row.closed_form[name]["0.05"];
      assert.ok(Math.abs(gap) < 0.012, `${name} at ${row.phi}: simulated ${row.size[name]["0.05"]}, closed form ${row.closed_form[name]["0.05"]}`);
    }
    for (const name of ["haircut_bonferroni_lo_one_sided", "luck_trials_lo"]) {
      assert.ok(row.size[name]["0.05"] < 0.06, `${name} at ${row.phi}: ${row.size[name]["0.05"]}`);
    }
  }
});
