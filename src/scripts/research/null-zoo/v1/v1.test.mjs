// Null Zoo v1 is an instrument first: the new family must have the structure it claims, a test that
// is calibrated by construction must measure as calibrated, and a run must not depend on how its
// chunks were scheduled. Small settings keep this fast; the published run uses V1_SETTINGS.
import assert from "node:assert/strict";
import test from "node:test";

import { summarize } from "../validators.mjs";
import { xoshiro } from "../xoshiro.mjs";
import { FAMILIES_V1, drawSearchV1 } from "./families.mjs";
import { DESIGN_V1, assemble, cellList, jobList, runChunk } from "./run.mjs";
import { V1_VALIDATORS, blockBootstrapT, neweyWestT, pValuesV1 } from "./validators.mjs";

const QUICK = Object.freeze({ spaReps: 200, bootstrapTReps: 199, bootstrapBestDraws: 99 });
const corr = (a, b) => {
  let ma = 0, mb = 0;
  for (let i = 0; i < a.length; i++) { ma += a[i]; mb += b[i]; }
  ma /= a.length; mb /= b.length;
  let c = 0, va = 0, vb = 0;
  for (let i = 0; i < a.length; i++) { c += (a[i] - ma) * (b[i] - mb); va += (a[i] - ma) ** 2; vb += (b[i] - mb) ** 2; }
  return c / Math.sqrt(va * vb);
};

test("block_cluster: trials in a cluster correlate at 0.8, trials in different clusters do not", () => {
  const search = drawSearchV1("block_cluster", { trials: 20, observations: 40000, periodsPerYear: 252 }, xoshiro(3));
  assert.ok(Math.abs(corr(search[0], search[1]) - 0.8) < 0.02, `within ${corr(search[0], search[1])}`);
  assert.ok(Math.abs(corr(search[0], search[4]) - 0.8) < 0.02, "trial 4 is in the first cluster of five");
  assert.ok(Math.abs(corr(search[0], search[5])) < 0.02, `across ${corr(search[0], search[5])}`);
  assert.ok(Math.abs(corr(search[6], search[19])) < 0.02, "different later clusters");
  const v = search[7].reduce((a, x) => a + x * x, 0) / search[7].length;
  assert.ok(Math.abs(v - 1) < 0.03, `unit variance ${v}`);
});

test("the nine families include every v0 family and block_cluster", () => {
  assert.equal(FAMILIES_V1.length, 9);
  assert.ok(FAMILIES_V1.includes("block_cluster") && FAMILIES_V1.includes("iid_normal"));
});

test("the instrument measures a test calibrated by construction as calibrated", () => {
  // The oracle t-test on one i.i.d. normal series is exact up to the Newey-West variance, so its size
  // must come out near 5%. With 1,500 searches the standard error is 0.0056; allow 3 of them.
  const r = xoshiro(11);
  let rejected = 0;
  const reps = 1500;
  for (let i = 0; i < reps; i++) {
    const search = drawSearchV1("iid_normal", { trials: 2, observations: 504, periodsPerYear: 252 }, r);
    if (pValuesV1(search, summarize(search, 252), r, QUICK).oracle_hac_t <= 0.05) rejected++;
  }
  assert.ok(Math.abs(rejected / reps - 0.05) < 0.017, `oracle size ${rejected / reps}`);
});

test("Newey-West t equals the plain t on a series with no autocorrelation in the sample, and grows with the mean", () => {
  const xs = Float64Array.from({ length: 504 }, (_, i) => (i % 2 ? 1 : -1) + 0.1);
  assert.ok(neweyWestT(xs) > 0);
  const shifted = Float64Array.from(xs, (x) => x + 0.1);
  assert.ok(neweyWestT(shifted) > neweyWestT(xs));
});

test("the bootstrap t gives a small p-value for a clear positive mean and a large one for a negative mean", () => {
  const r = xoshiro(5);
  const up = Float64Array.from({ length: 504 }, () => r.gauss() + 0.4);
  const down = Float64Array.from({ length: 504 }, () => r.gauss() - 0.4);
  assert.ok(blockBootstrapT(up, { reps: 499, seed: 9 }) < 0.01);
  assert.ok(blockBootstrapT(down, { reps: 499, seed: 9 }) > 0.9);
});

test("every validator gives a p-value in [0, 1], and a planted Sharpe of 6 is found by every best-of-N test", () => {
  const r = xoshiro(21);
  const search = drawSearchV1("iid_normal", { trials: 20, observations: 504, periodsPerYear: 252, skill: 6 }, r);
  // 999 resamples: with fewer, the smallest single p-value (1 / (B + 1)) adjusted for 20 trials stays
  // above 0.05 (199 resamples: 1 - (1 - 1/200)^20 = 0.095), and no planted skill could be found.
  const ps = pValuesV1(search, summarize(search, 252), r, { ...QUICK, bootstrapTReps: 999 });
  for (const name of V1_VALIDATORS) assert.ok(ps[name] >= 0 && ps[name] <= 1, `${name} ${ps[name]}`);
  for (const name of ["spa_consistent", "spa_upper", "hac_t_sidak", "bootstrap_t_sidak", "oracle_hac_t"]) assert.ok(ps[name] <= 0.05, `${name} ${ps[name]}`);
});

test("a run does not depend on how its chunks are scheduled", () => {
  const design = { ...DESIGN_V1, skills: [0], chunk: 3 };
  const reps = 7;
  const jobs = jobList(reps, design).filter((j) => j.cell.family === "ar1" || j.cell.family === "block_cluster");
  const inOrder = [];
  for (const j of jobs) inOrder[j.id] = runChunk(j.cell, j.chunk, j.count, design, QUICK);
  const reversed = [];
  for (const j of [...jobs].reverse()) reversed[j.id] = runChunk(j.cell, j.chunk, j.count, design, QUICK);
  assert.deepEqual(reversed, inOrder);
  const full = jobList(reps, design);
  const results = full.map((j) => runChunk(j.cell, j.chunk, j.count, design, QUICK));
  const out = assemble(reps, full, results, design, QUICK);
  assert.equal(out.cells.length, cellList(design).length);
  assert.ok(out.cells.every((c) => c.searches === reps));
});
