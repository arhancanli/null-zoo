// Robustness run for the paper. It reuses the published Null Zoo families and validators and
// varies only what a reviewer would ask about:
//   1. a stronger generator (xoshiro128** seeded by splitmix32) in place of xorshift32;
//   2. more replications per cell (default 10,000) so Monte Carlo errors are small;
//   3. the bootstrap with 1,999 resamples instead of 400 (with 400 and a Sidak adjustment for 20
//      trials the smallest attainable p-value is 1 - (400/401)^20 = 0.0487, so the test can reject
//      only when no resample reaches the observed Sharpe ratio, which costs power);
//   4. the two-sided haircut p-values also read one-sided (p / 2);
//   5. every p-value kept in memory, so size-adjusted power can be computed from each validator's
//      own null distribution in the same family;
//   6. two diagnostics: the cross-trial standard deviation of the annualized Sharpe ratios relative
//      to the i.i.d. standard error sqrt(q / T) (the DSR's benchmark scales with it), and the mean
//      sample skewness of the selected best series against the other series.
// Output keeps raw counts, so every rate's standard error can be recomputed from the file.
//   node analysis/extend.mjs [reps] [bootstrapDraws] > analysis/null-zoo-extended.json
import { FAMILIES, drawSearch } from "../src/scripts/research/null-zoo/families.mjs";
import { DESIGN } from "../src/scripts/research/null-zoo/run.mjs";
import { VALIDATORS, summarize } from "../src/scripts/research/null-zoo/validators.mjs";

// splitmix32 expands one seed into four state words; xoshiro128** draws 32-bit outputs.
export function xoshiro(seed) {
  let z = seed >>> 0;
  const split = () => {
    z = (z + 0x9e3779b9) >>> 0;
    let x = z;
    x = Math.imul(x ^ (x >>> 16), 0x85ebca6b) >>> 0;
    x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35) >>> 0;
    return (x ^ (x >>> 16)) >>> 0;
  };
  const s = [split(), split(), split(), split()];
  const rotl = (x, k) => ((x << k) | (x >>> (32 - k))) >>> 0;
  const next = () => {
    const result = Math.imul(rotl(Math.imul(s[1], 5) >>> 0, 7), 9) >>> 0;
    const t = (s[1] << 9) >>> 0;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] = (s[2] ^ t) >>> 0; s[3] = rotl(s[3], 11);
    s[0] >>>= 0; s[1] >>>= 0; s[2] >>>= 0;
    return result;
  };
  // 53-bit uniform in (0, 1) from two outputs.
  const u = () => ((next() >>> 5) * 67108864 + (next() >>> 6) + 0.5) / 9007199254740992;
  let spare = null;
  const gauss = () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    const r = Math.sqrt(-2 * Math.log(u()));
    const a = 2 * Math.PI * u();
    spare = r * Math.sin(a);
    return r * Math.cos(a);
  };
  return { u, gauss };
}

// Validators as scored here: the published ones, the haircut also read one-sided, and the bootstrap
// with more resamples.
const TWO_SIDED = ["haircut_bonferroni", "haircut_bonferroni_lo", "haircut_holm"];
export const SCORED = [
  "luck_trials", "luck_trials_lo", "luck_trials_nonnormal_se",
  ...TWO_SIDED, ...TWO_SIDED.map((v) => `${v}_one_sided`),
  "deflated_sharpe", "bootstrap_best",
];
function pValues(search, s, r, draws) {
  const out = {};
  for (const v of Object.keys(VALIDATORS)) out[v] = v === "bootstrap_best" ? VALIDATORS[v].p(search, s, r, draws) : VALIDATORS[v].p(search, s, r);
  for (const v of TWO_SIDED) out[`${v}_one_sided`] = out[v] / 2;
  return out;
}

export function runCell(family, { reps, skill, seed, draws, design = DESIGN }) {
  const r = xoshiro(seed);
  const p = Object.fromEntries(SCORED.map((v) => [v, new Float64Array(reps)]));
  const correct = new Uint8Array(reps);
  let sdRatio = 0, skewBest = 0, skewOthers = 0;
  const iidSe = Math.sqrt(design.periodsPerYear / design.observations);
  for (let rep = 0; rep < reps; rep++) {
    const search = drawSearch(family, { ...design, skill }, r);
    const s = summarize(search, design.periodsPerYear);
    correct[rep] = s.best === 0 ? 1 : 0;
    const ps = pValues(search, s, r, draws);
    for (const v of SCORED) p[v][rep] = ps[v];
    sdRatio += s.sdAnnual / iidSe;
    skewBest += s.stats[s.best].skew;
    skewOthers += s.stats.reduce((a, st, k) => a + (k === s.best ? 0 : st.skew), 0) / (s.trials - 1);
  }
  return { p, correct, diagnostics: { cross_trial_sd_over_iid_se: sdRatio / reps, mean_skew_best: skewBest / reps, mean_skew_others: skewOthers / reps } };
}

const count = (arr, c) => { let n = 0; for (const x of arr) if (x <= c) n++; return n; };
// The largest attainable cutoff whose null rejection rate does not exceed the level: the size-
// adjusted critical value. With ties (a discrete p-value) the rate can fall below the level.
function sizeAdjustedCutoff(nullP, level) {
  const sorted = Float64Array.from(nullP).sort();
  const k = Math.floor(level * sorted.length);
  if (k === 0) return -Infinity;
  let c = sorted[k - 1];
  // Step down past ties that would push the null rate above the level.
  let i = k - 1;
  while (i >= 0 && count(sorted, c) > k) { i--; c = i >= 0 ? sorted[i] : -Infinity; }
  return c;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const reps = Number(process.argv[2] ?? 10000);
  const draws = Number(process.argv[3] ?? 1999);
  const level = DESIGN.level;
  const rows = [];
  let seed = 20260929;
  for (const family of FAMILIES) {
    const nul = runCell(family, { reps, skill: 0, seed: seed++, draws });
    const alt = runCell(family, { reps, skill: DESIGN.skill, seed: seed++, draws });
    const row = { family, seeds: [seed - 2, seed - 1], size: {}, power: {}, correct: {}, size_adjusted: {} };
    for (const v of SCORED) {
      row.size[v] = count(nul.p[v], level);
      row.power[v] = count(alt.p[v], level);
      let c = 0;
      for (let i = 0; i < reps; i++) if (alt.p[v][i] <= level && alt.correct[i]) c++;
      row.correct[v] = c;
      const cut = sizeAdjustedCutoff(nul.p[v], level);
      row.size_adjusted[v] = { cutoff: cut, null_rate_count: count(nul.p[v], cut), power: count(alt.p[v], cut) };
    }
    row.best_is_skilled_trial = alt.correct.reduce((a, b) => a + b, 0);
    row.diagnostics = { null: nul.diagnostics, skill: alt.diagnostics };
    rows.push(row);
    process.stderr.write(`${family} `);
  }
  process.stderr.write("\n");
  console.log(JSON.stringify({
    schema: "canli.null-zoo.extended.v2",
    generator: "xoshiro128** seeded by splitmix32; Box-Muller normals",
    design: DESIGN,
    reps,
    bootstrap_draws: draws,
    notes: {
      one_sided: "*_one_sided reads each two-sided haircut p-value as p/2",
      size_adjusted: "power at the largest cutoff whose rejection rate in the same family's null arm is at most the level",
      counts: "size, power, correct and size_adjusted counts are out of reps",
    },
    rows,
  }, null, 1));
}
