// Null Zoo v1, autocorrelation levels: how often the Sharpe-based tests that ignore autocorrelation, and the same tests
// with Lo's (2002) correction, call a skill-less search skilled when every strategy's daily returns follow an AR(1)
// with lag-one autocorrelation 0, 0.02, 0.05, 0.1 or 0.2. v1's ar1 family has 0.2 only, a stress level; this adds the
// small values. Same design as v1: 20 strategies over 504 days, 10,000 searches per level, nominal 1%, 5% and 10%.
//
// Next to each simulated size is the closed form. Under an AR(1) the t-statistic of a mean has variance
// (1 + phi) / (1 - phi) instead of 1, so a test whose cutoff on the iid scale is c rejects one skill-less strategy with
// probability P(T > c / sqrt((1 + phi) / (1 - phi))), and the best of N with 1 - (1 - that)^N. Because a best-of-20
// cutoff sits far in the tail, a standard error a few percent too small raises the size by much more.
// phi 0 and 0.2 repeat v1's iid_normal and ar1 cells, as a check (ar1-levels.test.mjs).
//   node scripts/research/null-zoo/v1/ar1-levels.mjs [searches per level] > config/research/null-zoo-v1-ar1-levels.json
import { fileURLToPath } from "node:url";
import { studentTQuantileUpper, studentTUpper } from "../../../../js/student-t.js";
import { VALIDATORS, summarize } from "../validators.mjs";
import { xoshiro } from "../xoshiro.mjs";

export const AR1_DESIGN = Object.freeze({
  trials: 20, observations: 504, periodsPerYear: 252, levels: Object.freeze([0.01, 0.05, 0.1]),
  phis: Object.freeze([0, 0.02, 0.05, 0.1, 0.2]), baseSeed: 20261105,
});

// One-sided p-values, called as v1 calls them; v1 halves the haircut's two-sided p-values the same way.
export const AR1_TESTS = Object.freeze({
  luck_trials: (search, s) => VALIDATORS.luck_trials.p(search, s),
  luck_trials_lo: (search, s) => VALIDATORS.luck_trials_lo.p(search, s),
  haircut_bonferroni_one_sided: (search, s) => VALIDATORS.haircut_bonferroni.p(search, s) / 2,
  haircut_bonferroni_lo_one_sided: (search, s) => VALIDATORS.haircut_bonferroni_lo.p(search, s) / 2,
});

export const levelSeed = (design, i) => (design.baseSeed + i * 1000003) >>> 0;

/** One search of skill-less AR(1) strategies with coefficient phi and unit variance, each started from its stationary distribution. */
export function drawAr1Search(phi, { trials, observations }, r) {
  const scale = Math.sqrt(1 - phi * phi);
  const out = [];
  for (let k = 0; k < trials; k++) {
    const xs = new Float64Array(observations);
    let x = r.gauss();
    for (let t = 0; t < observations; t++) { x = phi * x + scale * r.gauss(); xs[t] = x; }
    out.push(xs);
  }
  return out;
}

/** Each plain test's cutoff on the iid t scale: the haircut's one-sided Bonferroni rejects when one strategy's p is at most level / N, luck-equivalent trials when it is at most 1 - (1 - level)^(1/N). */
export function cutoffs(level, { trials, observations }) {
  const df = observations - 1;
  return {
    haircut_bonferroni_one_sided: studentTQuantileUpper(level / trials, df),
    luck_trials: studentTQuantileUpper(-Math.expm1(Math.log1p(-level) / trials), df),
  };
}

/** The closed-form size of a best-of-N test with cutoff `cut` when the strategies' returns are AR(1) with coefficient phi. */
export function closedFormSize(cut, phi, { trials, observations }) {
  const one = studentTUpper(cut / Math.sqrt((1 + phi) / (1 - phi)), observations - 1);
  return -Math.expm1(trials * Math.log1p(-one));
}

/** Rejection counts of every test at every level over `reps` searches at one phi, from one seed. */
export function runLevel(phi, reps, seed, design = AR1_DESIGN) {
  const r = xoshiro(seed);
  const rejections = Object.fromEntries(Object.keys(AR1_TESTS).map((name) => [name, design.levels.map(() => 0)]));
  for (let rep = 0; rep < reps; rep++) {
    const search = drawAr1Search(phi, design, r);
    const s = summarize(search, design.periodsPerYear);
    for (const [name, test] of Object.entries(AR1_TESTS)) {
      const p = test(search, s);
      if (!(p >= 0 && p <= 1)) throw new RangeError(`${name} gave p = ${p} at phi ${phi}`);
      design.levels.forEach((a, i) => { if (p <= a) rejections[name][i]++; });
    }
  }
  return rejections;
}

const round6 = (x) => Number(x.toFixed(6));

export function study(reps, design = AR1_DESIGN) {
  const byLevel = (f) => Object.fromEntries(design.levels.map((a, i) => [String(a), f(a, i)]));
  return {
    schema: "canli.null-zoo.v1.ar1-levels",
    generator: "xoshiro128** seeded by splitmix32; one seed per autocorrelation level",
    design, reps_per_level: reps,
    monte_carlo_se_at_5_percent: round6(Math.sqrt((0.05 * 0.95) / reps)),
    closed_form: "1 - (1 - P(T > c / sqrt((1 + phi) / (1 - phi))))^N, with c the test's own cutoff for one strategy and T Student t with observations - 1 degrees of freedom",
    levels: design.phis.map((phi, i) => {
      const counts = runLevel(phi, reps, levelSeed(design, i), design);
      return {
        phi, searches: reps, seed: levelSeed(design, i),
        size: Object.fromEntries(Object.entries(counts).map(([name, c]) => [name, byLevel((_, j) => c[j] / reps)])),
        closed_form: Object.fromEntries(["haircut_bonferroni_one_sided", "luck_trials"].map((name) => [name, byLevel((a) => round6(closedFormSize(cutoffs(a, design)[name], phi, design)))])),
      };
    }),
  };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const reps = Number(process.argv[2] ?? 10000);
  const started = Date.now();
  const out = study(reps);
  for (const l of out.levels) process.stderr.write(`phi ${l.phi}: haircut ${l.size.haircut_bonferroni_one_sided["0.05"]} (closed form ${l.closed_form.haircut_bonferroni_one_sided["0.05"]}), Lo ${l.size.haircut_bonferroni_lo_one_sided["0.05"]}; luck ${l.size.luck_trials["0.05"]} (closed form ${l.closed_form.luck_trials["0.05"]}), Lo ${l.size.luck_trials_lo["0.05"]}\n`);
  process.stderr.write(`done in ${((Date.now() - started) / 1000).toFixed(0)} s\n`);
  process.stdout.write(`${JSON.stringify(out, null, 1)}\n`);
}
