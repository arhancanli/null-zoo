// Null Zoo v1 validators: every v0 validator, scored again on the stronger generator and the new
// family, plus the candidates for audit_backtest's calibrated headline test and a ceiling.
//
// With every variant's returns (the matrix a search produced):
//   spa_consistent, spa_upper   Hansen's SPA on one stationary bootstrap (js/snooping-core.js, the
//                               code validate_reality_check runs), consistent and upper p-values.
//   reality_check               White's Reality Check from the same bootstrap.
//   stepm_any                   Romano-Wolf StepM finds at least one superior variant at 5%. It is a
//                               decision, not a p-value, so it is scored at the 5% level only.
// With one series (the best trial's returns) and the number of trials:
//   hac_t_sidak                 Newey-West t of the mean (Bartlett, lag floor(4 (T/100)^(2/9))),
//                               Student-t tail, Sidak over the trials.
//   bootstrap_t_sidak           stationary-bootstrap t of the mean (mean block n^(1/3)), Sidak.
// Ceiling:
//   oracle_hac_t                knows which trial carries the skill and tests it alone. No valid
//                               best-of-N test can be more powerful, so it bounds the power bar.
import { VALIDATORS as V0 } from "../validators.mjs";
import { defaultBlock, snoopingTests, stationaryIndices } from "../../../../js/snooping-core.js";
import { studentTUpper } from "../../../../js/student-t.js";
import { bestOfTrialsProbability } from "../../../../js/luck-core.js";
import { makeRandom } from "../../../../js/selection-risk-core.js";

export const V1_SETTINGS = Object.freeze({ spaReps: 1000, bootstrapTReps: 4999, bootstrapBestDraws: 1999 });

const TWO_SIDED = ["haircut_bonferroni", "haircut_bonferroni_lo", "haircut_holm"];
const NEW = ["spa_consistent", "spa_upper", "reality_check", "stepm_any", "hac_t_sidak", "bootstrap_t_sidak", "oracle_hac_t"];
export const V1_VALIDATORS = Object.freeze([...Object.keys(V0), ...TWO_SIDED.map((v) => `${v}_one_sided`), ...NEW]);
export const DECISION_ONLY = Object.freeze(["stepm_any"]);
export const SIDES = Object.freeze(Object.fromEntries(V1_VALIDATORS.map((v) => [v, V0[v]?.sides ?? 1])));

// A 31-bit seed for a core that takes its own seed, drawn from the search's generator so the run
// stays reproducible from the cell seed alone.
const seedFrom = (r) => (r.next() >>> 1) || 1;

/** The t-statistic of a series' mean with a Newey-West (Bartlett) long-run variance. */
export function neweyWestT(xs) {
  const n = xs.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += xs[i];
  mean /= n;
  const lag = Math.floor(4 * Math.pow(n / 100, 2 / 9));
  let lrv = 0;
  for (let i = 0; i < n; i++) { const d = xs[i] - mean; lrv += d * d; }
  for (let l = 1; l <= lag; l++) {
    let g = 0;
    for (let i = l; i < n; i++) g += (xs[i] - mean) * (xs[i - l] - mean);
    lrv += 2 * (1 - l / (lag + 1)) * g;
  }
  lrv /= n;
  return lrv > 0 ? mean / Math.sqrt(lrv / n) : 0;
}

/**
 * One-sided p-value of "the mean is above zero" from a stationary-bootstrap t: each resample's mean
 * is studentized by its own standard deviation and centred on the sample mean, and the observed t is
 * read against that distribution. Blocks carry the dependence; studentizing corrects for skew.
 */
export function blockBootstrapT(xs, { reps, block = defaultBlock(xs.length), seed }) {
  const n = xs.length;
  let sum = 0, sq = 0;
  for (let i = 0; i < n; i++) { sum += xs[i]; sq += xs[i] * xs[i]; }
  const mean = sum / n;
  const sd = Math.sqrt(Math.max((sq - n * mean * mean) / (n - 1), 0));
  if (!(sd > 0)) return 1;
  const observed = mean / (sd / Math.sqrt(n));
  const next = makeRandom(seed);
  const idx = new Int32Array(n);
  let hits = 0;
  for (let b = 0; b < reps; b++) {
    stationaryIndices(n, block, next, idx);
    let s = 0, q = 0;
    for (let i = 0; i < n; i++) { const v = xs[idx[i]]; s += v; q += v * v; }
    const m = s / n;
    const v = (q - n * m * m) / (n - 1);
    // A resample with no spread has an infinite t on either side; it counts as beyond the observed.
    if (!(v > 0)) { hits++; continue; }
    if ((m - mean) / Math.sqrt(v / n) >= observed) hits++;
  }
  return (hits + 1) / (reps + 1);
}

function asMatrix(search) {
  const k = search.length;
  const n = search[0].length;
  const d = new Float64Array(n * k);
  for (let j = 0; j < k; j++) { const xs = search[j]; for (let t = 0; t < n; t++) d[t * k + j] = xs[t]; }
  return { d, n, k };
}

/** Every v1 validator's p-value for one search (`s` from summarize). Draws from `r` in a fixed order. */
export function pValuesV1(search, s, r, settings = V1_SETTINGS) {
  const out = {};
  for (const [name, v] of Object.entries(V0)) out[name] = name === "bootstrap_best" ? v.p(search, s, r, settings.bootstrapBestDraws) : v.p(search, s, r);
  for (const name of TWO_SIDED) out[`${name}_one_sided`] = out[name] / 2;

  const { d, n, k } = asMatrix(search);
  const snoop = snoopingTests(d, n, k, { reps: settings.spaReps, seed: seedFrom(r), alpha: 0.05 });
  out.spa_consistent = snoop.spa.p_consistent;
  out.spa_upper = snoop.spa.p_upper;
  out.reality_check = snoop.reality_check.p_value;
  out.stepm_any = snoop.stepm.superior.length ? 0 : 1;

  const best = search[s.best];
  out.hac_t_sidak = bestOfTrialsProbability(studentTUpper(neweyWestT(best), n - 1), s.trials);
  out.bootstrap_t_sidak = bestOfTrialsProbability(blockBootstrapT(best, { reps: settings.bootstrapTReps, seed: seedFrom(r) }), s.trials);
  out.oracle_hac_t = studentTUpper(neweyWestT(search[0]), n - 1);
  return out;
}
