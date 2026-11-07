// =============================================================================
// snooping-core.js
// -----------------------------------------------------------------------------
// Data-snooping tests for a search: did the best of K variants beat the benchmark by more than the
// best of K would by luck? One stationary bootstrap (Politis and Romano 1994) feeds three tests:
//
//   White's Reality Check (2000)     the best mean excess return against the bootstrap distribution
//                                    of the best recentred mean. Not studentized, so one volatile
//                                    variant can dominate it.
//   Hansen's SPA (2005)              the best t-statistic, each variant studentized by the
//                                    Politis-Romano variance of its mean. Three recentrings give a
//                                    lower bound, a consistent p-value and an upper bound: poor
//                                    variants are not allowed to inflate the p-value.
//   Romano and Wolf's StepM (2005)   step-down: which variants beat the benchmark, controlling the
//                                    familywise error at alpha.
//
// d[t][k] is variant k's return minus the benchmark's in period t; higher is better. Every draw
// comes from makeRandom(seed), so a result reproduces exactly from its inputs. The reference
// implementation it is checked against (Python's arch 8.0, and a separate numpy transcription of
// Hansen's formulas for the studentized SPA) draws with numpy, so agreement is within Monte Carlo
// error, not to the digit; scripts/research/reality-check/ holds the comparison.
// =============================================================================

import { makeRandom } from "./selection-risk-core.js";

/** Indices of one stationary-bootstrap resample of n periods with mean block length `block`. */
export function stationaryIndices(n, block, next, out = new Int32Array(n)) {
  const p = 1 / block;
  let at = Math.floor(next() * n);
  out[0] = at;
  for (let t = 1; t < n; t += 1) {
    at = next() < p ? Math.floor(next() * n) : at + 1 === n ? 0 : at + 1;
    out[t] = at;
  }
  return out;
}

/** round(n^(1/3)), at least 1: the default mean block length. */
export const defaultBlock = (n) => Math.max(1, Math.round(Math.cbrt(n)));

// Weights below this are dropped from the variance sum. Each term they multiply is bounded by the
// sample variance, so what is dropped is under 2e-14 * n of it (4e-10 at 20,000 periods).
const KAPPA_FLOOR = 1e-14;

/**
 * n times the variance of each column's mean under the stationary bootstrap, in closed form
 * (Politis and Romano 1994, as Hansen 2005 uses it): gamma_0 + 2 * sum_i kappa_i * gamma_i, with
 * kappa_i = (1 - i/n)(1 - p)^i + (i/n)(1 - p)^(n - i). Lags whose weight is below KAPPA_FLOOR are
 * skipped, which keeps the cost near n * k * 30 * block rather than n^2 * k.
 */
export function bootstrapVariance(d, n, k, block, means) {
  const p = 1 / block;
  const out = new Float64Array(k);
  const cross = (lag, weight) => {
    for (let t = 0; t + lag < n; t += 1) {
      const a = t * k;
      const b = (t + lag) * k;
      for (let j = 0; j < k; j += 1) out[j] += weight * (d[a + j] - means[j]) * (d[b + j] - means[j]);
    }
  };
  cross(0, 1 / n);
  const q = 1 - p;
  for (let i = 1; i < n; i += 1) {
    const kappa = (1 - i / n) * q ** i + (i / n) * q ** (n - i);
    if (kappa < KAPPA_FLOOR) {
      // The first term has decayed; the wrap-around term (i/n)(1 - p)^(n - i) only reaches the
      // floor again once n - i <= log(floor) / log(1 - p). Skip to there.
      const tail = q > 0 ? Math.log(KAPPA_FLOOR) / Math.log(q) : 0;
      const jump = Math.floor(n - tail);
      if (jump > i + 1) i = Math.min(jump, n) - 1;
      continue;
    }
    cross(i, (2 * kappa) / n);
  }
  return out;
}

/**
 * All three tests on one bootstrap. `d` is a row-major Float64Array of n periods by k variants.
 * Returns means, the variance estimates, the bootstrap maxima under each recentring and the
 * StepM rejections; p-values are shares of `reps` draws strictly above the observed statistic.
 */
export function snoopingTests(d, n, k, { block = defaultBlock(n), reps = 1000, seed = 42, alpha = 0.05, studentize = true } = {}) {
  if (!(n >= 2 && k >= 1)) throw new RangeError("need at least 2 periods and 1 variant");
  if (!(block >= 1 && block <= n)) throw new RangeError(`block_length must be from 1 to the number of periods (${n})`);
  if (!Number.isInteger(reps) || reps < 100) throw new RangeError("reps must be an integer of at least 100");
  if (!(alpha > 0 && alpha < 1)) throw new RangeError("alpha must be between 0 and 1");
  const means = new Float64Array(k);
  for (let t = 0; t < n; t += 1) for (let j = 0; j < k; j += 1) means[j] += d[t * k + j];
  for (let j = 0; j < k; j += 1) means[j] /= n;
  const variance = bootstrapVariance(d, n, k, block, means);
  const sd = new Float64Array(k);
  for (let j = 0; j < k; j += 1) sd[j] = Math.sqrt(Math.max(variance[j], 0));
  // A variant whose excess return never varies has no sampling distribution to studentize by.
  const usable = Array.from(sd, (s) => s > 0);
  if (!usable.some(Boolean)) throw new RangeError("every variant's excess return is constant; there is nothing to test");
  const root = Math.sqrt(n);
  const scale = (j) => (studentize ? root / sd[j] : 1);
  // Hansen's recentrings: lower keeps poor variants at zero, consistent drops only those far below
  // zero, upper recentres everything (White's null, the least favourable configuration).
  const threshold = (j) => -Math.sqrt((variance[j] / n) * 2 * Math.log(Math.log(n)));
  const centre = [
    Float64Array.from(means, (m) => Math.max(m, 0)),
    Float64Array.from(means, (m, j) => (m >= threshold(j) ? m : 0)),
    Float64Array.from(means),
  ];
  let observed = Number.NEGATIVE_INFINITY;
  let best = -1;
  for (let j = 0; j < k; j += 1) {
    if (!usable[j]) continue;
    const s = means[j] * scale(j);
    if (s > observed) { observed = s; best = j; }
  }
  let rcObserved = Number.NEGATIVE_INFINITY;
  for (let j = 0; j < k; j += 1) if (usable[j]) rcObserved = Math.max(rcObserved, means[j]);

  const next = makeRandom(seed);
  const idx = new Int32Array(n);
  const star = new Float64Array(k);
  const boot = new Float64Array(reps * k);   // bootstrap mean of each variant, per draw
  for (let r = 0; r < reps; r += 1) {
    stationaryIndices(n, block, next, idx);
    star.fill(0);
    for (let t = 0; t < n; t += 1) {
      const row = idx[t] * k;
      for (let j = 0; j < k; j += 1) star[j] += d[row + j];
    }
    for (let j = 0; j < k; j += 1) boot[r * k + j] = star[j] / n;
  }

  const spaStat = Math.max(observed, 0);
  const exceed = [0, 0, 0];
  let rcExceed = 0;
  for (let r = 0; r < reps; r += 1) {
    const maxima = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
    let rcMax = Number.NEGATIVE_INFINITY;
    for (let j = 0; j < k; j += 1) {
      if (!usable[j]) continue;
      const m = boot[r * k + j];
      for (let c = 0; c < 3; c += 1) maxima[c] = Math.max(maxima[c], (m - centre[c][j]) * scale(j));
      rcMax = Math.max(rcMax, m - means[j]);
    }
    for (let c = 0; c < 3; c += 1) if (Math.max(maxima[c], 0) > spaStat) exceed[c] += 1;
    if (rcMax > rcObserved) rcExceed += 1;
  }

  // StepM: reject every variant above the (1 - alpha) quantile of the bootstrap maximum over those
  // not yet rejected (consistent recentring), remove them, and repeat until nothing more goes.
  const rejected = new Set();
  const rounds = [];
  for (;;) {
    const live = [];
    for (let j = 0; j < k; j += 1) if (usable[j] && !rejected.has(j)) live.push(j);
    if (!live.length) break;
    const maxima = new Float64Array(reps);
    for (let r = 0; r < reps; r += 1) {
      let m = Number.NEGATIVE_INFINITY;
      for (const j of live) m = Math.max(m, (boot[r * k + j] - centre[1][j]) * scale(j));
      maxima[r] = m;
    }
    maxima.sort();
    const critical = quantile(maxima, 1 - alpha);
    const now = live.filter((j) => means[j] * scale(j) > critical);
    if (!now.length) break;
    for (const j of now) rejected.add(j);
    rounds.push({ critical, rejected: now });
  }

  return {
    n, k, block, reps, seed, alpha, studentize,
    means, variance, usable,
    best, observed,
    spa: { statistic: spaStat, p_lower: exceed[0] / reps, p_consistent: exceed[1] / reps, p_upper: exceed[2] / reps },
    reality_check: { statistic: rcObserved, p_value: rcExceed / reps },
    stepm: { superior: [...rejected].sort((a, b) => a - b), rounds },
  };
}

// numpy's default percentile (linear interpolation between order statistics), on sorted values.
function quantile(sorted, q) {
  const h = (sorted.length - 1) * q;
  const lo = Math.floor(h);
  const hi = Math.min(lo + 1, sorted.length - 1);
  return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo]);
}
