// =============================================================================
// snooping-core.js
// -----------------------------------------------------------------------------
// Data-snooping tests for a search: did the best of K variants beat the benchmark by more than the
// best of K would by luck? One stationary bootstrap (Politis and Romano 1994) feeds three tests:
//
//   White's Reality Check (2000)     the best mean excess return against the bootstrap distribution
//                                    of the best recentred mean. Not studentized, so one volatile
//                                    variant can dominate it.
//   Hansen's SPA (2005)              the best t-statistic. Three recentrings give a lower bound, a
//                                    consistent p-value and an upper bound: poor variants are not
//                                    allowed to inflate the p-value.
//   Romano and Wolf's StepM (2005)   step-down: which variants beat the benchmark, controlling the
//                                    familywise error at alpha.
//
// Studentizing is a bootstrap-t. The observed t-statistic divides each variant's mean by its sample
// standard deviation, and every resample divides its recentred means by that resample's own
// standard deviations, as Romano and Wolf (2005, footnote 22) recommend. Dividing every resample by
// one variance estimated once from the sample ignores that estimate's noise and its downward bias,
// and the test rejected too often (Null Zoo v1: 7.5% of skill-less searches at a nominal 5% with
// i.i.d. normal returns, 12.5% with negative skew). The Politis-Romano variance still sets Hansen's
// consistent-recentring threshold.
//
// p-values count ties, #(T* >= T) / reps. The SPA statistic is floored at zero, so when no variant
// beats the benchmark T = 0 and p = 1; a strict inequality counted only the few positive resamples
// there and rejected. StepM's critical value is floored at zero for the same reason.
//
// d[t][k] is variant k's return minus the benchmark's in period t; higher is better. Every draw
// comes from makeRandom(seed), so a result reproduces exactly from its inputs. The implementations
// it is checked against (Python's arch 8.0 without studentizing, and a separate numpy transcription
// of the bootstrap-t) draw with numpy, so agreement is within Monte Carlo error, not to the digit;
// scripts/research/reality-check/ holds the comparison.
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
 * Returns means, the Politis-Romano variances, the sample standard deviations, the p-values and the
 * StepM rejections. A p-value is the share of `reps` draws at least as large as the observed
 * statistic (the Reality Check's, which is not floored, strictly above it, as in arch).
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
  // Sample variances (divisor n - 1), the observed statistic's studentizer.
  const sampleVar = new Float64Array(k);
  for (let t = 0; t < n; t += 1) for (let j = 0; j < k; j += 1) sampleVar[j] += (d[t * k + j] - means[j]) ** 2;
  for (let j = 0; j < k; j += 1) sampleVar[j] /= n - 1;
  const sd = Float64Array.from(sampleVar, Math.sqrt);
  // A variant whose excess return never varies has no sampling distribution to studentize by.
  const usable = Array.from(sd, (s, j) => s > 0 && variance[j] > 0);
  if (!usable.some(Boolean)) throw new RangeError("every variant's excess return is constant; there is nothing to test");
  const root = Math.sqrt(n);
  // Hansen's recentrings: lower keeps poor variants at zero, consistent drops only those far below
  // zero, upper recentres everything (White's null, the least favourable configuration). A resample
  // mean recentred is (resample mean - sample mean) + offset, with offset = mean - centre.
  const threshold = (j) => -Math.sqrt((variance[j] / n) * 2 * Math.log(Math.log(n)));
  const offset = [
    Float64Array.from(means, (m) => Math.min(m, 0)),
    Float64Array.from(means, (m, j) => (m >= threshold(j) ? 0 : m)),
    new Float64Array(k),
  ];
  const statistic = (j) => (studentize ? (root * means[j]) / sd[j] : means[j]);
  let observed = Number.NEGATIVE_INFINITY;
  let best = -1;
  for (let j = 0; j < k; j += 1) {
    if (!usable[j]) continue;
    const s = statistic(j);
    if (s > observed) { observed = s; best = j; }
  }
  let rcObserved = Number.NEGATIVE_INFINITY;
  for (let j = 0; j < k; j += 1) if (usable[j]) rcObserved = Math.max(rcObserved, means[j]);

  const spaStat = Math.max(observed, 0);
  const exceed = [0, 0, 0];
  let rcExceed = 0;
  const next = makeRandom(seed);
  const idx = new Int32Array(n);
  const sum = new Float64Array(k);    // a resample's sum of deviations from the sample mean
  const sumSq = new Float64Array(k);  // and of their squares
  const stepT = new Float64Array(reps * k);   // each variant's consistent-recentred statistic, per draw
  for (let r = 0; r < reps; r += 1) {
    stationaryIndices(n, block, next, idx);
    sum.fill(0);
    sumSq.fill(0);
    for (let t = 0; t < n; t += 1) {
      const row = idx[t] * k;
      for (let j = 0; j < k; j += 1) {
        const x = d[row + j] - means[j];
        sum[j] += x;
        sumSq[j] += x * x;
      }
    }
    const maxima = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
    let rcMax = Number.NEGATIVE_INFINITY;
    for (let j = 0; j < k; j += 1) {
      if (!usable[j]) continue;
      const shift = sum[j] / n;
      let scale = 1;
      if (studentize) {
        // The resample's own standard deviation. A variant constant in this resample (it differs
        // from the benchmark in few periods and the resample missed them) keeps the sample's.
        const v = (sumSq[j] - sum[j] * shift) / (n - 1);
        scale = root / (v > Number.EPSILON * sampleVar[j] ? Math.sqrt(v) : sd[j]);
      }
      for (let c = 0; c < 3; c += 1) maxima[c] = Math.max(maxima[c], (shift + offset[c][j]) * scale);
      stepT[r * k + j] = (shift + offset[1][j]) * scale;
      rcMax = Math.max(rcMax, shift);
    }
    for (let c = 0; c < 3; c += 1) if (Math.max(maxima[c], 0) >= spaStat) exceed[c] += 1;
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
      for (const j of live) m = Math.max(m, stepT[r * k + j]);
      maxima[r] = m;
    }
    maxima.sort();
    // At least zero, as the SPA statistic is: a variant that does not beat the benchmark in sample
    // is never declared superior. It binds only when every live variant is far below zero.
    const critical = Math.max(quantile(maxima, 1 - alpha), 0);
    const now = live.filter((j) => statistic(j) > critical);
    if (!now.length) break;
    for (const j of now) rejected.add(j);
    rounds.push({ critical, rejected: now });
  }

  return {
    n, k, block, reps, seed, alpha, studentize,
    means, variance, sd, usable,
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
