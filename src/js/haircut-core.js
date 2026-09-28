// The haircut Sharpe ratio: Harvey and Liu, "Backtesting", Journal of Portfolio Management, 2015.
// A Sharpe ratio found among several tests is converted to a t-statistic, its p-value is adjusted
// for the number of tests, and the adjusted p-value is converted back to the Sharpe ratio a single
// test would have needed: the haircut Sharpe ratio. Checked against the authors' own Haircut_SR.m
// (run in GNU Octave) and against R's p.adjust in js/haircut-paper-vectors.test.js.
import { studentTQuantileUpper, studentTUpper } from "./student-t.js";

// Lo (2002) for returns with first-order autocorrelation rho sampled q times a year: the annualized
// Sharpe ratio is scaled by [1 + (2 rho / (1 - rho)) (1 - (1 - rho^q) / (q (1 - rho)))]^(-1/2), the
// form Haircut_SR.m applies to an annualized input.
export function autocorrelationFactor(rho, periodsPerYear) {
  const r = Number(rho);
  const q = Number(periodsPerYear);
  if (!(r > -1 && r < 1)) throw new RangeError("autocorrelation must be strictly between -1 and 1");
  if (r === 0) return 1;
  const inner = 1 + ((2 * r) / (1 - r)) * (1 - (1 - r ** q) / (q * (1 - r)));
  if (!(inner > 0)) throw new RangeError("This autocorrelation gives no valid annualization factor");
  return inner ** -0.5;
}

function requireCount(name, value, min) {
  const n = Number(value);
  if (!Number.isInteger(n) || n < min) throw new RangeError(`${name} must be an integer of at least ${min}`);
  return n;
}

// Holm (step-down) and Benjamini-Hochberg-Yekutieli adjusted p-values for one member of a family,
// computed exactly as R's p.adjust computes them (stable order, ties by position).
export function familyAdjusted(pValues, index) {
  const n = pValues.length;
  const ascending = pValues.map((p, i) => [p, i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const holm = new Array(n);
  let runningMax = 0;
  ascending.forEach(([p, i], k) => {
    runningMax = Math.max(runningMax, (n - k) * p);
    holm[i] = Math.min(1, runningMax);
  });
  const harmonic = pValues.reduce((sum, _, k) => sum + 1 / (k + 1), 0);
  const descending = pValues.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0] || b[1] - a[1]);
  const bhy = new Array(n);
  let runningMin = Infinity;
  descending.forEach(([p, i], k) => {
    const rank = n - k;
    runningMin = Math.min(runningMin, ((harmonic * n) / rank) * p);
    bhy[i] = Math.min(1, runningMin);
  });
  return { holm: holm[index], bhy: bhy[index] };
}

export function haircutSharpe({ sharpeAnnualized, periodsPerYear, observations, tests, autocorrelation = 0, otherSharpesAnnualized }) {
  const sharpe = Number(sharpeAnnualized);
  const q = Number(periodsPerYear);
  if (!(sharpe > 0 && Number.isFinite(sharpe))) throw new RangeError("The haircut applies to a positive, finite Sharpe ratio");
  if (!(q > 0 && q <= 10000)) throw new RangeError("periods_per_year must be greater than 0 and at most 10000");
  const T = requireCount("observations", observations, 3);
  const others = otherSharpesAnnualized === undefined ? null : otherSharpesAnnualized.map(Number);
  if (others && !others.every(Number.isFinite)) throw new RangeError("Every other Sharpe ratio must be a finite number");
  const m = others ? others.length + 1 : requireCount("tests", tests, 1);
  if (others && tests !== undefined && Number(tests) !== m) throw new RangeError("tests must equal the number of other Sharpe ratios plus one, or be left out");

  const factor = autocorrelationFactor(autocorrelation, q);
  const df = T - 1;
  const tOf = (annual) => ((annual * factor) / Math.sqrt(q)) * Math.sqrt(T);
  // Two-sided, as in Haircut_SR.m, but from the upper tail directly so that a large t keeps its p-value.
  const pOf = (annual) => {
    const t = tOf(annual);
    return 2 * (t >= 0 ? studentTUpper(t, df) : studentTUpper(-t, df));
  };
  const srCorrected = sharpe * factor;
  const tStat = tOf(sharpe);
  const pSingle = pOf(sharpe);
  const invert = (adjustedP) => {
    const p = Math.min(1, adjustedP);
    const tAdjusted = p >= 1 ? 0 : studentTQuantileUpper(p / 2, df);
    const haircutSharpe = (tAdjusted / Math.sqrt(T)) * Math.sqrt(q);
    return { adjusted_p: p, haircut_sharpe_annualized: haircutSharpe, haircut: (srCorrected - haircutSharpe) / srCorrected };
  };
  const result = {
    sharpe_annualized_corrected: srCorrected,
    autocorrelation_factor: factor,
    t_statistic: tStat,
    degrees_of_freedom: df,
    p_value_single: pSingle,
    tests: m,
    bonferroni: invert(m * pSingle),
    // Harvey and Liu's Eq. 4 for independent tests: 1 - (1 - p)^M.
    independent: invert(-Math.expm1(m * Math.log1p(-pSingle))),
  };
  if (others) {
    const family = [pSingle, ...others.map(pOf)];
    const adjusted = familyAdjusted(family, 0);
    result.holm = invert(adjusted.holm);
    result.bhy = invert(adjusted.bhy);
  }
  return result;
}
