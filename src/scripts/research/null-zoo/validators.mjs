// The validators the Null Zoo scores. Each takes one search (every trial's returns) and returns a
// p-value for "the best trial has skill": small means evidence. Each is called the way its own
// documentation says to call it; `sides` records whether its p-value is one- or two-sided.
import { calculateDsr, normalCdf } from "../../../js/dsr-core.js";
import { autocorrelationFactor, haircutSharpe } from "../../../js/haircut-core.js";
import { bestOfTrialsProbability, singleTrialProbability } from "../../../js/luck-core.js";

export function moments(xs) {
  const t = xs.length;
  let m = 0;
  for (let i = 0; i < t; i++) m += xs[i];
  m /= t;
  let m2 = 0, m3 = 0, m4 = 0;
  for (let i = 0; i < t; i++) { const d = xs[i] - m; const d2 = d * d; m2 += d2; m3 += d2 * d; m4 += d2 * d2; }
  m2 /= t; m3 /= t; m4 /= t;
  return { mean: m, sharpe: m / Math.sqrt((m2 * t) / (t - 1)), skew: m3 / m2 ** 1.5, kurtosis: m4 / m2 ** 2 };
}

// Summaries every validator shares, computed once per search.
export function summarize(search, periodsPerYear) {
  const stats = search.map(moments);
  let best = 0;
  for (let k = 1; k < stats.length; k++) if (stats[k].sharpe > stats[best].sharpe) best = k;
  const annual = stats.map((s) => s.sharpe * Math.sqrt(periodsPerYear));
  const meanAnnual = annual.reduce((a, b) => a + b, 0) / annual.length;
  const sdAnnual = Math.sqrt(annual.reduce((a, b) => a + (b - meanAnnual) ** 2, 0) / (annual.length - 1));
  return { stats, best, annual, sdAnnual, observations: search[0].length, periodsPerYear, trials: search.length };
}

// Lag-1 autocorrelation of one series, clamped inside (-1, 1) for Lo's correction.
export function lagOne(xs, mean) {
  let c = 0, v = 0;
  for (let i = 0; i < xs.length; i++) { const d = xs[i] - mean; v += d * d; if (i > 0) c += d * (xs[i - 1] - mean); }
  return Math.max(-0.95, Math.min(0.95, c / v));
}

// Lo (2002): the annualized Sharpe corrected for the best trial's own lag-1 autocorrelation.
const loSharpe = (search, s) => s.annual[s.best] * autocorrelationFactor(lagOne(search[s.best], s.stats[s.best].mean), s.periodsPerYear);

export const VALIDATORS = Object.freeze({
  luck_trials: {
    sides: 1,
    p: (_, s) => bestOfTrialsProbability(singleTrialProbability({ sharpe: s.annual[s.best], observations: s.observations, periodsPerYear: s.periodsPerYear }).probability, s.trials),
  },
  luck_trials_lo: {
    sides: 1,
    p: (search, s) => bestOfTrialsProbability(singleTrialProbability({ sharpe: loSharpe(search, s), observations: s.observations, periodsPerYear: s.periodsPerYear }).probability, s.trials),
  },
  luck_trials_nonnormal_se: {
    sides: 1,
    p: (_, s) => {
      const b = s.stats[s.best];
      const term = 1 - b.skew * b.sharpe + ((b.kurtosis - 1) / 4) * b.sharpe ** 2;
      return bestOfTrialsProbability(normalCdf(-(b.sharpe * Math.sqrt(s.observations - 1)) / Math.sqrt(term)), s.trials);
    },
  },
  haircut_bonferroni: {
    sides: 2,
    p: (_, s) => (s.annual[s.best] > 0 ? haircutSharpe({ sharpeAnnualized: s.annual[s.best], periodsPerYear: s.periodsPerYear, observations: s.observations, tests: s.trials }).bonferroni.adjusted_p : 1),
  },
  haircut_bonferroni_lo: {
    sides: 2,
    p: (search, s) => (s.annual[s.best] > 0
      ? haircutSharpe({ sharpeAnnualized: s.annual[s.best], periodsPerYear: s.periodsPerYear, observations: s.observations, tests: s.trials, autocorrelation: lagOne(search[s.best], s.stats[s.best].mean) }).bonferroni.adjusted_p
      : 1),
  },
  haircut_holm: {
    sides: 2,
    p: (_, s) => (s.annual[s.best] > 0
      ? haircutSharpe({ sharpeAnnualized: s.annual[s.best], periodsPerYear: s.periodsPerYear, observations: s.observations, otherSharpesAnnualized: s.annual.filter((_, k) => k !== s.best) }).holm.adjusted_p
      : 1),
  },
  // The deflated Sharpe ratio is a probability that the true Sharpe exceeds the expected best of N
  // skill-less trials; 1 - DSR is read as its p-value, as practitioners threshold DSR at 0.95.
  deflated_sharpe: {
    sides: 1,
    p: (_, s) => {
      const b = s.stats[s.best];
      return 1 - calculateDsr({
        observed_sharpe_annualized: s.annual[s.best], observations: s.observations, periods_per_year: s.periodsPerYear,
        skew: b.skew, non_excess_kurtosis: b.kurtosis, effective_independent_trials: s.trials, cross_trial_sharpe_sd_annualized: s.sdAnnual,
      }).deflated_sharpe_ratio;
    },
  },
  // Resamples the best trial's demeaned returns (a true mean of zero) and reads the tail at its Sharpe.
  bootstrap_best: {
    sides: 1,
    p: (search, s, r, draws = 400) => {
      const xs = search[s.best];
      const t = xs.length;
      const mean = s.stats[s.best].mean;
      const buf = new Float64Array(t);
      let hits = 0;
      for (let b = 0; b < draws; b++) {
        for (let i = 0; i < t; i++) buf[i] = xs[(r.u() * t) | 0] - mean;
        if (moments(buf).sharpe >= s.stats[s.best].sharpe) hits++;
      }
      return bestOfTrialsProbability((hits + 1) / (draws + 1), s.trials);
    },
  },
});
