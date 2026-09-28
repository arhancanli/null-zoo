// Luck-equivalent trials: how many skill-less strategies a search would have had to try for the best
// of them to reach the observed Sharpe ratio by luck alone. A reviewer's statistic, stated in the
// unit a research log records (trials), assembled from known results:
//
// - Under the null of no skill and normal returns, the Sharpe ratio's t-statistic, SR * sqrt(T) with
//   the Sharpe per period, is exactly Student t with T - 1 degrees of freedom, so one trial reaches
//   the observed Sharpe with probability p1 = P(t_{T-1} >= SR sqrt(T)).
// - The best of N independent skill-less trials reaches it with probability 1 - (1 - p1)^N (Sidak),
//   so the N at which that probability equals q is N_q = ln(1 - q) / ln(1 - p1).
// - The expected maximum of N standard normals (Bailey, Borwein, Lopez de Prado and Zhu 2014), the
//   deflated Sharpe ratio's benchmark, gives the N whose best is expected to reach it.
//
// Calibrated by Monte Carlo in js/luck-core.test.js; the full size study, with fixed seeds, is
// scripts/research/luck-trials-size-study.mjs and its output config/research/luck-trials-size-study.json.
// The size is correct for normal and for symmetric fat-tailed (Student t4) returns, conservative for
// positively skewed returns, and too small a count, so too kind to the strategy, for negatively
// skewed returns: with 252 observations a nominal 5% test rejected 10.8% of skill-less searches at
// skew -1.3 and 20.8% at skew -3.7.
import { expectedMaxStandardNormal } from "./dsr-core.js";
import { autocorrelationFactor } from "./haircut-core.js";
import { studentTUpper } from "./student-t.js";

export const TRIAL_CAP = 1e15;

function requirePositiveInteger(name, value, min) {
  const n = Number(value);
  if (!Number.isInteger(n) || n < min) throw new RangeError(`${name} must be an integer of at least ${min}`);
  return n;
}

// P(one skill-less trial shows a Sharpe at least this high): the Student t upper tail.
export function singleTrialProbability({ sharpe, observations, periodsPerYear }) {
  const sr = Number(sharpe);
  const periods = Number(periodsPerYear);
  const t = requirePositiveInteger("observations", observations, 3);
  if (!Number.isFinite(sr)) throw new RangeError("The Sharpe ratio must be a finite number");
  if (!(periods > 0 && Number.isFinite(periods))) throw new RangeError("periods per year must be a positive number");
  const tStatistic = (sr / Math.sqrt(periods)) * Math.sqrt(t);
  return { t_statistic: tStatistic, probability: studentTUpper(tStatistic, t - 1) };
}

// The N at which the best of N skill-less trials reaches the Sharpe with probability q, as a real
// number. Below 1 means a single trial already reaches it with probability above q; the count is
// capped at TRIAL_CAP, beyond which the tail probability is below what double precision resolves.
export function trialsAtProbability(probability, q) {
  const p = Number(probability);
  const level = Number(q);
  if (!(level > 0 && level < 1)) throw new RangeError("The probability level must be strictly between 0 and 1");
  if (!(p >= 0 && p <= 1)) throw new RangeError("The single-trial probability must be between 0 and 1");
  if (p === 0) return TRIAL_CAP;
  return Math.min(TRIAL_CAP, Math.log1p(-level) / Math.log1p(-p));
}

// P(the best of N skill-less trials shows a Sharpe at least this high): 1 - (1 - p1)^N.
export function bestOfTrialsProbability(probability, trials) {
  const n = requirePositiveInteger("trials", trials, 1);
  return -Math.expm1(n * Math.log1p(-Number(probability)));
}

// The largest N whose best skill-less trial is expected (the deflated Sharpe ratio's approximation)
// to stay at or below the t-statistic; 1 when even two are expected to exceed it.
export function expectedMaximumTrials(tStatistic) {
  const ceiling = Number(tStatistic);
  const fits = (n) => expectedMaxStandardNormal(n) <= ceiling;
  if (!fits(2)) return 1;
  let lo = 2;
  let hi = 4;
  while (fits(hi)) {
    lo = hi;
    if (hi >= TRIAL_CAP) return TRIAL_CAP;
    hi = Math.min(hi * 2, TRIAL_CAP);
  }
  while (hi - lo > 1) {
    const mid = Math.floor((lo + hi) / 2);
    if (fits(mid)) lo = mid;
    else hi = mid;
  }
  return lo;
}

// With the returns' lag-1 autocorrelation, the Sharpe is first corrected as Lo (2002), the form
// js/haircut-core.js applies; the Null Zoo measured that without it, positively autocorrelated
// returns (0.2) make every best-of-N test reject about four times as often as its level.
export function luckEquivalentTrials({ sharpe, observations, periodsPerYear, trials, autocorrelation = 0 }) {
  const factor = autocorrelationFactor(autocorrelation, periodsPerYear);
  const single = singleTrialProbability({ sharpe: Number(sharpe) * factor, observations, periodsPerYear });
  const result = {
    autocorrelation_factor: factor,
    t_statistic: single.t_statistic,
    single_trial_probability: single.probability,
    trials_for_even_odds: trialsAtProbability(single.probability, 0.5),
    trials_for_five_percent: trialsAtProbability(single.probability, 0.05),
    trials_expected_to_match: expectedMaximumTrials(single.t_statistic),
  };
  if (trials !== undefined) result.best_of_trials_probability = bestOfTrialsProbability(single.probability, trials);
  return result;
}
