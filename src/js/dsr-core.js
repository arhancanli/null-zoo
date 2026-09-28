// Pure deflated-Sharpe arithmetic. No DOM: shared by /tools/deflated-sharpe and the
// validation API so the two can never disagree. Bound to
// public/glassbox/deflated_sharpe_calculator_contract.json by checkGoldenVectors.
const EULER_MASCHERONI = 0.5772156649;

// Complementary error function to full double precision (relative error at most 1.5e-14 for
// x >= -20 and 6e-14 to -38, measured against the C library erfc). For |x| < 1.5 it uses the series
// erf(x) = 2/sqrt(pi) exp(-x^2) sum 2^n x^(2n+1) / (1*3*...*(2n+1)), which has no cancellation;
// beyond that, the continued fraction erfc(x) = exp(-x^2)/sqrt(pi) / (x + 1/2 / (x + 1 / (x + ...)))
// evaluated by the modified Lentz method. It replaced the Abramowitz and Stegun 7.1.26
// approximation (absolute error up to 1.5e-7, no relative accuracy in the tails) on 2026-09-25.
function erfc(value) {
  if (Number.isNaN(value)) return Number.NaN;
  if (value < 0) return 2 - erfc(-value);
  const x = value;
  if (x < 1.5) {
    let term = x;
    let sum = x;
    for (let n = 1; n < 200; n++) {
      term *= (2 * x * x) / (2 * n + 1);
      sum += term;
      if (term < sum * 1e-17) break;
    }
    return 1 - (2 / Math.sqrt(Math.PI)) * Math.exp(-x * x) * sum;
  }
  if (x > 27.3) return 0;
  // Lentz: f = b0 + a1/(b1 + a2/(b2 + ...)) with b_k = x, a_k = k/2.
  const tiny = 1e-300;
  let f = x;
  let c = x;
  let d = 0;
  for (let k = 1; k < 500; k++) {
    const a = k / 2;
    d = x + a * d;
    d = d === 0 ? tiny : d;
    c = x + a / c;
    c = c === 0 ? tiny : c;
    d = 1 / d;
    const delta = c * d;
    f *= delta;
    if (Math.abs(delta - 1) < 1e-16) break;
  }
  return Math.exp(-x * x) / (Math.sqrt(Math.PI) * f);
}

export function normalCdf(value) {
  if (value === Infinity) return 1;
  if (value === -Infinity) return 0;
  return 0.5 * erfc(-value / Math.SQRT2);
}

function acklamPpf(probability) {
  if (!(probability > 0 && probability < 1)) {
    throw new RangeError("Normal quantile probability must be between zero and one");
  }

  const a = [
    -3.969683028665376e1,
    2.209460984245205e2,
    -2.759285104469687e2,
    1.38357751867269e2,
    -3.066479806614716e1,
    2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1,
    1.615858368580409e2,
    -1.556989798598866e2,
    6.680131188771972e1,
    -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3,
    -3.223964580411365e-1,
    -2.400758277161838,
    -2.549732539343734,
    4.374664141464968,
    2.938163982698783,
  ];
  const d = [
    7.784695709041462e-3,
    3.224671290700398e-1,
    2.445134137142996,
    3.754408661907416,
  ];
  const low = 0.02425;
  const high = 1 - low;

  if (probability < low) {
    const q = Math.sqrt(-2 * Math.log(probability));
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
  if (probability <= high) {
    const q = probability - 0.5;
    const r = q * q;
    return (
      (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    );
  }
  const q = Math.sqrt(-2 * Math.log(1 - probability));
  return -(
    (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
  );
}

// Inverse normal CDF: Acklam's rational approximation (relative error near 1.15e-9) refined by two
// Halley steps against the full-precision normalCdf, which brings it to the precision of the CDF.
export function normalPpf(probability) {
  let x = acklamPpf(probability);
  for (let step = 0; step < 2; step++) {
    const error = normalCdf(x) - probability;
    const u = error * Math.sqrt(2 * Math.PI) * Math.exp((x * x) / 2);
    if (!Number.isFinite(u)) break;
    x -= u / (1 + (x * u) / 2);
  }
  return x;
}

function requireFinite(name, value) {
  if (!Number.isFinite(value)) throw new RangeError(`${name} must be a finite number`);
}

// Expected maximum of N independent standard Normal draws (Bailey, Borwein, López de Prado and Zhu
// 2014, Proposition 2.1; Bailey and López de Prado 2014, the deflated Sharpe ratio): the Sharpe
// ratio, in units of its standard deviation, that the best of N skill-less trials is expected to show.
export function expectedMaxStandardNormal(trials) {
  const n = Number(trials);
  if (!Number.isInteger(n) || n < 2) throw new RangeError("Effective independent trials must be an integer of at least 2");
  return (1 - EULER_MASCHERONI) * normalPpf(1 - 1 / n) + EULER_MASCHERONI * normalPpf(1 - 1 / (n * Math.E));
}

// Minimum Backtest Length (Bailey, Borwein, López de Prado and Zhu 2014, Theorem 3.1): the years of
// backtest needed so that the best of N skill-less trials is not expected to show an annualized
// Sharpe of targetSharpe in sample: ((1-g) Z^-1[1-1/N] + g Z^-1[1-1/(Ne)])^2 / targetSharpe^2,
// bounded above by 2 ln N / targetSharpe^2. Necessary, not sufficient, to avoid overfitting.
export function minimumBacktestLength({ trials, targetSharpe }) {
  const target = Number(targetSharpe);
  if (!(target > 0 && Number.isFinite(target))) throw new RangeError("The target Sharpe must be a positive number");
  const expectedMax = expectedMaxStandardNormal(trials);
  return {
    years: (expectedMax / target) ** 2,
    upper_bound_years: (2 * Math.log(Number(trials))) / target ** 2,
    expected_max_sharpe_one_year: expectedMax,
  };
}

// The largest number of independent trials whose best is still expected to stay below targetSharpe
// in sample over `years` of backtest (Eq. 3.1 solved for N). The expected maximum grows with N, so
// a doubling search then a bisection finds it exactly. Returns 1 when even two trials are too many.
export function maximumIndependentTrials({ years, targetSharpe }) {
  const y = Number(years);
  const target = Number(targetSharpe);
  if (!(y > 0 && Number.isFinite(y))) throw new RangeError("Backtest years must be a positive number");
  if (!(target > 0 && Number.isFinite(target))) throw new RangeError("The target Sharpe must be a positive number");
  const ceiling = target * Math.sqrt(y);
  const fits = (n) => expectedMaxStandardNormal(n) <= ceiling;
  if (!fits(2)) return 1;
  let lo = 2;
  let hi = 4;
  const LIMIT = 1e15;
  while (fits(hi)) {
    lo = hi;
    if (hi >= LIMIT) return LIMIT;
    hi = Math.min(hi * 2, LIMIT);
  }
  while (hi - lo > 1) {
    const mid = Math.floor((lo + hi) / 2);
    if (fits(mid)) lo = mid;
    else hi = mid;
  }
  return lo;
}

export function calculateDsr(input) {
  const values = Object.fromEntries(
    Object.entries(input).map(([key, value]) => [key, Number(value)]),
  );
  for (const [key, value] of Object.entries(values)) requireFinite(key, value);
  if (!Number.isInteger(values.observations) || values.observations < 2) {
    throw new RangeError("Return observations must be an integer of at least 2");
  }
  if (!(values.periods_per_year > 0)) {
    throw new RangeError("Periods per year must be greater than zero");
  }
  if (
    !Number.isInteger(values.effective_independent_trials) ||
    values.effective_independent_trials < 2
  ) {
    throw new RangeError("Effective independent trials must be an integer of at least 2");
  }
  if (values.cross_trial_sharpe_sd_annualized < 0) {
    throw new RangeError("Cross-trial Sharpe dispersion cannot be negative");
  }

  const annualizationScale = Math.sqrt(values.periods_per_year);
  const observedSharpePerPeriod = values.observed_sharpe_annualized / annualizationScale;
  const trialSdPerPeriod = values.cross_trial_sharpe_sd_annualized / annualizationScale;
  const trialVariancePerPeriod = trialSdPerPeriod ** 2;
  const quantile = expectedMaxStandardNormal(values.effective_independent_trials);
  const expectedMaxSharpePerPeriod = trialSdPerPeriod * quantile;
  const nonNormalityVarianceTerm =
    1 -
    values.skew * observedSharpePerPeriod +
    ((values.non_excess_kurtosis - 1) / 4) * observedSharpePerPeriod ** 2;
  if (!(nonNormalityVarianceTerm > 0)) {
    throw new RangeError(
      "These skew, kurtosis and Sharpe inputs produce a non-positive estimator variance term",
    );
  }
  const denominator = Math.sqrt(nonNormalityVarianceTerm);
  const sampleScale = Math.sqrt(values.observations - 1);
  const psrZ = observedSharpePerPeriod * sampleScale / denominator;
  const dsrZ =
    (observedSharpePerPeriod - expectedMaxSharpePerPeriod) * sampleScale / denominator;

  return {
    observed_sharpe_per_period: observedSharpePerPeriod,
    cross_trial_sharpe_variance_per_period: trialVariancePerPeriod,
    expected_max_sharpe_per_period: expectedMaxSharpePerPeriod,
    expected_max_sharpe_annualized: expectedMaxSharpePerPeriod * annualizationScale,
    psr_against_zero: normalCdf(psrZ),
    deflated_sharpe_ratio: normalCdf(dsrZ),
    non_normality_variance_term: nonNormalityVarianceTerm,
    search_haircut_annualized:
      values.observed_sharpe_annualized - expectedMaxSharpePerPeriod * annualizationScale,
    psr_z_score: psrZ,
    dsr_z_score: dsrZ,
  };
}

export function checkGoldenVectors(vectors, tolerance = 8e-7) {
  const failures = [];
  for (const vector of vectors) {
    const observed = calculateDsr(vector.inputs);
    for (const [key, expected] of Object.entries(vector.outputs)) {
      const error = Math.abs(observed[key] - expected);
      if (error > tolerance) failures.push({ vector: vector.id, key, expected, observed: observed[key], error });
    }
  }
  return failures;
}


// Probabilistic Sharpe ratio against a benchmark Sharpe, and the minimum track record length for
// it to clear that benchmark at a confidence level: Bailey and López de Prado, "The Sharpe Ratio
// Efficient Frontier", Journal of Risk 15(2), 2012, Eqs. (11) and (13). Inputs are annualized; the
// estimator works per period, like calculateDsr above.
function perPeriodInputs(values) {
  for (const [key, value] of Object.entries(values)) requireFinite(key, value);
  if (!(values.periods_per_year > 0)) throw new RangeError("Periods per year must be greater than zero");
  const scale = Math.sqrt(values.periods_per_year);
  const sr = values.observed_sharpe_annualized / scale;
  const benchmark = values.benchmark_sharpe_annualized / scale;
  const term = 1 - values.skew * sr + ((values.non_excess_kurtosis - 1) / 4) * sr ** 2;
  if (!(term > 0)) {
    throw new RangeError("These skew, kurtosis and Sharpe inputs produce a non-positive estimator variance term");
  }
  return { sr, benchmark, term };
}

export function probabilisticSharpe(input) {
  const values = {
    observed_sharpe_annualized: Number(input.observed_sharpe_annualized),
    benchmark_sharpe_annualized: Number(input.benchmark_sharpe_annualized ?? 0),
    observations: Number(input.observations),
    periods_per_year: Number(input.periods_per_year),
    skew: Number(input.skew),
    non_excess_kurtosis: Number(input.non_excess_kurtosis),
  };
  if (!Number.isInteger(values.observations) || values.observations < 2) {
    throw new RangeError("Return observations must be an integer of at least 2");
  }
  const { sr, benchmark, term } = perPeriodInputs(values);
  const z = ((sr - benchmark) * Math.sqrt(values.observations - 1)) / Math.sqrt(term);
  return { probabilistic_sharpe_ratio: normalCdf(z), z_score: z };
}

export function minimumTrackRecordLength(input) {
  const confidence = Number(input.confidence ?? 0.95);
  if (!(confidence > 0 && confidence < 1)) throw new RangeError("Confidence must be strictly between 0 and 1");
  const values = {
    observed_sharpe_annualized: Number(input.observed_sharpe_annualized),
    benchmark_sharpe_annualized: Number(input.benchmark_sharpe_annualized ?? 0),
    periods_per_year: Number(input.periods_per_year),
    skew: Number(input.skew),
    non_excess_kurtosis: Number(input.non_excess_kurtosis),
  };
  const { sr, benchmark, term } = perPeriodInputs(values);
  if (!(sr > benchmark)) {
    throw new RangeError("The observed Sharpe must exceed the benchmark; no track record length is enough otherwise");
  }
  const observations = 1 + term * (normalPpf(confidence) / (sr - benchmark)) ** 2;
  if (!Number.isFinite(observations)) {
    throw new RangeError("The observed Sharpe is too close to the benchmark; no finite track record length reaches this confidence");
  }
  return { observations, years: observations / values.periods_per_year, confidence };
}
