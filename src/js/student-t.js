// Student's t distribution to full double precision, for the haircut Sharpe ratio (Harvey and Liu,
// "Backtesting", 2015), which tests a Sharpe ratio's t-statistic against t with N - 1 degrees of
// freedom. Checked against R's pt and qt (js/fixtures/student-t-reference.json).
//
// The upper tail is computed directly, never as 1 - cdf: the authors' Haircut_SR.m computes
// 2 * (1 - tcdf(t, N - 1)), which rounds to 0 once t passes about 8 and makes the haircut Sharpe
// ratio infinite. P(T > t) = I_x(df / 2, 1 / 2) / 2 with x = df / (df + t^2) keeps full relative
// precision far into the tail.

const HALF_LOG_TWO_PI = 0.5 * Math.log(2 * Math.PI);

// Stirling's correction ln Gamma(x) - [(x - 1/2) ln x - x + ln(2 pi) / 2], for x >= 10: the series
// to the 1/x^13 term, whose error there is below 1e-16.
function stirlingCorrection(x) {
  const inv = 1 / x;
  const inv2 = inv * inv;
  return inv * (1 / 12 + inv2 * (-1 / 360 + inv2 * (1 / 1260 + inv2 * (-1 / 1680 + inv2 * (1 / 1188 + inv2 * (-691 / 360360 + inv2 * (1 / 156)))))));
}

// ln Gamma(x) for x > 0: shifted to x >= 10, where the series applies; ln Gamma(x) = ln Gamma(x + 1) -
// ln x undoes the shift.
export function logGamma(value) {
  let x = Number(value);
  if (!(x > 0) || !Number.isFinite(x)) throw new RangeError("logGamma needs a positive finite number");
  let shift = 0;
  while (x < 10) {
    shift -= Math.log(x);
    x += 1;
  }
  return (x - 0.5) * Math.log(x) - x + HALF_LOG_TWO_PI + stirlingCorrection(x) + shift;
}

// ln Gamma(a + b) - ln Gamma(a) without subtracting two large numbers when a is large (degrees of
// freedom in the thousands): (a - 1/2) ln(1 + b/a) + b ln(a + b) - b + the two corrections.
function logGammaRatio(a, b) {
  if (a < 10) return logGamma(a + b) - logGamma(a);
  return (a - 0.5) * Math.log1p(b / a) + b * Math.log(a + b) - b + stirlingCorrection(a + b) - stirlingCorrection(a);
}

// Continued fraction for the incomplete beta function (modified Lentz).
function betaFraction(a, b, x) {
  const TINY = 1e-300;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < TINY) d = TINY;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 100000; m += 1) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < TINY) d = TINY;
    c = 1 + aa / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < TINY) d = TINY;
    c = 1 + aa / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < 1e-16) return h;
  }
  throw new Error("The incomplete beta continued fraction did not converge");
}

// Regularized incomplete beta I_x(a, b) for b = 1/2 (all Student t needs), with y = 1 - x and both
// logarithms passed separately so that nothing is lost to cancellation when x or y is tiny.
function incompleteBeta(x, y, logX, logY, a, b) {
  if (x <= 0) return 0;
  if (y <= 0) return 1;
  const front = Math.exp(logGammaRatio(a, b) - logGamma(b) + a * logX + b * logY);
  if (x < (a + 1) / (a + b + 2)) return (front * betaFraction(a, b, x)) / a;
  return 1 - (front * betaFraction(b, a, y)) / b;
}

function requireDf(df) {
  const v = Number(df);
  if (!(v > 0) || !Number.isFinite(v)) throw new RangeError("Degrees of freedom must be a positive finite number");
  return v;
}

// P(T > t) for T ~ t(df).
export function studentTUpper(t, df) {
  const v = requireDf(df);
  const value = Number(t);
  if (Number.isNaN(value)) return Number.NaN;
  if (value === 0) return 0.5;
  if (value === Infinity) return 0;
  if (value === -Infinity) return 1;
  const t2 = value * value;
  const logX = -Math.log1p(t2 / v);
  const logY = Math.log(t2) - Math.log(v + t2);
  const half = 0.5 * incompleteBeta(Math.exp(logX), t2 / (v + t2), logX, logY, v / 2, 0.5);
  return value > 0 ? half : 1 - half;
}

export function studentTPdf(t, df) {
  const v = requireDf(df);
  const value = Number(t);
  return Math.exp(logGammaRatio(v / 2, 0.5) - 0.5 * Math.log(v * Math.PI) - ((v + 1) / 2) * Math.log1p((value * value) / v));
}

// The t with P(T > t) = upper: bisection on a bracket (the tail can be very heavy for small df),
// then Newton steps on log P(T > t), which is smooth and keeps relative precision in the tail.
export function studentTQuantileUpper(upper, df) {
  const v = requireDf(df);
  const p = Number(upper);
  if (!(p > 0 && p < 1)) throw new RangeError("The upper-tail probability must be strictly between 0 and 1");
  if (p === 0.5) return 0;
  if (p > 0.5) return -studentTQuantileUpper(1 - p, v);
  let lo = 0;
  let hi = 1;
  while (studentTUpper(hi, v) > p) {
    lo = hi;
    hi *= 2;
    if (!Number.isFinite(hi)) throw new RangeError("The upper-tail probability is too small for these degrees of freedom");
  }
  for (let i = 0; i < 200 && hi - lo > 1e-12 * hi; i += 1) {
    const mid = 0.5 * (lo + hi);
    if (studentTUpper(mid, v) > p) lo = mid;
    else hi = mid;
  }
  let t = 0.5 * (lo + hi);
  const target = Math.log(p);
  for (let i = 0; i < 4; i += 1) {
    const tail = studentTUpper(t, v);
    const step = (Math.log(tail) - target) / (-studentTPdf(t, v) / tail);
    if (!Number.isFinite(step)) break;
    t -= step;
  }
  return t;
}
