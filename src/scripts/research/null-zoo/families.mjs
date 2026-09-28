// The Null Zoo's return families. Each draws one research search: `trials` return series of
// `observations` periods, every one with population mean zero and population variance one, so the
// true Sharpe of every trial is zero. `skill` > 0 gives trial 0 a true annualized Sharpe of `skill`
// by adding a constant mean, which measures power. Every family is deterministic given its seed.

export function rng(seed) {
  let s = seed >>> 0 || 1;
  const u = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return (s + 0.5) / 4294967296; };
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

// A lognormal with log-sd 0.4, centred and scaled to unit variance: skew about 1.32.
const LN_S = 0.4;
const LN_MEAN = Math.exp(LN_S ** 2 / 2);
const LN_SD = Math.sqrt((Math.exp(LN_S ** 2) - 1) * Math.exp(LN_S ** 2));

// Per-series draw functions: (r, t) => next value, with state kept in a closure per series.
const SERIES = {
  iid_normal: (r) => () => r.gauss(),
  student_t4: (r) => () => { const z = r.gauss(); let c = 0; for (let i = 0; i < 4; i++) c += r.gauss() ** 2; return z / Math.sqrt(c / 4) / Math.SQRT2; },
  skew_negative: (r) => () => (LN_MEAN - Math.exp(LN_S * r.gauss())) / LN_SD,
  skew_positive: (r) => () => (Math.exp(LN_S * r.gauss()) - LN_MEAN) / LN_SD,
  // GARCH(1,1), alpha 0.10, beta 0.85, unconditional variance 1; starts at the unconditional variance.
  garch: (r) => { let h = 1; let e = 0; return () => { h = 0.05 + 0.1 * e * e + 0.85 * h; e = Math.sqrt(h) * r.gauss(); return e; }; },
  // AR(1) with coefficient 0.2 and stationary variance 1; starts from the stationary distribution.
  ar1: (r) => { let x = r.gauss(); return () => { x = 0.2 * x + Math.sqrt(1 - 0.04) * r.gauss(); return x; }; },
  // Two volatility regimes, 0.6 and 1.6, switching with probability 0.02 a period; unit variance.
  regimes: (r) => { let high = r.u() < 0.5; const scale = 1 / Math.sqrt((0.36 + 2.56) / 2); return () => { if (r.u() < 0.02) high = !high; return (high ? 1.6 : 0.6) * scale * r.gauss(); }; },
};

export const FAMILIES = Object.freeze([...Object.keys(SERIES), "correlated_trials"]);

export function drawSearch(family, { trials, observations, periodsPerYear, skill = 0 }, r) {
  const drift = skill / Math.sqrt(periodsPerYear);
  const out = [];
  if (family === "correlated_trials") {
    // One common factor: every pair of trials has correlation 0.5, so they are fewer effective trials.
    const factor = Float64Array.from({ length: observations }, () => r.gauss());
    for (let k = 0; k < trials; k++) {
      const xs = new Float64Array(observations);
      for (let t = 0; t < observations; t++) xs[t] = Math.SQRT1_2 * factor[t] + Math.SQRT1_2 * r.gauss() + (k === 0 ? drift : 0);
      out.push(xs);
    }
    return out;
  }
  const make = SERIES[family];
  if (!make) throw new RangeError(`Unknown family ${family}`);
  for (let k = 0; k < trials; k++) {
    const next = make(r);
    const xs = new Float64Array(observations);
    for (let t = 0; t < observations; t++) xs[t] = next() + (k === 0 ? drift : 0);
    out.push(xs);
  }
  return out;
}
