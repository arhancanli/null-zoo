// =============================================================================
// selection-risk-core.js
// -----------------------------------------------------------------------------
// The engine behind /tools/selection-risk: a sandbox where you search a strategy
// space over a series that provably has NO edge, and watch what your best result
// is worth once the search is counted.
//
// WHY SYNTHETIC, AND WHY THAT IS NOT A COMPROMISE.
// Two reasons, and the second is the better one.
//
//   1. Rights. config/data_source_rights_policy.json records every market-data
//      source this project uses as PROHIBITED or WITHHELD for raw redistribution,
//      with a project default of "do not bundle raw third-party rows". Shipping
//      real bars into a browser is not available, and no amount of wanting it
//      changes that.
//
//   2. Proof. On real data you can always argue the edge you found was real. Here
//      the generator is a driftless random walk, so the true Sharpe is ZERO by
//      construction. Every good result the visitor finds is therefore luck, with
//      no room for debate, which is exactly the point being demonstrated. A real
//      series would make the lesson weaker, not stronger.
//
// NO-LOOKAHEAD, the same rule the engine enforces. A signal is computed from
// closes up to and including bar t. The position it implies is held from the OPEN
// of bar t+1 to the OPEN of bar t+2. A decision never touches a price from its own
// bar or earlier than its own execution. `assertNoLookahead` below is the
// mechanical statement of that, and the test suite perturbs a bar to prove the
// property is real rather than asserted.
// =============================================================================

/** Deterministic PRNG (mulberry32). Same seed, same series, on every machine. */
export function makeRandom(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller, drawing from a supplied uniform generator so it stays deterministic. */
function gaussian(next) {
  let u = 0;
  let v = 0;
  while (u === 0) u = next();
  while (v === 0) v = next();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * A driftless geometric random walk, as OHLC-style bars with an open and a close.
 *
 * `drift` exists only so a caller can demonstrate the contrast; the sandbox pins it
 * to zero, which is what makes the true Sharpe zero.
 */
export function generateSeries({ seed, bars = 750, volatility = 0.011, drift = 0, gapVolatility = 0 }) {
  if (!Number.isInteger(bars) || bars < 32) throw new Error("bars must be an integer >= 32");
  if (!(volatility > 0)) throw new Error("volatility must be positive");
  const next = makeRandom(seed);
  const out = [];
  let level = 100;
  for (let i = 0; i < bars; i += 1) {
    // The overnight gap. With gapVolatility at zero, open[i+1] equals close[i] and
    // the series is continuous, which is what the Selection Risk Lab wants: it
    // studies search, not execution, and a gap would only add noise there.
    //
    // The Execution Reality Lab needs a non-zero gap, because without one "fill at
    // the next open" and "fill at the decision close" are the SAME price and the
    // most important comparison on that page silently collapses to a tie.
    if (gapVolatility > 0) level *= Math.exp(gapVolatility * gaussian(next));
    const open = level;
    level *= Math.exp(drift + volatility * gaussian(next));
    out.push({ i, open, close: level });
  }
  return out;
}

/** Simple moving average of `closes` ending at index `end`, inclusive. */
function sma(closes, end, window) {
  if (end + 1 < window) return Number.NaN;
  let total = 0;
  for (let k = end - window + 1; k <= end; k += 1) total += closes[k];
  return total / window;
}

/**
 * A fast/slow moving-average crossover, run under the engine's fill rule.
 *
 * Returns per-period returns, the equity path, and the summary statistics. The
 * returned `decisions` array records, for each realised return, which bar the
 * decision was taken on and which bars priced it, so a test can assert the
 * causality rather than trust the comment above it.
 */
export function runCrossover({ series, fast, slow, costBps = 0 }) {
  if (!Number.isInteger(fast) || !Number.isInteger(slow)) throw new Error("windows must be integers");
  if (fast < 1 || slow < 2) throw new Error("windows must be positive");
  if (fast >= slow) throw new Error("fast window must be shorter than slow");
  const closes = series.map((b) => b.close);
  const cost = costBps / 10000;

  const returns = [];
  const decisions = [];
  let previousPosition = 0;

  // t is the DECISION bar. Execution spans open[t+1] -> open[t+2], so the loop
  // stops early enough that both exist.
  for (let t = slow - 1; t + 2 < series.length; t += 1) {
    const position = sma(closes, t, fast) > sma(closes, t, slow) ? 1 : 0;
    const entry = series[t + 1].open;
    const exit = series[t + 2].open;
    const gross = position * (exit / entry - 1);
    const turnover = Math.abs(position - previousPosition);
    const net = gross - turnover * cost;
    returns.push(net);
    decisions.push({ decisionBar: t, entryBar: t + 1, exitBar: t + 2, position });
    previousPosition = position;
  }

  let equity = 1;
  const curve = [equity];
  for (const r of returns) {
    equity *= 1 + r;
    curve.push(equity);
  }

  return {
    returns,
    curve,
    decisions,
    trades: decisions.reduce(
      (n, d, k) => n + (k === 0 ? d.position : Math.abs(d.position - decisions[k - 1].position)),
      0,
    ),
    ...summarize(returns, curve),
  };
}

/** Per-period Sharpe, its annualised form, and the worst peak-to-trough drop. */
export function summarize(returns, curve) {
  const n = returns.length;
  if (n < 2) return { sharpePerPeriod: 0, sharpeAnnualised: 0, maxDrawdown: 0, totalReturn: 0 };
  const mean = returns.reduce((a, b) => a + b, 0) / n;
  const variance = returns.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1);
  const sd = Math.sqrt(variance);
  const sharpePerPeriod = sd > 0 ? mean / sd : 0;
  let peak = curve[0];
  let maxDrawdown = 0;
  for (const value of curve) {
    if (value > peak) peak = value;
    const drop = peak > 0 ? 1 - value / peak : 0;
    if (drop > maxDrawdown) maxDrawdown = drop;
  }
  // Sample skewness and NON-EXCESS kurtosis, which is what the deflation wants:
  // a Gaussian scores 3 here, not 0. Handing it excess kurtosis silently shifts
  // the estimator variance and the answer stays plausible, so the convention is
  // named rather than assumed.
  const m3 = returns.reduce((a, b) => a + (b - mean) ** 3, 0) / n;
  const m4 = returns.reduce((a, b) => a + (b - mean) ** 4, 0) / n;
  const populationSd = Math.sqrt(returns.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
  const skew = populationSd > 0 ? m3 / populationSd ** 3 : 0;
  const nonExcessKurtosis = populationSd > 0 ? m4 / populationSd ** 4 : 3;

  return {
    sharpePerPeriod,
    // 252 periods, the convention for a daily-frequency series.
    sharpeAnnualised: sharpePerPeriod * Math.sqrt(252),
    maxDrawdown,
    totalReturn: curve[curve.length - 1] - 1,
    skew,
    nonExcessKurtosis,
    observations: n,
  };
}

/**
 * The mechanical no-lookahead statement: for every realised return, the decision
 * bar strictly precedes the bar whose open is paid, which strictly precedes the
 * bar whose open is received.
 */
export function assertNoLookahead(result) {
  for (const d of result.decisions) {
    if (!(d.decisionBar < d.entryBar && d.entryBar < d.exitBar)) {
      return { ok: false, offending: d };
    }
  }
  return { ok: true, offending: null };
}

/** Sample variance of a list of Sharpe ratios, the V[SR] the deflation needs. */
export function sharpeVariance(sharpes) {
  const n = sharpes.length;
  if (n < 2) return 0;
  const mean = sharpes.reduce((a, b) => a + b, 0) / n;
  return sharpes.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1);
}
