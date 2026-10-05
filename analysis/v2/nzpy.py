"""Null Zoo v2: an independent Python implementation of the return families and the data-snooping tests.

Written from the published definitions (the family descriptions in the paper, White 2000, Hansen 2005,
Politis and Romano 1994), not translated from the JavaScript harness in src/, so that the v1 sizes it
reproduces are checked in a second language. Every draw comes from numpy's PCG64 through the
Generator passed in, so a cell reproduces exactly from its seed.

Conventions. A search is an (n, k) array: n periods of k strategies' returns in excess of the
benchmark, higher is better. Bootstrap p-values are returned as integer exceedance counts h out of
`reps` resamples (p = h / reps), so the stored histograms are lossless for every threshold.
"""
import numpy as np
from scipy import stats

Q = 252                     # periods per year
LN_S = 0.4                  # log-sd of the lognormal behind the skewed families (skewness about 1.32)
LN_MEAN = np.exp(LN_S ** 2 / 2)
LN_SD = np.sqrt((np.exp(LN_S ** 2) - 1) * np.exp(LN_S ** 2))

FAMILIES = ("iid_normal", "student_t4", "skew_negative", "skew_positive", "garch", "ar1",
            "regimes", "correlated_trials", "block_cluster")


def unit_family(family, rng, n, k):
    """k series of n periods from one Null Zoo family, each with population mean 0 and variance 1."""
    if family == "iid_normal":
        return rng.standard_normal((n, k))
    if family == "student_t4":
        z = rng.standard_normal((n, k))
        c = (rng.standard_normal((n, k, 4)) ** 2).sum(-1)
        return z / np.sqrt(c / 4) / np.sqrt(2)
    if family == "skew_negative":
        return (LN_MEAN - np.exp(LN_S * rng.standard_normal((n, k)))) / LN_SD
    if family == "skew_positive":
        return (np.exp(LN_S * rng.standard_normal((n, k))) - LN_MEAN) / LN_SD
    if family == "garch":
        # h_t = 0.05 + 0.10 e_{t-1}^2 + 0.85 h_{t-1}, unconditional variance 1, started at h = 1, e = 0.
        z = rng.standard_normal((n, k))
        x = np.empty((n, k))
        h = np.ones(k)
        e = np.zeros(k)
        for t in range(n):
            h = 0.05 + 0.1 * e * e + 0.85 * h
            e = np.sqrt(h) * z[t]
            x[t] = e
        return x
    if family == "ar1":
        # x_t = 0.2 x_{t-1} + sqrt(1 - 0.04) z_t, started from the stationary distribution.
        z = rng.standard_normal((n + 1, k))
        x = np.empty((n, k))
        prev = z[0]
        s = np.sqrt(1 - 0.04)
        for t in range(n):
            prev = 0.2 * prev + s * z[t + 1]
            x[t] = prev
        return x
    if family == "regimes":
        # Volatility 0.6 or 1.6 (rescaled to unit variance), switching with probability 0.02 a period.
        scale = 1 / np.sqrt((0.36 + 2.56) / 2)
        high = rng.random(k) < 0.5
        flips = rng.random((n, k)) < 0.02
        state = np.logical_xor(high[None, :], np.cumsum(flips, axis=0) % 2 == 1)
        return np.where(state, 1.6, 0.6) * scale * rng.standard_normal((n, k))
    if family == "correlated_trials":
        f = rng.standard_normal((n, 1))
        return np.sqrt(0.5) * f + np.sqrt(0.5) * rng.standard_normal((n, k))
    if family == "block_cluster":
        # Four clusters; trials in a cluster share a factor (pairwise correlation 0.8).
        size = int(np.ceil(k / 4))
        f = rng.standard_normal((n, 4))
        return np.sqrt(0.8) * np.repeat(f, size, axis=1)[:, :k] + np.sqrt(0.2) * rng.standard_normal((n, k))
    raise ValueError(f"unknown family {family}")


def draw_search(family, rng, n=504, k=20, sharpe=None, vol=None):
    """One search. `sharpe` (annualized, length k) sets each strategy's true mean; `vol` scales each one.

    With neither given, every strategy has mean zero and unit variance (the Null Zoo null).
    """
    x = unit_family(family, rng, n, k)
    if vol is not None:
        x = x * np.asarray(vol, dtype=float)[None, :]
    if sharpe is not None:
        sd = np.ones(k) if vol is None else np.asarray(vol, dtype=float)
        x = x + (np.asarray(sharpe, dtype=float) * sd / np.sqrt(Q))[None, :]
    return x


def stationary_indices(n, block, reps, rng):
    """Period indices of `reps` circular stationary-bootstrap resamples, shape (reps, n).

    Politis and Romano (1994): a resample starts at a uniform period; each later period continues the
    current block (the next period, wrapping at the end) with probability 1 - 1/block or starts a new
    block at a uniform period otherwise.
    """
    p = 1.0 / block
    flags = rng.random((reps, n)) < p
    flags[:, 0] = True
    starts = rng.integers(0, n, size=(reps, n))
    pos = np.where(flags, np.arange(n)[None, :], 0)
    last = np.maximum.accumulate(pos, axis=1)
    first = np.take_along_axis(starts, last, axis=1)
    return (first + (np.arange(n)[None, :] - last)) % n


def stationary_counts(n, block, reps, rng):
    """How often each period appears in each of `reps` stationary-bootstrap resamples, shape (reps, n)."""
    idx = stationary_indices(n, block, reps, rng)
    flat = (idx + n * np.arange(reps)[:, None]).ravel()
    return np.bincount(flat, minlength=reps * n).reshape(reps, n).astype(np.float64)


def pr_variance(x, block):
    """n times the stationary-bootstrap variance of each column's mean, in closed form.

    gamma_0 + 2 sum_{i=1}^{n-1} kappa_i gamma_i with kappa_i = (1 - i/n)(1 - p)^i + (i/n)(1 - p)^(n - i)
    and p = 1/block (Politis and Romano 1994; Hansen 2005). With block 1 it is the sample variance
    with divisor n.
    """
    n = x.shape[0]
    d = x - x.mean(0)
    f = np.fft.rfft(d, n=2 * n, axis=0)
    acov = np.fft.irfft(f * np.conj(f), n=2 * n, axis=0)[:n] / n
    p = 1.0 / block
    i = np.arange(1, n)
    kappa = (1 - i / n) * (1 - p) ** i + (i / n) * (1 - p) ** (n - i)
    return acov[0] + 2 * (kappa[:, None] * acov[1:]).sum(0)


def joint_tests(x, block, reps, rng, variants):
    """Exceedance counts h (p = h / reps) of the joint-resampling tests on one stationary bootstrap.

    variants is a subset of:
      spa_c, spa_u     Hansen (2005): T = max(0, max_k sqrt(n) m_k / w_k) with w_k^2 the Politis-Romano
                       variance at `block`, held fixed in the bootstrap; consistent / upper recentring.
      rc               White (2000): max_k m_k against max_k (m*_k - m_k); not studentized.
      spa_c_unstud     consistent recentring, statistic not studentized, no floor at zero; the
                       computation arch 8.0.0's SPA performs whatever its `studentize` flag says.
      spa_c_sd         as spa_c but studentized by the sample standard deviation (block 1 variance).
      spa_c_boot_t     consistent recentring, every resample re-studentized by its own standard
                       deviation (a bootstrap-t); the observed statistic uses the sample's.
    """
    n, k = x.shape
    m = x.mean(0)
    w2 = pr_variance(x, block)
    w = np.sqrt(w2)
    counts = stationary_counts(n, block, reps, rng)
    ms = counts @ x / n
    root = np.sqrt(n)
    thr = -np.sqrt(w2 / n * 2 * np.log(np.log(n)))
    centre_c = np.where(m >= thr, m, 0.0)
    out = {}
    if "spa_c" in variants or "spa_u" in variants:
        T = max(0.0, float(np.max(root * m / w)))
        if "spa_c" in variants:
            out["spa_c"] = int(np.sum(np.maximum(0.0, np.max(root * (ms - centre_c) / w, axis=1)) > T))
        if "spa_u" in variants:
            out["spa_u"] = int(np.sum(np.maximum(0.0, np.max(root * (ms - m) / w, axis=1)) > T))
    if "rc" in variants:
        out["rc"] = int(np.sum(np.max(ms - m, axis=1) > np.max(m)))
    if "spa_c_unstud" in variants:
        out["spa_c_unstud"] = int(np.sum(np.max(ms - centre_c, axis=1) > np.max(m)))
    if "spa_c_sd" in variants:
        s1 = np.sqrt(pr_variance(x, 1))
        T1 = max(0.0, float(np.max(root * m / s1)))
        out["spa_c_sd"] = int(np.sum(np.maximum(0.0, np.max(root * (ms - centre_c) / s1, axis=1)) > T1))
    if "spa_c_boot_t" in variants:
        sd = x.std(0, ddof=1)
        Tt = max(0.0, float(np.max(root * m / sd)))
        sq = counts @ (x * x) / n
        sd_star = np.sqrt(np.maximum(sq - ms * ms, 1e-300) * n / (n - 1))
        out["spa_c_boot_t"] = int(np.sum(np.maximum(0.0, np.max(root * (ms - centre_c) / sd_star, axis=1)) > Tt))
    return out


def sidak_t(x):
    """The v0/v1 luck-equivalent-trials test: the best Sharpe ratio's Student-t tail, Sidak over k."""
    n, k = x.shape
    t = x.mean(0) / x.std(0, ddof=1) * np.sqrt(n)
    p1 = float(stats.t.sf(np.max(t), n - 1))
    return 1.0 - (1.0 - p1) ** k


# ---- v2b additions (analysis/v2b/prereg-v2b.json). The functions above are unchanged, so the v2 run
# ---- still reproduces byte for byte.

def stationary_draw(n, block, reps, rng):
    """Like stationary_indices, but also returns the block-start flags, shape (reps, n) each."""
    p = 1.0 / block
    flags = rng.random((reps, n)) < p
    flags[:, 0] = True
    starts = rng.integers(0, n, size=(reps, n))
    pos = np.where(flags, np.arange(n)[None, :], 0)
    last = np.maximum.accumulate(pos, axis=1)
    first = np.take_along_axis(starts, last, axis=1)
    return (first + (np.arange(n)[None, :] - last)) % n, flags


def natural_block_variance(xs, flags):
    """n times the variance of each resample's mean estimated from its own blocks.

    xs: resampled series, shape (R, n, k); flags: block starts, shape (R, n). For each resample and
    column, sum the deviations from the resample mean within each block and average the squared block
    sums over n (the "natural" block-bootstrap variance of Gotze and Kunsch 1996). With block length 1
    it is the resample variance with divisor n.
    """
    R, n, k = xs.shape
    dev = xs - xs.mean(axis=1, keepdims=True)
    cs = np.cumsum(dev, axis=1)
    before = np.concatenate([np.zeros((R, 1, k)), cs[:, :-1, :]], axis=1)   # cumulative sum before t
    pos = np.where(flags, np.arange(n)[None, :], 0)
    start = np.maximum.accumulate(pos, axis=1)                                # first period of t's block
    partial = cs - np.take_along_axis(before, start[:, :, None], axis=1)      # block sum up to t
    end = np.concatenate([flags[:, 1:], np.ones((R, 1), dtype=bool)], axis=1) # t ends its block
    return np.einsum("rn,rnk->rk", end.astype(float), partial * partial) / n


def block_moments(x, idx, flags):
    """Per-resample mean, mean square and natural block variance, from block sums of the original series.

    Every stationary-bootstrap block is a contiguous (circular) stretch of x, so its sums are differences
    of cumulative sums; nothing is copied period by period. Returns arrays of shape (R, k) equal to
    xs.mean(1), (xs * xs).mean(1) and natural_block_variance(xs, flags) for xs = x[idx].
    """
    n, k = x.shape
    R = idx.shape[0]
    xx = np.concatenate([x, x], axis=0)
    C = np.vstack([np.zeros((1, k)), np.cumsum(xx, axis=0)])
    C2 = np.vstack([np.zeros((1, k)), np.cumsum(xx * xx, axis=0)])
    r_ids, t_ids = np.nonzero(flags)                       # row-major: sorted by resample, then period
    first = np.r_[True, r_ids[1:] != r_ids[:-1]]
    nxt = np.r_[t_ids[1:], n]
    nxt[np.r_[first[1:], True]] = n                        # a resample's last block runs to period n
    L = nxt - t_ids
    i0 = idx[r_ids, t_ids]
    S = C[i0 + L] - C[i0]
    Q = C2[i0 + L] - C2[i0]
    bounds = np.flatnonzero(first)
    sumS = np.add.reduceat(S, bounds, axis=0)
    sumQ = np.add.reduceat(Q, bounds, axis=0)
    sumS2 = np.add.reduceat(S * S, bounds, axis=0)
    sumLS = np.add.reduceat(L[:, None] * S, bounds, axis=0)
    sumL2 = np.add.reduceat((L * L).astype(float), bounds)
    ms = sumS / n
    vnb = (sumS2 - 2 * ms * sumLS + ms * ms * sumL2[:, None]) / n
    return ms, sumQ / n, vnb


def joint_tests_v2b(x, block, reps, rng):
    """Exceedance counts (p = h / reps) for the v2b tests on one stationary bootstrap of the search.

      rc                 White (2000), unchanged definition (strict inequality, no floor).
      spa_c_unstud       arch 8.0.0's computation (unstudentized, consistent recentring, no floor).
      spa_c_strict       Hansen (2005): fixed Politis-Romano studentizer, floor at zero, p = #(T* > T)/B.
      spa_c_ge           the same statistic with ties counted, p = #(T* >= T)/B (p = 1 when T = 0).
      spa_c_boot_t_ge    each resample re-studentized by its own standard deviation; observed statistic by
                         the sample standard deviation; ties counted.
      spa_c_nb_ge        each resample re-studentized by its natural block variance; observed statistic by
                         the Politis-Romano variance; ties counted.
    Columns with zero sample variance must be dropped by the caller.
    """
    n, k = x.shape
    m = x.mean(0)
    w2 = pr_variance(x, block)
    w = np.sqrt(w2)
    sd = x.std(0, ddof=1)
    root = np.sqrt(n)
    thr = -np.sqrt(w2 / n * 2 * np.log(np.log(n)))
    centre = np.where(m >= thr, m, 0.0)
    idx, flags = stationary_draw(n, block, reps, rng)
    if block == 1:
        # Every block is one period: count how often each period is drawn instead.
        counts = np.bincount((idx + n * np.arange(reps)[:, None]).ravel(), minlength=reps * n).reshape(reps, n).astype(float)
        ms = counts @ x / n
        sq = counts @ (x * x) / n
        vnb = sq - ms * ms
    else:
        ms, sq, vnb = block_moments(x, idx, flags)
    out = {}
    out["rc"] = int(np.sum(np.max(ms - m, axis=1) > np.max(m)))
    out["spa_c_unstud"] = int(np.sum(np.max(ms - centre, axis=1) > np.max(m)))
    T = max(0.0, float(np.max(root * m / w)))
    Ts = np.maximum(0.0, np.max(root * (ms - centre) / w, axis=1))
    out["spa_c_strict"] = int(np.sum(Ts > T))
    out["spa_c_ge"] = int(np.sum(Ts >= T))
    Tt = max(0.0, float(np.max(root * m / sd)))
    sd_star = np.sqrt(np.maximum(sq - ms * ms, 1e-300) * n / (n - 1))
    out["spa_c_boot_t_ge"] = int(np.sum(np.maximum(0.0, np.max(root * (ms - centre) / sd_star, axis=1)) >= Tt))
    Tn = np.maximum(0.0, np.max(root * (ms - centre) / np.sqrt(np.maximum(vnb, 1e-300)), axis=1))
    out["spa_c_nb_ge"] = int(np.sum(Tn >= T))
    return out
