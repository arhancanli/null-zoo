"""Instrument tests for analysis/v2/nzpy.py. Run: python -m unittest analysis/v2/test_nzpy.py (from the repo root)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nzpy  # noqa: E402


def direct_pr_variance(x, block):
    n = x.shape[0]
    d = x - x.mean(0)
    p = 1.0 / block
    v = (d ** 2).sum(0) / n
    for i in range(1, n):
        kappa = (1 - i / n) * (1 - p) ** i + (i / n) * (1 - p) ** (n - i)
        v = v + 2 * kappa * (d[: n - i] * d[i:]).sum(0) / n
    return v


def hansen_spa_transcription(d, block, reps, rng):
    """Hansen (2005) studentized SPA, loop by loop (from the canlicapital numpy reference)."""
    n, k = d.shape
    mean = d.mean(0)
    w = np.sqrt(direct_pr_variance(d, block))
    stat = max(0.0, float(np.max(np.sqrt(n) * mean / w)))
    thresh = -np.sqrt(w ** 2 / n * 2 * np.log(np.log(n)))
    g = np.where(mean >= thresh, mean, 0.0)
    p = 1.0 / block
    above = 0
    for _ in range(reps):
        idx = np.empty(n, dtype=np.int64)
        idx[0] = rng.integers(n)
        for t in range(1, n):
            idx[t] = rng.integers(n) if rng.random() < p else (idx[t - 1] + 1) % n
        m = d[idx].mean(0)
        above += max(0.0, float(np.max(np.sqrt(n) * (m - g) / w))) > stat
    return above / reps


class VarianceAndBootstrap(unittest.TestCase):
    def test_pr_variance_matches_the_direct_sum(self):
        x = np.random.default_rng(1).standard_normal((300, 4))
        for block in (1, 8, 22):
            self.assertLess(float(np.max(np.abs(nzpy.pr_variance(x, block) - direct_pr_variance(x, block)))), 1e-12)

    def test_block_one_is_the_sample_variance(self):
        x = np.random.default_rng(2).standard_normal((200, 3))
        self.assertLess(float(np.max(np.abs(nzpy.pr_variance(x, 1) - x.var(0)))), 1e-12)

    def test_resamples_have_n_periods_and_the_mean_block_length(self):
        rng = np.random.default_rng(3)
        n, block, reps = 504, 8, 4000
        idx = nzpy.stationary_indices(n, block, reps, rng)
        self.assertEqual(idx.shape, (reps, n))
        self.assertTrue(np.all((idx >= 0) & (idx < n)))
        starts = 1 + np.sum(idx[:, 1:] != (idx[:, :-1] + 1) % n, axis=1)
        expected = 1 + (n - 1) / block   # a new block starts with probability 1/block (a uniform restart
        # can land on the next period by chance, which this count misses: about 1/n of restarts)
        self.assertAlmostEqual(float(starts.mean()) / expected, 1.0, delta=0.02)
        counts = nzpy.stationary_counts(n, block, 50, np.random.default_rng(4))
        self.assertTrue(np.all(counts.sum(1) == n))

    def test_bootstrap_variance_of_the_mean_is_the_closed_form(self):
        rng = np.random.default_rng(5)
        x = nzpy.draw_search("ar1", rng, n=504, k=3)
        counts = nzpy.stationary_counts(504, 8, 20000, rng)
        ms = counts @ x / 504
        ratio = (ms.var(0) * 504) / nzpy.pr_variance(x, 8)
        self.assertTrue(np.all(np.abs(ratio - 1) < 0.05), ratio)


class Families(unittest.TestCase):
    def test_every_family_has_the_moments_it_claims(self):
        for fam in nzpy.FAMILIES:
            k = 20 if fam == "block_cluster" else 3
            y = nzpy.unit_family(fam, np.random.default_rng(6), 50000, k)
            self.assertLess(float(np.max(np.abs(y.mean(0)))), 0.03, fam)
            self.assertLess(float(np.max(np.abs(y.var(0) - 1))), 0.12, fam)
            sk = ((y - y.mean(0)) ** 3).mean(0) / y.var(0) ** 1.5
            ac = np.array([np.corrcoef(y[1:, j], y[:-1, j])[0, 1] for j in range(y.shape[1])])
            if fam == "skew_negative":
                self.assertTrue(np.all(sk < -1), sk)
            if fam == "skew_positive":
                self.assertTrue(np.all(sk > 1), sk)
            if fam == "ar1":
                self.assertTrue(np.all(np.abs(ac - 0.2) < 0.02), ac)
            if fam == "correlated_trials":
                self.assertAlmostEqual(float(np.corrcoef(y.T)[0, 1]), 0.5, delta=0.03)
            if fam == "block_cluster":
                c = np.corrcoef(y.T)
                self.assertAlmostEqual(float(c[0, 1]), 0.8, delta=0.03)
                self.assertAlmostEqual(float(c[0, 5]), 0.0, delta=0.03)
            if fam in ("garch", "regimes"):
                sq = y[:, 0] ** 2
                self.assertGreater(float(np.corrcoef(sq[1:], sq[:-1])[0, 1]), 0.05, fam)
            if fam == "student_t4":
                kurt = ((y - y.mean(0)) ** 4).mean(0) / y.var(0) ** 2
                self.assertTrue(np.all(kurt > 5), kurt)

    def test_sharpe_and_volatility_are_set_per_strategy(self):
        sharpe = np.array([2.0, 0.0, -3.0])
        vol = np.array([0.5, 1.0, 2.0])
        x = nzpy.draw_search("iid_normal", np.random.default_rng(7), n=400000, k=3, sharpe=sharpe, vol=vol)
        self.assertTrue(np.allclose(x.std(0), vol, rtol=0.01))
        self.assertTrue(np.allclose(x.mean(0) / x.std(0) * np.sqrt(252), sharpe, atol=0.05))


class Tests(unittest.TestCase):
    def test_studentized_spa_matches_the_loop_transcription(self):
        rng = np.random.default_rng(8)
        # Unequal volatilities, so studentizing by anything but the standard error changes the answer.
        x = nzpy.draw_search("iid_normal", rng, n=300, k=6, sharpe=[1.5, 0, 0, 0, 0, -1], vol=[0.3, 1, 2.5, 1, 3, 0.6])
        reps = 4000
        a = nzpy.joint_tests(x, 7, reps, np.random.default_rng(9), {"spa_c"})["spa_c"] / reps
        b = hansen_spa_transcription(x, 7, reps, np.random.default_rng(10))
        self.assertLess(abs(a - b), 4 * np.sqrt(max(a * (1 - a), 0.01) / reps) * np.sqrt(2), (a, b))

    def test_unstudentized_consistent_spa_is_arch_8_0_0(self):
        from arch.bootstrap import SPA
        rng = np.random.default_rng(11)
        # Eight clearly poor strategies: the consistent recentring drops them and the upper one does
        # not, so the two p-values differ and a swap between them would be caught.
        x = nzpy.draw_search("iid_normal", rng, n=504, k=10, sharpe=[1.0, 0] + [-4] * 8)
        reps = 5000
        ours = nzpy.joint_tests(x, 8, reps, np.random.default_rng(12), {"spa_c_unstud", "rc"})
        spa = SPA(np.zeros(504), -x, block_size=8, reps=reps, bootstrap="stationary", seed=np.random.default_rng(13))
        spa.compute()
        se = 4 * np.sqrt(0.25 / reps) * np.sqrt(2)
        self.assertGreater(float(spa.pvalues["upper"]) - float(spa.pvalues["consistent"]), 2 * se)
        self.assertLess(abs(ours["spa_c_unstud"] / reps - float(spa.pvalues["consistent"])), se)
        self.assertLess(abs(ours["rc"] / reps - float(spa.pvalues["upper"])), se)

    def test_sidak_t_is_the_formula(self):
        from scipy import stats
        x = nzpy.draw_search("iid_normal", np.random.default_rng(14), n=504, k=5)
        t = x.mean(0) / x.std(0, ddof=1) * np.sqrt(504)
        p1 = stats.t.sf(t.max(), 503)
        self.assertAlmostEqual(nzpy.sidak_t(x), 1 - (1 - p1) ** 5, places=12)


if __name__ == "__main__":
    unittest.main()


class V2bAdditions(unittest.TestCase):
    def test_flags_mark_every_block_start(self):
        n, reps = 504, 200
        idx, flags = nzpy.stationary_draw(n, 8, reps, np.random.default_rng(20))
        cont = idx[:, 1:] == (idx[:, :-1] + 1) % n
        self.assertTrue(np.all(flags[:, 1:] | cont))            # no jump without a flag
        self.assertTrue(np.all(flags[:, 0]))
        self.assertAlmostEqual(float(flags[:, 1:].mean()), 1 / 8, delta=0.01)

    def test_block_moments_equal_the_period_by_period_computation(self):
        rng = np.random.default_rng(21)
        for block in (4, 22):
            x = rng.standard_normal((300, 5))
            idx, flags = nzpy.stationary_draw(300, block, 200, rng)
            xs = x[idx]
            ms, sq, v = nzpy.block_moments(x, idx, flags)
            self.assertLess(float(np.abs(ms - xs.mean(1)).max()), 1e-12)
            self.assertLess(float(np.abs(sq - (xs * xs).mean(1)).max()), 1e-12)
            direct = np.zeros((200, 5))
            for r in range(200):
                d = xs[r] - xs[r].mean(0)
                starts = list(np.flatnonzero(flags[r])) + [300]
                for a, b in zip(starts[:-1], starts[1:]):
                    direct[r] += d[a:b].sum(0) ** 2
            self.assertLess(float(np.abs(v - direct / 300).max()), 1e-12)

    def test_natural_block_variance_tracks_the_long_run_variance(self):
        # Averaged over resamples, the block variance of a fixed sample is close to (a few percent below)
        # that sample's Politis-Romano variance at the same block length, which is the exact bootstrap
        # variance of the mean; it is not the plain sample variance.
        rng = np.random.default_rng(22)
        x = rng.standard_normal((504, 3))
        idx, flags = nzpy.stationary_draw(504, 8, 20000, rng)
        ms, _, v = nzpy.block_moments(x, idx, flags)
        pr = nzpy.pr_variance(x, 8)
        self.assertTrue(np.all(np.abs(504 * ms.var(0) / pr - 1) < 0.03), 504 * ms.var(0) / pr)
        self.assertTrue(np.all(np.abs(v.mean(0) / pr - 1) < 0.06), v.mean(0) / pr)

    def test_ties_give_p_one_when_no_strategy_beats_the_benchmark(self):
        x = nzpy.draw_search("iid_normal", np.random.default_rng(23), n=504, k=1, sharpe=[-4.0])
        out = nzpy.joint_tests_v2b(x, 8, 1000, np.random.default_rng(24))
        self.assertEqual(out["spa_c_ge"], 1000)
        self.assertEqual(out["spa_c_boot_t_ge"], 1000)
        self.assertEqual(out["spa_c_nb_ge"], 1000)
        self.assertLess(out["spa_c_strict"], 1000)

    def test_v2b_fixed_variance_test_matches_v2_in_distribution(self):
        x = nzpy.draw_search("iid_normal", np.random.default_rng(25), n=300, k=6, sharpe=[1.5, 0, 0, 0, 0, -1], vol=[0.3, 1, 2.5, 1, 3, 0.6])
        reps = 4000
        a = nzpy.joint_tests(x, 7, reps, np.random.default_rng(26), {"spa_c", "rc"})
        b = nzpy.joint_tests_v2b(x, 7, reps, np.random.default_rng(27))
        tol = 4 * np.sqrt(0.25 / reps) * np.sqrt(2)
        self.assertLess(abs(a["spa_c"] - b["spa_c_strict"]) / reps, tol)
        self.assertLess(abs(a["rc"] - b["rc"]) / reps, tol)


class Empirical(unittest.TestCase):
    def test_rules_trade_the_next_day_and_match_a_direct_loop(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "empirical"))
        import run_empirical as emp
        rng = np.random.default_rng(28)
        r = rng.normal(0.0004, 0.01, 700)
        rf = np.full(700, 0.0001)
        d, day = emp.excess_returns(r, rf)
        price = np.cumprod(1 + r)
        for j, L in ((0, 10), (9, 100), (19, 200)):
            for row in (0, 37, len(day) - 1):
                t = day[row] - 1                       # the signal day
                long = price[t] > price[t - L + 1:t + 1].mean()
                expect = (r[t + 1] if long else rf[t + 1]) - r[t + 1]
                self.assertAlmostEqual(d[row, j], expect, places=14)
        self.assertEqual(day[0], 200)
