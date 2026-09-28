# Null Zoo v0: backtest-overfitting corrections scored on ground truth

**Short title:** Null Zoo v0
**Author:** Arhan Canli
**Declared:** 2026-09-26, with every cell's seed in the published results.

Every correction for backtest overfitting makes a promise: that a skill-less strategy will pass it
no more often than its stated level. The Null Zoo tests the promise. It draws research searches in
which the truth is known, from return families with different shapes, and counts how often each
correction calls a skill-less search's best strategy skilled (its size) and how often it finds a
strategy that does have skill (its power).

## Design

Each search tries 20 strategies over 504 daily
observations; every strategy has a true Sharpe of zero, except in the power arm, where one has a
true annualized Sharpe of 2. Each cell runs 2000 searches with its own
fixed seed. The level is 5.0%. Results:
[`/glassbox/research/null-zoo-v0.json`](/glassbox/research/null-zoo-v0.json); generators and
checks: `scripts/research/null-zoo/` in the open repository.

## Size: how often a skill-less search is called skilled

| Returns | Luck-equivalent trials | Luck-equivalent trials, Lo-corrected | Luck trials, non-normal SE | Haircut Sharpe (Bonferroni) | Haircut Sharpe (Bonferroni, Lo-corrected) | Deflated Sharpe ratio at 0.95 | Bootstrap of the best trial |
|---|---|---|---|---|---|---|---|
| Independent normal returns | 0.059 | 0.061 | 0.059 | 0.028 | 0.036 | 0.000 | 0.048 |
| Fat tails (Student t) | 0.053 | 0.054 | 0.072 | 0.028 | 0.028 | 0.001 | 0.060 |
| Negative skew | 0.088 | 0.088 | 0.051 | 0.054 | 0.056 | 0.001 | 0.053 |
| Positive skew | 0.036 | 0.037 | 0.070 | 0.014 | 0.017 | 0.001 | 0.058 |
| Volatility clustering (GARCH) | 0.046 | 0.050 | 0.048 | 0.024 | 0.026 | 0.000 | 0.048 |
| Autocorrelated returns (AR(1)) | 0.200 | 0.059 | 0.201 | 0.132 | 0.027 | 0.002 | 0.171 |
| Two volatility regimes | 0.051 | 0.051 | 0.056 | 0.025 | 0.024 | 0.000 | 0.051 |
| Correlated trials (one common factor) | 0.035 | 0.035 | 0.034 | 0.017 | 0.019 | 0.013 | 0.032 |

## Power: how often a strategy with skill is found

| Returns | Luck-equivalent trials | Luck-equivalent trials, Lo-corrected | Luck trials, non-normal SE | Haircut Sharpe (Bonferroni) | Haircut Sharpe (Bonferroni, Lo-corrected) | Deflated Sharpe ratio at 0.95 | Bootstrap of the best trial |
|---|---|---|---|---|---|---|---|
| Independent normal returns | 0.533 | 0.529 | 0.528 | 0.430 | 0.428 | 0.090 | 0.471 |
| Fat tails (Student t) | 0.515 | 0.514 | 0.515 | 0.420 | 0.424 | 0.107 | 0.469 |
| Negative skew | 0.556 | 0.564 | 0.451 | 0.461 | 0.466 | 0.079 | 0.423 |
| Positive skew | 0.549 | 0.545 | 0.649 | 0.450 | 0.453 | 0.121 | 0.544 |
| Volatility clustering (GARCH) | 0.544 | 0.546 | 0.544 | 0.444 | 0.445 | 0.104 | 0.476 |
| Autocorrelated returns (AR(1)) | 0.603 | 0.358 | 0.604 | 0.508 | 0.268 | 0.071 | 0.544 |
| Two volatility regimes | 0.539 | 0.537 | 0.540 | 0.451 | 0.450 | 0.121 | 0.488 |
| Correlated trials (one common factor) | 0.501 | 0.502 | 0.498 | 0.403 | 0.405 | 0.225 | 0.442 |

## What it shows

- Autocorrelated returns break every best-of-N test that ignores them: luck-equivalent trials
  rejected 0.200 and the Bonferroni haircut 0.132
  of skill-less searches. Lo's correction brings luck-equivalent trials to 0.059.
- The deflated Sharpe ratio read as a test at 0.95 almost never calls noise skilled
  (0.000) but finds the skilled strategy in only 0.090
  of searches, against 0.533 for luck-equivalent trials. It subtracts the
  expected best of N and then asks for high confidence on top: an estimate, not a test at its level.
- Haircut Sharpe p-values are two-sided, as the authors define them, so their size sits near half
  the level by construction.

## Evidence boundary

These are synthetic searches. They measure how corrections behave on known return shapes; they do
not say which shape a real strategy has, and a correction that passes here can still fail on a
shape the zoo does not contain.
