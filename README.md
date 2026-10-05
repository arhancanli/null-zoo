# The Null Zoo

**Size and power of backtest-overfitting and data-snooping tests on synthetic searches with known ground truth.**

Arhan Canli · [paper page](https://arhancanli.github.io/null-zoo/) · [paper (PDF)](paper/main.pdf) · [DOI 10.5281/zenodo.23018987](https://doi.org/10.5281/zenodo.23018987) (all versions) · MIT licence

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23018987.svg)](https://doi.org/10.5281/zenodo.23018987)

Corrections for backtest overfitting and data snooping (the deflated Sharpe ratio, the multiple-testing
haircut, family-wise adjustments, White's Reality Check, Hansen's SPA test, the stepwise SPA test)
are derived under assumptions that real trading research violates. The Null Zoo draws complete
research searches, 20 strategies of 504 daily returns, from nine return families whose ground truth
is known by construction, and scores each correction by size (how often it calls a skill-less search's
best strategy skilled) and power (how often it finds a strategy that has skill). Every governed run was
pre-registered before it ran.

## Main findings

The exact figures, with their Monte Carlo errors, are in the paper; every one is generated from the
result files in this repository.

- **With block resampling, Hansen's studentized SPA test, and the stepwise SPA test built on it, are
  oversized at two years of daily data; White's Reality Check is close to its level except under
  autocorrelation.** Confirmed on fresh seeds and by a second implementation in another language.
- **The cause is a variance estimate the bootstrap holds fixed**: noisy and, computed from demeaned data,
  biased downward; under negative skew a mean and its standard deviation are also dependent.
  Re-studentizing every resample, as Romano and Wolf (2005) recommend, brings the size to about the
  Reality Check's. The excess shrinks with the sample and grows with the number of strategies and the
  block length; with the i.i.d. resampling of Hansen's own simulations it nearly disappears.
- **Every block bootstrap inherits the same estimate**, so long blocks make every test liberal, the
  Reality Check included. Re-studentizing by a natural block variance cures the autocorrelated case
  but is conservative elsewhere.
- **The Python package arch (8.0.0) does not studentize its SPA statistic**: its `studentize` option has
  no effect, so its test is the Reality Check with consistent recentring, much less oversized. A pending
  change that studentizes with a fixed variance would, on our implementation of its formula, make it
  the fixed-variance test.
- **Studentizing and recentring pay in the cases they were designed for**: studentizing finds a skilled
  low-volatility strategy that ranking by mean misses (and misses a skilled high-volatility one that it
  finds); consistent recentring raises power when many strategies are clearly poor. A floored SPA
  statistic with a strict inequality can reject when no strategy beats the benchmark; counting ties
  fixes it.
- **Autocorrelation of 0.2 roughly quadruples the false-positive rate of the single-series tests that
  ignore it**; Lo's factor repairs the AR(1) case.
- **The deflated Sharpe ratio with its authors' 0.95 rule tests a harder hypothesis than "no skill"**
  and passes few strategies with a true Sharpe ratio of 2 over two years.
- **On real data the choice matters at the margin**: on 20 moving-average rules in 51 two-year windows
  of US market data since 1927, the tests disagree in 5 windows.

## Layout

| Path | What it is |
|---|---|
| `src/` | The JavaScript benchmark as maintained in [arhancanli/canlicapital](https://github.com/arhancanli/canlicapital), copied unchanged: v0 at the commit in `src/SOURCE_COMMIT`, v1 and v1b (harness, pre-registrations, results) at the commit in `src/SOURCE_COMMIT_V1`. |
| `analysis/v2/` | The pre-registered v2 study: a second implementation in Python (`nzpy.py`), written by the same author from the published definitions, with instrument tests, the pre-registration (`prereg-v2.json`, committed before the run), the run and scoring scripts, their outputs and an exploratory check that is labelled as such. |
| `analysis/v2b/` | The pre-registered v2b study (number of strategies and block length, block re-studentization, unequal volatilities and poor alternatives with ties counted): pre-registration, run and scoring scripts, outputs. |
| `analysis/empirical/` | The v2b empirical illustration: 20 moving-average timing rules on the Fama/French daily market data. The data file is downloaded, not committed; `data-manifest.json` records its SHA-256. |
| `analysis/provenance.json` | Commit hashes and times of every pre-registration and result, and the re-run checks. |
| `analysis/tables.mjs` | Builds every table and every result quoted in the paper from the result files; the paper reads each one by key. |
| `analysis/*.json` | Earlier v0-design runs: the robustness run and the paired bootstrap-resolution run. |
| `paper/` | LaTeX source and the compiled PDF. |

## Reproduce

Requires Node.js 24 and Python 3.12 (`pip install -r analysis/v2/requirements.txt`). Everything is
deterministic given the seeds recorded in the result files, and independent of the number of workers.

```sh
npm test                                      # v0 and v1 instrument tests
python -m unittest analysis/v2/test_nzpy.py   # v2 instrument tests
npm run reproduce:v0                          # rerun the published v0 benchmark and compare byte for byte
npm run reproduce:v1                          # ~30-45 min: rerun v1 and compare byte for byte
npm run v2                                    # ~20-40 min: rerun the v2 experiments and score them
python analysis/empirical/run_empirical.py fetch   # download the French data file (checked against its SHA-256)
npm run v2b                                   # ~1-2 h: rerun the v2b experiments and the illustration, and score them
npm run tables                                # regenerate paper/generated/* from the result files
```

To build the PDF, run `tectonic main.tex` (or `latexmk -pdf main.tex`) in `paper/`.

Continuous integration runs the instrument tests, the byte-for-byte reproduction of v0, the v2 and v2b
scoring against the committed result files and a check that the committed tables match the result files.
It does not re-run v1, v1b, v2 or v2b, which take hours. The Python runs used numpy 2.5.3; numpy does not
promise identical random streams across versions, so a byte-for-byte re-run needs that version.

## Citation

See [`CITATION.cff`](CITATION.cff).

## Competing interests

The author designed the luck-equivalent-trials statistic scored here. The author's firm, Canli Capital,
publishes a validation API (free at the time of writing) and MCP servers that implement the
luck-equivalent-trials test, the haircut, the deflated Sharpe ratio and the data-snooping tests; the v1
and v1b runs set their defaults (the Reality Check for searches, LET-Lo for single series), and the firm
may benefit from wider use of these tools.
