# The Null Zoo

**Size and power of backtest-overfitting and data-snooping tests on synthetic searches with known ground truth.**

Arhan Canli · [paper page](https://arhancanli.github.io/null-zoo/) · [paper (PDF)](paper/main.pdf) · [DOI 10.5281/zenodo.23018987](https://doi.org/10.5281/zenodo.23018987) (all versions) · MIT licence

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23018987.svg)](https://doi.org/10.5281/zenodo.23018987)

Corrections for backtest overfitting and data snooping (the deflated Sharpe ratio, the multiple-testing
haircut, family-wise adjustments, White's Reality Check, Hansen's SPA test, Romano and Wolf's StepM)
are derived under assumptions that real trading research violates. The Null Zoo draws complete
research searches, 20 strategies of 504 daily returns, from nine return families whose ground truth
is known by construction, and scores each correction by size (how often it calls a skill-less search's
best strategy skilled) and power (how often it finds a strategy that has skill). Every governed run was
pre-registered before it ran.

## Main findings

The exact figures, with their Monte Carlo errors, are in the paper; every one is generated from the
result files in this repository.

- **Hansen's studentized SPA test, and StepM with it, is oversized at two years of daily data; White's
  Reality Check is not.** Confirmed on fresh seeds and by an independent implementation in a second
  language.
- **The cause is a variance estimate the bootstrap holds fixed**, compounded under negative skew by the
  dependence between a mean and its estimated standard deviation. Re-studentizing every resample (a
  bootstrap-t) restores calibration. The excess shrinks with the sample, grows with the number of
  strategies and with the block length.
- **The Python package arch (8.0.0) does not studentize its SPA statistic**: its `studentize` option
  changes nothing, which is why its test stays close to its level. A pending change that studentizes
  with a fixed variance would, on our implementation of its formula, make it oversized.
- **Studentizing and recentring still pay when done right**: with unequal volatilities or many poor
  strategies, a bootstrap-t SPA test with consistent recentring is far more powerful than the Reality
  Check. A floored SPA statistic with a strict inequality can reject when no strategy beats the
  benchmark; counting ties fixes it.
- **Autocorrelation of 0.2 roughly quadruples the false-positive rate of the single-series tests that
  ignore it**; Lo's factor repairs the AR(1) case.
- **The deflated Sharpe ratio with its authors' 0.95 rule tests a harder hypothesis than "no skill"**
  and passes few strategies with a true Sharpe ratio of 2 over two years.

## Layout

| Path | What it is |
|---|---|
| `src/` | The JavaScript benchmark as maintained in [arhancanli/canlicapital](https://github.com/arhancanli/canlicapital), copied unchanged: v0 at the commit in `src/SOURCE_COMMIT`, v1 and v1b (harness, pre-registrations, results) at the commit in `src/SOURCE_COMMIT_V1`. |
| `analysis/v2/` | The pre-registered v2 study: an independent Python implementation (`nzpy.py`) with instrument tests, the pre-registration (`prereg-v2.json`, committed before the run), the run and scoring scripts, their outputs and an exploratory check that is labelled as such. |
| `analysis/tables.mjs` | Builds every table and every number in the paper from the result files. The paper reads each number by key, so none is typed by hand. |
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
npm run tables                                # regenerate paper/generated/* from the result files
```

To build the PDF, run `tectonic main.tex` (or `latexmk -pdf main.tex`) in `paper/`.

Continuous integration runs the instrument tests, the byte-for-byte reproduction of v0, the v2 scoring
against the committed result files and a check that the committed tables match the result files.

## Citation

See [`CITATION.cff`](CITATION.cff).

## Competing interests

The author designed the luck-equivalent-trials statistic scored here and maintains open-source
implementations of several of the scored methods, including the data-snooping tests; the v1 and v1b
runs were also used to choose a default test for that software.
