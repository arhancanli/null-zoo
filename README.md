# The Null Zoo

**Size and power of backtest-overfitting corrections on synthetic searches with known ground truth.**

Arhan Canli · [paper page](https://arhancanli.github.io/null-zoo/) · [paper (PDF)](paper/main.pdf) · [DOI 10.5281/zenodo.23018988](https://doi.org/10.5281/zenodo.23018988) · MIT licence

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23018988.svg)](https://doi.org/10.5281/zenodo.23018988)

Corrections for backtest overfitting (the deflated Sharpe ratio, the multiple-testing haircut,
family-wise adjustments, bootstrap tests) are derived under assumptions that real trading research
violates. The Null Zoo draws complete research searches, 20 strategies of 504 daily returns, from
eight return families whose ground truth is known by construction, and scores each correction by
size (how often it calls a skill-less search's best strategy skilled) and power (how often it finds
a strategy that has skill).

## Main findings

- A lag-one autocorrelation of 0.2 raises the false-positive rate of every test that ignores it,
  except the near-zero deflated Sharpe ratio, from 5% to about 19–20%. Lo's autocorrelation-adjusted
  annualisation factor repairs it in the AR(1) case, at a small cost in size elsewhere.
- No validator is calibrated across negative skew, positive skew and fat tails. Under fat tails,
  selection favours series with a large positive outlier.
- The deflated Sharpe ratio with its authors' 0.95 rule tests a harder hypothesis than "no skill":
  under i.i.d. returns it fails to accept about 90% of searches containing a strategy with a true
  Sharpe ratio of 2 over two years.
- A bootstrap's number of resamples materially affects its power (paired comparison on the same
  searches).

Every number in the paper is generated from the result files in this repository by
`analysis/tables.mjs`; none is typed by hand.

## Layout

| Path | What it is |
|---|---|
| `src/` | The benchmark as published in [arhancanli/canlicapital](https://github.com/arhancanli/canlicapital) at the commit in `src/SOURCE_COMMIT`: return families, validators, instrument tests, core libraries and the published v0 result file. Copied unchanged. |
| `analysis/extend.mjs` | Robustness run: xoshiro128** generator, 10,000 searches per cell, bootstrap with 1,999 resamples, haircut read one-sided, size-adjusted power and diagnostics. |
| `analysis/bootstrap-resolution.mjs` | Paired run: the same searches scored by the bootstrap with 400 and 1,999 resamples. |
| `analysis/tables.mjs` | Builds every table and every number in the paper from the result files. |
| `analysis/*.json` | Result files: `null-zoo-extended.json` (robustness run), `null-zoo-extended-v1.json` (first robustness run, 400 resamples), `bootstrap-resolution.json`. |
| `paper/` | LaTeX source and the compiled PDF. |

## Reproduce

Requires Node.js 24. Everything is deterministic given the seeds recorded in the result files.

```sh
npm test                 # instrument tests: every family has the properties it claims
npm run reproduce:v0     # rerun the published benchmark and compare byte for byte
npm run tables           # regenerate paper/generated/* from the result files
npm run robustness       # ~30 min: rerun analysis/null-zoo-extended.json
npm run bootstrap-res    # ~12 min: rerun analysis/bootstrap-resolution.json
```

To build the PDF, run `tectonic main.tex` (or `latexmk -pdf main.tex`) in `paper/`.

Continuous integration runs the instrument tests, the byte-for-byte reproduction of the published
run, and a check that the committed tables and numbers match what `analysis/tables.mjs` generates.

## Citation

See [`CITATION.cff`](CITATION.cff).

## Competing interests

The author designed the luck-equivalent-trials statistic scored here and maintains open-source
implementations of several of the scored methods.
