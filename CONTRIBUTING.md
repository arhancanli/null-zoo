# Contributing to the Null Zoo

Thank you for checking our numbers. The most useful contributions are the ones that could prove us wrong.

## Report a reproduction mismatch

Run the scripts as the README describes. If a number you get differs from the one in the paper or in
`analysis/`, open a **Reproduction mismatch** issue with the command, your output, your Python version
and operating system, and the commit you ran. A mismatch is treated as a possible error in the paper
until it is explained.

## Report an error in a reported number or claim

Open a **Correction** issue naming the table, figure or sentence and what you believe is wrong. Confirmed
errors are corrected in the next version and credited by name in the changelog.

## Code changes

1. Open an issue first for anything larger than a typo, so the change can be checked against the
   pre-registration (results that were pre-registered are not re-run under new settings).
2. Keep each pull request to one change, with a test or a reproduction command.
3. Commits must not change a published result without saying so in the pull request description.

## Questions

Use [Discussions](https://github.com/arhancanli/null-zoo/discussions) for questions about the method,
the simulations or how to apply the corrections to your own backtests.
