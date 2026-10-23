# Contributing to bm25s

## Contributor eligibility

To reduce spam from AI agents, all pull request authors must meet this requirement:

> Pull request authors must have a GitHub account at least six calendar months old and at least 100 GitHub contributions dated on or before the date one calendar month before the check.

The automated check uses UTC dates and GitHub contribution history visible to the
repository's automation. It counts contributions across the account's history,
excluding the most recent month. For example, a check on October 1 counts
contributions dated September 1 or earlier and requires an account created on
April 1 or earlier. Private activity that GitHub does not expose cannot be counted.

Every new, reopened, or updated PR receives an automated reply reporting its
account age and qualifying contribution count. PRs that do not meet either
requirement are automatically closed with a link to this policy. Passing this
check does not imply approval of the code or authorize running PR tests.

## Preparing a pull request

Keep changes focused, explain the problem and resulting behavior, and include
regression coverage for bug fixes. Run the relevant tests before submitting, for
example:

```bash
python -m unittest discover tests/core
python -m unittest discover tests/numba
```

Contributors remain responsible for understanding and validating their changes,
including changes prepared with AI assistance.
