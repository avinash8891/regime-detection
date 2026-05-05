# Review and CI Setup

This repository uses three review/check layers:

1. GitHub CI for deterministic checks.
2. Cubic GitHub PR review configured by `cubic.yaml`.
3. Codex PR review through GitHub Actions using the installed plugin prompts.

## CI

Workflow:

```text
.github/workflows/ci.yml
```

Runs on pushes to `main`/`v1-of-regime-detection` and on pull requests.

Checks:

- git whitespace validation against the PR/base diff;
- shell syntax for review scripts;
- pytest when a `tests/` directory exists.

## Cubic PR Review

Config:

```text
cubic.yaml
```

This config is consumed by Cubic's GitHub App. The app must be installed for `avinash8891/regime-detection`; once installed, Cubic reviews new PRs automatically using this repo-level config.

The config emphasizes:

- V1 no future-data leakage;
- no non-trading-date rollback;
- V1/V2 scope boundaries;
- fixture provenance.

Local Cubic pre-push review remains available through:

```text
scripts/cubic_review.sh
```

## Codex PR Review

Workflow:

```text
.github/workflows/codex-review.yml
```

Required GitHub secret:

```text
OPENAI_API_KEY
```

Optional GitHub variable:

```text
CODEX_REVIEW_MODEL
```

Default model:

```text
gpt-5.4
```

The workflow installs `@openai/codex`, checks out `anthropics/claude-plugins-official`, then runs:

```text
scripts/codex_code_simplifier.sh
scripts/codex_pr_review_toolkit.sh
```

The scripts load the plugin prompt files from the checked-out plugin repository:

```text
code-simplifier/1.0.0/agents/code-simplifier.md
pr-review-toolkit/local/agents/*.md
```

If `OPENAI_API_KEY` is unavailable, the workflow logs a skip message instead of failing infrastructure-only.
