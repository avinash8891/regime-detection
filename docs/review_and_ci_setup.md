# Review and CI Setup

This repository uses two PR review layers:

1. Cubic GitHub PR review configured by `cubic.yaml`.
2. Codex PR review through the official Codex GitHub integration.

There are no repository-owned GitHub Actions workflows. Codex review triggers are configured in Codex settings, and Cubic review triggers are configured by the Cubic GitHub App.

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

Codex GitHub review is not run by a repository GitHub Actions workflow and does not use an `OPENAI_API_KEY` secret. It is enabled outside the repo through Codex cloud and Codex code review settings.

Required setup:

1. Connect this repository in Codex cloud.
2. Open Codex code review settings.
3. Turn on Code review for `avinash8891/regime-detection`.
4. Optionally turn on Automatic reviews if every PR should be reviewed without a manual trigger.

Manual trigger in a pull request comment:

```text
@codex review
```

Codex reads repository guidance from `AGENTS.md`. This repo keeps Codex-specific review focus under the `Review guidelines` section there.

Local Codex review scripts remain available for pre-push or manual local use:

```text
scripts/codex_code_simplifier.sh
scripts/codex_pr_review_toolkit.sh
```

Those scripts use the local Codex CLI login from `~/.codex/auth.json`. They may be pinned to a model with:

```text
CODEX_REVIEW_MODEL=gpt-5.2
```
