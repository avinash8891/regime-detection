# Regime Data Manifest Lockfiles

This directory is the tracked home for small, non-secret artifact manifests.
The artifacts themselves stay in object storage and materialize into the
gitignored `data/raw/` cache.

- `runs/`: immutable run lockfiles, for example `regime_engine_2026-05-17.yaml`
  Small `.md` pointer docs are allowed for bulk generated manifests that must
  be regenerated from artifact-store metadata instead of tracked in full.
- `latest.yaml`: optional reviewed alias to one immutable run lockfile

Do not place durable manifests under `data/` or `.context/`; those paths are
ignored and will not survive a fresh workspace.
