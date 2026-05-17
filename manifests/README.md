# Regime Data Manifest Lockfiles

This directory is the tracked home for small, non-secret artifact manifests.
The artifacts themselves stay in object storage and materialize into the
gitignored `data/raw/` cache.

- `runs/`: immutable run lockfiles, for example `regime_engine_2026-05-17.yaml`
- `latest.yaml`: optional reviewed alias to one immutable run lockfile

Do not place durable manifests under `data/` or `.context/`; those paths are
ignored and will not survive a fresh workspace.
