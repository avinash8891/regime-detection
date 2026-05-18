MANIFEST ?=
DATA_ROOT ?= data/raw
OPERATOR_ENV_FILE ?=
OPERATOR_ENV_ARG = $(if $(OPERATOR_ENV_FILE),--operator-env-file "$(OPERATOR_ENV_FILE)",)

.PHONY: regime-data profile-30d

regime-data:
	@if [ ! -f "$(MANIFEST)" ]; then \
		echo "Missing manifest lockfile: $(MANIFEST)"; \
		echo "Pass MANIFEST=manifests/runs/<reviewed-run-lockfile>.yaml"; \
		exit 2; \
	fi
	python3 scripts/materialize_regime_data.py \
		--manifest "$(MANIFEST)" \
		--local-root "$(DATA_ROOT)" $(OPERATOR_ENV_ARG)

profile-30d:
	@if [ ! -f "$(MANIFEST)" ]; then \
		echo "Missing manifest lockfile: $(MANIFEST)"; \
		echo "Pass MANIFEST=manifests/runs/<reviewed-run-lockfile>.yaml"; \
		exit 2; \
	fi
	python3 scripts/profile_engine_30d.py \
		--manifest "$(MANIFEST)" \
		--data-root "$(DATA_ROOT)" $(OPERATOR_ENV_ARG)
