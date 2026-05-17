MANIFEST ?= manifests/latest.yaml
DATA_ROOT ?= data/raw

.PHONY: regime-data profile-30d

regime-data:
	@if [ ! -f "$(MANIFEST)" ]; then \
		echo "Missing manifest lockfile: $(MANIFEST)"; \
		echo "Pass MANIFEST=manifests/runs/regime_engine_YYYY-MM-DD.yaml"; \
		exit 2; \
	fi
	python3 scripts/materialize_regime_data.py \
		--manifest "$(MANIFEST)" \
		--local-root "$(DATA_ROOT)"

profile-30d:
	@if [ ! -f "$(MANIFEST)" ]; then \
		echo "Missing manifest lockfile: $(MANIFEST)"; \
		echo "Pass MANIFEST=manifests/runs/regime_engine_YYYY-MM-DD.yaml"; \
		exit 2; \
	fi
	python3 scripts/profile_engine_30d.py \
		--manifest "$(MANIFEST)" \
		--data-root "$(DATA_ROOT)"
