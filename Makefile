PYTHON ?= python3
SITE_DIR ?= site
TIER ?= quick
BACKEND ?= qwen_vllm
BENCH_PYTHON = $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))
GENERATED_PATHS := \
	experiments/*/metadata.yaml \
	knowledge/artifact_index.md \
	knowledge/artifact_manifest_index.csv \
	knowledge/artifact_manifest_index.md \
	knowledge/claims/index.csv \
	knowledge/claims/index.md \
	knowledge/experiment_catalog.csv \
	knowledge/experiment_catalog.md \
	knowledge/experiment_manifest.json \
	knowledge/experiment_readiness.csv \
	knowledge/experiment_readiness.md \
	knowledge/future_experiment_queue.csv \
	knowledge/future_experiment_queue.md \
	knowledge/readme_coverage.md \
	knowledge/research_program_index.csv \
	knowledge/research_program_index.md \
	knowledge/source_tracks.csv \
	knowledge/source_tracks.md \
	knowledge/tag_index.md

.PHONY: parked catalog catalog-test validate active-experiments py-compile check-links check-text check-footguns generated-clean lint site site-check check related new-program new-experiment from-queue site-dates site-content briefs-gate bench bench-validate

catalog:
	$(PYTHON) scripts/build_knowledgebase.py

catalog-test:
	$(PYTHON) -B -m unittest scripts.tests.test_build_knowledgebase

validate:
	$(PYTHON) scripts/validate_repository.py

active-experiments:
	$(PYTHON) -B scripts/list_active_experiments.py

py-compile:
	$(PYTHON) scripts/check_python_syntax.py

check-links:
	$(PYTHON) scripts/check_markdown_links.py

check-text:
	$(PYTHON) scripts/check_repository_text.py

# Ratcheting footgun gate: rules the repo has ALREADY paid for, enforced on CHANGED files so new
# code must be clean while 299 experiments of history stay grandfathered (`--all` audits everything).
check-footguns:
	$(PYTHON) scripts/check_footguns.py

generated-clean:
	git diff --exit-code -- $(GENERATED_PATHS)

lint: py-compile check-links check-text check-footguns

site: catalog
	$(PYTHON) scripts/build_site.py --out $(SITE_DIR)

site-check: site
	$(PYTHON) scripts/check_site.py "$(SITE_DIR)"

# Keep site content current as experiments are added (see docs/site_maintenance.md).
# Dates auto-fill from git for post-import experiments; charts/briefs are agent-authored.
site-dates:
	$(PYTHON) scripts/extract_experiment_dates.py --apply

site-content: site-dates
	$(PYTHON) scripts/site_content_status.py

check: catalog catalog-test generated-clean validate active-experiments lint site-check briefs-gate

# Enforce that every experiment has a plain-language practitioner brief for the
# site. Fails the check until briefs are authored — see docs/site_maintenance.md.
briefs-gate:
	$(PYTHON) scripts/site_content_status.py --strict

bench:
	cd benchmarks/menagerie && PYTHONDONTWRITEBYTECODE=1 $(BENCH_PYTHON) run.py --tier $(TIER) --backend $(BACKEND)

bench-validate:
	cd benchmarks/menagerie && PYTHONDONTWRITEBYTECODE=1 $(BENCH_PYTHON) validate_suite.py

related:
	$(PYTHON) scripts/find_related.py "$(QUERY)" $(EXTRA_ARGS)

# Match a NEWLY FOUND bottleneck against the corpus's own unrun follow-ups (claim next_tests/avoid,
# future queue, open questions, docs). `related` matches ideas against past EXPERIMENTS; this matches
# findings against PARKED work -- the gap that let a session re-derive C59's layer-looping next-test
# from scratch while never surfacing C62's termination intervention.
parked:
	$(PYTHON) scripts/mine_parked.py "$(QUERY)"

new-program:
	$(PYTHON) scripts/scaffold_research_program.py "$(PROGRAM)" --title "$(TITLE)" --focus "$(FOCUS)" $(EXTRA_ARGS)

new-experiment:
	$(PYTHON) scripts/scaffold_experiment.py "$(EXPERIMENT)" --program "$(PROGRAM)" --title "$(TITLE)" --summary "$(SUMMARY)" $(EXTRA_ARGS)

from-queue:
	$(PYTHON) scripts/scaffold_from_queue.py "$(PROPOSAL)" $(EXTRA_ARGS)
