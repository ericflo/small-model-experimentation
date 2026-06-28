.PHONY: catalog validate check new-program new-experiment

catalog:
	python3 scripts/build_knowledgebase.py

validate:
	python3 scripts/validate_repository.py

check: catalog validate

new-program:
	python3 scripts/scaffold_research_program.py "$(PROGRAM)" --title "$(TITLE)" --focus "$(FOCUS)" $(EXTRA_ARGS)

new-experiment:
	python3 scripts/scaffold_experiment.py "$(EXPERIMENT)" --program "$(PROGRAM)" --title "$(TITLE)" --summary "$(SUMMARY)" $(EXTRA_ARGS)
