.PHONY: catalog validate

catalog:
	python3 scripts/build_knowledgebase.py

validate:
	python3 scripts/validate_repository.py

