.PHONY: pipeline ingest prepare deduplicate regex-extract extract validate clean-enrich export \
        dry-run list-steps all clean-data analyze test lint type-check

# -----------------------------------------------------------------------
# Orchestrator targets
# -----------------------------------------------------------------------

pipeline:
	python orchestrate.py

ingest:
	python orchestrate.py --only ingest

prepare:
	python orchestrate.py --only prepare

deduplicate:
	python orchestrate.py --only deduplicate

regex-extract:
	python orchestrate.py --only regex_extract

extract:
	python orchestrate.py --only extract

validate:
	python orchestrate.py --only validate

clean-enrich:
	python orchestrate.py --only clean_enrich

export:
	python orchestrate.py --only export

resume-from-%:
	python orchestrate.py --from $*

dry-run:
	python orchestrate.py --dry-run

list-steps:
	python orchestrate.py --list

# -----------------------------------------------------------------------
# Legacy / convenience targets
# -----------------------------------------------------------------------

# Legacy: runs the full pipeline (same as 'pipeline')
all: pipeline

# Legacy: re-run only the clean+export steps
clean-data:
	python orchestrate.py --from clean_enrich

analyze:
	cd notebooks && jupyter notebook

# -----------------------------------------------------------------------
# Dev targets
# -----------------------------------------------------------------------

test:
	poetry run pytest tests/ -v

lint:
	poetry run ruff check . --fix

type-check:
	poetry run mypy src/extraction/ src/shared/ --ignore-missing-imports
