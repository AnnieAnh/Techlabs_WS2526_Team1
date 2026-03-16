.PHONY: pipeline ingest prepare deduplicate regex-extract extract validate clean-enrich export \
        dry-run list-steps all clean-data analyze test lint type-check

# -----------------------------------------------------------------------
# Orchestrator targets
# -----------------------------------------------------------------------

pipeline:
	poetry run python orchestrate.py

ingest:
	poetry run python orchestrate.py --only ingest

prepare:
	poetry run python orchestrate.py --only prepare

deduplicate:
	poetry run python orchestrate.py --only deduplicate

regex-extract:
	poetry run python orchestrate.py --only regex_extract

extract:
	poetry run python orchestrate.py --only extract

validate:
	poetry run python orchestrate.py --only validate

clean-enrich:
	poetry run python orchestrate.py --only clean_enrich

export:
	poetry run python orchestrate.py --only export

resume-from-%:
	poetry run python orchestrate.py --from $*

dry-run:
	poetry run python orchestrate.py --dry-run

list-steps:
	poetry run python orchestrate.py --list

# -----------------------------------------------------------------------
# Legacy / convenience targets
# -----------------------------------------------------------------------

# Legacy: runs the full pipeline (same as 'pipeline')
all: pipeline

# Legacy: re-run only the clean+export steps
clean-data:
	poetry run python orchestrate.py --from clean_enrich

analyze:
	cd notebooks && poetry run jupyter notebook

# -----------------------------------------------------------------------
# Dev targets
# -----------------------------------------------------------------------

test:
	poetry run pytest tests/ -v

lint:
	poetry run ruff check . --fix

type-check:
	poetry run mypy src/extraction/ src/shared/ src/ingestion/ src/cleaning/ src/analysis/ src/steps/ --ignore-missing-imports
