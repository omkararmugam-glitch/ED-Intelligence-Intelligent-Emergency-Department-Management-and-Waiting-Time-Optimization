# ED Intelligence -- developer tasks.
#
# Targets call the project virtualenv's interpreter directly, so nothing needs
# activating. They do need GNU Make and a POSIX shell (Git Bash, WSL, macOS or
# Linux) because `clean` uses rm and find. The README lists a one-line direct
# equivalent for every target, for plain Windows cmd/PowerShell.

PYTHON_BOOTSTRAP ?= py -3.11
VENV := .venv
ifeq ($(OS),Windows_NT)
PY := $(VENV)/Scripts/python.exe
else
PY := $(VENV)/bin/python
endif
CONFIG ?= configs/default.yaml
DATA ?= synthetic
STAGES ?=

.PHONY: setup test test-fast synthetic run stages clean clean-artifacts

## setup: create the virtualenv and install pinned dependencies
setup:
	$(PYTHON_BOOTSTRAP) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -c "import sys; print(\x27Python\x27, sys.version)"

## test: run the whole test suite
test:
	$(PY) -m pytest

## test-fast: skip the tests with a multi-second runtime budget
test-fast:
	$(PY) -m pytest -m "not slow"

## synthetic: regenerate the synthetic dataset only
synthetic:
	$(PY) -m src.pipeline run --config $(CONFIG) --data synthetic --stages load

## run: run the pipeline (override DATA=real, STAGES=load,validate, CONFIG=...)
run:
	$(PY) -m src.pipeline run --config $(CONFIG) --data $(DATA) $(if $(STAGES),--stages $(STAGES),)

## stages: list the pipeline stages and which are implemented
stages:
	$(PY) -m src.pipeline list-stages

## clean: remove generated data, artifacts and caches (keeps data/raw and the venv)
clean: clean-artifacts
	rm -rf data/processed/*.csv data/synthetic/*.csv
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

## clean-artifacts: remove only artifacts/ outputs
clean-artifacts:
	rm -rf artifacts/models/* artifacts/metrics/* artifacts/figures/* artifacts/tables/*
	rm -f artifacts/report.md artifacts/pipeline.log
