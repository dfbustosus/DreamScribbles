PYTHON ?= python
PIP ?= pip
PACKAGE = dreamscribbles

.PHONY: install fmt fmt-check lint type test run download-models

install:
	$(PIP) install -U pip
	$(PIP) install -e .[dev]

fmt:
	ruff check --fix .
	black .
	isort .

fmt-check:
	ruff check .
	black --check .
	isort --check-only .

lint: fmt-check
	ruff check .

type:
	mypy src/$(PACKAGE)

test:
	pytest -q

run:
	$(PYTHON) -m $(PACKAGE)

download-models:
	$(PYTHON) scripts/download_models.py
