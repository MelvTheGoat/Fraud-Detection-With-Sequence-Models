PY ?= python
RESULTS ?= results
DATA ?= data/synthetic

.PHONY: help install simulate baseline train experiment report serve bench test lint typecheck all clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the package and dev extras
	$(PY) -m pip install -e ".[serve,explain,dev]"

simulate: ## Generate the synthetic transaction dataset
	$(PY) -m src.simulate --out $(DATA)

baseline: ## Train the LightGBM baseline
	$(PY) -m src.baselines --data $(DATA) --results $(RESULTS)

train: ## Train all sequence models across seeds and imbalance strategies
	$(PY) -m src.train --data $(DATA) --results $(RESULTS)

experiment: ## Full pipeline: simulate -> baseline -> train -> evaluate -> calibrate -> policy -> monitor
	$(PY) -m src.experiment --data $(DATA) --results $(RESULTS)

report: ## Render the README's results tables from results/
	$(PY) -m src.report --results $(RESULTS)

bench: ## Benchmark ONNX Runtime against raw PyTorch
	$(PY) -m src.api.benchmark --results $(RESULTS)

serve: ## Run the scoring API locally
	$(PY) -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000

test: ## Run the test suite (synthetic data only)
	$(PY) -m pytest

lint: ## ruff check
	$(PY) -m ruff check src tests

typecheck: ## mypy
	$(PY) -m mypy src

all: lint typecheck test experiment ## Everything

clean:
	rm -rf $(RESULTS)/* artifacts/* data/synthetic data/processed .pytest_cache .mypy_cache .ruff_cache
