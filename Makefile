# SAMO-DL Makefile - Minimal Code Quality Edition
.PHONY: help format lint test quality-check clean

help: ## Show this help message
	@echo "SAMO-DL Code Quality Commands"
	@echo "============================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

format: ## Format code with black and isort
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

lint: ## Run basic linting (flake8, pylint)
	flake8 src/ tests/ scripts/
	pylint src/ tests/ scripts/ || true

test: ## Run tests with coverage
	pytest tests/ --cov=src --cov-report=term-missing

quality-check: ## Run all quality checks
	@echo "Running code quality checks..."
	black --check src/ tests/ scripts/
	isort --check-only src/ tests/ scripts/
	flake8 src/ tests/ scripts/
	pylint src/ tests/ scripts/ || true
	bandit -r src/ -s B101,B601
	safety check --json --output safety-report.json || true

clean: ## Clean up generated files
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf bandit-report.json safety-report.json
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
