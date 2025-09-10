# SAMO-DL Makefile
# Provides convenient commands for development and deployment

.PHONY: help install install-dev test lint format quality-check clean setup

help: ## Show this help message
	@echo "SAMO-DL Development Commands"
	@echo "============================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e .[dev]

setup: ## Set up development environment
	python scripts/setup_dev_environment.py

test: ## Run tests with coverage
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

test-unit: ## Run unit tests only
	pytest tests/ -m unit

test-integration: ## Run integration tests only
	pytest tests/ -m integration

lint: ## Run all linting tools
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	pylint src/ tests/

format: ## Format code with black and isort
	black src/ tests/
	isort src/ tests/

quality-check: ## Run comprehensive quality checks
	python scripts/run_quality_checks.py

security: ## Run security checks
	bandit -r src/ -f json -o bandit-report.json
	safety check --json --output safety-report.json

type-check: ## Run type checking
	mypy src/

clean: ## Clean up generated files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf bandit-report.json
	rm -rf safety-report.json
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

pre-commit: ## Run pre-commit on all files
	pre-commit run --all-files

ci: ## Run CI pipeline locally
	$(MAKE) clean
	$(MAKE) install-dev
	$(MAKE) quality-check
	$(MAKE) test

run-api: ## Run the API server
	python src/unified_api_server.py

run-dev: ## Run development server with auto-reload
	FLASK_ENV=development python src/unified_api_server.py
