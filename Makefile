# SAMO-DL Makefile - Minimal Code Quality Edition
.PHONY: help format lint test quality-check clean

help: ## Show this help message
	@echo "SAMO-DL Code Quality Commands"
	@echo "============================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

format: ## Format code with ruff
	ruff format src/ tests/ scripts/

lint: ## Run comprehensive linting with ruff and pylint
	ruff check --fix src/ tests/ scripts/
	pylint --rcfile=pyproject.toml src/ tests/ scripts/

test: ## Run tests with coverage
	pytest tests/ --cov=src --cov-report=term-missing

quality-check: ## Run all quality checks (matches pre-commit)
	@echo "Running comprehensive code quality checks..."
	ruff check --diff src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/
	pylint --rcfile=pyproject.toml src/ tests/ scripts/
	mypy src/ scripts/training/ scripts/database/
	bandit -r src/ -c pyproject.toml
	bandit -r tests/ -c pyproject.toml -s B101
	safety scan --json --output safety-report.json
	docformatter --check --wrap-summaries=88 --wrap-descriptions=88 src/ tests/ scripts/

check-scope: ## Check PR scope compliance
	python scripts/check_pr_scope.py --strict

pre-commit-install: ## Install pre-commit hooks
	pre-commit install
	git config commit.template .gitmessage.txt

clean: ## Clean up generated files
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf bandit-report.json safety-report.json
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
