#!/bin/bash

# ============================================================================
# SAMO Security Pipeline Setup Script
# This script sets up pre-commit hooks and security tools
# ============================================================================

set -euo pipefail

echo "🔒 Setting up SAMO Security Pipeline..."

# Check if we're in the right directory
if [ ! -f "app.py" ] || [ ! -d ".circleci" ]; then
    echo "❌ Error: Please run this script from the SAMO project root directory"
    exit 1
fi

# Install pre-commit hooks
echo "📦 Installing pre-commit hooks..."
if ! command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit..."
    pip install pre-commit
else
    echo "pre-commit already installed"
fi

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install --install-hooks

# Create .markdownlint.yaml configuration
echo "📝 Creating markdownlint configuration..."
cat > .markdownlint.yaml << 'EOF'
# Markdown linting rules
default: true
MD013: false  # Line length
MD033: false  # Allow HTML
MD041: false  # First line in file should be a top level heading
MD024: false  # Multiple headings with the same content
MD026: false  # Trailing punctuation in heading
MD029: false  # Ordered list item prefix
MD007: false  # Unordered list indentation
MD012: false  # Multiple consecutive blank lines
MD025: false  # Single top level heading
MD002: false  # First heading should be a top level heading
MD018: false  # No space after hash on atx style heading
MD036: false  # No emphasis for headings
MD038: false  # Spaces inside code span elements
MD039: false  # Spaces inside link text
MD040: false  # Fenced code blocks should have a language specified
MD046: false  # Code block style
MD047: false  # Files should end with a single newline character
EOF

# Create .ruff.toml configuration
echo "🐍 Creating Ruff configuration..."
cat > .ruff.toml << 'EOF'
# Ruff configuration for SAMO project
target-version = "py312"
line-length = 88
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
    "N",   # pep8-naming
    "Q",   # flake8-quotes
    "SIM", # flake8-simplify
    "TCH", # flake8-type-checking
    "ARG", # flake8-unused-arguments
    "PIE", # flake8-pie
    "T20", # flake8-print
    "PYI", # flake8-pyi
    "PT",  # flake8-pytest-style
    "RSE", # flake8-raise
    "RET", # flake8-return
    "SLF", # flake8-self
    "SLOT", # flake8-slots
    "TID", # flake8-tidy-imports
    "TCH", # flake8-type-checking
    "ARG", # flake8-unused-arguments
    "PIE", # flake8-pie
    "T20", # flake8-print
    "PYI", # flake8-pyi
    "PT",  # flake8-pytest-style
    "RSE", # flake8-raise
    "RET", # flake8-return
    "SLF", # flake8-self
    "SLOT", # flake8-slots
    "TID", # flake8-tidy-imports
]

ignore = [
    "E501",  # Line too long (handled by Black)
    "B008",  # Do not perform function calls in argument defaults
    "C901",  # Function is too complex
    "PLR0913", # Too many arguments to function call
    "PLR0912", # Too many branches
    "PLR0915", # Too many statements
    "PLR0911", # Too many return statements
]

# Exclude patterns
exclude = [
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "build",
    "dist",
    ".eggs",
    "*.egg-info",
    "htmlcov",
    ".tox",
    ".pytest_cache",
    "notebooks",
    "scripts/legacy",
]

# Per-file ignores
[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]
"tests/**/*.py" = ["PLR2004", "S101"]
"scripts/**/*.py" = ["PLR2004", "S101"]
"notebooks/**/*.py" = ["E", "W", "F", "I", "B", "C4", "UP", "N", "Q", "SIM", "TCH", "ARG", "PIE", "T20", "PYI", "PT", "RSE", "RET", "SLF", "SLOT", "TID"]

# Import sorting
[tool.ruff.isort]
known-first-party = ["src", "scripts", "tests"]
known-third-party = ["fastapi", "flask", "torch", "transformers", "numpy", "pandas"]
section-order = ["future", "standard-library", "third-party", "first-party", "local-folder"]
EOF

# Create .mypy.ini configuration
echo "🔍 Creating MyPy configuration..."
cat > .mypy.ini << 'EOF'
[mypy]
python_version = 3.12
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True

# Ignore missing imports for external packages
ignore_missing_imports = True

# Per-module options
[mypy-src.*]
disallow_untyped_defs = False

[mypy-scripts.*]
disallow_untyped_defs = False

[mypy-tests.*]
disallow_untyped_defs = False

[mypy-notebooks.*]
disallow_untyped_defs = False
EOF

# Create .black configuration
echo "⚫ Creating Black configuration..."
cat > pyproject.toml << 'EOF'
[tool.black]
line-length = 88
target-version = ['py312']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
  | notebooks
  | scripts/legacy
)/
'''

[tool.isort]
profile = "black"
line_length = 88
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
known_first_party = ["src", "scripts", "tests"]
known_third_party = ["fastapi", "flask", "torch", "transformers", "numpy", "pandas"]
EOF

# Install development dependencies
echo "📚 Installing development dependencies..."
pip install -r requirements-dev.txt

# Install security tools
echo "🔒 Installing security tools..."
pip install safety bandit semgrep pip-audit

# Test the setup
echo "🧪 Testing the setup..."
echo "Testing pre-commit hooks..."
pre-commit run --all-files || echo "⚠️ Some pre-commit hooks failed (this is normal on first run)"

echo "Testing security tools..."
python -c "import safety, bandit, semgrep; print('✅ Security tools imported successfully')"

echo ""
echo "🎉 Security Pipeline Setup Complete!"
echo ""
echo "📋 What was set up:"
echo "  ✅ Pre-commit hooks installed"
echo "  ✅ Markdown linting configuration"
echo "  ✅ Ruff (Python linter) configuration"
echo "  ✅ MyPy (type checker) configuration"
echo "  ✅ Black (code formatter) configuration"
echo "  ✅ Security tools installed"
echo ""
echo "🚀 Next steps:"
echo "  1. Commit these configuration files"
echo "  2. The pre-commit hooks will run automatically on future commits"
echo "  3. Use 'pre-commit run --all-files' to check all files"
echo "  4. The CircleCI security pipeline will run on PRs and daily"
echo ""
echo "🔧 Available commands:"
echo "  - pre-commit run --all-files    # Check all files"
echo "  - pre-commit run                 # Check staged files"
echo "  - ruff check .                   # Run linter"
echo "  - black .                        # Format code"
echo "  - mypy .                         # Type check"
echo "  - safety check                   # Check dependencies"
echo "  - bandit -r .                    # Security scan"
