#!/bin/bash

# SAMO-DL Code Quality Prevention System Setup
# This script sets up the complete code quality infrastructure to prevent
# ALL recurring DeepSource issues from ever happening again.

set -e  # Exit on any error

echo "🚀 Setting up SAMO-DL Code Quality Prevention System..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] && [ ! -f "requirements.txt" ]; then
    print_error "This script must be run from the SAMO-DL project root directory"
    exit 1
fi

print_status "Current directory: $(pwd)"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
print_status "Python version: $PYTHON_VERSION"

if [ "$(echo "$PYTHON_VERSION >= 3.8" | bc -l 2>/dev/null || echo "0")" -eq 0 ]; then
    print_warning "Python 3.8+ is recommended for optimal performance"
fi

# Install pre-commit if not already installed
if ! command -v pre-commit &> /dev/null; then
    print_status "Installing pre-commit..."
    pip install pre-commit
    print_success "pre-commit installed successfully"
else
    print_status "pre-commit is already installed"
fi

# Install additional quality tools
print_status "Installing code quality tools..."

# Core formatting and linting tools
pip install black isort ruff mypy

# Security scanning tools
pip install bandit safety

# Documentation and string formatting tools
pip install docformatter flynt

print_success "All code quality tools installed successfully"

# Install pre-commit hooks
print_status "Installing pre-commit hooks..."
pre-commit install
print_success "Pre-commit hooks installed successfully"

# Install additional hooks for different stages
print_status "Installing additional pre-commit hooks..."
pre-commit install --hook-type commit-msg
pre-commit install --hook-type pre-push
print_success "Additional hooks installed successfully"

# Verify the setup
print_status "Verifying the setup..."

# Check if .pre-commit-config.yaml exists
if [ ! -f ".pre-commit-config.yaml" ]; then
    print_error ".pre-commit-config.yaml not found!"
    exit 1
fi

# Check if custom scripts exist
if [ ! -f "scripts/maintenance/code_quality_enforcer.py" ]; then
    print_error "code_quality_enforcer.py not found!"
    exit 1
fi

if [ ! -f "scripts/maintenance/auto_fix_code_quality.py" ]; then
    print_error "auto_fix_code_quality.py not found!"
    exit 1
fi

print_success "All required files found"

# Test the pre-commit configuration
print_status "Testing pre-commit configuration..."
if pre-commit run --all-files --hook-stage manual; then
    print_success "Pre-commit configuration is valid"
else
    print_warning "Some pre-commit hooks failed (this is normal for the first run)"
fi

# Create a test to verify the system works
print_status "Creating a test file to verify the system..."
cat > test_code_quality.py << 'EOF'
#!/usr/bin/env python3
"""Test file for code quality system."""

import os
import sys
from typing import List, Dict, Optional

def test_function_with_long_line_that_should_be_broken_down_into_multiple_lines_because_it_exceeds_the_eighty_eight_character_limit():
    """This function has a very long name that should trigger line length warnings."""
    return "test"

def test_function_with_trailing_whitespace():    
    """This function has trailing whitespace."""
    return "test"

def test_function_without_docstring():
    return "test"

if __name__ == "__main__":
    print("Testing code quality system")
EOF

print_success "Test file created"

# Test the auto-fix system
print_status "Testing auto-fix system..."
python scripts/maintenance/auto_fix_code_quality.py . --dry-run
print_success "Auto-fix system test completed"

# Clean up test file
rm test_code_quality.py
print_status "Test file cleaned up"

# Create a quick reference guide
print_status "Creating quick reference guide..."
cat > CODE_QUALITY_QUICK_REFERENCE.md << 'EOF'
# SAMO-DL Code Quality Quick Reference

## 🚀 Quick Start

### Manual Quality Check
```bash
# Check all files for quality issues
python scripts/maintenance/code_quality_enforcer.py .

# Auto-fix common issues (dry run first)
python scripts/maintenance/auto_fix_code_quality.py . --dry-run

# Apply fixes
python scripts/maintenance/auto_fix_code_quality.py .
```

### Pre-commit Hooks (Automatic)
The system automatically runs on every commit. To run manually:
```bash
# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
pre-commit run ruff --all-files
```

## 🛠️ Available Tools

### Formatting
- **Black**: Code formatting (88 character line length)
- **isort**: Import sorting and organization
- **docformatter**: Docstring formatting

### Linting
- **Ruff**: Fast Python linting
- **MyPy**: Type checking
- **Custom Enforcer**: SAMO-DL specific rules

### Security
- **Bandit**: Security vulnerability scanning
- **Safety**: Dependency vulnerability scanning

### Auto-fixing
- **Custom Auto-fixer**: Automatically fixes common issues
- **flynt**: Converts .format() to f-strings

## 📋 Quality Rules Enforced

### Critical (Blocking)
- FLK-W291: Trailing whitespace
- FLK-W292: Missing newlines at end of file
- FLK-W293: Blank line whitespace
- FLK-E501: Line length violations
- FLK-E128: Continuation line indentation
- FLK-E301: Missing blank lines

### Warnings
- PYL-R1705: Unnecessary else/elif after return
- PTC-W0027: f-strings without expressions
- PY-W2000: Unused imports
- FLK-W505: Doc line length
- PY-D0003: Missing docstrings
- PY-R1000: High cyclomatic complexity

## 🔧 Configuration Files

- `.pre-commit-config.yaml`: Pre-commit hooks configuration
- `pyproject.toml`: Tool configurations (Black, Ruff, MyPy)
- `scripts/maintenance/`: Custom quality enforcement scripts

## 🚨 Troubleshooting

### Pre-commit hooks not running
```bash
# Reinstall hooks
pre-commit install

# Clear cache
pre-commit clean
```

### Specific tool issues
```bash
# Update all tools
pip install --upgrade black isort ruff mypy bandit safety docformatter flynt

# Check tool versions
black --version
ruff --version
mypy --version
```

## 📚 Best Practices

1. **Commit often** - Pre-commit hooks catch issues early
2. **Run checks locally** - Use `python scripts/maintenance/code_quality_enforcer.py .`
3. **Auto-fix first** - Use auto-fixer before manual fixes
4. **Check before push** - Run `pre-commit run --all-files`

## 🎯 Success Metrics

- ✅ Zero recurring DeepSource warnings
- ✅ Consistent code formatting across the project
- ✅ Automated quality enforcement
- ✅ Reduced manual code review time
EOF

print_success "Quick reference guide created"

# Final verification
print_status "Running final verification..."

# Check if hooks are properly installed
if [ -d ".git/hooks" ]; then
    if [ -f ".git/hooks/pre-commit" ]; then
        print_success "Pre-commit hook is properly installed"
    else
        print_error "Pre-commit hook is not properly installed"
        exit 1
    fi
else
    print_error ".git directory not found. Are you in a git repository?"
    exit 1
fi

# Summary
echo ""
echo "🎉 SAMO-DL Code Quality Prevention System Setup Complete!"
echo "========================================================"
echo ""
echo "✅ What was installed:"
echo "   - pre-commit framework"
echo "   - Black (code formatter)"
echo "   - isort (import sorter)"
echo "   - Ruff (fast linter)"
echo "   - MyPy (type checker)"
echo "   - Bandit (security scanner)"
echo "   - Safety (dependency scanner)"
echo "   - docformatter (docstring formatter)"
echo "   - flynt (f-string converter)"
echo "   - Custom SAMO-DL quality enforcer"
echo "   - Custom SAMO-DL auto-fixer"
echo ""
echo "✅ What was configured:"
echo "   - Pre-commit hooks for automatic quality enforcement"
echo "   - Custom quality rules for SAMO-DL"
echo "   - Auto-fixing capabilities for common issues"
echo ""
echo "✅ How to use:"
echo "   - Quality checks run automatically on every commit"
echo "   - Manual checks: python scripts/maintenance/code_quality_enforcer.py ."
echo "   - Auto-fixes: python scripts/maintenance/auto_fix_code_quality.py ."
echo "   - Quick reference: CODE_QUALITY_QUICK_REFERENCE.md"
echo ""
echo "🚀 Your codebase is now protected from recurring quality issues!"
echo ""
echo "Next steps:"
echo "1. Make a test commit to verify the system works"
echo "2. Review the quick reference guide"
echo "3. Run quality checks on existing code: python scripts/maintenance/code_quality_enforcer.py ."
echo "4. Apply auto-fixes: python scripts/maintenance/auto_fix_code_quality.py ."
echo ""

print_success "Setup completed successfully!"
