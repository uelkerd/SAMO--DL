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
