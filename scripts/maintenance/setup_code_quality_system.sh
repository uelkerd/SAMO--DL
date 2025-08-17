#!/bin/bash
"""
Setup Script for SAMO-DL Code Quality Prevention System

This script sets up the comprehensive code quality prevention system:
1. Installs pre-commit hooks
2. Installs required Python packages
3. Configures the system
4. Runs initial quality checks

Usage: ./setup_code_quality_system.sh
"""

set -e  # Exit on any error

echo "🚀 Setting up SAMO-DL Code Quality Prevention System"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: This script must be run from the SAMO-DL project root"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
echo "🐍 Python version: $python_version"

# Install pre-commit
echo "📦 Installing pre-commit..."
if ! command -v pre-commit &> /dev/null; then
    pip install pre-commit
else
    echo "✅ pre-commit already installed"
fi

# Install required Python packages for code quality
echo "📦 Installing code quality packages..."
pip install black ruff isort flynt docformatter

# Install pre-commit hooks
echo "🔧 Installing pre-commit hooks..."
pre-commit install

# Make scripts executable
echo "🔧 Making maintenance scripts executable..."
chmod +x scripts/maintenance/*.py
chmod +x scripts/maintenance/*.sh

# Test the code quality enforcer
echo "🧪 Testing code quality enforcer..."
python scripts/maintenance/code_quality_enforcer.py . --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Code quality enforcer working correctly"
else
    echo "❌ Code quality enforcer test failed"
    exit 1
fi

# Test the auto-fix script
echo "🧪 Testing auto-fix script..."
python scripts/maintenance/auto_fix_code_quality.py --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Auto-fix script working correctly"
else
    echo "❌ Auto-fix script test failed"
    exit 1
fi

# Run initial quality check
echo "🔍 Running initial code quality check..."
python scripts/maintenance/code_quality_enforcer.py . > /tmp/initial_check.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Initial quality check passed"
else
    echo "⚠️  Initial quality check found issues (see /tmp/initial_check.txt)"
    echo "💡 This is expected - the system will prevent these from being committed"
fi

# Create .gitignore entries for quality tools
echo "📝 Updating .gitignore..."
if ! grep -q "# Code quality tools" .gitignore 2>/dev/null; then
    cat >> .gitignore << 'EOF'

# Code quality tools
.pre-commit-config.yaml
.pre-commit-hooks.yaml
EOF
    echo "✅ Added code quality tools to .gitignore"
else
    echo "✅ .gitignore already contains code quality entries"
fi

# Create a quick reference guide
echo "📚 Creating quick reference guide..."
cat > CODE_QUALITY_QUICK_REFERENCE.md << 'EOF'
# Code Quality Quick Reference

## 🚀 **NEW APPROACH: Prevention Instead of Reaction**

This system **prevents** code quality issues from being committed instead of fixing them after the fact.

## 🔧 **Available Tools**

### 1. **Pre-commit Hooks (Automatic)**
- **Black**: Code formatting
- **Ruff**: Fast linting and formatting
- **isort**: Import sorting
- **flynt**: f-string conversion
- **docformatter**: Docstring formatting
- **Custom SAMO-DL enforcer**: Catches all recurring issues

### 2. **Manual Tools**
```bash
# Check code quality
python scripts/maintenance/code_quality_enforcer.py .

# Auto-fix common issues
python scripts/maintenance/auto_fix_code_quality.py .

# Auto-fix with preview (dry-run)
python scripts/maintenance/auto_fix_code_quality.py --dry-run .
```

### 3. **Traditional Tools (Still Available)**
```bash
# Ruff linting
ruff check src

# Black formatting
black src

# isort import sorting
isort src
```

## 🚨 **What Gets Blocked**

The pre-commit hooks will **block commits** if any of these issues are found:
- Unnecessary else/elif after return (PYL-R1705)
- f-strings without expressions (PTC-W0027)
- Unused imports (PY-W2000)
- Continuation line indentation (FLK-E128)
- Missing blank lines (FLK-E301)
- Line length violations (FLK-E501)
- Trailing whitespace (FLK-W291)
- Missing newlines (FLK-W292)
- Blank line whitespace (FLK-W293)
- Doc line length (FLK-W505)
- Missing docstrings (PY-D0003)
- High cyclomatic complexity (PY-R1000)

## 💡 **Best Practices**

1. **Always run pre-commit hooks** (they run automatically)
2. **Use the auto-fix script** for common issues
3. **Fix issues before committing** - don't bypass the system
4. **Run quality checks locally** before pushing

## 🆘 **Troubleshooting**

### Bypass Pre-commit (Emergency Only)
```bash
git commit --no-verify -m "Emergency commit"
```

### Reinstall Hooks
```bash
pre-commit uninstall
pre-commit install
```

### Update Hooks
```bash
pre-commit autoupdate
```

## 📊 **Quality Metrics**

The system tracks:
- Files checked
- Issues found by type
- Auto-fixes applied
- Prevention success rate

## 🎯 **Goals**

- **Zero recurring issues** in DeepSource
- **Consistent code quality** across the project
- **Developer productivity** through automation
- **Professional codebase** appearance
EOF

echo "✅ Quick reference guide created: CODE_QUALITY_QUICK_REFERENCE.md"

# Final setup
echo ""
echo "🎉 SAMO-DL Code Quality Prevention System Setup Complete!"
echo "========================================================"
echo ""
echo "✅ What's been set up:"
echo "   • Pre-commit hooks installed and configured"
echo "   • Code quality tools installed"
echo "   • Maintenance scripts made executable"
echo "   • Initial quality check completed"
echo "   • Quick reference guide created"
echo ""
echo "🚀 Next steps:"
echo "   1. Try making a small change and committing it"
echo "   2. The system will automatically check quality"
echo "   3. If issues are found, fix them before committing"
echo "   4. Use the auto-fix script for common issues"
echo ""
echo "📚 Documentation:"
echo "   • CODE_QUALITY_QUICK_REFERENCE.md - Quick guide"
echo "   • .pre-commit-config.yaml - Hook configuration"
echo "   • scripts/maintenance/ - Quality tools"
echo ""
echo "🔧 Commands to remember:"
echo "   python scripts/maintenance/code_quality_enforcer.py .     # Check quality"
echo "   python scripts/maintenance/auto_fix_code_quality.py .     # Auto-fix issues"
echo "   pre-commit run --all-files                               # Run all hooks"
echo ""
echo "🎯 The system will now PREVENT code quality issues from being committed!"
echo "   No more recurring DeepSource warnings!"
