#!/bin/bash
# Git Hooks Setup Script
# Configures the repository to use hooks from scripts/ directory

set -euo pipefail

echo "🔧 Configuring Git hooks for SAMO project..."

# Set hooks path to scripts directory
git config core.hooksPath scripts

# Verify configuration
HOOKS_PATH=$(git config --get core.hooksPath)
if [ "$HOOKS_PATH" = "scripts" ]; then
    echo "✅ Git hooks configured successfully"
    echo "   Hooks path: $HOOKS_PATH"
    echo "   Hooks will now run from the scripts/ directory"
else
    echo "❌ Failed to configure Git hooks"
    exit 1
fi

echo ""
echo "📋 Available hooks:"
for hook in scripts/pre-push scripts/post-merge scripts/post-commit scripts/post-checkout; do
    if [ -f "$hook" ]; then
        echo "   - $(basename "$hook")"
    fi
done

echo ""
echo "🎉 Setup complete! Git hooks are now active."