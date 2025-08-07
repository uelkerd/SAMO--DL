#!/bin/bash
# Setup script to install pre-commit hooks for repository bloat prevention
# This helps prevent future issues like the one we just resolved

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Setting up pre-commit hooks for repository bloat prevention...${NC}"

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Not in a git repository. Please run this script from the project root.${NC}"
    exit 1
fi

# Create .git/hooks directory if it doesn't exist
mkdir -p .git/hooks

# Copy the pre-commit hook
if [ -f "scripts/pre-commit-hook.sh" ]; then
    cp scripts/pre-commit-hook.sh .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit
    echo -e "${GREEN}✅ Pre-commit hook installed successfully${NC}"
else
    echo -e "${YELLOW}⚠️  Pre-commit hook script not found at scripts/pre-commit-hook.sh${NC}"
    exit 1
fi

# Create a simple post-commit hook to show repository size
cat > .git/hooks/post-commit << 'EOF'
#!/bin/bash
# Post-commit hook to show repository size information

echo ""
echo "📊 Repository size after commit:"
git count-objects -vH | grep -E "(size|pack)"
echo ""
echo "💡 To check for large files: git ls-files | xargs wc -l | sort -nr | head -10"
EOF

chmod +x .git/hooks/post-commit

# Check if health check script exists
if [ -f "scripts/check-repo-health.sh" ]; then
    echo -e "${GREEN}✅ Repository health check script found${NC}"
else
    echo -e "${YELLOW}⚠️  Repository health check script not found at scripts/check-repo-health.sh${NC}"
fi

# Check if branch cleanup script exists
if [ -f "scripts/cleanup-branches.sh" ]; then
    echo -e "${GREEN}✅ Branch cleanup script found${NC}"
else
    echo -e "${YELLOW}⚠️  Branch cleanup script not found at scripts/cleanup-branches.sh${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Pre-commit setup complete!${NC}"
echo ""
echo "📋 What was installed:"
echo "  ✅ Pre-commit hook (prevents large file commits)"
echo "  ✅ Post-commit hook (shows repository size)"
echo "  ✅ Repository health check script"
echo "  ✅ Branch cleanup script"
echo ""
echo "🔧 Usage:"
echo "  • Pre-commit hook runs automatically on every commit"
echo "  • Check repository health: ./scripts/check-repo-health.sh"
echo "  • Clean up branches: ./scripts/cleanup-branches.sh"
echo ""
echo "💡 Tips:"
echo "  • The pre-commit hook will block commits with files >1MB"
echo "  • Model artifacts (*.pt, *.pth, etc.) are automatically blocked"
echo "  • Run health checks regularly to monitor repository size"
echo ""
echo -e "${BLUE}🚀 Your repository is now protected against bloat!${NC}" 