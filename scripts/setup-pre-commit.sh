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

# Create a repository health check script
cat > scripts/check-repo-health.sh << 'EOF'
#!/bin/bash
# Repository health check script

echo "🔍 Repository Health Check"
echo "=========================="

echo ""
echo "📊 Repository Size:"
git count-objects -vH

echo ""
echo "📁 Largest Files:"
git ls-files | xargs wc -l 2>/dev/null | sort -nr | head -10

echo ""
echo "🔍 Potential Model Artifacts:"
find . -name "*.pt" -o -name "*.pth" -o -name "*.bin" -o -name "*.safetensors" -o -name "*.onnx" -o -name "*.arrow" -o -name "merges.txt" 2>/dev/null | head -10

echo ""
echo "📈 Recent Commits:"
git log --oneline -5

echo ""
echo "🌿 Current Branch:"
git branch --show-current
EOF

chmod +x scripts/check-repo-health.sh

echo -e "${GREEN}✅ Repository health check script created${NC}"

# Create a branch cleanup script
cat > scripts/cleanup-branches.sh << 'EOF'
#!/bin/bash
# Branch cleanup script

echo "🧹 Branch Cleanup"
echo "================="

echo ""
echo "📋 Local branches:"
git branch

echo ""
echo "🗑️  To clean up merged branches:"
echo "git branch --merged | grep -v '\\*\\|main\\|master' | xargs -n 1 git branch -d"

echo ""
echo "🔍 To check for large files in recent commits:"
echo "git log --name-only --pretty=format: | sort | uniq | xargs wc -l 2>/dev/null | sort -nr | head -10"
EOF

chmod +x scripts/cleanup-branches.sh

echo -e "${GREEN}✅ Branch cleanup script created${NC}"

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