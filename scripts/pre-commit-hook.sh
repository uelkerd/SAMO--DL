#!/bin/bash
# Pre-commit hook to prevent large files and model artifacts from being committed
# This helps prevent repository bloat issues like the one we just resolved

set -e

# Configuration
MAX_SIZE=1048576  # 1MB in bytes
LARGE_FILE_WARNING_SIZE=524288  # 512KB warning threshold

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo "🔍 Running pre-commit checks..."

# Check for large files
large_files_found=false
while IFS= read -r file; do
    if [ -f "$file" ]; then
        # Get file size (portable method)
        size=$(wc -c < "$file" | tr -d '[:space:]')

        if [ "$size" -gt "$MAX_SIZE" ]; then
            human_size=$(command -v numfmt >/dev/null && numfmt --to=iec "$size" || echo "${size}B")
            echo -e "${RED}❌ ERROR: $file is larger than 1MB ($human_size)${NC}"
            echo "   This file should not be committed to version control."
            echo "   Consider adding it to .gitignore if it's a model artifact or data file."
            large_files_found=true
        elif [ "$size" -gt "$LARGE_FILE_WARNING_SIZE" ]; then
            human_size=$(command -v numfmt >/dev/null && numfmt --to=iec "$size" || echo "${size}B")
            echo -e "${YELLOW}⚠️  WARNING: $file is larger than 512KB ($human_size)${NC}"
            echo "   Consider if this file should be in version control."
        fi
    fi
done < <(git diff --cached --name-only)

# Check for model artifacts and other problematic files
model_artifacts_found=false
while IFS= read -r file; do
    case "$file" in
        *.pt|*.pth|*.bin|*.safetensors|*.onnx|*.arrow|merges.txt|*.pkl|*.pickle|*.h5|*.hdf5)
            echo -e "${RED}❌ ERROR: $file is a model artifact and should not be committed${NC}"
            echo "   Model files should be excluded from version control."
            echo "   Add this file pattern to .gitignore"
            model_artifacts_found=true
            ;;
        *.log|logs/*|*.tmp|*.temp)
            echo -e "${YELLOW}⚠️  WARNING: $file appears to be a log or temporary file${NC}"
            echo "   Consider if this should be in version control."
            ;;
    esac
done < <(git diff --cached --name-only)

# Check for files in model directories
while IFS= read -r file; do
    if [[ "$file" == models/* ]] || [[ "$file" == deployment/model*/* ]]; then
        echo -e "${YELLOW}⚠️  WARNING: $file is in a model directory${NC}"
        echo "   Ensure this is not a large model file that should be excluded."
    fi
done < <(git diff --cached --name-only)

# Summary
if [ "$large_files_found" = true ] || [ "$model_artifacts_found" = true ]; then
    echo ""
    echo -e "${RED}🚫 Commit blocked due to large files or model artifacts${NC}"
    echo "Please fix the issues above before committing."
    echo ""
    echo "Quick fixes:"
    echo "1. Remove large files: git reset HEAD <file>"
    echo "2. Add to .gitignore: echo 'pattern' >> .gitignore"
    echo "3. Use git-lfs for large files that must be tracked"
    exit 1
else
    echo -e "${GREEN}✅ Pre-commit checks passed${NC}"
    echo "No large files or model artifacts detected."
fi

echo ""
echo "💡 Tips to prevent repository bloat:"
echo "- Keep branches focused on single features"
echo "- Regularly check repository size: git count-objects -vH"
echo "- Use .gitignore to exclude model artifacts"
echo "- Consider git-lfs for large files that must be tracked"
