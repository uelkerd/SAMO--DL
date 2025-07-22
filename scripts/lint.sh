#!/bin/bash
# SAMO Deep Learning - Code Quality Maintenance Script
# Usage: ./scripts/lint.sh [command]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}🔧 SAMO Deep Learning - Code Quality Tools${NC}"
echo -e "${BLUE}===========================================${NC}"

# Function to check if ruff is installed
check_ruff() {
    if ! command -v ruff &> /dev/null; then
        echo -e "${RED}❌ Ruff not found. Please install it first:${NC}"
        echo -e "${YELLOW}   conda activate samo-dl && conda install ruff${NC}"
        echo -e "${YELLOW}   OR: pip install ruff${NC}"
        exit 1
    fi
}

# Function to run ruff check
run_check() {
    echo -e "${BLUE}📋 Running Ruff linter check...${NC}"
    check_ruff
    
    if ruff check .; then
        echo -e "${GREEN}✅ All linting checks passed!${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  Linting issues found (see above)${NC}"
        return 1
    fi
}

# Function to run ruff format check
run_format_check() {
    echo -e "${BLUE}🎨 Checking code formatting...${NC}"
    check_ruff
    
    if ruff format --check .; then
        echo -e "${GREEN}✅ Code formatting is correct!${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  Code formatting issues found${NC}"
        return 1
    fi
}

# Function to fix issues automatically
run_fix() {
    echo -e "${BLUE}🔧 Auto-fixing linting issues...${NC}"
    check_ruff
    
    echo -e "${YELLOW}Fixing auto-fixable issues...${NC}"
    ruff check --fix .
    
    echo -e "${YELLOW}Formatting code...${NC}"
    ruff format .
    
    echo -e "${GREEN}✅ Auto-fix complete! Please review changes.${NC}"
}

# Function to run full quality check
run_full_check() {
    echo -e "${BLUE}🚀 Running comprehensive code quality check...${NC}"
    
    local all_passed=true
    
    # Linting check
    if ! run_check; then
        all_passed=false
    fi
    
    echo ""
    
    # Format check
    if ! run_format_check; then
        all_passed=false
    fi
    
    echo ""
    echo -e "${BLUE}📊 Summary:${NC}"
    if $all_passed; then
        echo -e "${GREEN}✅ All quality checks passed! Ready for commit.${NC}"
        return 0
    else
        echo -e "${RED}❌ Quality issues found. Run './scripts/lint.sh fix' to auto-fix.${NC}"
        return 1
    fi
}

# Function to show statistics
show_stats() {
    echo -e "${BLUE}📈 Code quality statistics:${NC}"
    check_ruff
    
    echo -e "${YELLOW}File coverage:${NC}"
    find . -name "*.py" -not -path "./.venv/*" -not -path "./node_modules/*" | wc -l | xargs echo "Python files:"
    
    echo -e "${YELLOW}Ruff configuration:${NC}"
    echo "Configuration file: pyproject.toml"
    echo "Target Python version: 3.10"
    echo "Line length: 88"
    
    echo -e "${YELLOW}Running quick analysis...${NC}"
    ruff check --statistics . || true
}

# Function to show help
show_help() {
    echo -e "${BLUE}Available commands:${NC}"
    echo -e "${GREEN}  check${NC}       - Run linting checks only"
    echo -e "${GREEN}  format-check${NC} - Check code formatting only"  
    echo -e "${GREEN}  fix${NC}         - Auto-fix issues and format code"
    echo -e "${GREEN}  full${NC}        - Run complete quality check (default)"
    echo -e "${GREEN}  stats${NC}       - Show code quality statistics"
    echo -e "${GREEN}  help${NC}        - Show this help message"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo -e "${YELLOW}  ./scripts/lint.sh${NC}           # Run full check"
    echo -e "${YELLOW}  ./scripts/lint.sh fix${NC}       # Auto-fix issues"
    echo -e "${YELLOW}  ./scripts/lint.sh check${NC}     # Quick lint check"
    echo ""
    echo -e "${BLUE}Integration with editors:${NC}"
    echo -e "${YELLOW}  VS Code:${NC} Install the Ruff extension"
    echo -e "${YELLOW}  PyCharm:${NC} Configure external tool for ruff"
    echo -e "${YELLOW}  Vim/Neovim:${NC} Use ruff-lsp or ALE"
}

# Main command processing
case "${1:-full}" in
    "check")
        run_check
        ;;
    "format-check")
        run_format_check
        ;;
    "fix")
        run_fix
        ;;
    "full")
        run_full_check
        ;;
    "stats")
        show_stats
        ;;
    "help" | "-h" | "--help")
        show_help
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac 