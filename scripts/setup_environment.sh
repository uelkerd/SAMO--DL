#!/bin/bash

# SAMO Deep Learning Environment Setup Script
# This script sets up the complete development environment for the SAMO project

set -e  # Exit on any error

# Resolve repository root path once for consistent path resolution
resolve_repo_path() {
    local script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    echo "$(cd "$script_dir/.." && pwd)"
}

# Cache the repo path for use throughout the script
REPO_ROOT="$(resolve_repo_path)"

echo "🚀 Setting up SAMO Deep Learning Environment..."

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

# Check if conda is available
check_conda() {
    print_status "Checking conda installation..."
    
    # First try to find conda in PATH
    if command -v conda >/dev/null 2>&1; then
        CONDA_PATH="$(command -v conda)"
    else
        # Try different conda locations
        CONDA_PATHS=(
            "/opt/homebrew/anaconda3/bin/conda"
            "/usr/local/anaconda3/bin/conda"
            "/opt/anaconda3/bin/conda"
            "$HOME/anaconda3/bin/conda"
            "$HOME/miniconda3/bin/conda"
            "$HOME/miniforge3/bin/conda"
        )
        
        CONDA_PATH=""
        for path in "${CONDA_PATHS[@]}"; do
            if [ -f "$path" ]; then
                CONDA_PATH="$path"
                break
            fi
        done
    fi
    
    if [ -z "$CONDA_PATH" ]; then
        print_error "Conda not found. Please install Anaconda or Miniconda first."
        print_status "Download from: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    
    print_success "Found conda at: ${CONDA_PATH}"
    export PATH="$(dirname "$CONDA_PATH"):$PATH"
}

# Initialize conda
init_conda() {
    print_status "Initializing conda..."
    
    # Source conda initialization
    CONDA_BASE=$(dirname "$(dirname "$CONDA_PATH")")
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    
    if [ $? -eq 0 ]; then
        print_success "Conda initialized successfully"
    else
        print_error "Failed to initialize conda"
        exit 1
    fi
}

# Create or update environment
setup_environment() {
    print_status "Setting up conda environment 'samo-dl'..."
    
    # Check if environment exists
    local env_file="$REPO_ROOT/environment.yml"
    if [ ! -f "$env_file" ]; then
        print_error "Environment file not found at: $env_file"
        exit 1
    fi
    
    if conda env list | grep -q "samo-dl"; then
        print_warning "Environment 'samo-dl' already exists. Updating..."
        conda env update -f "$env_file"
    else
        print_status "Creating new environment 'samo-dl'..."
        conda env create -f "$env_file"
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Environment setup completed"
        echo "📄 Environment file used: environment.yml"
        echo "🏷️  Environment name: samo-dl-stable"
        echo "ℹ️  For development use: environment.dev.yml → samo-dl-dev"
        echo "ℹ️  For ML training use: environment.ml.yml → samo-dl-ml"
    else
        print_error "Failed to setup environment"
        exit 1
    fi
}

# Activate environment and install additional dependencies
activate_and_setup() {
    print_status "Activating environment and installing additional dependencies..."
    
    conda activate samo-dl
    
    # Install additional pip packages
    pip install --upgrade pip
    
    # Resolve paths for constraints and requirements files
    local constraints_file="$REPO_ROOT/dependencies/constraints.txt"
    local repo_requirements="$REPO_ROOT/requirements.txt"
    local local_requirements="requirements.txt"
    
    # Check constraints file exists
    if [ ! -f "$constraints_file" ]; then
        print_warning "Constraints file not found at: $constraints_file"
        print_warning "Installing without constraints (not recommended)"
        constraints_file=""
    else
        print_status "Using constraints from: $constraints_file"
    fi
    
    # Determine which requirements file to use with fallback logic
    local requirements_file=""
    local requirements_source=""
    
    if [ -f "$local_requirements" ]; then
        requirements_file="$local_requirements"
        requirements_source="local requirements.txt"
    elif [ -f "$repo_requirements" ]; then
        requirements_file="$repo_requirements"
        requirements_source="canonical requirements.txt from repo root"
    else
        print_warning "No requirements.txt found (checked local and repo root)"
        print_warning "Skipping additional package installation"
        return
    fi
    
    print_status "Installing packages from: $requirements_source"
    
    # Build pip install command with constraints if available
    local pip_cmd="pip install"
    if [ -n "$constraints_file" ]; then
        pip_cmd="$pip_cmd -c \"$constraints_file\""
    fi
    pip_cmd="$pip_cmd -r \"$requirements_file\""
    
    # Execute the installation
    eval "$pip_cmd" || print_warning "Failed to install some packages from $requirements_source"
    
    # Install pre-commit hooks
    print_status "Setting up pre-commit hooks..."
    pre-commit install
    
    print_success "Environment activation completed"
}

# Test the environment
test_environment() {
    print_status "Testing environment setup..."
    
    # Test Python version
    python_version=$(python --version 2>&1)
    print_status "Python version: ${python_version}"
    
    # Test key imports
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
    python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
    
    print_success "Environment test completed successfully"
}

# Setup database connection
setup_database() {
    print_status "Setting up database connection..."
    
    # Check if .env file exists
    local env_file="$REPO_ROOT/.env"
    local env_template="$REPO_ROOT/.env.template"
    
    if [ ! -f "$env_file" ]; then
        print_warning "No .env file found at: $env_file"
        print_warning "Creating from template..."
        if [ -f "$env_template" ]; then
            cp "$env_template" "$env_file"
            print_warning "Please edit .env file with your database credentials"
        else
            print_warning "No .env.template found at: $env_template"
            print_warning "Please create .env file manually"
        fi
    fi
    
    # Test database connection if .env exists
    if [ -f "$env_file" ]; then
        python "$REPO_ROOT/scripts/database/check_pgvector.py" 2>/dev/null || print_warning "Database connection test failed"
    fi
}

# Main execution
main() {
    echo "=========================================="
    echo "SAMO Deep Learning Environment Setup"
    echo "=========================================="
    
    check_conda
    init_conda
    setup_environment
    activate_and_setup
    test_environment
    setup_database
    
    echo ""
    echo "=========================================="
    print_success "Environment setup completed!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Activate environment: conda activate samo-dl"
    echo "2. Edit .env file with your database credentials"
    echo "3. Run training: python -m src.models.emotion_detection.training_pipeline"
    echo "4. Test APIs: python src/unified_ai_api.py"
    echo ""
    echo "For more information, see ENVIRONMENT_SETUP.md"
}

# Run main function
main "$@" 