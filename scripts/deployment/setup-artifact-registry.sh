#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

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

# Check for required tools
check_required_tools() {
    print_status "Checking for required tools..."
    
    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    print_success "All required tools are available."
}

# Get project ID
get_project_id() {
    if [ -z "$PROJECT_ID" ]; then
        PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
        if [ -z "$PROJECT_ID" ]; then
            print_error "No project ID found. Please set PROJECT_ID environment variable or run 'gcloud config set project YOUR_PROJECT_ID'"
            exit 1
        fi
    fi
    print_status "Using project ID: $PROJECT_ID"
}

# Enable required APIs
enable_apis() {
    print_status "Enabling required APIs..."
    
    gcloud services enable artifactregistry.googleapis.com --project="$PROJECT_ID"
    
    print_success "Artifact Registry API enabled successfully."
}

# Create Artifact Registry repository
create_artifact_repository() {
    local repo_name="${_ARTIFACT_REPO:-samo-dl-repo}" # Default to 'samo-dl-repo' if not set
    local location="${_REGION:-us-central1}" # Default to 'us-central1' if not set
    
    print_status "Creating Artifact Registry repository '$repo_name' in '$location'..."
    
    if gcloud artifacts repositories describe "$repo_name" --location="$location" --project="$PROJECT_ID" &>/dev/null; then
        print_warning "Repository '$repo_name' already exists in '$location'. Skipping creation."
    else
        gcloud artifacts repositories create "$repo_name" \
            --repository-format=docker \
            --location="$location" \
            --description="SAMO-DL Docker images repository" \
            --project="$PROJECT_ID"
        print_success "Repository '$repo_name' created successfully."
    fi
}

# Main function
main() {
    print_status "Setting up Google Artifact Registry..."
    
    check_required_tools
    get_project_id
    enable_apis
    create_artifact_repository
    
    print_success "Artifact Registry setup complete!"
    print_status "Repository name: ${_ARTIFACT_REPO:-samo-dl-repo}"
    print_status "Project ID: $PROJECT_ID"
    print_status "You can now use this repository for your Docker images in Cloud Build."
}

# Run main function
main "$@"