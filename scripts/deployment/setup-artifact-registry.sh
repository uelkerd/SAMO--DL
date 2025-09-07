#!/bin/bash

# Setup Google Artifact Registry for Docker images
# This script creates an Artifact Registry repository for storing Docker images

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
    gcloud services enable cloudbuild.googleapis.com --project="$PROJECT_ID"
    
    print_success "APIs enabled successfully."
}

# Create Artifact Registry repository
create_repository() {
    local repo_name="${1:-samo-dl-repo}"
    local region="${2:-us-central1}"
    
    print_status "Creating Artifact Registry repository..."
    
    # Check if repository already exists
    if gcloud artifacts repositories describe "$repo_name" --location="$region" --project="$PROJECT_ID" &>/dev/null; then
        print_warning "Repository '$repo_name' already exists in region '$region'."
    else
        # Create new repository
        gcloud artifacts repositories create "$repo_name" \
            --repository-format=docker \
            --location="$region" \
            --description="SAMO-DL Docker images repository" \
            --project="$PROJECT_ID"
        
        print_success "Repository '$repo_name' created successfully in region '$region'."
    fi
}

# Configure Docker authentication
configure_docker_auth() {
    local region="${1:-us-central1}"
    
    print_status "Configuring Docker authentication for Artifact Registry..."
    
    gcloud auth configure-docker "$region-docker.pkg.dev" --quiet
    
    print_success "Docker authentication configured."
}

# Grant Cloud Build access
grant_cloud_build_access() {
    local repo_name="${1:-samo-dl-repo}"
    local region="${2:-us-central1}"
    local project_number
    
    print_status "Granting Cloud Build access to Artifact Registry..."
    
    # Get project number
    project_number=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
    
    # Grant access to Cloud Build service account
    gcloud artifacts repositories add-iam-policy-binding "$repo_name" \
        --location="$region" \
        --member="serviceAccount:$project_number@cloudbuild.gserviceaccount.com" \
        --role="roles/artifactregistry.writer" \
        --project="$PROJECT_ID"
    
    print_success "Cloud Build access granted to Artifact Registry."
}

# Main function
main() {
    local repo_name="${1:-samo-dl-repo}"
    local region="${2:-us-central1}"
    
    print_status "Setting up Google Artifact Registry for Docker images..."
    
    check_required_tools
    get_project_id
    enable_apis
    create_repository "$repo_name" "$region"
    configure_docker_auth "$region"
    grant_cloud_build_access "$repo_name" "$region"
    
    print_success "Artifact Registry setup complete!"
    print_status "Repository: $repo_name"
    print_status "Region: $region"
    print_status "Project ID: $PROJECT_ID"
    print_status "Docker registry: $region-docker.pkg.dev/$PROJECT_ID/$repo_name"
    print_status ""
    print_status "You can now use the enhanced cloudbuild.yaml configuration with:"
    print_status "gcloud builds submit --config cloudbuild.yaml"
}

# Show usage if help requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [REPO_NAME] [REGION]"
    echo ""
    echo "Arguments:"
    echo "  REPO_NAME    Artifact Registry repository name (default: samo-dl-repo)"
    echo "  REGION       GCP region (default: us-central1)"
    echo ""
    echo "Examples:"
    echo "  $0                           # Use defaults"
    echo "  $0 my-repo                   # Custom repository name"
    echo "  $0 my-repo europe-west1      # Custom repository name and region"
    exit 0
fi

# Run main function
main "$@"
