#!/bin/bash

# Setup Google Secret Manager for secure API key storage
# This script creates a secret in Google Secret Manager for the API key

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
    
    if ! command -v openssl &> /dev/null; then
        print_error "openssl is not installed. Please install it first."
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
    
    gcloud services enable secretmanager.googleapis.com --project="$PROJECT_ID"
    gcloud services enable cloudbuild.googleapis.com --project="$PROJECT_ID"
    gcloud services enable run.googleapis.com --project="$PROJECT_ID"
    
    print_success "APIs enabled successfully."
}

# Create secret in Secret Manager
create_secret() {
    local secret_name="admin-api-key"
    local api_key="$1"
    
    print_status "Creating secret in Secret Manager..."
    
    # Check if secret already exists
    if gcloud secrets describe "$secret_name" --project="$PROJECT_ID" &>/dev/null; then
        print_warning "Secret '$secret_name' already exists. Updating version..."
        
        # Add new version to existing secret
        echo -n "$api_key" | gcloud secrets versions add "$secret_name" --data-file=- --project="$PROJECT_ID"
    else
        # Create new secret
        gcloud secrets create "$secret_name" --replication-policy="automatic" --project="$PROJECT_ID"
        echo -n "$api_key" | gcloud secrets versions add "$secret_name" --data-file=- --project="$PROJECT_ID"
    fi
    
    print_success "Secret created/updated successfully."
}

# Grant Cloud Build access to the secret
grant_cloud_build_access() {
    local secret_name="admin-api-key"
    local project_number
    
    print_status "Granting Cloud Build access to secret..."
    
    # Get project number
    project_number=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
    
    # Grant access to Cloud Build service account
    gcloud secrets add-iam-policy-binding "$secret_name" \
        --member="serviceAccount:$project_number@cloudbuild.gserviceaccount.com" \
        --role="roles/secretmanager.secretAccessor" \
        --project="$PROJECT_ID"
    
    print_success "Cloud Build access granted."
}

# Generate API key if not provided
generate_api_key() {
    if [ -z "$API_KEY" ]; then
        print_status "Generating new secure API key..."
        API_KEY=$(openssl rand -hex 32)
        print_success "API key generated successfully."
        print_warning "IMPORTANT: Save this API key securely - it will not be shown again!"
        print_status "API key: $API_KEY" > /tmp/samo_api_key.txt
        print_status "API key saved to: /tmp/samo_api_key.txt"
        print_status "Please copy and store it securely, then delete the temporary file."
    else
        print_status "Using provided API key."
    fi
}

# Main function
main() {
    print_status "Setting up Google Secret Manager for secure API key storage..."
    
    check_required_tools
    get_project_id
    enable_apis
    generate_api_key
    create_secret "$API_KEY"
    grant_cloud_build_access
    
    print_success "Secret Manager setup complete!"
    print_status "You can now use the consolidated cloudbuild.yaml configuration."
    print_status "Secret name: admin-api-key"
    print_status "Project ID: $PROJECT_ID"
    print_status "Deploy with: gcloud builds submit --config cloudbuild.yaml"
}

# Run main function
main "$@"