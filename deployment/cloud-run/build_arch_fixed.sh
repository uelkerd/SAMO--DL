#!/bin/bash

# Architecture-Fixed Cloud Run Build Script
# This script builds Docker images with explicit x86_64 architecture targeting

set -e

# Configuration
PROJECT_ID="the-tendril-466607-n8"
IMAGE_NAME="gcr.io/$PROJECT_ID/cloud-run-api:arch-fixed"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker buildx is available
check_buildx() {
    log_info "Checking Docker buildx support..."
    
    if ! docker buildx version &> /dev/null; then
        log_warning "Docker buildx not available, using standard build"
        return 1
    fi
    
    log_success "Docker buildx is available"
    return 0
}

# Build with explicit architecture targeting
build_with_arch() {
    log_info "Building Docker image with explicit x86_64 architecture..."
    
    if check_buildx; then
        # Use buildx for multi-platform support
        log_info "Using Docker buildx for architecture-specific build..."
        docker buildx build \
            --platform linux/amd64 \
            --tag $IMAGE_NAME \
            --file Dockerfile.arch_fixed \
            --push \
            .
    else
        # Fallback to standard build with platform flag
        log_info "Using standard Docker build with platform flag..."
        docker build \
            --platform linux/amd64 \
            --tag $IMAGE_NAME \
            --file Dockerfile.arch_fixed \
            .
        
        log_info "Pushing image to registry..."
        docker push $IMAGE_NAME
    fi
    
    log_success "Architecture-fixed image built and pushed successfully"
}

# Deploy the architecture-fixed image
deploy_arch_fixed() {
    log_info "Deploying architecture-fixed Cloud Run service..."
    
    gcloud run deploy arch-fixed-test \
        --image $IMAGE_NAME \
        --region us-central1 \
        --platform managed \
        --allow-unauthenticated \
        --max-instances 1 \
        --min-instances 0 \
        --memory 512Mi \
        --cpu 1 \
        --timeout 60 \
        --port 8080
    
    log_success "Architecture-fixed service deployed successfully"
}

# Test the deployment
test_deployment() {
    log_info "Testing architecture-fixed deployment..."
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe arch-fixed-test --region=us-central1 --format='value(status.url)')
    
    log_info "Service URL: $SERVICE_URL"
    
    # Wait for service to be ready
    log_info "Waiting for service to be ready..."
    sleep 30
    
    # Test health endpoint
    log_info "Testing health endpoint..."
    if curl -f "$SERVICE_URL/health"; then
        log_success "Health check passed!"
    else
        log_error "Health check failed!"
        return 1
    fi
    
    # Test root endpoint
    log_info "Testing root endpoint..."
    if curl -f "$SERVICE_URL/"; then
        log_success "Root endpoint test passed!"
    else
        log_error "Root endpoint test failed!"
        return 1
    fi
    
    log_success "All tests passed!"
}

# Main execution
main() {
    log_info "Starting architecture-fixed Cloud Run deployment"
    
    build_with_arch
    deploy_arch_fixed
    test_deployment
    
    log_success "Architecture-fixed deployment completed successfully!"
    
    # Show service information
    SERVICE_URL=$(gcloud run services describe arch-fixed-test --region=us-central1 --format='value(status.url)')
    echo
    echo "=== ARCHITECTURE-FIXED DEPLOYMENT SUCCESS ==="
    echo "Service Name: arch-fixed-test"
    echo "Service URL: $SERVICE_URL"
    echo "Architecture: linux/amd64 (Cloud Run compatible)"
    echo
    echo "=== TESTING ==="
    echo "Health check: curl $SERVICE_URL/health"
    echo "Root endpoint: curl $SERVICE_URL/"
}

# Run main function
main "$@" 