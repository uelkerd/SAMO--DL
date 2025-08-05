#!/bin/bash

# Cloud Run Deployment Script for SAMO Emotion Detection API
# This script automates the deployment process to Google Cloud Run

set -e  # Exit on any error

# Configuration
PROJECT_ID="the-tendril-466607-n8"
REGION="us-central1"
SERVICE_NAME="emotion-detection-api"
REPOSITORY="emotion-detection-repo"
IMAGE_NAME="cloud-run-api"
MODEL_PATH="deployment/models/default"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Check if we're authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        log_error "Not authenticated with gcloud. Please run 'gcloud auth login' first."
        exit 1
    fi
    
    # Check if project is set
    if [ "$(gcloud config get-value project)" != "$PROJECT_ID" ]; then
        log_warning "Setting project to $PROJECT_ID"
        gcloud config set project $PROJECT_ID
    fi
    
    log_success "Prerequisites check passed"
}

# Prepare the deployment directory
prepare_deployment() {
    log_info "Preparing deployment directory..."
    
    # Create cloud-run directory if it doesn't exist
    mkdir -p deployment/cloud-run
    
    # Copy model files
    if [ ! -d "deployment/cloud-run/model" ]; then
        log_info "Copying model files..."
        cp -r $MODEL_PATH deployment/cloud-run/model/
    fi
    
    # Verify model files exist
    if [ ! -f "deployment/cloud-run/model/pytorch_model.bin" ] && [ ! -f "deployment/cloud-run/model/model.safetensors" ]; then
        log_error "Model files not found in $MODEL_PATH"
        exit 1
    fi
    
    log_success "Deployment directory prepared"
}

# Build the Docker image
build_image() {
    log_info "Building Docker image..."
    
    cd deployment/cloud-run
    
    # Build the image
    docker build -t gcr.io/$PROJECT_ID/$IMAGE_NAME:latest .
    
    # Tag for Artifact Registry
    docker tag gcr.io/$PROJECT_ID/$IMAGE_NAME:latest \
        $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:latest
    
    cd ../..
    
    log_success "Docker image built successfully"
}

# Push the image to Artifact Registry
push_image() {
    log_info "Pushing image to Artifact Registry..."
    
    # Configure docker to use gcloud as a credential helper
    gcloud auth configure-docker $REGION-docker.pkg.dev
    
    # Push the image
    docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:latest
    
    log_success "Image pushed to Artifact Registry"
}

# Deploy to Cloud Run
deploy_service() {
    log_info "Deploying to Cloud Run..."
    
    # Deploy the service
    gcloud run deploy $SERVICE_NAME \
        --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:latest \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --memory 2Gi \
        --cpu 2 \
        --max-instances 10 \
        --min-instances 0 \
        --timeout 300 \
        --concurrency 80 \
        --set-env-vars "MODEL_PATH=/app/model,LOG_LEVEL=INFO"
    
    log_success "Service deployed to Cloud Run"
}

# Get service URL
get_service_url() {
    log_info "Getting service URL..."
    
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
        --region $REGION \
        --format="value(status.url)")
    
    log_success "Service URL: $SERVICE_URL"
    echo $SERVICE_URL > .cloud_run_url
}

# Test the deployment
test_deployment() {
    log_info "Testing deployment..."
    
    SERVICE_URL=$(cat .cloud_run_url)
    
    # Test health endpoint
    log_info "Testing health endpoint..."
    HEALTH_RESPONSE=$(curl -s "$SERVICE_URL/health")
    
    if echo "$HEALTH_RESPONSE" | grep -q '"status":"healthy"'; then
        log_success "Health check passed"
    else
        log_error "Health check failed: $HEALTH_RESPONSE"
        return 1
    fi
    
    # Test prediction endpoint
    log_info "Testing prediction endpoint..."
    PREDICTION_RESPONSE=$(curl -s -X POST "$SERVICE_URL/predict" \
        -H "Content-Type: application/json" \
        -d '{"text": "I am feeling very happy today!"}')
    
    if echo "$PREDICTION_RESPONSE" | grep -q '"emotion"'; then
        log_success "Prediction test passed"
        log_info "Response: $PREDICTION_RESPONSE"
    else
        log_error "Prediction test failed: $PREDICTION_RESPONSE"
        return 1
    fi
    
    log_success "All tests passed"
}

# Display deployment information
display_info() {
    log_info "Deployment completed successfully!"
    echo
    echo "Service Information:"
    echo "  Name: $SERVICE_NAME"
    echo "  URL: $(cat .cloud_run_url)"
    echo "  Region: $REGION"
    echo "  Project: $PROJECT_ID"
    echo
    echo "API Endpoints:"
    echo "  Health Check: $(cat .cloud_run_url)/health"
    echo "  Prediction: $(cat .cloud_run_url)/predict"
    echo
    echo "Example Usage:"
    echo "  curl -X POST $(cat .cloud_run_url)/predict \\"
    echo "    -H \"Content-Type: application/json\" \\"
    echo "    -d '{\"text\": \"I am feeling very happy today!\"}'"
    echo
    echo "Monitoring:"
    echo "  Logs: gcloud logging tail \"resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME\""
    echo "  Metrics: https://console.cloud.google.com/run/detail/$REGION/$SERVICE_NAME/metrics"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    rm -f .cloud_run_url
}

# Main deployment process
main() {
    log_info "Starting Cloud Run deployment for SAMO Emotion Detection API"
    
    # Set up cleanup on exit
    trap cleanup EXIT
    
    # Run deployment steps
    check_prerequisites
    prepare_deployment
    build_image
    push_image
    deploy_service
    get_service_url
    test_deployment
    display_info
    
    log_success "Deployment completed successfully!"
}

# Run main function
main "$@" 