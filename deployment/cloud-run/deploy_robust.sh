#!/bin/bash

# Robust Cloud Run Deployment Script
# Based on Google Cloud Run troubleshooting documentation
# https://cloud.google.com/run/docs/troubleshooting

set -e

# Configuration
PROJECT_ID="the-tendril-466607-n8"
REGION="us-central1"
SERVICE_NAME="samo-emotion-api"
IMAGE_NAME="gcr.io/$PROJECT_ID/cloud-run-api:robust"

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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if gcloud is authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        log_error "Not authenticated with gcloud. Please run 'gcloud auth login' first."
        exit 1
    fi
    
    # Check if project is set
    if [ "$(gcloud config get-value project)" != "$PROJECT_ID" ]; then
        log_warning "Setting project to $PROJECT_ID"
        gcloud config set project $PROJECT_ID
    fi
    
    # Check if required APIs are enabled
    log_info "Checking required APIs..."
    gcloud services enable run.googleapis.com --quiet || log_warning "Failed to enable Cloud Run API"
    gcloud services enable cloudbuild.googleapis.com --quiet || log_warning "Failed to enable Cloud Build API"
    
    log_success "Prerequisites check completed"
}

# Build and push image
build_and_push() {
    log_info "Building and pushing Docker image..."
    
    # Build image
    docker build -f Dockerfile.robust -t $IMAGE_NAME .
    
    # Push image
    docker push $IMAGE_NAME
    
    log_success "Image built and pushed successfully"
}

# Deploy with comprehensive configuration
deploy_service() {
    log_info "Deploying Cloud Run service with robust configuration..."
    
    # Deploy with comprehensive settings
    gcloud run deploy $SERVICE_NAME \
        --image $IMAGE_NAME \
        --region $REGION \
        --platform managed \
        --allow-unauthenticated \
        --max-instances 3 \
        --min-instances 0 \
        --memory 2Gi \
        --cpu 2 \
        --timeout 300 \
        --concurrency 10 \
        --port 8080 \
        --set-env-vars "PYTHONUNBUFFERED=1" \
        --set-env-vars "PORT=8080" \
        --set-env-vars "FLASK_ENV=production" \
        --set-env-vars "FLASK_DEBUG=0" \
        --update-env-vars "GOOGLE_CLOUD_PROJECT=$PROJECT_ID" \
        --service-account="$(gcloud iam service-accounts list --filter="displayName:Cloud Run" --format='value(email)' | head -1)" \
        --no-cpu-throttling \
        --execution-environment=gen2
    
    log_success "Service deployed successfully"
}

# Test the deployment
test_deployment() {
    log_info "Testing deployment..."
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')
    
    log_info "Service URL: $SERVICE_URL"
    
    # Test health endpoint
    log_info "Testing health endpoint..."
    curl -f "$SERVICE_URL/health" || log_warning "Health check failed"
    
    # Test root endpoint
    log_info "Testing root endpoint..."
    curl -f "$SERVICE_URL/" || log_warning "Root endpoint failed"
    
    log_success "Deployment testing completed"
}

# Show logs for debugging
show_logs() {
    log_info "Showing recent logs for debugging..."
    gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME" --limit=20 --format="value(timestamp,severity,textPayload)" || log_warning "Failed to fetch logs"
}

# Main deployment process
main() {
    log_info "Starting robust Cloud Run deployment for SAMO Emotion Detection API"
    
    check_prerequisites
    build_and_push
    deploy_service
    
    # Wait for deployment to stabilize
    log_info "Waiting for deployment to stabilize..."
    sleep 30
    
    test_deployment
    show_logs
    
    log_success "Robust deployment completed!"
    
    # Show service information
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')
    echo
    echo "=== DEPLOYMENT SUMMARY ==="
    echo "Service Name: $SERVICE_NAME"
    echo "Service URL: $SERVICE_URL"
    echo "Region: $REGION"
    echo "Project: $PROJECT_ID"
    echo
    echo "=== TESTING ==="
    echo "Health check: curl $SERVICE_URL/health"
    echo "Root endpoint: curl $SERVICE_URL/"
    echo "Prediction: curl -X POST $SERVICE_URL/predict -H 'Content-Type: application/json' -d '{\"text\":\"I am happy\"}'"
    echo
    echo "=== MONITORING ==="
    echo "Logs: gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME\" --limit=50"
    echo "Metrics: https://console.cloud.google.com/run/detail/$REGION/$SERVICE_NAME/metrics?project=$PROJECT_ID"
}

# Run main function
main "$@" 