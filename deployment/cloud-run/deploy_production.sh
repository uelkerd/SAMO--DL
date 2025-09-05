#!/bin/bash

# Production Deployment Script for ONNX API Server
# Uses Gunicorn WSGI server for production deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Configuration
PROJECT_ID="${PROJECT_ID:-the-tendril-466607-n8}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-samo-emotion-api-onnx-production}"
IMAGE_NAME="${IMAGE_NAME:-samo-emotion-api-onnx-production}"
REPOSITORY="${REPOSITORY:-samo-dl}"

echo "🚀 Production Deployment: ONNX API Server with Gunicorn"
echo "========================================================"

# Check if we're in the right directory
if [ ! -f "onnx_api_server.py" ]; then
    print_error "Please run this script from the deployment/cloud-run directory"
    exit 1
fi

print_status "Configuration:"
print_status "  Project ID: ${PROJECT_ID}"
print_status "  Region: ${REGION}"
print_status "  Service Name: ${SERVICE_NAME}"
print_status "  Image Name: ${IMAGE_NAME}"
print_status "  Repository: ${REPOSITORY}"

# Step 1: Build production Docker image
print_status "Step 1: Building production Docker image..."
docker build -f deployment/docker/Dockerfile.production -t "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest" .

if [ $? -ne 0 ]; then
    print_error "Docker build failed!"
    exit 1
fi

# Step 2: Tag for Artifact Registry
print_status "Step 2: Tagging image for Artifact Registry..."
docker tag "gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest" "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest"

# Step 3: Push to Artifact Registry
print_status "Step 3: Pushing image to Artifact Registry..."
docker push "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest"

if [ $? -ne 0 ]; then
    print_error "Docker push failed!"
    exit 1
fi

# Step 4: Deploy to Cloud Run with production settings
print_status "Step 4: Deploying to Cloud Run with production settings..."

gcloud run deploy "${SERVICE_NAME}" \
    --image="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest" \
    --region="${REGION}" \
    --platform=managed \
    --allow-unauthenticated \
    --port=8080 \
    --memory=2Gi \
    --cpu=2 \
    --max-instances=10 \
    --min-instances=1 \
    --concurrency=80 \
    --timeout=300 \
    --set-env-vars="FLASK_ENV=production,ENVIRONMENT=production,GUNICORN_WORKERS=2,MAX_LENGTH=512"

if [ $? -ne 0 ]; then
    print_error "Cloud Run deployment failed!"
    exit 1
fi

# Step 5: Get service URL
print_status "Step 5: Getting service URL..."
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --region="${REGION}" --format="value(status.url)")

print_success "Production deployment completed successfully!"
print_success "Service URL: ${SERVICE_URL}"

# Step 6: Test the deployment
print_status "Step 6: Testing production deployment..."

# Wait for service to be ready
print_status "Waiting for service to be ready..."
HEALTH_URL="${SERVICE_URL}/health"
TIMEOUT=60
INTERVAL=3
ELAPSED=0

until curl -sf "${HEALTH_URL}"; do
    if [ ${ELAPSED} -ge ${TIMEOUT} ]; then
        print_error "Service did not become healthy within ${TIMEOUT} seconds."
        exit 1
    fi
    print_status "Waiting for service... (${ELAPSED}/${TIMEOUT} seconds)"
    sleep ${INTERVAL}
    ELAPSED=$((ELAPSED + INTERVAL))
done

print_success "Service is healthy!"

# Test health endpoint
print_status "Testing health endpoint..."
curl -f "${SERVICE_URL}/health" || {
    print_error "Health check failed!"
    exit 1
}

# Test prediction endpoint
print_status "Testing prediction endpoint..."
curl -X POST "${SERVICE_URL}/predict" \
    -H "Content-Type: application/json" \
    -d '{"text": "I am feeling happy today!"}' || {
    print_error "Prediction test failed!"
    exit 1
}

print_success "✅ Production deployment completed successfully!"
print_success "🎯 Service is operational at: ${SERVICE_URL}"
print_success "📊 Health endpoint: ${SERVICE_URL}/health"
print_success "🔮 Prediction endpoint: ${SERVICE_URL}/predict"
print_success "📈 Metrics endpoint: ${SERVICE_URL}/metrics"

# PLACEHOLDER: NEXT DEPLOYMENT STEP - Deploy enhanced API with Priority 1 Features
# TODO: Update Docker image to include JWT authentication
# TODO: Add voice transcription and summarization endpoints
# TODO: Configure WebSocket support for real-time processing
# TODO: Deploy monitoring dashboard alongside API

echo ""
print_success "Production Deployment Summary:"
echo "  - Service: ${SERVICE_NAME}"
echo "  - Region: ${REGION}"
echo "  - Image: ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest"
echo "  - Server: Gunicorn WSGI (production-ready)"
echo "  - Workers: 2 (configurable via GUNICORN_WORKERS)"
echo "  - Max Length: 512 characters (configurable via MAX_LENGTH)"
echo "  - Status: ✅ PRODUCTION READY" 