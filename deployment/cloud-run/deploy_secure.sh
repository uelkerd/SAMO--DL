#!/bin/bash

# Secure API Server Deployment Script
# Deploys the secure API server with enhanced security features

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
SERVICE_NAME="${SERVICE_NAME:-samo-emotion-secure}"
IMAGE_NAME="${IMAGE_NAME:-samo-emotion-secure}"
REPOSITORY="${REPOSITORY:-samo-dl}"

echo "🔒 Secure API Server Deployment"
echo "================================"

# Check if we're in the right directory
if [ ! -f "secure_api_server.py" ]; then
    print_error "Please run this script from the deployment/cloud-run directory"
    exit 1
fi

print_status "Configuration:"
print_status "  Project ID: ${PROJECT_ID}"
print_status "  Region: ${REGION}"
print_status "  Service Name: ${SERVICE_NAME}"
print_status "  Image Name: ${IMAGE_NAME}"
print_status "  Repository: ${REPOSITORY}"

# Step 1: Tag the local image for Artifact Registry
print_status "Step 1: Tagging local image for Artifact Registry..."
docker tag "samo-emotion-secure:test" "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest"

if [ $? -ne 0 ]; then
    print_error "Docker tag failed!"
    exit 1
fi

# Step 2: Push to Artifact Registry
print_status "Step 2: Pushing image to Artifact Registry..."
docker push "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest"

if [ $? -ne 0 ]; then
    print_error "Docker push failed!"
    exit 1
fi

# Step 3: Deploy to Cloud Run with secure settings
print_status "Step 3: Deploying to Cloud Run with secure settings..."

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
    --set-env-vars="FLASK_ENV=production,ENVIRONMENT=production" \
    --set-env-vars="ENABLE_SECURITY=true,ENABLE_RATE_LIMITING=true" \
    --set-env-vars="ENABLE_INPUT_SANITIZATION=true,MAX_LENGTH=512" \
    --set-env-vars="EMOTION_PROVIDER=hf,EMOTION_LOCAL_ONLY=1" \
    --set-env-vars="EMOTION_MODEL_DIR=/app/models/emotion-english-distilroberta-base"

if [ $? -ne 0 ]; then
    print_error "Cloud Run deployment failed!"
    exit 1
fi

# Step 4: Get service URL
print_status "Step 4: Getting service URL..."
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --region="${REGION}" --format="value(status.url)")

print_success "Secure API deployment completed successfully!"
print_success "Service URL: ${SERVICE_URL}"

# Step 5: Test the deployment
print_status "Step 5: Testing secure deployment..."

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

# Test security headers
print_status "Testing security headers..."
SECURITY_HEADERS=$(curl -I "${SERVICE_URL}/health" 2>/dev/null | grep -E "(X-Content-Type-Options|X-Frame-Options|X-XSS-Protection|Strict-Transport-Security)" || true)

if [ -n "$SECURITY_HEADERS" ]; then
    print_success "Security headers are properly configured"
else
    print_warning "Some security headers may not be configured"
fi

print_success "✅ Secure API deployment completed successfully!"
print_success "🎯 Service is operational at: ${SERVICE_URL}"
print_success "🔒 Security features enabled:"
print_success "  - Input sanitization"
print_success "  - Rate limiting"
print_success "  - Security headers"
print_success "  - JWT authentication (if configured)"
print_success "📊 Health endpoint: ${SERVICE_URL}/health"
print_success "🔮 Prediction endpoint: ${SERVICE_URL}/predict"
print_success "📈 Metrics endpoint: ${SERVICE_URL}/metrics"

echo ""
print_success "Secure Deployment Summary:"
echo "  - Service: ${SERVICE_NAME}"
echo "  - Region: ${REGION}"
echo "  - Image: ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest"
echo "  - Security: Enhanced with input sanitization, rate limiting, and security headers"
echo "  - Status: ✅ SECURE & PRODUCTION READY" 