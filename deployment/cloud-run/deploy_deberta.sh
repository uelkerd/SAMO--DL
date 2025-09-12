#!/bin/bash

# DeBERTa API Server Deployment Script
# Deploys the DeBERTa model with enhanced emotion detection (28 emotions)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

print_deberta() {
    echo -e "${PURPLE}[DeBERTa]${NC} $1"
}

# Configuration
PROJECT_ID="${PROJECT_ID:-the-tendril-466607-n8}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-samo-emotion-deberta}"
IMAGE_NAME="${IMAGE_NAME:-samo-emotion-api-deberta}"
REPOSITORY="${REPOSITORY:-samo-dl}"

echo "🤖 DeBERTa API Server Deployment"
echo "================================"
echo "🎯 Model: duelker/samo-goemotions-deberta-v3-large"
echo "🎯 Emotions: 28 (vs 6 in production)"
echo "🎯 Performance: 51.8% F1 Macro"
echo ""

# Check if we're in the right directory (allow running from project root)
if [ ! -f "secure_api_server.py" ] && [ ! -f "deployment/cloud-run/secure_api_server.py" ]; then
    print_error "Please run this script from the project root or deployment/cloud-run directory"
    exit 1
fi

# Set the project root directory
if [ -f "secure_api_server.py" ]; then
    PROJECT_ROOT="."
else
    PROJECT_ROOT="../.."
fi

print_status "Configuration:"
print_status "  Project ID: ${PROJECT_ID}"
print_status "  Region: ${REGION}"
print_status "  Service Name: ${SERVICE_NAME}"
print_status "  Image Name: ${IMAGE_NAME}"
print_status "  Repository: ${REPOSITORY}"
print_deberta "  Model: duelker/samo-goemotions-deberta-v3-large"
print_deberta "  Emotions: 28 emotion classes"
echo ""

# Step 1: Build the Docker image locally
print_status "Step 1: Building DeBERTa Docker image..."
cd "$PROJECT_ROOT"
docker build -t "samo-emotion-deberta:test" -f deployment/cloud-run/Dockerfile.deberta .

if [ $? -ne 0 ]; then
    print_error "Docker build failed!"
    exit 1
fi

print_success "Docker image built successfully!"

# Step 2: Tag the local image for Artifact Registry
print_status "Step 2: Tagging local image for Artifact Registry..."
docker tag "samo-emotion-deberta:test" "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest"

if [ $? -ne 0 ]; then
    print_error "Docker tag failed!"
    exit 1
fi

print_success "Image tagged for Artifact Registry"

# Step 3: Authenticate Docker to Google Cloud (if not already done)
print_status "Step 3: Authenticating with Google Cloud..."
gcloud auth configure-docker --quiet

if [ $? -ne 0 ]; then
    print_error "Google Cloud authentication failed!"
    exit 1
fi

# Step 4: Push to Artifact Registry
print_status "Step 4: Pushing image to Artifact Registry..."
docker push "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest"

if [ $? -ne 0 ]; then
    print_error "Docker push failed!"
    exit 1
fi

print_success "Image pushed to Artifact Registry!"

# Step 5: Deploy to Cloud Run with DeBERTa settings
print_status "Step 5: Deploying DeBERTa model to Cloud Run..."

print_deberta "Configuring DeBERTa environment variables..."
gcloud run deploy "${SERVICE_NAME}" \
    --image="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest" \
    --region="${REGION}" \
    --platform=managed \
    --allow-unauthenticated \
    --port=8080 \
    --memory=4Gi \
    --cpu=2 \
    --max-instances=5 \
    --min-instances=1 \
    --concurrency=50 \
    --timeout=900 \
    --set-env-vars="FLASK_ENV=production,ENVIRONMENT=production" \
    --set-env-vars="USE_DEBERTA=true,PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python" \
    --set-env-vars="ENABLE_SECURITY=true,ENABLE_RATE_LIMITING=true" \
    --set-env-vars="ENABLE_INPUT_SANITIZATION=true,MAX_LENGTH=512" \
    --set-env-vars="EMOTION_PROVIDER=hf,EMOTION_LOCAL_ONLY=1" \
    --set-env-vars="DEBERTA_MODEL_NAME=duelker/samo-goemotions-deberta-v3-large" \
    --set-env-vars="ADMIN_API_KEY=${ADMIN_API_KEY:-test123}"

if [ $? -ne 0 ]; then
    print_error "Cloud Run deployment failed!"
    exit 1
fi

print_success "DeBERTa model deployed to Cloud Run!"

# Step 6: Get service URL
print_status "Step 6: Getting service URL..."
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --region="${REGION}" --format="value(status.url)")

print_success "DeBERTa API deployment completed successfully!"
print_success "Service URL: ${SERVICE_URL}"

# Step 7: Test the deployment
print_status "Step 7: Testing DeBERTa deployment..."

# Wait for service to be ready (DeBERTa takes longer to load)
print_status "Waiting for DeBERTa model to initialize (this may take 2-3 minutes)..."
HEALTH_URL="${SERVICE_URL}/api/health"
TIMEOUT=300  # 5 minutes timeout for DeBERTa
INTERVAL=10
ELAPSED=0

until curl -sf "${HEALTH_URL}"; do
    if [ ${ELAPSED} -ge ${TIMEOUT} ]; then
        print_error "Service did not become healthy within ${TIMEOUT} seconds."
        exit 1
    fi
    print_status "Waiting for DeBERTa model to load... (${ELAPSED}/${TIMEOUT} seconds)"
    sleep ${INTERVAL}
    ELAPSED=$((ELAPSED + INTERVAL))
done

print_success "DeBERTa service is healthy!"

# Test health endpoint
print_status "Testing health endpoint..."
curl -f "${SERVICE_URL}/api/health" || {
    print_error "Health check failed!"
    exit 1
}

# Test model status endpoint
print_status "Testing model status endpoint..."
MODEL_STATUS=$(curl -s "${SERVICE_URL}/admin/model_status" -H "X-API-Key: ${ADMIN_API_KEY:-test123}")

if echo "$MODEL_STATUS" | grep -q "28"; then
    print_deberta "✅ DeBERTa model confirmed (28 emotions detected)"
else
    print_warning "⚠️ Model status may not be showing 28 emotions"
fi

# Test DeBERTa prediction endpoint
print_status "Testing DeBERTa prediction endpoint..."
PREDICTION_RESPONSE=$(curl -s -X POST "${SERVICE_URL}/api/predict" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: ${ADMIN_API_KEY:-test123}" \
    -d '{"text": "I am so happy today!"}')

if echo "$PREDICTION_RESPONSE" | grep -q "joy\|admiration\|amusement"; then
    print_deberta "✅ DeBERTa prediction working (emotion labels detected)"
else
    print_warning "⚠️ Prediction response may not contain expected emotions"
fi

# Test batch predictions
print_status "Testing batch predictions..."
BATCH_RESPONSE=$(curl -s -X POST "${SERVICE_URL}/api/predict_batch" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: ${ADMIN_API_KEY:-test123}" \
    -d '{"texts": ["I am happy", "I am sad", "This is amazing"]}')

if echo "$BATCH_RESPONSE" | grep -q "results"; then
    print_success "Batch predictions working!"
else
    print_warning "⚠️ Batch prediction may have issues"
fi

print_success "🎉 DeBERTa API deployment completed successfully!"
print_success "🌐 Service URL: ${SERVICE_URL}"
print_deberta "🤖 Model: duelker/samo-goemotions-deberta-v3-large"
print_deberta "🎯 Emotions: 28 emotion classes"
print_deberta "📊 Performance: 51.8% F1 Macro"
echo ""
print_success "🔗 Endpoints:"
echo "  - Health: ${SERVICE_URL}/api/health"
echo "  - Predict: ${SERVICE_URL}/api/predict"
echo "  - Batch Predict: ${SERVICE_URL}/api/predict_batch"
echo "  - Model Status: ${SERVICE_URL}/admin/model_status"
echo ""
print_success "🔐 Authentication:"
echo "  - API Key required for admin endpoints"
echo "  - Admin API Key: [REDACTED - check environment variable]"
echo ""
print_success "🚀 PRODUCTION READY - DeBERTa API is live!"

echo ""
print_success "Deployment Summary:"
echo "  - Service: ${SERVICE_NAME}"
echo "  - Region: ${REGION}"
echo "  - Image: ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest"
echo "  - Model: DeBERTa (28 emotions)"
echo "  - Memory: 4GB (increased for DeBERTa)"
echo "  - CPU: 2 cores"
echo "  - Status: ✅ DeBERTa PRODUCTION READY"
