#!/bin/bash
set -e

# Minimal Working Cloud Run Deployment Script
# Uses known compatible PyTorch/transformers versions

echo "🚀 Starting minimal working Cloud Run deployment..."

# Configuration
PROJECT_ID="the-tendril-466607-n8"
REGION="us-central1"
SERVICE_NAME="samo-emotion-api-minimal"
IMAGE_NAME="samo-emotion-api-minimal"
REPOSITORY="samo-dl"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Step 1: Verify model exists
print_status "Step 1: Verifying model file..."
cd /Users/minervae/Projects/SAMO--GENERAL/SAMO--DL

if [ ! -f "models/best_simple_model.pth" ]; then
    print_error "Model file not found: models/best_simple_model.pth"
    exit 1
fi

print_status "✅ Model file verified"

# Step 2: Build and push Docker image
print_status "Step 2: Building and pushing Docker image..."

cd deployment/cloud-run

# Build image
print_status "Building Docker image..."
docker build -f Dockerfile.minimal -t gcr.io/$PROJECT_ID/$IMAGE_NAME:latest .

if [ $? -ne 0 ]; then
    print_error "Docker build failed!"
    exit 1
fi

# Tag for Artifact Registry
print_status "Tagging image for Artifact Registry..."
docker tag gcr.io/$PROJECT_ID/$IMAGE_NAME:latest $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:latest

# Push to Artifact Registry
print_status "Pushing image to Artifact Registry..."
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:latest

if [ $? -ne 0 ]; then
    print_error "Docker push failed!"
    exit 1
fi

# Step 3: Deploy to Cloud Run
print_status "Step 3: Deploying to Cloud Run..."

gcloud run deploy $SERVICE_NAME \
    --image=$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:latest \
    --region=$REGION \
    --platform=managed \
    --allow-unauthenticated \
    --port=8080 \
    --memory=2Gi \
    --cpu=2 \
    --max-instances=10 \
    --min-instances=1 \
    --concurrency=80 \
    --timeout=300

if [ $? -ne 0 ]; then
    print_error "Cloud Run deployment failed!"
    exit 1
fi

# Step 4: Get service URL
print_status "Step 4: Getting service URL..."
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")

print_status "Service deployed successfully!"
print_status "Service URL: $SERVICE_URL"

# Step 5: Test the deployment
print_status "Step 5: Testing deployment..."

# Wait for service to be ready
print_status "Waiting for service to be ready..."
sleep 60

# Test health endpoint
print_status "Testing health endpoint..."
curl -f "$SERVICE_URL/health" || {
    print_error "Health check failed!"
    exit 1
}

# Test prediction endpoint
print_status "Testing prediction endpoint..."
curl -X POST "$SERVICE_URL/predict" \
    -H "Content-Type: application/json" \
    -d '{"text": "I am feeling happy today!"}' || {
    print_error "Prediction test failed!"
    exit 1
}

print_status "✅ Minimal deployment completed successfully!"
print_status "🎯 Service is operational at: $SERVICE_URL"
print_status "📊 Health endpoint: $SERVICE_URL/health"
print_status "🔮 Prediction endpoint: $SERVICE_URL/predict"
print_status "📈 Metrics endpoint: $SERVICE_URL/metrics"

echo ""
print_status "Deployment Summary:"
echo "  - Service: $SERVICE_NAME"
echo "  - Region: $REGION"
echo "  - Image: $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:latest"
echo "  - Model Type: PyTorch (known compatible versions)"
echo "  - Status: ✅ OPERATIONAL" 