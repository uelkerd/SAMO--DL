#!/bin/bash
set -e

# ONNX-Based Cloud Run Deployment Script
# Eliminates PyTorch dependency issues completely

# Default configuration - can be overridden by environment variables or command-line args
PROJECT_ID="${PROJECT_ID:-the-tendril-466607-n8}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-samo-emotion-api-onnx}"
IMAGE_NAME="${IMAGE_NAME:-samo-emotion-api-onnx}"
REPOSITORY="${REPOSITORY:-samo-dl}"
ONNX_MODEL_PATH="${ONNX_MODEL_PATH:-deployment/cloud-run/model/bert_emotion_classifier.onnx}"
DEPLOYMENT_DIR="${DEPLOYMENT_DIR:-deployment/cloud-run}"
CONVERSION_SCRIPT="${CONVERSION_SCRIPT:-scripts/deployment/convert_model_to_onnx.py}"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --service-name)
            SERVICE_NAME="$2"
            shift 2
            ;;
        --image-name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --repository)
            REPOSITORY="$2"
            shift 2
            ;;
        --onnx-model-path)
            ONNX_MODEL_PATH="$2"
            shift 2
            ;;
        --deployment-dir)
            DEPLOYMENT_DIR="$2"
            shift 2
            ;;
        --conversion-script)
            CONVERSION_SCRIPT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --project-id PROJECT_ID     GCP Project ID (default: $PROJECT_ID)"
            echo "  --region REGION             GCP Region (default: $REGION)"
            echo "  --service-name NAME         Cloud Run service name (default: $SERVICE_NAME)"
            echo "  --image-name NAME           Docker image name (default: $IMAGE_NAME)"
            echo "  --repository NAME           Artifact Registry repository (default: $REPOSITORY)"
            echo "  --onnx-model-path PATH      Path to ONNX model file (default: $ONNX_MODEL_PATH)"
            echo "  --deployment-dir PATH       Deployment directory (default: $DEPLOYMENT_DIR)"
            echo "  --conversion-script PATH    ONNX conversion script (default: $CONVERSION_SCRIPT)"
            echo "  --help                      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🚀 Starting ONNX-based Cloud Run deployment..."

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

# Print configuration
print_status "Configuration:"
print_status "  Project ID: $PROJECT_ID"
print_status "  Region: $REGION"
print_status "  Service Name: $SERVICE_NAME"
print_status "  Image Name: $IMAGE_NAME"
print_status "  Repository: $REPOSITORY"
print_status "  ONNX Model Path: $ONNX_MODEL_PATH"
print_status "  Deployment Dir: $DEPLOYMENT_DIR"
print_status "  Conversion Script: $CONVERSION_SCRIPT"

# Get the script directory and navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Step 1: Convert model to ONNX
print_status "Step 1: Converting PyTorch model to ONNX format..."

if [ ! -f "$PROJECT_ROOT/$ONNX_MODEL_PATH" ]; then
    print_status "Converting model to ONNX..."
    cd "$PROJECT_ROOT"
    python "$CONVERSION_SCRIPT"
    if [ $? -ne 0 ]; then
        print_error "ONNX conversion failed!"
        exit 1
    fi
else
    print_status "ONNX model already exists, skipping conversion"
fi

# Step 2: Build and push Docker image
print_status "Step 2: Building and pushing Docker image..."

cd "$PROJECT_ROOT/$DEPLOYMENT_DIR"

# Build image
print_status "Building Docker image..."
docker build -f Dockerfile.onnx -t gcr.io/$PROJECT_ID/$IMAGE_NAME:latest .

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

# Wait for service to be ready with intelligent polling
print_status "Waiting for service to be ready..."
HEALTH_URL="$SERVICE_URL/health"
TIMEOUT=60
INTERVAL=3
ELAPSED=0

until curl -sf "$HEALTH_URL"; do
    if [ $ELAPSED -ge $TIMEOUT ]; then
        print_error "Service did not become healthy within $TIMEOUT seconds."
        exit 1
    fi
    print_status "Waiting for service... ($ELAPSED/$TIMEOUT seconds)"
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

print_status "Service is healthy!"

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

print_status "✅ ONNX-based deployment completed successfully!"
print_status "🎯 Service is operational at: $SERVICE_URL"
print_status "📊 Health endpoint: $SERVICE_URL/health"
print_status "🔮 Prediction endpoint: $SERVICE_URL/predict"
print_status "📈 Metrics endpoint: $SERVICE_URL/metrics"

echo ""
print_status "Deployment Summary:"
echo "  - Service: $SERVICE_NAME"
echo "  - Region: $REGION"
echo "  - Image: $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:latest"
echo "  - Model Type: ONNX (no PyTorch dependencies)"
echo "  - Status: ✅ OPERATIONAL" 