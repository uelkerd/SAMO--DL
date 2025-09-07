#!/bin/bash
# Build optimized emotion detection model for Cloud Run
# CPU-only, minimal dependencies, smaller image size

set -e

# Check for required tools
check_required_tools() {
    echo "🔧 Checking for required tools..."
    
    if ! command -v jq &> /dev/null; then
        echo "❌ Error: jq is not installed. Please install jq before running this script."
        echo "   Install with: brew install jq (macOS) or apt-get install jq (Ubuntu)"
        exit 1
    fi
    
    if ! command -v curl &> /dev/null; then
        echo "❌ Error: curl is not installed. Please install curl before running this script."
        exit 1
    fi
    
    echo "✅ All required tools are available."
}

# Set default values for parameters
API_PORT=8081
CONTAINER_PORT=8080
API_KEY=${API_KEY:-"test-key-123"}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --api-port)
            API_PORT="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Usage: $0 [--api-port PORT] [--api-key KEY]"
            exit 1
            ;;
    esac
done

# Run checks
check_required_tools

echo "🚀 Building OPTIMIZED emotion detection model..."
echo "📦 Using PRODUCTION ARCHITECTURE from PRs #136, #137, #138"
echo "🔐 Flask-RESTX, security headers, rate limiting, batch processing"
echo "🎯 CPU-only PyTorch for Cloud Run deployment"
echo ""

# Build the optimized image with production architecture
echo "🔨 Building Docker image with REAL production architecture..."
docker buildx build \
    -f deployment/docker/Dockerfile.optimized \
    -t emotion-detection-api:optimized \
    --progress=plain \
    --no-cache \
    .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo ""
echo "✅ Build completed!"
echo ""

# Show image size comparison
echo "📊 Image size comparison:"
echo "Original image:"
docker images emotion-detection-api:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

echo ""
echo "Optimized image:"
docker images emotion-detection-api:optimized --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

echo ""
echo "🎯 Expected savings: ~1.2GB (CPU-only PyTorch vs full PyTorch)"
echo ""

# Test the optimized image
echo "🧪 Testing optimized image..."
echo "Starting container for quick test..."

# Run a quick test with extended timeout and environment variables
docker run --rm -d --name emotion-test-optimized -p "${API_PORT}":"${CONTAINER_PORT}" \
    -e ADMIN_API_KEY="${API_KEY}" \
    -e MODEL_CACHE_DIR=/app/models \
    emotion-detection-api:optimized \
    gunicorn --bind :"${CONTAINER_PORT}" --workers 1 --timeout 300 secure_api_server:app

if [ $? -ne 0 ]; then
    echo "❌ Failed to start container!"
    exit 1
fi

# Wait for container to start and model to load (pre-downloaded during build)
echo "⏳ Waiting for container to start..."

MAX_ATTEMPTS=30
SLEEP_SECONDS=2
HEALTH_URL="http://localhost:${API_PORT}/api/health"
attempt=1
while [ $attempt -le $MAX_ATTEMPTS ]; do
    echo "🔄 Checking health endpoint (attempt $attempt/$MAX_ATTEMPTS)..."
    if curl -s "$HEALTH_URL" | jq '.' > /dev/null 2>&1; then
        echo "✅ Container is healthy!"
        break
    fi
    sleep "$SLEEP_SECONDS"
    attempt=$((attempt+1))
done

if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
    echo "❌ Container did not become healthy after $((MAX_ATTEMPTS * SLEEP_SECONDS)) seconds."
    docker logs emotion-test-optimized
    docker stop emotion-test-optimized 2>/dev/null || echo "Container already stopped/removed"
    exit 1
fi

# Test health endpoint (Flask-RESTX endpoint)
echo "🔍 Testing health endpoint..."
health_response=$(curl -s "$HEALTH_URL")
echo "$health_response" | jq '.' || { echo "Health check failed (using /api/health)"; exit 1; }

# Test prediction endpoint (Flask-RESTX endpoint)
echo "🔍 Testing prediction endpoint..."
predict_response=$(curl -s -X POST "http://localhost:${API_PORT}/api/predict" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: ${API_KEY}" \
    -d '{"text": "I am feeling happy today!"}')
echo "$predict_response" | jq '.' || { echo "Prediction test failed"; exit 1; }

# Test batch prediction endpoint (Flask-RESTX endpoint)
echo "🔍 Testing batch prediction endpoint..."
batch_response=$(curl -s -X POST "http://localhost:${API_PORT}/api/predict_batch" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: ${API_KEY}" \
    -d '{"texts": ["I am happy", "I feel sad", "I am excited"]}')
echo "$batch_response" | jq '.' || { echo "Batch prediction test failed"; exit 1; }

# Clean up
echo "🧹 Cleaning up test container..."
docker stop emotion-test-optimized 2>/dev/null || echo "Container already stopped/removed"

echo ""
echo "✅ PRODUCTION ARCHITECTURE build and test completed!"
echo "🏗️ Flask-RESTX API with security, rate limiting, batch processing"
echo "🎯 CPU-only PyTorch for Cloud Run deployment"
echo "📦 Ready for Cloud Run deployment with FULL production features!"