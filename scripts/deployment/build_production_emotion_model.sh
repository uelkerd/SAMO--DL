#!/bin/bash
# Build production emotion detection model for Cloud Run
# Uses the actual production architecture from PRs #136, #137, #138

set -euo pipefail

# Check for required tools
check_required_tools() {
    echo "🔧 Checking for required tools..."

    if ! command -v docker &> /dev/null; then
        echo "❌ Error: docker is not installed."
        exit 1
    fi

    if ! command -v jq &> /dev/null; then
        echo "❌ Error: jq is not installed. Please install jq before running this script."
        echo "   Install with: brew install jq (macOS) or apt-get install jq (Ubuntu)"
        exit 1
    fi

    if ! command -v curl &> /dev/null; then
        echo "❌ Error: curl is not installed. Please install curl before running this script."
        exit 1
    fi

    if ! command -v openssl &> /dev/null; then
        echo "❌ Error: openssl is not installed. Please install openssl or set API_KEY env."
        exit 1
    fi

    echo "✅ All required tools are available."
}

# Set default values for parameters
API_PORT=8080
CONTAINER_PORT=8080
# Generate a random API key if not provided
if [ -z "${API_KEY}" ]; then
    API_KEY="test-key-$(date +%s)-$(openssl rand -hex 8)"
fi

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

echo "🚀 Building PRODUCTION emotion detection model..."
echo "📦 Using FULL PRODUCTION ARCHITECTURE from PRs #136, #137, #138"
echo "🔐 Flask-RESTX, security headers, advanced rate limiting, batch processing"
echo "🎯 CPU-only PyTorch (avoiding CUDA download issues)"
echo ""

# Build the production image with CPU-only PyTorch
echo "🔨 Building Docker image with FULL production architecture..."
docker buildx build \
    -f deployment/docker/Dockerfile.production \
    -t emotion-detection-api:production \
    --progress=plain \
    --no-cache \
    --load \
    .

if ! docker image inspect emotion-detection-api:production >/dev/null 2>&1; then
    echo "❌ Docker build produced no local image (missing --load?)"
    exit 1
fi

echo ""
echo "✅ Production build completed!"
echo ""

# Show image size comparison
echo "📊 Image size comparison:"
echo "Original image:"
docker images emotion-detection-api:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

echo ""
echo "Optimized image:"
docker images emotion-detection-api:optimized --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

echo ""
echo "Production image:"
docker images emotion-detection-api:production --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

echo ""
echo "🧪 Testing production image..."
echo "Starting container for quick test..."

# Add cleanup trap for robustness
cleanup() { docker rm -f emotion-test-production >/dev/null 2>&1 || true; }
trap cleanup EXIT

# Start the production container
docker run --rm -d \
    -p "${API_PORT}":"${CONTAINER_PORT}" \
    --name emotion-test-production \
    -e ADMIN_API_KEY="${API_KEY}" \
    emotion-detection-api:production

if [ $? -ne 0 ]; then
    echo "❌ Failed to start container!"
    exit 1
fi

echo "⏳ Waiting for container to start..."

# Poll health endpoint with timeout
max_attempts=30
attempt=1
until curl -sf "http://localhost:${API_PORT}/api/health" > /dev/null; do
    if [ "$attempt" -ge "$max_attempts" ]; then
        echo "❌ Container did not become healthy after $((max_attempts)) attempts."
        docker logs emotion-test-production
        docker stop emotion-test-production 2>/dev/null || echo "Container already stopped"
        exit 1
    fi
    echo "Waiting for health endpoint... (attempt $attempt/$max_attempts)"
    attempt=$((attempt + 1))
    sleep 1
done

echo "✅ Container is healthy!"

echo "🔍 Testing health endpoint..."
health_response=$(curl -s "http://localhost:${API_PORT}/api/health")
echo "$health_response" | jq -e '.status=="ok"' >/dev/null || { echo "Health check failed"; exit 1; }

echo ""
echo "🔍 Testing prediction endpoint..."
predict_response=$(curl -s -X POST "http://localhost:${API_PORT}/api/predict" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: ${API_KEY}" \
    -d '{"text": "I am so happy today!"}')
echo "$predict_response" | jq -e '.label and (.score|type=="number")' >/dev/null || { echo "Prediction test failed"; exit 1; }

echo ""
echo "🔍 Testing batch prediction endpoint..."
batch_response=$(curl -s -X POST "http://localhost:${API_PORT}/api/predict_batch" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: ${API_KEY}" \
    -d '{"texts": ["I am happy", "I am sad", "I am excited"]}')
echo "$batch_response" | jq -e 'type=="array" and length==3' >/dev/null || { echo "Batch prediction test failed"; exit 1; }

echo ""
echo "🧹 Cleaning up test container..."
docker stop emotion-test-production 2>/dev/null || echo "Container already stopped"

echo ""
echo "🎉 Production image is ready!"
echo "📋 Features included:"
echo "  ✅ Flask-RESTX endpoints"
echo "  ✅ Batch processing"
echo "  ✅ Security headers"
echo "  ✅ Rate limiting"
echo "  ✅ Input validation"
echo "  ✅ Error handling"
echo "  ✅ Health checks"
echo "  ✅ API key authentication"
echo "  ✅ Swagger documentation"
echo ""
echo "🚀 Ready for Cloud Run deployment!"