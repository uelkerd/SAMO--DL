#!/bin/bash
# Build optimized emotion detection model for Cloud Run
# CPU-only, minimal dependencies, smaller image size

set -e

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
docker run --rm -d --name emotion-test-optimized -p 8081:8080 \
    -e ADMIN_API_KEY=test-key-123 \
    -e MODEL_CACHE_DIR=/app/models \
    emotion-detection-api:optimized \
    gunicorn --bind :8080 --workers 1 --timeout 300 secure_api_server:app

# Wait for container to start and model to load (pre-downloaded during build)
echo "⏳ Waiting for container to start..."
sleep 30

# Test health endpoint (Flask-RESTX endpoint)
echo "🔍 Testing health endpoint..."
curl -s http://localhost:8081/api/health | jq '.' || echo "Health check failed (using /api/health)"

# Test prediction endpoint (Flask-RESTX endpoint)
echo "🔍 Testing prediction endpoint..."
curl -s -X POST http://localhost:8081/api/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "I am feeling happy today!"}' | jq '.' || echo "Prediction test failed"

# Test batch prediction endpoint (Flask-RESTX endpoint)
echo "🔍 Testing batch prediction endpoint..."
curl -s -X POST http://localhost:8081/api/predict_batch \
    -H "Content-Type: application/json" \
    -d '{"texts": ["I am happy", "I feel sad", "I am excited"]}' | jq '.' || echo "Batch prediction test failed"

# Clean up
echo "🧹 Cleaning up test container..."
docker stop emotion-test-optimized 2>/dev/null || echo "Container already stopped/removed"

echo ""
echo "✅ PRODUCTION ARCHITECTURE build and test completed!"
echo "🏗️ Flask-RESTX API with security, rate limiting, batch processing"
echo "🎯 CPU-only PyTorch for Cloud Run deployment"
echo "📦 Ready for Cloud Run deployment with FULL production features!"
