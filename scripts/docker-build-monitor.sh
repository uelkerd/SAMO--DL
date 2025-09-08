#!/bin/bash

# Docker Build Monitor Script
# Helps monitor and troubleshoot Docker builds

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKERFILE="${1:-deployment/docker/Dockerfile.optimized-secure}"
IMAGE_NAME="${2:-samo-complete-api:latest}"

echo "🐳 Docker Build Monitor"
echo "======================"
echo "Project Root: $PROJECT_ROOT"
echo "Dockerfile: $DOCKERFILE"
echo "Image Name: $IMAGE_NAME"
echo ""

# Check if build is already running
if pgrep -f "docker build" > /dev/null; then
    echo "⚠️  Docker build process already running!"
    echo "Process details:"
    ps aux | grep "docker build" | grep -v grep
    echo ""
    echo "To stop the build, run: docker build --no-cache --progress=plain -t $IMAGE_NAME -f $DOCKERFILE ."
    exit 1
fi

# Pre-build checks
echo "🔍 Pre-build checks..."
echo "Checking disk space..."
df -h | grep -E "(Filesystem|Size|Avail)"
echo ""

echo "Checking Docker system..."
docker system df
echo ""

echo "Checking network connectivity..."
ping -c 2 huggingface.co > /dev/null 2>&1 && echo "✓ Hugging Face reachable" || echo "✗ Hugging Face unreachable"
ping -c 2 cdn-lfs.huggingface.co > /dev/null 2>&1 && echo "✓ Hugging Face CDN reachable" || echo "✗ Hugging Face CDN unreachable"
echo ""

# Start build with monitoring
echo "🏗️  Starting Docker build..."
echo "Command: docker build --no-cache --progress=plain -t $IMAGE_NAME -f $DOCKERFILE ."
echo ""

# Start build and capture start time
START_TIME=$(date +%s)
docker build --no-cache --progress=plain -t $IMAGE_NAME -f $DOCKERFILE . 2>&1 | tee build.log
BUILD_EXIT_CODE=${PIPESTATUS[0]}

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Build completed successfully!"
    echo "Duration: $DURATION seconds ($(($DURATION / 60)) minutes)"
    echo "Image: $IMAGE_NAME"
    echo ""

    # Show image size
    docker images $IMAGE_NAME --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
else
    echo ""
    echo "❌ Build failed with exit code $BUILD_EXIT_CODE"
    echo "Duration: $DURATION seconds ($(($DURATION / 60)) minutes)"
    echo ""
    echo "Last 20 lines of build output:"
    tail -20 build.log
    echo ""
    echo "💡 Troubleshooting tips:"
    echo "1. Check build.log for detailed error messages"
    echo "2. Try: docker system prune -a (removes unused images)"
    echo "3. Try: docker buildx prune (clears build cache)"
    echo "4. Use Dockerfile.fast-build for faster builds (models downloaded at runtime)"
fi
