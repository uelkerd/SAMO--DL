#!/bin/bash

# Build secure emotion detection model for Cloud Run
# Addresses Docker Scout vulnerability findings

set -e

echo "🔒 Building security-hardened emotion detection model..."

# Build the secure image
echo "📦 Building Docker image with security updates..."
docker build \
    -f deployment/docker/Dockerfile.optimized-secure \
    -t emotion-detection-api:secure \
    .

echo "✅ Secure image built successfully!"

# Run Docker Scout scan on the new image
echo "🔍 Running Docker Scout vulnerability scan..."
docker scout quickview emotion-detection-api:secure

echo "📊 Detailed vulnerability report:"
docker scout cves emotion-detection-api:secure

echo "🎯 Security recommendations:"
docker scout recommendations emotion-detection-api:secure

echo "✅ Security scan completed!"
echo "📝 Image tagged as: emotion-detection-api:secure"
