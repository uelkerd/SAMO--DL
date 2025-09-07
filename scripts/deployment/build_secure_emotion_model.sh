#!/bin/bash

# Build secure emotion detection model for Cloud Run
# Addresses Docker Scout vulnerability findings

set -e

# Check for required tools
check_required_tools() {
    echo "🔧 Checking for required tools..."
    
    if ! command -v jq &> /dev/null; then
        echo "❌ Error: jq is not installed. Please install jq before running this script."
        echo "   Install with: brew install jq (macOS) or apt-get install jq (Ubuntu)"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        echo "❌ Error: docker is not installed. Please install docker before running this script."
        exit 1
    fi
    
    if ! docker scout --version &> /dev/null; then
        echo "❌ Error: Docker Scout is not available. Please ensure Docker Desktop is up to date."
        exit 1
    fi
    
    echo "✅ All required tools are available."
}

# Run checks
check_required_tools

echo "🔒 Building security-hardened emotion detection model..."

# Build the secure image
echo "📦 Building Docker image with security updates..."
docker buildx build \
    -f deployment/docker/Dockerfile.optimized-secure \
    -t emotion-detection-api:secure \
    --progress=plain \
    .

if [ "$?" -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo "✅ Secure image built successfully!"

# Run Docker Scout scan on the new image
echo "🔍 Running Docker Scout vulnerability scan..."
docker scout quickview emotion-detection-api:secure
if [ "$?" -ne 0 ]; then
    echo "❌ Docker Scout quickview scan failed!"
    exit 1
fi

echo "📊 Detailed vulnerability report:"
docker scout cves emotion-detection-api:secure --output json > scout_cves.json
if [ "$?" -ne 0 ]; then
    echo "❌ Docker Scout CVE scan failed!"
    exit 1
fi

# Check for critical vulnerabilities
if [ -f scout_cves.json ]; then
    critical_count=$(jq '[.cves[] | select(.severity == "critical")] | length' scout_cves.json 2>/dev/null || echo "0")
    if [ "$critical_count" -gt 0 ]; then
        echo "❌ Critical vulnerabilities detected ($critical_count). Halting deployment!"
        jq '.cves[] | select(.severity == "critical")' scout_cves.json
        exit 2
    fi
    echo "✅ No critical vulnerabilities detected."
    
    # Display vulnerability summary
    echo "📊 Vulnerability summary:"
    jq -r '.summary' scout_cves.json
else
    echo "⚠️ Could not generate vulnerability report."
fi

echo "🎯 Security recommendations:"
docker scout recommendations emotion-detection-api:secure
if [ "$?" -ne 0 ]; then
    echo "❌ Docker Scout recommendations scan failed!"
    exit 1
fi

echo "✅ Security scan completed!"
echo "📝 Image tagged as: emotion-detection-api:secure"

# Clean up temporary files
rm -f scout_cves.json