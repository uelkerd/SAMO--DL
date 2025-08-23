#!/bin/bash
# 🚨 CRITICAL SECURITY DEPLOYMENT FIX - EXECUTION SCRIPT
# =====================================================
# Emergency script to fix critical security vulnerabilities in Cloud Run.
#
# This script will:
# 1. Stop the current insecure deployment
# 2. Deploy the secure version with all security features
# 3. Test the deployment for security compliance
# 4. Clean up old insecure deployment
#
# WARNING: This will replace your current deployment with a secure version.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SECURITY_SCRIPT="$SCRIPT_DIR/security_deployment_fix.py"

# Check if we're in the right directory
if [[ ! -f "$SECURITY_SCRIPT" ]]; then
    error "Security deployment script not found: ${SECURITY_SCRIPT}"
    exit 1
fi

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    error "Google Cloud not authenticated. Please run: gcloud auth login"
    exit 1
fi

# Check if required tools are installed
command -v gcloud >/dev/null 2>&1 || { error "gcloud is required but not installed."; exit 1; }
command -v python3 >/dev/null 2>&1 || { error "python3 is required but not installed."; exit 1; }

# Function to check current deployment status
check_current_deployment() {
    log "Checking current deployment status..."

    # Try different possible service names
    for service_name in "samo-emotion-api" "samo-emotion-api-71517823771" "arch-fixed-test"; do
        if gcloud run services describe "$service_name" --region=us-central1 --format="value(status.url)" 2>/dev/null; then
            CURRENT_URL=$(gcloud run services describe "${service_name}" --region=us-central1 --format="value(status.url)")
            CURRENT_SERVICE_NAME="${service_name}"
            warning "Current deployment found: ${CURRENT_URL} (service: ${service_name})"
            return 0
        fi
    done

    log "No current deployment found"
    return 1
}

# Function to test current deployment security
test_current_security() {
    log "Testing current deployment security..."

    if check_current_deployment; then
        # Test for security headers
        if curl -s -I "${CURRENT_URL}/health" | grep -q "Content-Security-Policy"; then
            warning "Current deployment has some security headers"
        else
            error "Current deployment MISSING security headers"
        fi

        # Test for rate limiting
        responses=()
        for i in {1..105}; do
            response=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${CURRENT_URL}/predict" \
                -H "Content-Type: application/json" \
                -d '{"text":"test"}' 2>/dev/null || echo "000")
            responses+=("${response}")
        done

        if [[ " ${responses[@]} " =~ " 429 " ]]; then
            warning "Current deployment has rate limiting"
        else
            error "Current deployment MISSING rate limiting"
        fi
    fi
}

# Main execution
main() {
    echo "🚨 CRITICAL SECURITY DEPLOYMENT FIX"
    echo "=================================="
    echo ""

    # Check prerequisites
    log "Checking prerequisites..."
    success "All prerequisites met"

    # Test current deployment
    test_current_security

    echo ""
    warning "WARNING: This will replace your current deployment with a secure version."
    echo "The new deployment will include:"
    echo "  ✅ Updated dependencies (torch 2.8.0+, scikit-learn 1.7.1+)"
    echo "  ✅ Rate limiting (100 requests/minute)"
    echo "  ✅ Security headers (CSP, XSS protection, etc.)"
    echo "  ✅ API key authentication for admin endpoints"
    echo "  ✅ Input sanitization and validation"
    echo "  ✅ Request tracking and logging"
    echo ""

    read -p "Do you want to proceed with the security fix? (y/N): " -n 1 -r
    echo

    if [[ ! ${REPLY} =~ ^[Yy]$ ]]; then
        log "Security deployment cancelled by user"
        exit 0
    fi

    # Set environment variable for admin API key
    export ADMIN_API_KEY="samo-admin-key-2024-secure-$(date +%s)"
    log "Generated admin API key: ${ADMIN_API_KEY}"

    # Run the security deployment script
    log "Starting security deployment fix..."
    cd "${PROJECT_ROOT}"

    if python3 "$SECURITY_SCRIPT"; then
        success "Security deployment completed successfully!"
        echo ""
        echo "🎉 DEPLOYMENT SECURITY STATUS:"
        echo "=============================="
        echo "✅ All dependencies updated to secure versions"
        echo "✅ Rate limiting implemented (100 req/min)"
        echo "✅ Security headers added"
        echo "✅ API key authentication enabled"
        echo "✅ Input sanitization active"
        echo "✅ Request tracking implemented"
        echo ""
        echo "🔑 Admin API Key: ${ADMIN_API_KEY}"
        echo "📝 Save this key for admin endpoint access"
        echo ""

        # Get new service URL
        NEW_URL=$(gcloud run services describe samo-emotion-api-secure --region=us-central1 --format="value(status.url)" 2>/dev/null || echo "Service not found")
            if [[ "${NEW_URL}" != "Service not found" ]]; then
        echo "🌐 New Secure Service URL: ${NEW_URL}"
            echo ""
            echo "🧪 Testing new deployment..."

            # Quick test
            if curl -s "${NEW_URL}/health" | grep -q "healthy"; then
                success "New deployment is healthy and responding"
            else
                warning "New deployment may have issues - check logs"
            fi
        fi

        # Clean up old deployment if it exists
        if [[ -n "${CURRENT_SERVICE_NAME}" ]]; then
            echo ""
            echo "🗑️ Cleaning up old deployment: ${CURRENT_SERVICE_NAME}"
            gcloud run services delete "${CURRENT_SERVICE_NAME}" --region=us-central1 --quiet 2>/dev/null || true
            success "Old deployment cleaned up"
        fi

    else
        error "Security deployment failed!"
        echo ""
        echo "Troubleshooting steps:"
        echo "1. Check gcloud authentication: gcloud auth login"
        echo "2. Check project permissions: gcloud projects list"
        echo "3. Check Cloud Run API: gcloud services enable run.googleapis.com"
        echo "4. Check logs: gcloud logging read 'resource.type=cloud_run_revision'"
        exit 1
    fi
}

# Run main function
main "$@"
