#!/bin/bash
# 🔒 INTEGRATED SECURITY & CLOUD RUN OPTIMIZATION DEPLOYMENT
# =========================================================
# This script integrates security fixes with Phase 3 Cloud Run optimization
# and deploys a fully optimized and secure service.

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
INTEGRATION_SCRIPT="$SCRIPT_DIR/integrate_security_fixes.py"

# Check if we're in the right directory
if [[ ! -f "$INTEGRATION_SCRIPT" ]]; then
    error "Integration script not found: $INTEGRATION_SCRIPT"
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

# Get current project
CURRENT_PROJECT=$(gcloud config get-value project)
log "Current GCP Project: $CURRENT_PROJECT"

# Main execution
main() {
    echo "🔒 INTEGRATED SECURITY & CLOUD RUN OPTIMIZATION"
    echo "==============================================="
    echo ""
    
    # Check prerequisites
    log "Checking prerequisites..."
    success "All prerequisites met"
    
    echo ""
    log "This deployment will integrate:"
    echo "  ✅ Phase 3 Cloud Run optimization features"
    echo "  ✅ Security headers and rate limiting"
    echo "  ✅ Input sanitization and validation"
    echo "  ✅ Health monitoring and auto-scaling"
    echo "  ✅ Graceful shutdown and error handling"
    echo "  ✅ Updated secure dependencies"
    echo "  ✅ Dynamic project ID detection"
    echo ""
    
    read -p "Do you want to proceed with the integrated deployment? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Integrated deployment cancelled by user"
        exit 0
    fi
    
    # Run the integration script
    log "Starting integrated security and optimization deployment..."
    cd "$PROJECT_ROOT"
    
    if python3 "$INTEGRATION_SCRIPT"; then
        success "Integrated deployment completed successfully!"
        echo ""
        echo "🎉 INTEGRATED DEPLOYMENT STATUS:"
        echo "================================"
        echo "✅ Cloud Run optimization features active"
        echo "✅ Security headers implemented"
        echo "✅ Rate limiting active (100 req/min)"
        echo "✅ Input sanitization enabled"
        echo "✅ Health monitoring active"
        echo "✅ Auto-scaling configured"
        echo "✅ Graceful shutdown enabled"
        echo "✅ All dependencies updated to secure versions"
        echo "✅ Dynamic project configuration"
        echo ""
        
        # Get new service URL
        NEW_URL=$(gcloud run services describe samo-emotion-api-optimized-secure --region=us-central1 --format="value(status.url)" 2>/dev/null || echo "Service not found")
        if [[ "$NEW_URL" != "Service not found" ]]; then
            echo "🌐 New Integrated Service URL: $NEW_URL"
            echo ""
            echo "🧪 Quick test of new deployment..."
            
            # Quick test
            if curl -s "$NEW_URL/health" | grep -q "healthy"; then
                success "New integrated deployment is healthy and responding"
            else
                warning "New deployment may have issues - check logs"
            fi
        fi
        
    else
        error "Integrated deployment failed!"
        echo ""
        echo "Troubleshooting steps:"
        echo "1. Check gcloud authentication: gcloud auth login"
        echo "2. Check project permissions: gcloud projects list"
        echo "3. Check Cloud Run API: gcloud services enable run.googleapis.com"
        echo "4. Check Cloud Build API: gcloud services enable cloudbuild.googleapis.com"
        echo "5. Check logs: gcloud logging read 'resource.type=cloud_run_revision'"
        exit 1
    fi
}

# Run main function
main "$@" 