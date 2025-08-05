#!/bin/bash

# Immediate Cost Optimization Script
# This script optimizes current costs by scaling down expensive resources

set -e

PROJECT_ID="the-tendril-466607-n8"
REGION="us-central1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo "=== IMMEDIATE COST OPTIMIZATION ==="
echo "Project: $PROJECT_ID"
echo "Date: $(date)"
echo

# 1. Stop the expensive compute instance
log_info "Stopping expensive compute instance..."
if gcloud compute instances describe samo-dl-training-cpu --zone=us-central1-a &> /dev/null; then
    gcloud compute instances stop samo-dl-training-cpu --zone=us-central1-a
    log_success "Stopped compute instance: samo-dl-training-cpu"
else
    log_info "Compute instance already stopped or doesn't exist"
fi

# 2. Scale down Cloud Run service
log_info "Scaling down Cloud Run service..."
if gcloud run services describe emotion-detection-api --region=$REGION &> /dev/null; then
    gcloud run services update emotion-detection-api \
        --region=$REGION \
        --max-instances=1 \
        --min-instances=0
    log_success "Scaled down Cloud Run service to max 1 instance"
else
    log_info "Cloud Run service not found"
fi

# 3. Scale down Vertex AI endpoints
log_info "Scaling down Vertex AI endpoints..."
ENDPOINTS=$(gcloud ai endpoints list --region=$REGION --format="value(name)")
if [ ! -z "$ENDPOINTS" ]; then
    echo "$ENDPOINTS" | while read endpoint; do
        if [ ! -z "$endpoint" ]; then
            log_info "Scaling down Vertex AI endpoint: $endpoint"
            # Use the correct syntax for Vertex AI endpoint updates
            gcloud ai endpoints update $endpoint \
                --region=$REGION \
                --deployed-model-id=$(gcloud ai endpoints describe $endpoint --region=$REGION --format="value(deployedModels[0].id)") \
                --traffic-split=$(gcloud ai endpoints describe $endpoint --region=$REGION --format="value(deployedModels[0].id)")=0 || log_warning "Failed to scale down endpoint $endpoint"
        fi
    done
    log_success "Scaled down Vertex AI endpoints"
else
    log_info "No Vertex AI endpoints found"
fi

# 4. Check for other expensive resources
log_info "Checking for other expensive resources..."

# Check for running VMs
RUNNING_VMS=$(gcloud compute instances list --filter="status=RUNNING" --format="value(name,zone,machineType)" 2>/dev/null || echo "")
if [ ! -z "$RUNNING_VMS" ]; then
    echo "Running VMs found:"
    echo "$RUNNING_VMS"
    log_warning "Consider stopping these VMs if not needed"
else
    log_success "No running VMs found"
fi

# Check for expensive storage
log_info "Checking storage usage..."
gsutil ls -L gs://* 2>/dev/null | grep -E "(Bucket:|Total size:)" || log_info "No storage buckets found"

echo
log_success "Immediate cost optimization completed!"
echo
echo "Estimated cost savings:"
echo "  - Compute instance (n1-standard-16): ~$0.50/hour saved"
echo "  - Cloud Run scaling: Reduced from potential high usage"
echo "  - Vertex AI scaling: Reduced replica costs"
echo
echo "Monitor costs: ./scripts/cost-controls/check_costs.sh"
echo "Emergency shutdown: ./scripts/cost-controls/emergency_cost_reduction.sh"
echo "Billing dashboard: https://console.cloud.google.com/billing" 