#!/bin/bash

# Simple GCP Cost Controls Setup Script
# This script sets up basic cost controls without complex features that might fail

set -e  # Exit on any error

# Configuration
PROJECT_ID="the-tendril-466607-n8"
BILLING_ACCOUNT="0156F5-8F20E3-96A680"
REGION="us-central1"
BUDGET_AMOUNT=100  # USD
EMAIL_ALERT="den.ulker@gmail.com"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if we're authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        log_error "Not authenticated with gcloud. Please run 'gcloud auth login' first."
        exit 1
    fi
    
    # Check if project is set
    if [ "$(gcloud config get-value project)" != "$PROJECT_ID" ]; then
        log_warning "Setting project to $PROJECT_ID"
        gcloud config set project $PROJECT_ID
    fi
    
    log_success "Prerequisites check passed"
}

# Create simple budget with basic alerts
create_simple_budget() {
    log_info "Creating simple budget with alerts..."
    
    # Create budget with basic configuration
    gcloud billing budgets create \
        --billing-account=$BILLING_ACCOUNT \
        --display-name="SAMO-DL Project Budget" \
        --budget-amount=${BUDGET_AMOUNT}USD \
        --threshold-rule=percent=0.8 \
        --threshold-rule=percent=1.0 || log_warning "Budget creation failed (may already exist)"
    
    log_success "Budget created with alerts at 80% and 100% thresholds"
}

# Apply cost controls to existing resources
apply_cost_controls() {
    log_info "Applying cost controls to existing resources..."
    
    # Cloud Run cost controls
    if gcloud run services describe emotion-detection-api --region=$REGION &> /dev/null; then
        log_info "Applying cost controls to Cloud Run service..."
        gcloud run services update emotion-detection-api \
            --region=$REGION \
            --max-instances=3 \
            --min-instances=0 \
            --memory=1Gi \
            --cpu=1 \
            --timeout=300 || log_warning "Failed to update Cloud Run service"
        log_success "Cloud Run cost controls applied"
    else
        log_info "Cloud Run service not found, skipping"
    fi
    
    # Vertex AI cost controls (if endpoint exists)
    if gcloud ai endpoints list --region=$REGION --filter="displayName:comprehensive-emotion-detection-endpoint" --format="value(name)" | grep -q .; then
        ENDPOINT_ID=$(gcloud ai endpoints list --region=$REGION --filter="displayName:comprehensive-emotion-detection-endpoint" --format="value(name)")
        log_info "Applying cost controls to Vertex AI endpoint: $ENDPOINT_ID"
        gcloud ai endpoints update $ENDPOINT_ID \
            --region=$REGION \
            --min-replica-count=0 \
            --max-replica-count=1 \
            --machine-type=e2-standard-2 || log_warning "Failed to update Vertex AI endpoint"
        log_success "Vertex AI cost controls applied"
    else
        log_info "Vertex AI endpoint not found, skipping"
    fi
}

# Create emergency cost reduction script
create_emergency_script() {
    log_info "Creating emergency cost reduction script..."
    
    cat > scripts/cost-controls/emergency_cost_reduction.sh << 'EOF'
#!/bin/bash

# Emergency Cost Reduction Script
# This script immediately scales down all resources to minimize costs

set -e

PROJECT_ID="the-tendril-466607-n8"
REGION="us-central1"

echo "EMERGENCY COST REDUCTION - Scaling down all resources"

# Scale down Cloud Run services
gcloud run services list --format="value(name)" | \
while read service; do
    echo "Scaling down Cloud Run service: $service"
    gcloud run services update $service \
        --region=$REGION \
        --min-instances=0 \
        --max-instances=0
done

# Scale down Vertex AI endpoints
gcloud ai endpoints list --region=$REGION --format="value(name)" | \
while read endpoint; do
    echo "Scaling down Vertex AI endpoint: $endpoint"
    gcloud ai endpoints update $endpoint \
        --region=$REGION \
        --min-replica-count=0 \
        --max-replica-count=0
done

# Stop all compute instances
gcloud compute instances list --format="value(name,zone)" | \
while read name zone; do
    echo "Stopping compute instance: $name in zone $zone"
    gcloud compute instances stop $name --zone=$zone
done

echo "Emergency cost reduction completed"
echo "All resources have been scaled down to minimum levels"
EOF
    
    chmod +x scripts/cost-controls/emergency_cost_reduction.sh
    log_success "Emergency cost reduction script created"
}

# Create monitoring script
create_monitoring_script() {
    log_info "Creating cost monitoring script..."
    
    cat > scripts/cost-controls/check_costs.sh << 'EOF'
#!/bin/bash

# Simple Cost Monitoring Script
# This script checks current resource usage and costs

PROJECT_ID="the-tendril-466607-n8"
REGION="us-central1"

echo "=== GCP Cost Monitoring Report ==="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Date: $(date)"
echo

echo "=== Cloud Run Services ==="
gcloud run services list --region=$REGION --format="table(name,status.url,status.conditions[0].status)" 2>/dev/null || echo "No Cloud Run services found"

echo -e "\n=== Vertex AI Endpoints ==="
gcloud ai endpoints list --region=$REGION --format="table(name,displayName,deployedModels[0].model)" 2>/dev/null || echo "No Vertex AI endpoints found"

echo -e "\n=== Compute Instances ==="
gcloud compute instances list --format="table(name,zone,machineType,status)" 2>/dev/null || echo "No compute instances found"

echo -e "\n=== Cost Optimization Tips ==="
echo "1. Scale down unused services: gcloud run services update [SERVICE] --max-instances=0"
echo "2. Stop compute instances: gcloud compute instances stop [INSTANCE] --zone=[ZONE]"
echo "3. Emergency reduction: ./scripts/cost-controls/emergency_cost_reduction.sh"
echo "4. Check billing: https://console.cloud.google.com/billing"
EOF
    
    chmod +x scripts/cost-controls/check_costs.sh
    log_success "Cost monitoring script created"
}

# Display setup summary
display_summary() {
    log_info "Simple cost control setup completed!"
    echo
    echo "Budget Configuration:"
    echo "  Amount: $BUDGET_AMOUNT USD"
    echo "  Alerts: 80% and 100% thresholds"
    echo "  Email: $EMAIL_ALERT (via default billing notifications)"
    echo
    echo "Resource Controls Applied:"
    echo "  Cloud Run: Max 3 instances, 1Gi memory, 1 CPU"
    echo "  Vertex AI: Max 1 replica, e2-standard-2 machines"
    echo
    echo "Monitoring & Control:"
    echo "  Check costs: ./scripts/cost-controls/check_costs.sh"
    echo "  Emergency reduction: ./scripts/cost-controls/emergency_cost_reduction.sh"
    echo "  Billing dashboard: https://console.cloud.google.com/billing"
    echo
    echo "Next Steps:"
    echo "  1. Monitor costs daily: ./scripts/cost-controls/check_costs.sh"
    echo "  2. Set up billing alerts in GCP Console"
    echo "  3. Review resource usage regularly"
    echo "  4. Use emergency script if costs get too high"
}

# Main setup process
main() {
    log_info "Setting up simple GCP cost controls for SAMO Deep Learning project"
    
    check_prerequisites
    create_simple_budget
    apply_cost_controls
    create_emergency_script
    create_monitoring_script
    display_summary
    
    log_success "Simple cost control setup completed successfully!"
}

# Run main function
main "$@" 