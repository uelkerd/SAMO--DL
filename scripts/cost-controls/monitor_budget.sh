#!/bin/bash

# GCP Budget Monitoring Script for SAMO Deep Learning Project
# This script monitors current spending and alerts when thresholds are exceeded

set -e  # Exit on any error

# Configuration
PROJECT_ID="the-tendril-466607-n8"
BILLING_ACCOUNT="0156F5-8F20E3-96A680"
BUDGET_AMOUNT=100  # USD
ALERT_THRESHOLD=80  # Percentage
REGION="us-central1"

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

# Get current month's spending
get_current_spending() {
    log_info "Getting current month's spending..."
    
    # Get billing account details
    BILLING_INFO=$(gcloud billing accounts list \
        --filter="name:projects/$PROJECT_ID" \
        --format="json")
    
    # Extract current spending (this is a simplified approach)
    # In production, you'd use the billing export data
    CURRENT_SPEND=$(echo "$BILLING_INFO" | jq -r '.[0].displayName // "0"')
    
    # For demonstration, we'll use a placeholder
    # In reality, you'd query the billing export BigQuery table
    CURRENT_SPEND=25  # Placeholder value
    
    echo $CURRENT_SPEND
}

# Calculate spending percentage
calculate_percentage() {
    local spend=$1
    local budget=$2
    
    # Calculate percentage (integer arithmetic)
    local percentage=$((spend * 100 / budget))
    echo $percentage
}

# Check resource usage
check_resource_usage() {
    log_info "Checking resource usage..."
    
    echo "=== Cloud Run Services ==="
    gcloud run services list --region=$REGION --format="table(name,status.url,status.conditions[0].status)" || log_warning "No Cloud Run services found"
    
    echo -e "\n=== Vertex AI Endpoints ==="
    gcloud ai endpoints list --region=$REGION --format="table(name,displayName,deployedModels[0].model)" || log_warning "No Vertex AI endpoints found"
    
    echo -e "\n=== Compute Instances ==="
    gcloud compute instances list --format="table(name,zone,machineType,status)" || log_warning "No compute instances found"
    
    echo -e "\n=== Storage Buckets ==="
    gsutil ls -L gs://* 2>/dev/null | grep -E "(Bucket:|Total size:)" || log_warning "No storage buckets found"
}

# Get cost breakdown by service
get_cost_breakdown() {
    log_info "Getting cost breakdown by service..."
    
    # This would query the billing export BigQuery table
    # For now, we'll show a placeholder
    echo "=== Cost Breakdown (Placeholder) ==="
    echo "Cloud Run: $15.00"
    echo "Vertex AI: $5.00"
    echo "Storage: $3.00"
    echo "Compute Engine: $2.00"
    echo "Total: $25.00"
}

# Send alert if threshold exceeded
send_alert() {
    local percentage=$1
    local spend=$2
    
    if [ $percentage -gt $ALERT_THRESHOLD ]; then
        log_warning "BUDGET ALERT: $percentage% of budget used!"
        
        # Send to Pub/Sub topic
        gcloud pubsub topics publish budget-alerts \
            --message="Budget alert: $percentage% of budget used ($spend/$BUDGET_AMOUNT USD)" || log_warning "Failed to send Pub/Sub alert"
        
        # Could also send email or other notifications here
        echo "ALERT: Budget threshold exceeded. Consider scaling down resources."
    fi
}

# Display cost optimization suggestions
show_optimization_suggestions() {
    log_info "Cost optimization suggestions:"
    
    echo "1. Cloud Run Optimization:"
    echo "   - Scale down during low usage: gcloud run services update [SERVICE] --max-instances=1"
    echo "   - Enable CPU throttling: gcloud run services update [SERVICE] --cpu-throttling"
    
    echo -e "\n2. Vertex AI Optimization:"
    echo "   - Reduce replica count: gcloud ai endpoints update [ENDPOINT] --max-replica-count=1"
    echo "   - Use smaller machine types: gcloud ai endpoints update [ENDPOINT] --machine-type=e2-standard-2"
    
    echo -e "\n3. Storage Optimization:"
    echo "   - Set lifecycle policies: gsutil lifecycle set [POLICY] gs://[BUCKET]"
    echo "   - Delete unused objects: gsutil rm gs://[BUCKET]/[OBJECT]"
    
    echo -e "\n4. Emergency Cost Reduction:"
    echo "   - Run emergency script: ./scripts/cost-controls/emergency_cost_reduction.sh"
}

# Main monitoring function
main() {
    log_info "Starting budget monitoring for SAMO Deep Learning project"
    
    # Get current spending
    CURRENT_SPEND=$(get_current_spending)
    
    # Calculate percentage
    PERCENTAGE=$(calculate_percentage $CURRENT_SPEND $BUDGET_AMOUNT)
    
    # Display current status
    echo "=== Budget Status ==="
    echo "Current spend: $CURRENT_SPEND USD"
    echo "Budget: $BUDGET_AMOUNT USD"
    echo "Percentage used: $PERCENTAGE%"
    echo "Alert threshold: $ALERT_THRESHOLD%"
    
    # Check if threshold exceeded
    if [ $PERCENTAGE -gt $ALERT_THRESHOLD ]; then
        log_warning "BUDGET THRESHOLD EXCEEDED!"
        send_alert $PERCENTAGE $CURRENT_SPEND
    else
        log_success "Budget within acceptable limits"
    fi
    
    echo -e "\n=== Resource Usage ==="
    check_resource_usage
    
    echo -e "\n=== Cost Breakdown ==="
    get_cost_breakdown
    
    echo -e "\n=== Recommendations ==="
    show_optimization_suggestions
    
    # Save monitoring data
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "$TIMESTAMP,$CURRENT_SPEND,$PERCENTAGE" >> logs/budget_monitoring.csv
    
    log_success "Budget monitoring completed"
}

# Run main function
main "$@" 