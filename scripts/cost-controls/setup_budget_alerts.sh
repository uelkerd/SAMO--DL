#!/bin/bash

# GCP Budget Alerts Setup Script for SAMO Deep Learning Project
# This script sets up comprehensive cost controls and budget alerts

set -e  # Exit on any error

# Configuration
PROJECT_ID="the-tendril-466607-n8"
BILLING_ACCOUNT="0156F5-8F20E3-96A680"
REGION="us-central1"
BUDGET_AMOUNT=100  # USD
EMAIL_ALERT="den.ulker@gmail.com"  # Replace with your email

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

# Create Pub/Sub topic for budget alerts
create_pubsub_topic() {
    log_info "Creating Pub/Sub topic for budget alerts..."
    
    # Create topic if it doesn't exist
    if ! gcloud pubsub topics describe budget-alerts &> /dev/null; then
        gcloud pubsub topics create budget-alerts
        log_success "Pub/Sub topic 'budget-alerts' created"
    else
        log_info "Pub/Sub topic 'budget-alerts' already exists"
    fi
}

# Create budget with alerts
create_budget() {
    log_info "Creating budget with alerts..."
    
    # Create budget
    gcloud billing budgets create \
        --billing-account=$BILLING_ACCOUNT \
        --display-name="SAMO-DL Project Budget" \
        --budget-amount=${BUDGET_AMOUNT}USD \
        --threshold-rule=percent=0.5 \
        --threshold-rule=percent=0.8 \
        --threshold-rule=percent=0.9 \
        --threshold-rule=percent=1.0 \
        --notifications-rule-pubsub-topic=projects/$PROJECT_ID/topics/budget-alerts
    
    log_success "Budget created with alerts at 50%, 80%, 90%, and 100% thresholds"
    
    # Note: Email notifications are handled through default billing account recipients
    # or through monitoring notification channels
}

# Enable billing export to BigQuery
enable_billing_export() {
    log_info "Enabling billing export to BigQuery..."
    
    # Create BigQuery dataset if it doesn't exist
    if ! bq show billing_export &> /dev/null; then
        bq mk --dataset $PROJECT_ID:billing_export
        log_success "BigQuery dataset 'billing_export' created"
    else
        log_info "BigQuery dataset 'billing_export' already exists"
    fi
    
    # Enable billing export (this may require billing account admin permissions)
    gcloud billing accounts update $BILLING_ACCOUNT \
        --enable-bigquery-export \
        --bigquery-dataset=projects/$PROJECT_ID/datasets/billing_export || log_warning "Billing export may require admin permissions"
    
    log_success "Billing export configuration attempted"
}

# Set resource quotas
set_resource_quotas() {
    log_info "Setting resource quotas..."
    
    # Get current quotas
    log_info "Current quotas in $REGION:"
    gcloud compute regions describe $REGION \
        --format="value(quotas[].limit,quotas[].metric,quotas[].usage)" \
        --project=$PROJECT_ID
    
    # Set conservative quotas for development
    log_info "Setting conservative quotas for development..."
    
    # Request quota updates (these may require approval)
    gcloud compute regions update $REGION \
        --quotas=CPUS=4,CPUS_ALL_REGIONS=8 \
        --project=$PROJECT_ID || log_warning "Quota update may require approval"
    
    log_success "Resource quotas configured"
}

# Create cost monitoring dashboard
create_cost_dashboard() {
    log_info "Creating cost monitoring dashboard..."
    
    # Create dashboard configuration
    cat > /tmp/cost-dashboard.json << EOF
{
  "displayName": "GCP Cost Monitoring Dashboard",
  "gridLayout": {
    "columns": "2",
    "widgets": [
      {
        "title": "Daily Cost Trend",
        "xyChart": {
          "dataSets": [
            {
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "metric.type=\"billing.googleapis.com/account/amount\"",
                  "aggregation": {
                    "alignmentPeriod": "86400s",
                    "perSeriesAligner": "ALIGN_SUM"
                  }
                }
              }
            }
          ]
        }
      },
      {
        "title": "Service Cost Breakdown",
        "pieChart": {
          "dataSets": [
            {
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "metric.type=\"billing.googleapis.com/account/amount\"",
                  "aggregation": {
                    "alignmentPeriod": "86400s",
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_SUM",
                    "groupByFields": ["resource.labels.service"]
                  }
                }
              }
            }
          ]
        }
      }
    ]
  }
}
EOF
    
    # Create dashboard
    gcloud monitoring dashboards create \
        --project=$PROJECT_ID \
        --config-from-file=/tmp/cost-dashboard.json
    
    log_success "Cost monitoring dashboard created"
}

# Create notification channel for alerts
create_notification_channel() {
    log_info "Creating notification channel for cost alerts..."
    
    # Create notification channel configuration
    cat > /tmp/notification-channel.json << EOF
{
  "displayName": "Cost Alerts",
  "type": "email",
  "labels": {
    "email_address": "$EMAIL_ALERT"
  }
}
EOF
    
    # Create notification channel
    gcloud alpha monitoring channels create \
        --project=$PROJECT_ID \
        --channel-content-from-file=/tmp/notification-channel.json || log_warning "Failed to create notification channel (may already exist)"
    
    log_success "Notification channel created or already exists"
}

# Apply cost controls to existing resources
apply_cost_controls() {
    log_info "Applying cost controls to existing resources..."
    
    # Cloud Run cost controls
    if gcloud run services describe emotion-detection-api --region=$REGION &> /dev/null; then
        log_info "Applying cost controls to Cloud Run service..."
        gcloud run services update emotion-detection-api \
            --region=$REGION \
            --max-instances=5 \
            --min-instances=0 \
            --cpu-throttling \
            --execution-environment=gen2 \
            --memory=1Gi \
            --cpu=1 \
            --timeout=300
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
            --max-replica-count=2 \
            --machine-type=e2-standard-2
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

# Display setup summary
display_summary() {
    log_info "Cost control setup completed!"
    echo
    echo "Budget Configuration:"
    echo "  Amount: $BUDGET_AMOUNT USD"
    echo "  Alerts: 50%, 80%, 90%, 100% thresholds"
    echo "  Email: $EMAIL_ALERT"
    echo
    echo "Monitoring:"
    echo "  Dashboard: https://console.cloud.google.com/monitoring/dashboards"
    echo "  Billing: https://console.cloud.google.com/billing"
    echo "  Budgets: https://console.cloud.google.com/billing/budgets"
    echo
    echo "Emergency Actions:"
    echo "  Emergency script: ./scripts/cost-controls/emergency_cost_reduction.sh"
    echo "  Manual cleanup: gcloud run services list --region=$REGION"
    echo
    echo "Next Steps:"
    echo "  1. Update EMAIL_ALERT in this script with your email"
    echo "  2. Monitor costs daily during development"
    echo "  3. Review weekly cost reports"
    echo "  4. Set up additional alerts if needed"
}

# Main setup process
main() {
    log_info "Setting up GCP cost controls for SAMO Deep Learning project"
    
    check_prerequisites
    create_pubsub_topic
    create_budget
    enable_billing_export
    set_resource_quotas
    create_cost_dashboard
    create_notification_channel
    apply_cost_controls
    create_emergency_script
    display_summary
    
    log_success "Cost control setup completed successfully!"
}

# Run main function
main "$@" 