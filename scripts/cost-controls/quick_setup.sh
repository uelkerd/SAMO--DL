#!/bin/bash

# Quick Setup Script for GCP Cost Controls
# This script sets up all cost controls automatically

set -e  # Exit on any error

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

# Check if email is provided
if [ -z "$1" ]; then
    log_error "Please provide your email address as an argument"
    echo "Usage: $0 your-email@domain.com"
    exit 1
fi

EMAIL_ALERT="$1"

log_info "Setting up GCP cost controls for email: $EMAIL_ALERT"

# Update the email in the setup script
log_info "Updating email configuration..."
sed -i.bak "s/your-email@domain.com/$EMAIL_ALERT/g" scripts/cost-controls/setup_budget_alerts.sh

# Run the main setup script
log_info "Running budget alerts setup..."
./scripts/cost-controls/setup_budget_alerts.sh

# Test the monitoring script
log_info "Testing budget monitoring..."
./scripts/cost-controls/monitor_budget.sh

log_success "Cost controls setup completed!"
echo
echo "Your GCP cost controls are now active:"
echo "  - Budget alerts configured for $EMAIL_ALERT"
echo "  - Monitoring dashboard created"
echo "  - Emergency cost reduction script ready"
echo
echo "Monitor your costs:"
echo "  - Run: ./scripts/cost-controls/monitor_budget.sh"
echo "  - Dashboard: https://console.cloud.google.com/monitoring/dashboards"
echo "  - Billing: https://console.cloud.google.com/billing"
echo
echo "Emergency actions:"
echo "  - Emergency reduction: ./scripts/cost-controls/emergency_cost_reduction.sh"
echo "  - Recovery: ./scripts/cost-controls/recover_services.sh" 