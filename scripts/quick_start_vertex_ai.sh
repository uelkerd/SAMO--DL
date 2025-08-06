#!/bin/bash

echo "🚀 SAMO Deep Learning - Vertex AI Quick Start"
echo "=============================================="
echo "This script will set up Vertex AI and solve the 0.0000 loss issue"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "Please run this script from the SAMO--DL project root directory"
    exit 1
fi

print_status "Starting Vertex AI setup..."

# Step 1: Check if gcloud is installed
print_status "Checking Google Cloud CLI..."
if ! command -v gcloud &> /dev/null; then
    print_warning "Google Cloud CLI not found. Installing..."
    curl https://sdk.cloud.google.com | bash
    exec -l "${SHELL}"
    print_success "Google Cloud CLI installed"
else
    print_success "Google Cloud CLI found: $(gcloud --version | head -n 1)"
fi

# Step 2: Check authentication
print_status "Checking GCP authentication..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    print_warning "Not authenticated with GCP. Please authenticate..."
    gcloud auth login
    print_success "GCP authentication completed"
else
    print_success "Already authenticated with GCP"
fi

# Step 3: Get project ID
print_status "Getting GCP project ID..."
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)

if [ -z "$PROJECT_ID" ]; then
    print_warning "No project ID set. Please enter your GCP project ID:"
    read -p "Project ID: " PROJECT_ID
    gcloud config set project "$PROJECT_ID"
    print_success "Project ID set to: ${PROJECT_ID}"
else
    print_success "Using project ID: ${PROJECT_ID}"
fi

# Step 4: Enable required APIs
print_status "Enabling required APIs..."
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable logging.googleapis.com
print_success "Required APIs enabled"

# Step 5: Install Vertex AI dependencies
print_status "Installing Vertex AI dependencies..."
pip install --upgrade google-cloud-aiplatform google-cloud-storage google-cloud-logging google-auth
print_success "Vertex AI dependencies installed"

# Step 6: Set environment variables
print_status "Setting environment variables..."
export GOOGLE_CLOUD_PROJECT="${PROJECT_ID}"
export VERTEX_AI_REGION="us-central1"
print_success "Environment variables set"

# Step 7: Run Vertex AI setup
print_status "Running Vertex AI setup..."
python scripts/vertex_ai_setup.py
if [ $? -eq 0 ]; then
    print_success "Vertex AI setup completed"
else
    print_error "Vertex AI setup failed"
    exit 1
fi

# Step 8: Run validation
print_status "Running validation to identify 0.0000 loss root cause..."
python scripts/vertex_ai_training.py --validation_mode
if [ $? -eq 0 ]; then
    print_success "Validation completed successfully"
else
    print_warning "Validation found issues. Check logs for details."
fi

# Step 9: Provide next steps
echo ""
echo "🎉 VERTEX AI SETUP COMPLETED!"
echo "=============================="
echo ""
echo "📋 Next Steps:"
echo "1. Check Vertex AI Console: https://console.cloud.google.com/vertex-ai"
echo "2. Review validation results above"
echo "3. Start training with:"
echo "   python scripts/vertex_ai_training.py --use_focal_loss --class_weights"
echo ""
echo "🔧 Configuration:"
echo "   • Learning rate: 2e-6 (optimized for stability)"
echo "   • Focal loss: Enabled (addresses class imbalance)"
echo "   • Class weights: Enabled (handles imbalanced data)"
echo "   • GPU: NVIDIA_TESLA_T4 (automatic allocation)"
echo ""
echo "📊 Expected Results:"
echo "   • F1 Score: 13.2% → >75% (target)"
echo "   • Training Loss: Non-zero, decreasing values"
echo "   • No more 0.0000 loss issues"
echo ""
echo "💡 Benefits:"
echo "   • Managed infrastructure (no more terminal issues)"
echo "   • Automatic hyperparameter tuning"
echo "   • Built-in monitoring and alerting"
echo "   • Scalable training and deployment"
echo "   • Cost optimization"
echo ""
echo "🚀 Ready to solve the 0.0000 loss issue and achieve >75% F1 score!" 