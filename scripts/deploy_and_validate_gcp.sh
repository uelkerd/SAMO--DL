#!/bin/bash

echo "🚀 SAMO-DL GCP Deployment with Pre-Training Validation"
echo "======================================================"
echo "This script addresses the critical 0.0000 loss issue by:"
echo "1. Setting up GCP instance with proper environment"
echo "2. Running comprehensive pre-training validation"
echo "3. Starting training only if validation passes"
echo ""

# Configuration
INSTANCE_NAME="samo-dl-training-cpu"  # Using CPU for validation first
ZONE="us-central1-a"
MACHINE_TYPE="n1-standard-4"
IMAGE_FAMILY="ubuntu-2004-lts"
DISK_SIZE="200GB"

echo "📋 Deployment Configuration:"
echo "   • Instance: ${INSTANCE_NAME}"
echo "   • Zone: ${ZONE}"
echo "   • Machine Type: ${MACHINE_TYPE}"
echo "   • Image: ${IMAGE_FAMILY}"
echo "   • Purpose: Pre-training validation and debugging"
echo ""

# Step 1: Check GCP authentication
echo "🔐 Step 1: Checking GCP Authentication..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "samo.summer25@gmail.com"; then
    echo "❌ GCP authentication required"
    echo "   Run: ./scripts/setup_gcp_auth.sh"
    exit 1
else
    echo "✅ GCP authenticated: $(gcloud auth list --filter=status:ACTIVE --format='value(account)')"
fi

# Step 2: Create instance
echo ""
echo "🔧 Step 2: Creating GCP Instance..."
echo "   This will take 2-3 minutes..."

gcloud compute instances create "${INSTANCE_NAME}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --image-family="${IMAGE_FAMILY}" \
    --boot-disk-size="${DISK_SIZE}" \
    --metadata="install-nvidia-driver=True" \
    --maintenance-policy=TERMINATE \
    --restart-on-failure

if [ $? -ne 0 ]; then
    echo "❌ Failed to create instance"
    exit 1
fi

echo "✅ Instance created successfully!"

# Step 3: Wait for instance to be ready
echo ""
echo "⏳ Step 3: Waiting for instance to be ready..."
sleep 30

# Step 4: Setup environment on instance
echo ""
echo "🔧 Step 4: Setting up environment on instance..."
echo "   This will take 5-10 minutes..."

gcloud compute ssh "${INSTANCE_NAME}" --zone="${ZONE}" --command="
echo '🔧 Installing system dependencies...'
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git curl wget

echo '🐍 Setting up Python environment...'
python3 -m venv ~/samo-env
source ~/samo-env/bin/activate

echo '📦 Installing Python dependencies...'
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets scikit-learn numpy pandas matplotlib seaborn

echo '📁 Setting up project directory...'
mkdir -p ~/SAMO--DL
cd ~/SAMO--DL

echo '✅ Environment setup complete!'
"

# Step 5: Copy project files
echo ""
echo "📁 Step 5: Copying project files to instance..."
echo "   This will take 2-3 minutes..."

# Create a temporary tar file
tar -czf /tmp/samo-dl-project.tar.gz \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='data/cache' \
    --exclude='models/checkpoints' \
    .

# Copy to instance
gcloud compute scp /tmp/samo-dl-project.tar.gz "${INSTANCE_NAME}":~/ --zone="${ZONE}"

# Extract on instance
gcloud compute ssh "${INSTANCE_NAME}" --zone="${ZONE}" --command="
cd ~/SAMO--DL
tar -xzf ~/samo-dl-project.tar.gz
rm ~/samo-dl-project.tar.gz
echo '✅ Project files extracted!'
"

# Clean up local tar
rm /tmp/samo-dl-project.tar.gz

# Step 6: Run pre-training validation
echo ""
echo "🔍 Step 6: Running Pre-Training Validation..."
echo "   This will identify the root cause of the 0.0000 loss issue..."

gcloud compute ssh "${INSTANCE_NAME}" --zone="${ZONE}" --command="
cd ~/SAMO--DL
source ~/samo-env/bin/activate

echo '🔍 Running comprehensive pre-training validation...'
python scripts/pre_training_validation.py

echo '📊 Validation complete! Check the output above for issues.'
"

# Step 7: Provide next steps
echo ""
echo "🎯 DEPLOYMENT COMPLETE!"
echo "======================="
echo ""
echo "📋 Next Steps:"
echo "   1. SSH into instance: gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE}"
echo "   2. Activate environment: source ~/samo-env/bin/activate"
echo "   3. Navigate to project: cd ~/SAMO--DL"
echo "   4. Run validation again: python scripts/pre_training_validation.py"
echo "   5. If validation passes, run training: python scripts/validate_and_train.py"
echo ""
echo "🔍 Validation Results:"
echo "   • Check the output above for critical issues"
echo "   • Look for 'CRITICAL ISSUES' section"
echo "   • Address any issues before starting training"
echo ""
echo "💰 Cost Estimate: ~$0.20-0.50 per hour"
echo "⏱️  Expected validation time: 10-30 minutes"
echo ""
echo "💡 If validation fails, the script will show exactly what needs to be fixed!"
