#!/bin/bash

echo "🚀 Creating GPU Instance for SAMO-DL Training"
echo "=============================================="

# Set variables
INSTANCE_NAME="samo-dl-training"
ZONE="us-central1-a"
MACHINE_TYPE="n1-standard-4"
ACCELERATOR="type=nvidia-tesla-t4,count=1"
IMAGE_FAMILY="debian-11"
DISK_SIZE="50GB"

echo "📋 Instance Configuration:"
echo "   • Name: ${INSTANCE_NAME}"
echo "   • Zone: ${ZONE}"
echo "   • Machine Type: ${MACHINE_TYPE}"
echo "   • GPU: ${ACCELERATOR}"
echo "   • Image: ${IMAGE_FAMILY}"
echo "   • Disk Size: ${DISK_SIZE}"

echo ""
echo "🔧 Creating instance..."

# Create the instance
gcloud compute instances create "${INSTANCE_NAME}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --accelerator="${ACCELERATOR}" \
    --image-family="${IMAGE_FAMILY}" \
    --boot-disk-size="${DISK_SIZE}" \
    --metadata="install-nvidia-driver=True" \
    --maintenance-policy=TERMINATE \
    --restart-on-failure

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ GPU Instance created successfully!"
    echo ""
    echo "📋 Next Steps:"
    echo "   1. SSH into instance: gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE}"
    echo "   2. Set up environment: sudo apt-get update && sudo apt-get install -y python3-pip python3-venv git"
    echo "   3. Clone repository: git clone https://github.com/YOUR_USERNAME/SAMO--DL.git"
    echo "   4. Install dependencies: pip install torch transformers datasets scikit-learn"
    echo "   5. Run training: python scripts/focal_loss_training.py"
    echo ""
    echo "💰 Estimated cost: ~$0.50-2.00 per hour"
    echo "⏱️  Expected training time: 2-4 hours"
else
    echo ""
    echo "❌ Failed to create instance. Check the error message above."
    echo "🔧 Common issues:"
    echo "   • Insufficient quota for GPU instances"
    echo "   • Billing not enabled for the project"
    echo "   • API not enabled"
fi
