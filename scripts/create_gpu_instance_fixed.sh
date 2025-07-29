#!/bin/bash

echo "🚀 Creating GPU Instance for SAMO-DL Training (Fixed)"
echo "====================================================="

# Set variables
INSTANCE_NAME="samo-dl-training"
ZONE="us-central1-a"
MACHINE_TYPE="n1-standard-4"
ACCELERATOR="type=nvidia-tesla-t4,count=1"
IMAGE_FAMILY="ubuntu-2004-lts"  # More reliable image family
DISK_SIZE="200GB"  # Increased for better performance

echo "📋 Instance Configuration:"
echo "   • Name: $INSTANCE_NAME"
echo "   • Zone: $ZONE"
echo "   • Machine Type: $MACHINE_TYPE"
echo "   • GPU: $ACCELERATOR"
echo "   • Image: $IMAGE_FAMILY"
echo "   • Disk Size: $DISK_SIZE"

echo ""
echo "🔧 Creating instance..."

# Create the instance with Ubuntu image
gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --accelerator=$ACCELERATOR \
    --image-family=$IMAGE_FAMILY \
    --boot-disk-size=$DISK_SIZE \
    --metadata="install-nvidia-driver=True" \
    --maintenance-policy=TERMINATE \
    --restart-on-failure

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ GPU Instance created successfully!"
    echo ""
    echo "📋 Next Steps:"
    echo "   1. SSH into instance: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
    echo "   2. Set up environment: sudo apt-get update && sudo apt-get install -y python3-pip python3-venv git"
    echo "   3. Clone repository: git clone https://github.com/YOUR_USERNAME/SAMO--DL.git"
    echo "   4. Install dependencies: pip install torch transformers datasets scikit-learn"
    echo "   5. Run training: python scripts/focal_loss_training.py"
    echo ""
    echo "💰 Estimated cost: ~$0.50-2.00 per hour"
    echo "⏱️  Expected training time: 2-4 hours"
else
    echo ""
    echo "❌ Failed to create instance. Trying alternative approach..."
    echo ""
    echo "🔧 Alternative: Using Deep Learning VM image..."

    # Try with Deep Learning VM image
    gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --accelerator=$ACCELERATOR \
        --image-family=deeplearning-platform-release \
        --image-project=deeplearning-platform-release \
        --boot-disk-size=$DISK_SIZE \
        --maintenance-policy=TERMINATE \
        --restart-on-failure

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ GPU Instance created successfully with Deep Learning VM!"
        echo "🎉 This image comes with PyTorch and CUDA pre-installed!"
        echo ""
        echo "📋 Next Steps:"
        echo "   1. SSH into instance: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
        echo "   2. Clone repository: git clone https://github.com/YOUR_USERNAME/SAMO--DL.git"
        echo "   3. Install additional dependencies: pip install transformers datasets scikit-learn"
        echo "   4. Run training: python scripts/focal_loss_training.py"
    else
        echo ""
        echo "❌ Both attempts failed. Please check:"
        echo "   • Billing is enabled for the project"
        echo "   • GPU quota is available"
        echo "   • APIs are enabled"
    fi
fi
