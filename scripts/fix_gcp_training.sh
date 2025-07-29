#!/bin/bash

echo "🔧 Fixing GCP Training Issue"
echo "============================"

# Copy the fixed script to GCP
echo "📋 Copying fixed focal loss training script to GCP..."
gcloud compute scp scripts/focal_loss_training_fixed.py minervae@samo-dl-training-cpu:~/SAMO-DL/scripts/ --zone=us-central1-a

if [ $? -eq 0 ]; then
    echo "✅ Script copied successfully!"
    echo ""
    echo "🚀 Running fixed focal loss training on GCP..."
    echo "Command: python3 scripts/focal_loss_training_fixed.py --epochs 5 --batch_size 8 --gamma 2.0 --alpha 0.25 --lr 2e-5 --max_length 256"
    echo ""
    echo "📋 SSH into GCP and run:"
    echo "   gcloud compute ssh samo-dl-training-cpu --zone=us-central1-a"
    echo "   cd ~/SAMO-DL"
    echo "   python3 scripts/focal_loss_training_fixed.py --epochs 5 --batch_size 8 --gamma 2.0 --alpha 0.25 --lr 2e-5 --max_length 256"
    echo ""
    echo "💡 Expected Timeline:"
    echo "   • Training time: 6-10 hours"
    echo "   • Expected F1 improvement: 13.2% → 35-45%"
    echo "   • Cost: ~$2-20 for complete training"
else
    echo "❌ Failed to copy script. Please check GCP connection."
fi
