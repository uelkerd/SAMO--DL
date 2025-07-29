#!/bin/bash

echo "🚀 Running Focal Loss Training on GCP"
echo "====================================="

# Change to project root directory
cd ~/SAMO-DL

echo "📁 Current directory: $(pwd)"
echo "📋 Available scripts:"
ls -la scripts/ | grep focal

echo ""
echo "🔧 Running focal loss training..."

# Run the training script from project root
python3 scripts/focal_loss_training_simple.py

echo ""
echo "✅ Training script completed!"
