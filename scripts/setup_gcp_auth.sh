#!/bin/bash

echo "🔐 GCP Authentication Setup for samo.summer25@gmail.com"
echo "========================================================"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Installing..."

    # Try Homebrew first
    if command -v brew &> /dev/null; then
        echo "📦 Installing via Homebrew..."
        brew install google-cloud-sdk
    else
        echo "⚠️  Homebrew not found. Please install gcloud manually:"
        echo "   https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
else
    echo "✅ gcloud CLI found: $(gcloud --version | head -1)"
fi

echo ""
echo "🔑 Setting up authentication..."

# Initialize gcloud
echo "📋 Running gcloud init..."
echo "   This will open your browser for authentication with samo.summer25@gmail.com"
gcloud init

# Set up application default credentials
echo ""
echo "🔐 Setting up application default credentials..."
gcloud auth application-default login

# Show current configuration
echo ""
echo "📊 Current Configuration:"
echo "========================="
echo "Account: $(gcloud config get-value account)"
echo "Project: $(gcloud config get-value project)"
echo "Region: $(gcloud config get-value compute/region)"
echo "Zone: $(gcloud config get-value compute/zone)"

# Enable required APIs
echo ""
echo "🚀 Enabling required APIs..."
PROJECT_ID=$(gcloud config get-value project)
gcloud services enable compute.googleapis.com --project="${PROJECT_ID}"
gcloud services enable aiplatform.googleapis.com --project="${PROJECT_ID}"

echo ""
echo "✅ GCP Setup Complete!"
echo "🎯 Ready to create GPU instance and run training"
echo ""
echo "📋 Next steps:"
echo "   1. Create GPU instance: gcloud compute instances create samo-dl-training ..."
echo "   2. SSH into instance: gcloud compute ssh samo-dl-training"
echo "   3. Run training: python scripts/focal_loss_training.py"
