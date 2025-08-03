#!/bin/bash
# 🚀 DEPLOYMENT SCRIPT
# ====================

echo "🚀 DEPLOYING EMOTION DETECTION MODEL"
echo "===================================="

# Check if model directory exists
if [ ! -d "./model" ]; then
    echo "❌ Model directory not found!"
    echo "Please ensure the trained model is in ./model/"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Test the model
echo "🧪 Testing model..."
python test_examples.py

# Start API server
echo "🌐 Starting API server..."
echo "Server will be available at: http://localhost:5000"
python api_server.py
