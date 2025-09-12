#!/bin/bash
# Start local deployment

echo "🚀 STARTING LOCAL DEPLOYMENT"
echo "============================"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements-api.txt

# Start API server
echo "🌐 Starting API server..."
echo "Server will be available at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

python api_server.py
