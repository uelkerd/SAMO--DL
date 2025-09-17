#!/bin/bash
# Start simple local development server

echo "🚀 STARTING SIMPLE LOCAL DEVELOPMENT SERVER"
echo "==========================================="

# Install minimal dependencies
echo "📦 Installing minimal dependencies..."
pip install -r requirements-simple.txt

# Start simple server
echo "🌐 Starting simple development server..."
echo "Server will be available at: http://localhost:8000"
echo "Website files served with CORS enabled"
echo "Press Ctrl+C to stop the server"
echo ""

python simple_server.py