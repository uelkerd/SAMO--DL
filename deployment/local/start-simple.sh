#!/usr/bin/env bash
# Start simple local development server

# Enable strict bash options for fail-fast behavior
set -euo pipefail
IFS=$'\n\t'

# Change to script's directory for location independence
cd "$(dirname "$0")"

echo "🚀 STARTING SIMPLE LOCAL DEVELOPMENT SERVER"
echo "==========================================="

# Install minimal dependencies
echo "📦 Installing minimal dependencies..."
python3 -m pip install -r requirements-simple.txt

# Start simple server
echo "🌐 Starting simple development server..."
echo "Server will be available at: http://localhost:8000"
echo "Website files served with CORS enabled"
echo "Press Ctrl+C to stop the server"
echo ""

python3 simple_server.py