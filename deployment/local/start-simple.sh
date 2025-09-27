#!/usr/bin/env bash
# Start simple local development server

# Enable strict bash options for fail-fast behavior
set -euo pipefail
IFS=$'\n\t'

# Change to script's directory for location independence
cd "$(dirname "$0")"

echo "🚀 STARTING SIMPLE LOCAL DEVELOPMENT SERVER"
echo "==========================================="

# Check if Python 3 is available
if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ Error: python3 not found in PATH"
    echo "Please install Python 3 and try again"
    exit 127
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if requirements file exists
if [ ! -f requirements-simple.txt ]; then
    echo "❌ Error: requirements-simple.txt not found"
    echo "Expected location: $(pwd)/requirements-simple.txt"
    exit 1
fi

echo "📦 Installing minimal dependencies..."

# Determine if we're in a virtual environment
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "🔧 Installing dependencies for user (not in virtual environment)"
    USER_FLAG="--user"
else
    echo "🔧 Installing dependencies in virtual environment: $VIRTUAL_ENV"
    USER_FLAG=""
fi

# Install dependencies
if ! python3 -m pip install $USER_FLAG -r requirements-simple.txt; then
    echo "❌ Failed to install dependencies"
    echo "You may need to run: python3 -m pip install --upgrade pip"
    exit 1
fi

echo "✅ Dependencies installed successfully"

# Get port from environment or use default
PORT="${PORT:-8000}"

# Check if port is available
if command -v netstat >/dev/null 2>&1; then
    if netstat -an | grep -q ":$PORT "; then
        echo "⚠️  Warning: Port $PORT appears to be in use"
        echo "You can set a different port with: PORT=8001 $0"
    fi
fi

echo "🌐 Starting simple development server..."
echo "📁 Serving website files with CORS enabled"
echo "🔗 Server will be available at: http://localhost:${PORT}"
echo ""
echo "Available pages:"
echo "  • Main page: http://localhost:${PORT}/"

# Check for HTML files in website directory
WEBSITE_DIR="../../website"
if [ -d "$WEBSITE_DIR" ]; then
    for html_file in "$WEBSITE_DIR"/*.html; do
        if [ -f "$html_file" ]; then
            filename=$(basename "$html_file")
            if [ "$filename" != "index.html" ]; then
                page_name=$(basename "$filename" .html)
                echo "  • ${page_name^}: http://localhost:${PORT}/${filename}"
            fi
        fi
    done
fi

echo "  • Health check: http://localhost:${PORT}/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "==========================================="
echo ""

# Start the server
exec python3 simple_server.py --port "${PORT}"
