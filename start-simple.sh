#!/usr/bin/env bash
# Start simple local development server

set -euo pipefail
IFS=$'\n\t'

# Get script directory and change to it
cd "$(dirname "$0")"

echo "🚀 STARTING SIMPLE LOCAL DEVELOPMENT SERVER"
echo "==========================================="

# Check Python availability
echo "📦 Installing minimal dependencies..."
if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ Error: python3 not found in PATH" >&2
    exit 127
fi

# Check if requirements file exists
if [ ! -f requirements-simple.txt ]; then
    echo "❌ Error: requirements-simple.txt not found next to script" >&2
    exit 1
fi

# Install dependencies (use --user flag if not in virtual environment)
if [ -z "${VIRTUAL_ENV:-}" ]; then
    USER_FLAG="--user"
else
    USER_FLAG=""
fi

python3 -m pip install $USER_FLAG -r requirements-simple.txt

echo "🌐 Starting simple development server..."

# Set default port
PORT="${PORT:-8000}"

echo "Server will be available at: http://localhost:${PORT}"
echo "Website files served with CORS enabled"
echo "🎯 Connects to Cloud Run API: https://samo-unified-api-optimized-frrnetyhfa-uc.a.run.app"
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
exec python3 simple_server.py --port "${PORT}"