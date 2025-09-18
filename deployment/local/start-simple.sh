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
command -v python3 >/dev/null || { echo "python3 not found in PATH" >&2; exit 127; }
[ -f requirements-simple.txt ] || { echo "requirements-simple.txt not found next to script" >&2; exit 1; }
if [ -z "${VIRTUAL_ENV:-}" ]; then USER_FLAG="--user"; else USER_FLAG=""; fi
python3 -m pip install $USER_FLAG -r requirements-simple.txt

# Start simple server
echo "🌐 Starting simple development server..."
PORT="${PORT:-8000}"
echo "Server will be available at: http://localhost:${PORT}"
echo "Website files served with CORS enabled"
echo "Press Ctrl+C to stop the server"
echo ""

exec python3 simple_server.py --port "${PORT}"