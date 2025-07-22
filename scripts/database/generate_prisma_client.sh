#!/bin/bash

# Exit on error
set -e

echo "Generating Prisma client..."

# Navigate to project root (assuming this script is in scripts/database)
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Check for Prisma installation
if ! command -v npx &> /dev/null; then
    echo "Error: npx is not installed. Please install Node.js and npm."
    exit 1
fi

# Generate Prisma client
npx prisma generate

echo "Prisma client generated successfully!"
