#!/bin/bash

# Sync documentation from root docs/ to website/docs/
# This ensures GitHub Pages has access to all documentation files

set -e

echo "🔄 Syncing documentation for GitHub Pages..."

# Remove existing website/docs if it exists
if [ -d "website/docs" ]; then
    echo "📁 Removing existing website/docs/"
    rm -rf website/docs
fi

# Copy docs to website/docs
echo "📋 Copying docs/ to website/docs/"
cp -r docs website/docs

# Verify the copy was successful
if [ -f "website/docs/api/API_DOCUMENTATION.md" ]; then
    echo "✅ Documentation sync completed successfully!"
    echo "📊 Files copied:"
    find website/docs -name "*.md" | wc -l | xargs echo "   Markdown files:"
    echo "   📁 API Documentation: website/docs/api/"
    echo "   🚀 Deployment Guides: website/docs/deployment/"
    echo "   📖 User Guides: website/docs/guides/"
else
    echo "❌ Documentation sync failed!"
    exit 1
fi

echo "🌐 GitHub Pages will now serve documentation at:"
echo "   - docs/api/API_DOCUMENTATION.md"
echo "   - docs/deployment/PRODUCTION_DEPLOYMENT_GUIDE.md"
echo "   - docs/guides/USER_GUIDE.md"
echo "   - docs/README.md"
echo ""
echo "💡 Run this script whenever you update documentation in the root docs/ folder"