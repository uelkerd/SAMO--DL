#!/bin/bash
# Script to clean up old Docker images from Artifact Registry

echo "🧹 Starting Docker image cleanup..."

# Function to delete old images (keep only latest 2)
cleanup_repo() {
    local repo=$1
    echo "Cleaning up repository: $repo"
    
    # Get all images, sort by creation time (newest first)
    gcloud artifacts docker images list $repo --format="value(package,version,createTime)" | \
    sort -k3 -r | \
    tail -n +3 | \
    while read package version createTime; do
        if [ ! -z "$package" ]; then
            echo "Deleting old image: $package:$version"
            gcloud artifacts docker images delete $package:$version --quiet || true
        fi
    done
}

# Clean up each repository
echo "Cleaning emotion-detection-repo..."
cleanup_repo "us-central1-docker.pkg.dev/the-tendril-466607-n8/emotion-detection-repo"

echo "Cleaning samo-dl repo..."
cleanup_repo "us-central1-docker.pkg.dev/the-tendril-466607-n8/samo-dl"

echo "Cleaning cloud-run-source-deploy..."
cleanup_repo "us-central1-docker.pkg.dev/the-tendril-466607-n8/cloud-run-source-deploy"

echo "✅ Cleanup complete!"
echo "💰 This should significantly reduce your storage costs!"
