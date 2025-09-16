#!/bin/bash
# Script to clean up old Docker images from Artifact Registry

set -euo pipefail

echo "🧹 Starting Docker image cleanup..."

# Function to delete old images (keep only latest 2 per package)
cleanup_repo() {
    local repo=$1
    local keep_count=${2:-2}  # Default to keeping 2 most recent per package
    echo "Cleaning up repository: $repo (keeping $keep_count most recent per package)"
    
    # Get all images in CSV format with stable delimiter, no header
    local temp_file
    temp_file=$(mktemp)
    
    # Use CSV format with comma delimiter and no header
    gcloud artifacts docker images list "$repo" \
        --format="csv(package,version,createTime)" \
        --filter="createTime!=null" > "$temp_file"
    
    # Process each package separately to keep N most recent per package
    local packages
    packages=$(cut -d',' -f1 "$temp_file" | sort -u)
    
    for package in $packages; do
        echo "Processing package: $package"
        
        # Get images for this package, sort by creation time (newest first)
        local package_images
        package_images=$(grep "^$package," "$temp_file" | sort -t',' -k3 -r)
        
        # Count total images for this package
        local total_count
        total_count=$(echo "$package_images" | wc -l)
        
        if [ "$total_count" -le "$keep_count" ]; then
            echo "  Package $package has $total_count images (≤ $keep_count), skipping deletion"
            continue
        fi
        
        # Calculate how many to delete
        local delete_count=$((total_count - keep_count))
        echo "  Package $package has $total_count images, deleting $delete_count oldest"
        
        # Get images to delete (skip the first $keep_count, delete the rest)
        echo "$package_images" | tail -n +$((keep_count + 1)) | while IFS=',' read -r pkg version createTime; do
            # Handle both tagged versions and digest-only versions
            if [[ "$version" =~ ^sha256: ]]; then
                # This is a digest-only version
                echo "  Deleting digest-only image: $pkg@$version"
                gcloud artifacts docker images delete "$pkg@$version" --quiet || true
            else
                # This is a tagged version
                echo "  Deleting tagged image: $pkg:$version"
                gcloud artifacts docker images delete "$pkg:$version" --delete-tags --quiet || true
            fi
        done
    done
    
    # Clean up temp file
    rm -f "$temp_file"
}

# Clean up each repository (keeping 2 most recent per package by default)
echo "Cleaning emotion-detection-repo..."
cleanup_repo "us-central1-docker.pkg.dev/the-tendril-466607-n8/emotion-detection-repo" 2

echo "Cleaning samo-dl repo..."
cleanup_repo "us-central1-docker.pkg.dev/the-tendril-466607-n8/samo-dl" 2

echo "Cleaning cloud-run-source-deploy..."
cleanup_repo "us-central1-docker.pkg.dev/the-tendril-466607-n8/cloud-run-source-deploy" 2

echo "✅ Cleanup complete!"
echo "💰 This should significantly reduce your storage costs!"
echo "📊 Each package now has at most 2 most recent images"
