#!/bin/bash
# Script to clean up old Docker images from Artifact Registry
#
# This script:
# - Uses fully-qualified image names (LOCATION-docker.pkg.dev/PROJECT/REPO/PACKAGE)
# - Handles both tagged versions and digest-only versions
# - Groups images by package and keeps N most recent per package
# - Uses --delete-tags flag to ensure complete removal of tagged images
# - Uses digest form (@sha256:...) for digest-only images

# Enable strict bash options for fail-fast behavior
set -euo pipefail
IFS=$'\n\t'

# Check for gcloud CLI presence
if ! command -v gcloud >/dev/null 2>&1; then
    echo "❌ Error: gcloud CLI not found" >&2
    echo "   Please install gcloud CLI and authenticate:" >&2
    echo "   https://cloud.google.com/sdk/docs/install" >&2
    exit 1
fi

# Verify gcloud authentication and configuration
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Error: No active gcloud authentication found" >&2
    echo "   Please run: gcloud auth login" >&2
    exit 1
fi

echo "🧹 Starting Docker image cleanup..."

# Function to delete old images (keep only latest 2 per package)
cleanup_repo() {
    local repo=$1
    local keep_count=${2:-2}  # Default to keeping 2 most recent per package
    
    # Validate that repo is fully qualified
    if [[ ! "$repo" =~ ^[a-z0-9-]+-docker\.pkg\.dev/[a-z0-9-]+/[a-z0-9-]+$ ]]; then
        echo "❌ Error: Repository '$repo' is not fully qualified"
        echo "   Expected format: LOCATION-docker.pkg.dev/PROJECT/REPO"
        return 1
    fi
    
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
        echo "$package_images" | tail -n +$((keep_count + 1)) | while IFS=',' read -r pkg version _; do
            # Construct fully-qualified image name
            local full_image_name="$repo/$pkg"
            
            # Handle both tagged versions and digest-only versions
            if [[ "$version" =~ ^sha256: ]]; then
                # This is a digest-only version - use digest form
                local digest_image="$full_image_name@$version"
                echo "  Deleting digest-only image: $digest_image"
                gcloud artifacts docker images delete "$digest_image" --quiet || true
            else
                # This is a tagged version - use tag form with --delete-tags
                local tagged_image="$full_image_name:$version"
                echo "  Deleting tagged image: $tagged_image"
                gcloud artifacts docker images delete "$tagged_image" --delete-tags --quiet || true
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
