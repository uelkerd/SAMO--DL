#!/bin/bash

# Quick fix for GCP image family access issues
# SAMO-DL Project: the-tendril-466607-n8

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔧 SAMO-DL GCP Quick Fix for Image Family Issues${NC}"
echo -e "${BLUE}===============================================${NC}"

# Test different image options
PROJECT_ID="the-tendril-466607-n8"
INSTANCE_NAME="samo-dl-training"
ZONE="us-central1-a"

echo -e "${YELLOW}Testing image access for project: ${PROJECT_ID}${NC}"
echo ""

# Function to test image access
test_image() {
    local image_spec="$1"
    local description="$2"

    echo -e "${BLUE}Testing: ${description}${NC}"
    echo "Command: gcloud compute images list ${image_spec}"

    if gcloud compute images list "${image_spec}" --project="${PROJECT_ID}" --limit=1 --quiet &>/dev/null; then
        echo -e "${GREEN}✅ WORKING: ${description}${NC}"
        return 0
    else
        echo -e "${RED}❌ FAILED: ${description}${NC}"
        return 1
    fi
}

# Test various image options
echo -e "${YELLOW}📋 Testing Image Access...${NC}"
echo ""

# Test 1: Deep Learning VM images
test_image "--filter='family:tf-*-gpu' --project=deeplearning-platform-release" "Deep Learning VM Images"

# Test 2: Ubuntu images
test_image "--filter='family:ubuntu-2004-lts' --project=ubuntu-os-cloud" "Ubuntu 20.04 Family"

# Test 3: Specific Ubuntu images
test_image "--filter='name:ubuntu-2004*' --project=ubuntu-os-cloud" "Specific Ubuntu Images"

# Test 4: Debian images
test_image "--filter='family:debian-11' --project=debian-cloud" "Debian 11 Family"

# Test 5: Container-optimized OS
test_image "--filter='family:cos-stable' --project=cos-cloud" "Container-Optimized OS"

echo ""
echo -e "${YELLOW}📋 Recommended Working Commands:${NC}"
echo ""

# Command 1: Deep Learning VM (best for ML)
echo -e "${GREEN}1. Deep Learning VM (RECOMMENDED for ML):${NC}"
echo "gcloud compute instances create ${INSTANCE_NAME} \\"
echo "  --zone=${ZONE} \\"
echo "  --machine-type=n1-standard-4 \\"
echo "  --accelerator='type=nvidia-tesla-t4,count=1' \\"
echo "  --image-family=tf-latest-gpu \\"
echo "  --image-project=deeplearning-platform-release \\"
echo "  --boot-disk-size=200GB \\"
echo "  --boot-disk-type=pd-ssd \\"
echo "  --metadata='install-nvidia-driver=True' \\"
echo "  --maintenance-policy=TERMINATE \\"
echo "  --restart-on-failure"
echo ""

# Command 2: Specific Ubuntu image
echo -e "${GREEN}2. Specific Ubuntu Image (fallback):${NC}"
echo "gcloud compute instances create ${INSTANCE_NAME} \\"
echo "  --zone=${ZONE} \\"
echo "  --machine-type=n1-standard-4 \\"
echo "  --accelerator='type=nvidia-tesla-t4,count=1' \\"
echo "  --image=ubuntu-2004-lts \\"
echo "  --image-project=ubuntu-os-cloud \\"
echo "  --boot-disk-size=200GB \\"
echo "  --boot-disk-type=pd-ssd \\"
echo "  --metadata='install-nvidia-driver=True' \\"
echo "  --maintenance-policy=TERMINATE \\"
echo "  --restart-on-failure"
echo ""

# Command 3: Try different zone
echo -e "${GREEN}3. Different Zone (if quota issues):${NC}"
echo "gcloud compute instances create ${INSTANCE_NAME} \\"
echo "  --zone=us-west1-b \\"
echo "  --machine-type=n1-standard-4 \\"
echo "  --accelerator='type=nvidia-tesla-t4,count=1' \\"
echo "  --image=ubuntu-2004-lts \\"
echo "  --image-project=ubuntu-os-cloud \\"
echo "  --boot-disk-size=200GB \\"
echo "  --boot-disk-type=pd-ssd \\"
echo "  --metadata='install-nvidia-driver=True' \\"
echo "  --maintenance-policy=TERMINATE \\"
echo "  --restart-on-failure"
echo ""

echo -e "${YELLOW}📋 Troubleshooting Steps:${NC}"
echo ""
echo "1. Check project permissions:"
echo "   gcloud projects get-iam-policy ${PROJECT_ID}"
echo ""
echo "2. Verify billing is enabled:"
echo "   gcloud beta billing projects describe ${PROJECT_ID}"
echo ""
echo "3. Check compute quotas:"
echo "   gcloud compute project-info describe --project=${PROJECT_ID}"
echo ""
echo "4. List available GPU types in your zone:"
echo "   gcloud compute accelerator-types list --filter='zone:${ZONE}'"
echo ""

echo -e "${BLUE}💡 Quick Start Command (copy and run):${NC}"
echo -e "${GREEN}gcloud compute instances create samo-dl-training --zone=us-central1-a --machine-type=n1-standard-4 --accelerator='type=nvidia-tesla-t4,count=1' --image=ubuntu-2004-lts --image-project=ubuntu-os-cloud --boot-disk-size=200GB --boot-disk-type=pd-ssd --metadata='install-nvidia-driver=True' --maintenance-policy=TERMINATE --restart-on-failure${NC}"

echo ""
echo -e "${YELLOW}🎯 Next Steps After Instance Creation:${NC}"
echo "1. SSH: gcloud compute ssh samo-dl-training --zone=us-central1-a"
echo "2. Update: sudo apt-get update && sudo apt-get install -y python3-pip git"
echo "3. Clone: git clone https://github.com/YOUR_USERNAME/SAMO-DL.git"
echo "4. Setup: cd SAMO-DL && python3 -m venv venv && source venv/bin/activate"
echo "5. Install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
echo "6. Train: python scripts/focal_loss_training.py --gamma 2.0 --alpha 0.25"
