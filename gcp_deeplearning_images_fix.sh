#!/bin/bash

# Find correct Deep Learning VM images for SAMO-DL

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔍 Finding Correct Deep Learning VM Images${NC}"
echo -e "${BLUE}==========================================${NC}"

echo -e "${YELLOW}📋 Checking available Deep Learning VM families...${NC}"
echo ""

# List all available Deep Learning VM image families
echo -e "${BLUE}Available Deep Learning VM families:${NC}"
gcloud compute images list --project=deeplearning-platform-release --filter="family:*" --format="table(family)" --sort-by=family | head -20

echo ""
echo -e "${YELLOW}📋 Looking for GPU-enabled images...${NC}"
echo ""

# Find GPU-specific images
echo -e "${BLUE}GPU-enabled Deep Learning VM images:${NC}"
gcloud compute images list --project=deeplearning-platform-release --filter="name:*gpu*" --format="table(name,family,status)" --limit=10

echo ""
echo -e "${YELLOW}📋 Latest PyTorch/TensorFlow images...${NC}"
echo ""

# Find latest PyTorch images
echo -e "${BLUE}PyTorch GPU images:${NC}"
gcloud compute images list --project=deeplearning-platform-release --filter="family~pytorch.*gpu" --format="table(name,family,status)" --limit=5

echo ""
# Find latest TensorFlow images
echo -e "${BLUE}TensorFlow GPU images:${NC}"
gcloud compute images list --project=deeplearning-platform-release --filter="family~tf.*gpu" --format="table(name,family,status)" --limit=5

echo ""
echo -e "${YELLOW}📋 Recommended Working Commands:${NC}"
echo ""

# Get the most recent GPU images
PYTORCH_FAMILY=$(gcloud compute images list --project=deeplearning-platform-release --filter="family~pytorch.*gpu" --format="value(family)" --limit=1 2>/dev/null)
TF_FAMILY=$(gcloud compute images list --project=deeplearning-platform-release --filter="family~tf.*gpu" --format="value(family)" --limit=1 2>/dev/null)

if [[ -n "$PYTORCH_FAMILY" ]]; then
    echo -e "${GREEN}1. PyTorch GPU Image (RECOMMENDED for SAMO-DL):${NC}"
    echo "gcloud compute instances create samo-dl-training \\"
    echo "  --zone=us-central1-a \\"
    echo "  --machine-type=n1-standard-4 \\"
    echo "  --accelerator='type=nvidia-tesla-t4,count=1' \\"
    echo "  --image-family=${PYTORCH_FAMILY} \\"
    echo "  --image-project=deeplearning-platform-release \\"
    echo "  --boot-disk-size=200GB \\"
    echo "  --boot-disk-type=pd-ssd \\"
    echo "  --metadata='install-nvidia-driver=True' \\"
    echo "  --maintenance-policy=TERMINATE \\"
    echo "  --restart-on-failure"
    echo ""
fi

if [[ -n "$TF_FAMILY" ]]; then
    echo -e "${GREEN}2. TensorFlow GPU Image (alternative):${NC}"
    echo "gcloud compute instances create samo-dl-training \\"
    echo "  --zone=us-central1-a \\"
    echo "  --machine-type=n1-standard-4 \\"
    echo "  --accelerator='type=nvidia-tesla-t4,count=1' \\"
    echo "  --image-family=${TF_FAMILY} \\"
    echo "  --image-project=deeplearning-platform-release \\"
    echo "  --boot-disk-size=200GB \\"
    echo "  --boot-disk-type=pd-ssd \\"
    echo "  --metadata='install-nvidia-driver=True' \\"
    echo "  --maintenance-policy=TERMINATE \\"
    echo "  --restart-on-failure"
    echo ""
fi

echo -e "${GREEN}3. Ubuntu 20.04 LTS (reliable fallback):${NC}"
echo "gcloud compute instances create samo-dl-training \\"
echo "  --zone=us-central1-a \\"
echo "  --machine-type=n1-standard-4 \\"
echo "  --accelerator='type=nvidia-tesla-t4,count=1' \\"
echo "  --image-family=ubuntu-2004-lts \\"
echo "  --image-project=ubuntu-os-cloud \\"
echo "  --boot-disk-size=200GB \\"
echo "  --boot-disk-type=pd-ssd \\"
echo "  --metadata='install-nvidia-driver=True' \\"
echo "  --maintenance-policy=TERMINATE \\"
echo "  --restart-on-failure"
echo ""

echo -e "${GREEN}4. Specific Ubuntu Image (guaranteed to work):${NC}"
echo "gcloud compute instances create samo-dl-training \\"
echo "  --zone=us-central1-a \\"
echo "  --machine-type=n1-standard-4 \\"
echo "  --accelerator='type=nvidia-tesla-t4,count=1' \\"
echo "  --image=ubuntu-2004-focal-v20240830 \\"
echo "  --image-project=ubuntu-os-cloud \\"
echo "  --boot-disk-size=200GB \\"
echo "  --boot-disk-type=pd-ssd \\"
echo "  --metadata='install-nvidia-driver=True' \\"
echo "  --maintenance-policy=TERMINATE \\"
echo "  --restart-on-failure"
echo ""

echo -e "${BLUE}💡 If all Deep Learning VMs fail, use Ubuntu and install PyTorch manually:${NC}"
echo -e "${YELLOW}This takes 5-10 extra minutes but is 100% reliable${NC}"
