#!/bin/bash

# Find available GPU zones for SAMO-DL training

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔍 Finding Available GPU Zones for SAMO-DL${NC}"
echo -e "${BLUE}=========================================${NC}"

# List of zones to try (ordered by preference)
ZONES=(
    "us-west1-b"
    "us-west1-c"
    "us-west1-a"
    "us-central1-b"
    "us-central1-c"
    "us-central1-f"
    "us-east1-b"
    "us-east1-c"
    "us-east1-d"
    "europe-west1-b"
    "europe-west1-c"
    "europe-west1-d"
    "asia-southeast1-a"
    "asia-southeast1-b"
)

GPU_TYPES=("nvidia-tesla-t4" "nvidia-tesla-k80" "nvidia-tesla-p4")
MACHINE_TYPES=("n1-standard-4" "n1-standard-2" "n1-highmem-2")

echo -e "${YELLOW}📋 Checking GPU availability in different zones...${NC}"
echo ""

# Function to test zone availability
test_zone_gpu() {
    local zone="$1"
    local gpu_type="$2"
    local machine_type="$3"

    echo -ne "${BLUE}Testing ${zone} with ${gpu_type} on ${machine_type}...${NC} "

    # Try to create a test instance (dry run)
    if gcloud compute instances create test-gpu-check \
        --zone="${zone}" \
        --machine-type="${machine_type}" \
        --accelerator="type=${gpu_type},count=1" \
        --image-family=pytorch-latest-gpu \
        --image-project=deeplearning-platform-release \
        --dry-run \
        --quiet 2>/dev/null; then
        echo -e "${GREEN}✅ AVAILABLE${NC}"

        # Generate working command
        echo ""
        echo -e "${GREEN}🚀 WORKING COMMAND:${NC}"
        echo "gcloud compute instances create samo-dl-training \\"
        echo "  --zone=${zone} \\"
        echo "  --machine-type=${machine_type} \\"
        echo "  --accelerator='type=${gpu_type},count=1' \\"
        echo "  --image-family=pytorch-latest-gpu \\"
        echo "  --image-project=deeplearning-platform-release \\"
        echo "  --boot-disk-size=200GB \\"
        echo "  --boot-disk-type=pd-ssd \\"
        echo "  --metadata='install-nvidia-driver=True' \\"
        echo "  --maintenance-policy=TERMINATE \\"
        echo "  --restart-on-failure"
        echo ""
        return 0
    else
        echo -e "${RED}❌ NOT AVAILABLE${NC}"
        return 1
    fi
}

# Quick availability check for top zones
echo -e "${YELLOW}🚀 Quick Check - Top 3 Zones:${NC}"
for zone in "us-west1-b" "us-central1-b" "europe-west1-b"; do
    if test_zone_gpu "$zone" "nvidia-tesla-t4" "n1-standard-4"; then
        echo -e "${GREEN}✅ Found available zone: ${zone}${NC}"
        echo -e "${YELLOW}Copy and run the command above!${NC}"
        exit 0
    fi
done

echo ""
echo -e "${YELLOW}🔄 Extended Search (checking more zones)...${NC}"

# Extended search
for zone in "${ZONES[@]}"; do
    for gpu in "${GPU_TYPES[@]}"; do
        for machine in "${MACHINE_TYPES[@]}"; do
            if test_zone_gpu "$zone" "$gpu" "$machine"; then
                echo -e "${GREEN}✅ Success! Use the command above.${NC}"
                exit 0
            fi
        done
    done
done

echo ""
echo -e "${RED}❌ No GPU resources found in any zone.${NC}"
echo ""
echo -e "${YELLOW}💡 Alternative Options:${NC}"
echo "1. Try again in 30-60 minutes (resources refresh frequently)"
echo "2. Use CPU-only training (much slower but works):"
echo ""
echo "gcloud compute instances create samo-dl-training \\"
echo "  --zone=us-central1-a \\"
echo "  --machine-type=n1-standard-4 \\"
echo "  --image-family=pytorch-latest-gpu \\"
echo "  --image-project=deeplearning-platform-release \\"
echo "  --boot-disk-size=100GB \\"
echo "  --boot-disk-type=pd-ssd"
echo ""
echo "3. Request GPU quota increase if you have 0 quota"
