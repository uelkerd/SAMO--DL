#!/bin/bash

# SAMO-DL Ubuntu ML Environment Setup
# Run this after SSH'ing into your Ubuntu instance

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 SAMO-DL Ubuntu ML Environment Setup${NC}"
echo -e "${BLUE}====================================${NC}"

# Check if we're on the right instance
echo -e "${YELLOW}📋 System Information:${NC}"
echo "Hostname: $(hostname)"
echo "OS: $(lsb_release -d | cut -f2)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'Checking...')"
echo ""

# Update system packages
echo -e "${YELLOW}🔄 Updating system packages...${NC}"
sudo apt-get update -q
sudo apt-get install -y python3-pip python3-venv git curl wget software-properties-common

# Check Python version
python_version=$(python3 --version)
echo -e "${GREEN}✅ Python: ${python_version}${NC}"

# Install CUDA if needed (the metadata should handle this, but let's verify)
echo -e "${YELLOW}🔧 Checking CUDA installation...${NC}"
if nvidia-smi &>/dev/null; then
    echo -e "${GREEN}✅ NVIDIA drivers working${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo -e "${YELLOW}⚠️  Installing NVIDIA drivers...${NC}"
    sudo apt-get install -y nvidia-driver-525
    echo "Please reboot the instance after this script completes: sudo reboot"
fi

# Create project directory
echo -e "${YELLOW}📁 Setting up project directory...${NC}"
mkdir -p ~/SAMO-DL
cd ~/SAMO-DL

# Create Python virtual environment
echo -e "${YELLOW}🐍 Creating Python virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
echo -e "${YELLOW}⚡ Installing PyTorch with CUDA support...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install ML dependencies
echo -e "${YELLOW}📦 Installing ML dependencies...${NC}"
pip install transformers==4.36.0
pip install datasets==2.14.0
pip install scikit-learn==1.3.0
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install tqdm==4.66.0
pip install matplotlib==3.7.2
pip install seaborn==0.12.2

# Install API dependencies
echo -e "${YELLOW}🌐 Installing API dependencies...${NC}"
pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install python-multipart==0.0.6

# Test PyTorch CUDA
echo -e "${YELLOW}🧪 Testing PyTorch CUDA setup...${NC}"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠️ CUDA not available - will use CPU (much slower)')
"

# Create directories
echo -e "${YELLOW}📂 Creating project directories...${NC}"
mkdir -p scripts models logs configs data

# Create a simple test script
echo -e "${YELLOW}📝 Creating test script...${NC}"
cat > scripts/test_environment.py << 'EOF'
#!/usr/bin/env python3
"""
Test script to validate the ML environment setup
"""

import torch
import transformers
import sklearn
import numpy as np
import pandas as pd

def test_environment():
    print("🧪 Testing ML Environment")
    print("=" * 40)

    # Test PyTorch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Test other libraries
    print(f"✅ Transformers: {transformers.__version__}")
    print(f"✅ Scikit-learn: {sklearn.__version__}")
    print(f"✅ NumPy: {np.__version__}")
    print(f"✅ Pandas: {pd.__version__}")

    # Test basic tensor operations
    x = torch.randn(1000, 1000)
    if torch.cuda.is_available():
        x = x.cuda()
        print("✅ GPU tensor operations working")
    else:
        print("⚠️ Using CPU for tensor operations")

    print("\n🎉 Environment setup successful!")
    return True

if __name__ == "__main__":
    test_environment()
EOF

# Make test script executable
chmod +x scripts/test_environment.py

# Run environment test
echo -e "${YELLOW}🧪 Running environment test...${NC}"
python3 scripts/test_environment.py

echo ""
echo -e "${GREEN}🎉 Ubuntu ML Environment Setup Complete!${NC}"
echo ""
echo -e "${YELLOW}📋 Next Steps:${NC}"
echo "1. Clone your SAMO-DL repository:"
echo "   git clone https://github.com/YOUR_USERNAME/SAMO-DL.git ."
echo ""
echo "2. Start focal loss training:"
echo "   python3 scripts/focal_loss_training.py --gamma 2.0 --alpha 0.25"
echo ""
echo "3. Monitor training:"
echo "   watch -n 5 nvidia-smi"
echo ""
echo -e "${BLUE}💡 To reactivate environment later: source ~/SAMO-DL/venv/bin/activate${NC}"
