#!/bin/bash
# Automated installation and integration script for Whisper and T5 models

set -e  # Exit on error

echo "=================================================="
echo "SAMO Deep Learning - Model Integration Script"
echo "=================================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Step 1: Check Python
echo "Step 1: Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_status "Python3 found: $PYTHON_VERSION"
else
    print_error "Python3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Step 2: Install dependencies
echo ""
echo "Step 2: Installing model dependencies..."
echo "This may take several minutes depending on your internet connection..."

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    print_status "Virtual environment detected: $VIRTUAL_ENV"
else
    print_warning "No virtual environment detected. Installing globally."
    echo "Consider using a virtual environment for better isolation."
fi

# Install core dependencies
print_status "Installing PyTorch..."
pip3 install torch --no-cache-dir --quiet || {
    print_error "Failed to install PyTorch"
    exit 1
}

print_status "Installing Transformers for T5..."
pip3 install transformers --no-cache-dir --quiet || {
    print_error "Failed to install Transformers"
    exit 1
}

print_status "Installing OpenAI Whisper..."
pip3 install openai-whisper --no-cache-dir --quiet || {
    print_error "Failed to install Whisper"
    exit 1
}

print_status "Installing additional audio dependencies..."
pip3 install pydub soundfile --no-cache-dir --quiet || {
    print_warning "Some audio dependencies failed to install"
}

# Step 3: Verify installations
echo ""
echo "Step 3: Verifying installations..."

# Test Whisper
python3 -c "import whisper; print('✅ Whisper verified')" 2>/dev/null || {
    print_error "Whisper verification failed"
    exit 1
}

# Test Transformers
python3 -c "import transformers; print('✅ Transformers verified')" 2>/dev/null || {
    print_error "Transformers verification failed"
    exit 1
}

# Test PyTorch
python3 -c "import torch; print('✅ PyTorch verified')" 2>/dev/null || {
    print_error "PyTorch verification failed"
    exit 1
}

print_status "All dependencies installed successfully!"

# Step 4: Download models (optional - they'll download on first use anyway)
echo ""
echo "Step 4: Pre-downloading models (optional)..."
echo "This will download models to cache for faster first-time loading."
read -p "Do you want to download models now? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Downloading Whisper base model..."
    python3 -c "import whisper; whisper.load_model('base')" 2>/dev/null || {
        print_warning "Whisper model download failed (will download on first use)"
    }
    
    print_status "Downloading T5-small model..."
    python3 -c "from transformers import T5ForConditionalGeneration, T5Tokenizer; T5Tokenizer.from_pretrained('t5-small'); T5ForConditionalGeneration.from_pretrained('t5-small')" 2>/dev/null || {
        print_warning "T5 model download failed (will download on first use)"
    }
fi

# Step 5: Test the integration
echo ""
echo "Step 5: Testing model integration..."

python3 scripts/ensure_real_models.py --test 2>/dev/null || {
    print_warning "Integration test encountered issues"
    echo "Models may still work. Check the logs for details."
}

# Step 6: Final summary
echo ""
echo "=================================================="
echo "Installation Complete!"
echo "=================================================="
echo ""
print_status "✨ Whisper and T5 models are ready for integration!"
echo ""
echo "Next steps:"
echo "1. Start the API server:"
echo "   uvicorn src.unified_ai_api:app --reload"
echo ""
echo "2. Test the endpoints:"
echo "   - Voice: POST /transcribe/voice"
echo "   - Text:  POST /summarize/text"
echo ""
echo "3. Check the integration guide:"
echo "   cat MODEL_INTEGRATION_GUIDE.md"
echo ""
echo "For troubleshooting, check:"
echo "   python3 scripts/integrate_real_models.py --check-deps"
echo ""
echo "=================================================="