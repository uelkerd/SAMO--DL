#!/bin/bash
# Install dependencies for Whisper and T5 models

echo "Installing model dependencies for SAMO Deep Learning..."
echo "============================================"

# Install core dependencies
echo "Installing core dependencies..."
pip3 install --upgrade pip
pip3 install numpy torch --no-cache-dir

echo ""
echo "Installing Whisper..."
pip3 install openai-whisper --no-cache-dir

echo ""
echo "Installing Transformers for T5..."
pip3 install transformers --no-cache-dir

echo ""
echo "Installing additional audio dependencies..."
pip3 install pydub soundfile librosa --no-cache-dir

echo ""
echo "============================================"
echo "Installation complete!"
echo ""
echo "Verifying installations..."
python3 -c "import whisper; print('✅ Whisper installed')" 2>/dev/null || echo "❌ Whisper not installed"
python3 -c "import transformers; print('✅ Transformers installed')" 2>/dev/null || echo "❌ Transformers not installed"
python3 -c "import torch; print('✅ PyTorch installed')" 2>/dev/null || echo "❌ PyTorch not installed"