# 🚀 Whisper & T5 Model Integration Guide

## Overview

This guide will help you integrate real OpenAI Whisper and T5 models into the SAMO Deep Learning API to replace the current mock implementations.

## Current Status

- **Whisper (Voice Transcription)**: Code structure ready, needs model installation
- **T5 (Text Summarization)**: Code structure ready, needs model installation
- **API Integration**: Updated to load real models with proper fallbacks

## 📦 Step 1: Install Dependencies

### Option A: Quick Install (Recommended)
```bash
# Install all model dependencies
pip install openai-whisper transformers torch --upgrade

# Verify installation
python3 -c "import whisper; print('✅ Whisper installed')"
python3 -c "import transformers; print('✅ Transformers installed')"
python3 -c "import torch; print('✅ PyTorch installed')"
```

### Option B: Install from Requirements
```bash
# Install from project requirements
pip install -r dependencies/requirements_unified.txt
pip install -r dependencies/requirements-audio.txt
pip install -r dependencies/requirements-ml.txt
```

### Option C: Minimal Install (Lighter weight)
```bash
# Install minimal versions for testing
pip install openai-whisper==20231117 transformers==4.35.0 torch==2.0.0
```

## 🔧 Step 2: Verify Model Loading

### Test Individual Models
```bash
# Test Whisper model loading
python3 -c "
import whisper
model = whisper.load_model('base')
print('✅ Whisper base model loaded successfully')
print(f'Model: {model}')
"

# Test T5 model loading  
python3 -c "
from transformers import T5ForConditionalGeneration, T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')
print('✅ T5-small model loaded successfully')
print(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')
"
```

### Test Integration
```bash
# Run the integration test script
python3 scripts/ensure_real_models.py --test
```

## 📝 Step 3: Update API Configuration

The API has been updated to automatically load real models when available. The changes include:

1. **Enhanced Model Loading** (`src/unified_ai_api.py`):
   - Attempts to load real models first
   - Verifies models are working
   - Falls back to mock implementations if real models fail

2. **Model Verification**:
   - T5: Tests with simple summarization
   - Whisper: Checks model info and configuration

## 🧪 Step 4: Test the Integration

### Start the API Server
```bash
# Start the API with real models
cd /workspace
uvicorn src.unified_ai_api:app --host 0.0.0.0 --port 8000 --reload
```

### Test Voice Transcription
```bash
# Test with a sample audio file
curl -X POST "http://localhost:8000/transcribe/voice" \
  -H "accept: application/json" \
  -F "audio_file=@samples/test_audio.mp3" \
  -F "language=en" \
  -F "model_size=base"
```

### Test Text Summarization
```bash
# Test with sample text
curl -X POST "http://localhost:8000/summarize/text" \
  -H "accept: application/json" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "text=I had a wonderful day today. The weather was perfect and I spent time with family. We went to the park and had a picnic. The children played while we relaxed. It was one of those perfect moments.&model=t5-small&max_length=50"
```

## 🎯 Step 5: Verify Production Readiness

### Check Model Performance
```python
# Create test_models.py
import time
import whisper
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Test Whisper
print("Testing Whisper...")
start = time.time()
model = whisper.load_model("base")
load_time = time.time() - start
print(f"Whisper load time: {load_time:.2f}s")

# Test T5
print("\nTesting T5...")
start = time.time()
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
load_time = time.time() - start
print(f"T5 load time: {load_time:.2f}s")

# Test summarization
text = "summarize: " + "This is a test. " * 20
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
start = time.time()
outputs = model.generate(inputs["input_ids"], max_length=50)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
inference_time = time.time() - start
print(f"\nT5 inference time: {inference_time:.2f}s")
print(f"Summary: {summary}")
```

### Monitor Resource Usage
```bash
# Check memory usage
python3 -c "
import psutil
import whisper
import torch
from transformers import T5ForConditionalGeneration

# Load models
whisper_model = whisper.load_model('base')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Check memory
process = psutil.Process()
memory_mb = process.memory_info().rss / 1024 / 1024
print(f'Memory usage with models loaded: {memory_mb:.0f} MB')
"
```

## 🐛 Troubleshooting

### Common Issues and Solutions

1. **ImportError: No module named 'whisper'**
   ```bash
   pip install openai-whisper
   ```

2. **CUDA/GPU errors**
   ```python
   # Force CPU usage in your code
   device = "cpu"
   model = whisper.load_model("base", device=device)
   ```

3. **Memory errors with T5**
   ```python
   # Use smaller model or reduce batch size
   model = T5ForConditionalGeneration.from_pretrained("t5-small")
   # Or use t5-base, t5-large for better quality
   ```

4. **Slow model loading**
   ```python
   # Cache models locally
   import os
   os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/transformers'
   os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
   ```

5. **Audio format not supported**
   ```bash
   # Install audio dependencies
   pip install pydub soundfile librosa ffmpeg-python
   ```

## 📊 Performance Expectations

### Whisper Model Sizes
| Model | Parameters | Relative Speed | English-only |
|-------|-----------|----------------|--------------|
| tiny  | 39 M      | ~32x           | Yes          |
| base  | 74 M      | ~16x           | Yes          |
| small | 244 M     | ~6x            | Yes          |
| medium| 769 M     | ~2x            | Yes          |
| large | 1550 M    | 1x             | No           |

### T5 Model Sizes
| Model    | Parameters | Speed  | Quality |
|----------|-----------|--------|---------|
| t5-small | 60 M      | Fast   | Good    |
| t5-base  | 220 M     | Medium | Better  |
| t5-large | 770 M     | Slow   | Best    |

## ✅ Verification Checklist

- [ ] Dependencies installed (whisper, transformers, torch)
- [ ] Whisper model loads without errors
- [ ] T5 model loads without errors
- [ ] API starts with real models loaded
- [ ] Voice transcription endpoint works
- [ ] Text summarization endpoint works
- [ ] Response times are acceptable (<2s for most requests)
- [ ] Memory usage is within limits (<4GB)

## 🚀 Next Steps

1. **Optimize for Production**:
   - Consider using ONNX Runtime for faster inference
   - Implement model caching and warm-up
   - Add request queuing for heavy loads

2. **Enhance Quality**:
   - Fine-tune models on journal-specific data
   - Implement confidence thresholds
   - Add language detection for Whisper

3. **Monitor Performance**:
   - Track inference times
   - Monitor memory usage
   - Log model predictions for quality assessment

## 📚 Additional Resources

- [OpenAI Whisper Documentation](https://github.com/openai/whisper)
- [Hugging Face T5 Documentation](https://huggingface.co/docs/transformers/model_doc/t5)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [SAMO API Documentation](docs/api/API_DOCUMENTATION.md)

---

**Need Help?** Check the logs at `/workspace/logs/` or run the diagnostic script:
```bash
python3 scripts/integrate_real_models.py --check-deps
```