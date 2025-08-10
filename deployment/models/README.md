# Models Directory

This directory is the **primary location** for your trained emotion detection models.

## What to put here:

### 🎯 From Colab Training:
Place your downloaded model files from Google Colab here:

- **`best_domain_adapted_model.pth`** ✅ (most common)
- **`comprehensive_emotion_model_final/`** (HuggingFace directory format)
- **`emotion_model_ensemble_final/`** (ensemble model directory)
- **`emotion_model_specialized_final/`** (specialized model directory)
- **`domain_adapted_model/`** (domain adaptation model directory)

### 📁 Expected Structure:
```
deployment/models/
├── best_domain_adapted_model.pth          # PyTorch model file
├── comprehensive_emotion_model_final/      # HuggingFace model directory
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── label_encoder.pkl                       # Label encoder (if available)
└── model_config.json                       # Model metadata (if available)
```

## How to use:

1. **Download** your trained model from Google Colab
2. **Place** it in this directory (`/Users/minervae/Projects/SAMO--GENERAL/SAMO--DL/deployment/models/`)
3. **Run** the upload script:
   ```bash
   python scripts/deployment/upload_model_to_huggingface.py
   ```

The script will automatically find your model here and upload it to HuggingFace Hub for production deployment.

## Model Types Supported:

- **`.pth` files** (PyTorch state dicts) → Automatically converted to HuggingFace format
- **HuggingFace directories** → Directly uploaded with metadata updates
- **Checkpoint files** → Extracted and converted

## Notes:

- Only put **trained/fine-tuned** models here, not base models
- The script prioritizes this directory over all other locations
- Models should be trained on your specific emotion classes: `anxious`, `calm`, `content`, `excited`, `frustrated`, `grateful`, `happy`, `hopeful`, `overwhelmed`, `proud`, `sad`, `tired`

---

**Need help?** Check the complete guide: [`deployment/CUSTOM_MODEL_DEPLOYMENT_GUIDE.md`](../CUSTOM_MODEL_DEPLOYMENT_GUIDE.md)