# Model Versions

This directory contains different versions of the emotion detection model.

## Model Structure

```
models/
├── model_1_fallback/     # Working model with configuration persistence fix
├── default/              # Comprehensive model (to be trained)
└── models_index.json     # Index of all models
```

## Model Versions

### Model 1 (Fallback) - `model_1_fallback/`
- **Version**: 1.0
- **Status**: Ready for deployment
- **Performance**: 91.67% test accuracy
- **Features**: 
  - Configuration persistence fix
  - DistilRoBERTa architecture
  - 12 emotion classes
- **Use Case**: Fallback model, production deployment

### Default Model - `default/`
- **Version**: 2.0
- **Status**: Pending training
- **Expected Features**:
  - All features from Model 1
  - Focal loss
  - Class weighting
  - Advanced data augmentation
  - Comprehensive validation
- **Use Case**: Primary production model (once trained)

## Usage

### For Production Deployment
```python
# Use default model (once trained)
model_path = "deployment/models/default"

# Fallback to model_1 if needed
fallback_path = "deployment/models/model_1_fallback"
```

### For Testing
```python
# Test specific model version
model_path = "deployment/models/model_1_fallback"
```

## Model Metadata

Each model directory contains:
- `model_metadata.json`: Detailed model information
- Model files (config.json, model.safetensors, etc.)
- Training artifacts

## Notes

- Model 1 is the working fallback with configuration persistence fix
- Default model will be trained using the comprehensive notebook
- Always test models before deployment
- Keep fallback models for safety
