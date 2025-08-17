#!/usr/bin/env python3
"""
Reorganize Model Directory
==========================

This script reorganizes the deployment model directory to:
1. Save the current working model as model_1 (fallback)
2. Prepare structure for the comprehensive model as default
3. Create clear versioning and documentation
"""


import os
import shutil
import json
from datetime import datetime


def reorganize_model_directory():
    """Reorganize the model directory with versioning."""
    
    print("📁 REORGANIZING MODEL DIRECTORY")
    print("=" * 50)
    
    # Define paths
    current_model_path = "deployment/model"
    models_dir = "deployment/models"
    model_1_path = os.path.join(models_dir, "model_1_fallback")
    default_model_path = os.path.join(models_dir, "default")
    
    # Create models directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"✅ Created models directory: {models_dir}")
    
    # 1. Save current model as model_1 (fallback)
    print(f"\n💾 SAVING CURRENT MODEL AS FALLBACK")
    print("-" * 40)
    
    if os.path.exists(current_model_path):
        # Copy current model to model_1_fallback
        if os.path.exists(model_1_path):
            shutil.rmtree(model_1_path)
        
        shutil.copytree(current_model_path, model_1_path)
        print(f"✅ Saved current model as: {model_1_path}")
        
        # Create model metadata
        model_1_metadata = {
            "version": "1.0",
            "name": "model_1_fallback",
            "description": "Working model with configuration persistence fix",
            "created_date": datetime.now().isoformat(),
            "performance": {
                "test_accuracy": "91.67%",
                "average_confidence": "0.298",
                "architecture": "DistilRoBERTa",
                "num_labels": 12,
                "problem_type": "single_label_classification"
            },
            "training_details": {
                "dataset_size": "60 samples (48 train, 12 validation)",
                "training_epochs": 3,
                "final_f1_score": "0.8889",
                "final_accuracy": "0.9167"
            },
            "status": "fallback_model",
            "notes": "Successfully resolved configuration persistence issue. Ready for deployment."
        }
        
        # Save metadata
        metadata_path = os.path.join(model_1_path, "model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(model_1_metadata, f, indent=2)
        print(f"✅ Created model metadata: {metadata_path}")
        
    else:
        print(f"❌ Current model not found at: {current_model_path}")
        return
    
    # 2. Create default model directory structure
    print(f"\n📂 CREATING DEFAULT MODEL STRUCTURE")
    print("-" * 40)
    
    if os.path.exists(default_model_path):
        shutil.rmtree(default_model_path)
    
    os.makedirs(default_model_path)
    print(f"✅ Created default model directory: {default_model_path}")
    
    # Create placeholder metadata for default model
    default_metadata = {
        "version": "2.0",
        "name": "default_comprehensive",
        "description": "Comprehensive model with all advanced features (to be trained)",
        "created_date": "pending",
        "performance": {
            "test_accuracy": "pending",
            "average_confidence": "pending",
            "architecture": "DistilRoBERTa",
            "num_labels": 12,
            "problem_type": "single_label_classification"
        },
        "training_details": {
            "dataset_size": "240+ samples with augmentation",
            "training_epochs": "5",
            "features": [
                "Focal loss",
                "Class weighting",
                "Advanced data augmentation",
                "Comprehensive validation",
                "Configuration persistence"
            ]
        },
        "status": "pending_training",
        "notes": "Will be trained using COMPREHENSIVE_ULTIMATE_TRAINING_COLAB.ipynb"
    }
    
    # Save default metadata
    default_metadata_path = os.path.join(default_model_path, "model_metadata.json")
    with open(default_metadata_path, 'w') as f:
        json.dump(default_metadata, f, indent=2)
    print(f"✅ Created default model metadata: {default_metadata_path}")
    
    # 3. Create models index file
    print(f"\n📋 CREATING MODELS INDEX")
    print("-" * 40)
    
    models_index = {
        "models_directory": models_dir,
        "current_default": "default",
        "fallback_model": "model_1_fallback",
        "models": {
            "model_1_fallback": {
                "path": "model_1_fallback",
                "version": "1.0",
                "status": "ready",
                "description": "Working model with configuration persistence fix"
            },
            "default": {
                "path": "default",
                "version": "2.0",
                "status": "pending",
                "description": "Comprehensive model with all advanced features"
            }
        },
        "last_updated": datetime.now().isoformat(),
        "notes": "Use default model for production, model_1_fallback as backup"
    }
    
    index_path = os.path.join(models_dir, "models_index.json")
    with open(index_path, 'w') as f:
        json.dump(models_index, f, indent=2)
    print(f"✅ Created models index: {index_path}")
    
    # 4. Create README for models directory
    print(f"\n📖 CREATING MODELS README")
    print("-" * 40)
    
    readme_content = """# Model Versions

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
"""
    
    readme_path = os.path.join(models_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"✅ Created models README: {readme_path}")
    
    # 5. Create symlink for easy access
    print(f"\n🔗 CREATING SYMLINKS")
    print("-" * 40)
    
    # Create symlink from deployment/model to default model
    symlink_path = "deployment/model"
    if os.path.exists(symlink_path):
        if os.path.islink(symlink_path):
            os.unlink(symlink_path)
        else:
            # Backup the original model directory
            backup_path = "deployment/model_backup"
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)
            shutil.move(symlink_path, backup_path)
            print(f"✅ Backed up original model to: {backup_path}")
    
    # Create symlink to default model
    try:
        os.symlink(default_model_path, symlink_path)
        print(f"✅ Created symlink: {symlink_path} -> {default_model_path}")
    except Exception as e:
        print(f"⚠️ Could not create symlink: {e}")
        print(f"   You can manually link {symlink_path} to {default_model_path}")
    
    # 6. Summary
    print(f"\n📋 REORGANIZATION SUMMARY")
    print("=" * 50)
    
    print("✅ Model directory reorganized successfully!")
    print()
    print("📁 New Structure:")
    print(f"   {models_dir}/")
    print(f"   ├── model_1_fallback/     # Your working model (91.67% accuracy)")
    print(f"   ├── default/              # Ready for comprehensive model")
    print(f"   ├── models_index.json     # Model registry")
    print(f"   └── README.md             # Documentation")
    print()
    print("🎯 Next Steps:")
    print("   1. Train the comprehensive model using COMPREHENSIVE_ULTIMATE_TRAINING_COLAB.ipynb")
    print("   2. Save the trained model to deployment/models/default/")
    print("   3. Update the default model metadata")
    print("   4. Test the new model")
    print()
    print("🛡️ Safety:")
    print("   - Model 1 is preserved as fallback")
    print("   - Original model backed up to deployment/model_backup/")
    print("   - Clear versioning and documentation")

if __name__ == "__main__":
    reorganize_model_directory() 