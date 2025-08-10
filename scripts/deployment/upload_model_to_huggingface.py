#!/usr/bin/env python3
"""
🚀 UPLOAD CUSTOM TRAINED MODEL TO HUGGINGFACE HUB
=================================================
Upload your custom-trained emotion detection model to HuggingFace Hub 
so it can be used in production deployment.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from huggingface_hub import HfApi, login, create_repo
from sklearn.preprocessing import LabelEncoder
import pickle

def print_banner():
    """Print banner"""
    print("🚀 UPLOAD CUSTOM MODEL TO HUGGINGFACE HUB")
    print("=" * 60)
    print("This script will:")
    print("  1. Find your best trained model")
    print("  2. Prepare it for HuggingFace Hub")
    print("  3. Upload it to your HuggingFace account")
    print("  4. Update deployment configurations")
    print()

def find_best_trained_model() -> Optional[str]:
    """Find the best trained model from common locations."""
    print("🔍 SEARCHING FOR TRAINED MODELS")
    print("=" * 40)
    
    # Priority order of model locations
    model_search_paths = [
        # From Colab downloads (most likely location)
        os.path.expanduser("~/Downloads/best_domain_adapted_model.pth"),
        os.path.expanduser("~/Downloads/comprehensive_emotion_model_final"),
        os.path.expanduser("~/Desktop/best_domain_adapted_model.pth"),
        os.path.expanduser("~/Desktop/comprehensive_emotion_model_final"),
        
        # From local training scripts
        "./models/checkpoints/focal_loss_best_model.pt",
        "./models/checkpoints/simple_working_model.pt",
        "./models/checkpoints/minimal_working_model.pt",
        
        # From notebook exports
        "./emotion_model_ensemble_final",
        "./emotion_model_specialized_final",
        "./emotion_model_fixed_bulletproof_final",
        "./comprehensive_emotion_model_final",
        "./domain_adapted_model",
        "./emotion_model",
        
        # Individual files
        "./best_domain_adapted_model.pth",
        "./best_simple_model.pth",
        "./best_focal_model.pth",
    ]
    
    found_models = []
    
    for path in model_search_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                # Check if it's a complete HuggingFace model directory
                config_file = os.path.join(path, "config.json")
                tokenizer_file = os.path.join(path, "tokenizer.json")
                if os.path.exists(config_file):
                    size = sum(os.path.getsize(os.path.join(path, f)) 
                             for f in os.listdir(path) 
                             if os.path.isfile(os.path.join(path, f)))
                    found_models.append((path, size, "huggingface_dir"))
                    print(f"✅ Found HF model directory: {path} ({size:,} bytes)")
            else:
                # Individual model file
                size = os.path.getsize(path)
                found_models.append((path, size, "model_file"))
                print(f"✅ Found model file: {path} ({size:,} bytes)")
    
    if not found_models:
        print("❌ No trained models found!")
        print("\n📋 To use this script, you need to:")
        print("  1. Download your trained model from Colab")
        print("  2. Place it in Downloads/ or Desktop/")
        print("  3. Run this script again")
        return None
    
    print(f"\n📊 Found {len(found_models)} model(s)")
    
    # Return the largest model (likely the best one)
    best_model = max(found_models, key=lambda x: x[1])
    print(f"🎯 Selected best model: {best_model[0]} ({best_model[1]:,} bytes)")
    
    return best_model[0]

def setup_huggingface_auth():
    """Setup HuggingFace authentication."""
    print("\n🔐 HUGGINGFACE AUTHENTICATION")
    print("=" * 40)
    
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if not hf_token:
        print("❌ HUGGINGFACE_TOKEN environment variable not set")
        print("\n📋 To authenticate:")
        print("  1. Go to https://huggingface.co/settings/tokens")
        print("  2. Create a new token with 'write' permissions")
        print("  3. Set it as environment variable:")
        print("     export HUGGINGFACE_TOKEN='your_token_here'")
        print("  4. Or run: huggingface-cli login")
        
        # Try interactive login
        try:
            login()
            print("✅ Successfully logged in via interactive login!")
            return True
        except Exception as e:
            print(f"❌ Interactive login failed: {e}")
            return False
    else:
        login(token=hf_token)
        print("✅ Successfully authenticated with token!")
        return True

def prepare_model_for_upload(model_path: str, temp_dir: str) -> Dict[str, Any]:
    """Prepare model for HuggingFace Hub upload."""
    print(f"\n🔧 PREPARING MODEL: {model_path}")
    print("=" * 40)
    
    os.makedirs(temp_dir, exist_ok=True)
    
    # Define emotion labels (based on your training)
    emotion_labels = [
        'anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful',
        'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired'
    ]
    
    # Create label mappings
    id2label = {i: label for i, label in enumerate(emotion_labels)}
    label2id = {label: i for i, label in enumerate(emotion_labels)}
    
    if os.path.isdir(model_path):
        # Already a HuggingFace directory - copy and update
        print("📁 Processing HuggingFace model directory...")
        
        # Copy all files
        for file in os.listdir(model_path):
            src = os.path.join(model_path, file)
            dst = os.path.join(temp_dir, file)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print(f"  ✅ Copied: {file}")
        
        # Update config if needed
        config_path = os.path.join(temp_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            config.update({
                'id2label': id2label,
                'label2id': label2id,
                'num_labels': len(emotion_labels)
            })
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print("  ✅ Updated config.json")
        
    else:
        # Individual .pth file - need to reconstruct HuggingFace model
        print("🔄 Converting .pth file to HuggingFace format...")
        
        # Load the state dict
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Determine base model (make educated guess)
        base_model_name = "distilroberta-base"  # Most commonly used in your training
        
        print(f"  📦 Using base model: {base_model_name}")
        
        # Load base model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=len(emotion_labels),
            id2label=id2label,
            label2id=label2id
        )
        
        # Load trained weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("  ✅ Loaded model_state_dict")
        else:
            model.load_state_dict(checkpoint)
            print("  ✅ Loaded state_dict directly")
        
        # Save in HuggingFace format
        model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(temp_dir)
        print("  ✅ Saved in HuggingFace format")
    
    # Create model card
    model_card = f"""---
language: en
tags:
- emotion-detection
- text-classification
- pytorch
- transformers
license: apache-2.0
datasets:
- custom-journal-entries
metrics:
- f1
- accuracy
---

# SAMO-DL Custom Emotion Detection Model

This model is a fine-tuned version of a transformer model for emotion detection, specifically trained on journal entries and personal text data.

## Model Details

- **Model Type:** Emotion Classification
- **Language:** English
- **Training Data:** Custom journal entries + domain adaptation
- **Labels:** {len(emotion_labels)} emotion categories
- **Architecture:** Transformer-based (DistilRoBERTa/BERT)

## Emotions Detected

{', '.join(emotion_labels)}

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("your-username/samo-dl-emotion-model")
model = AutoModelForSequenceClassification.from_pretrained("your-username/samo-dl-emotion-model")

text = "I'm feeling really happy today!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=-1)

emotion = model.config.id2label[predicted_class.item()]
confidence = predictions[0][predicted_class].item()

print(f"Emotion: {{emotion}} ({{confidence:.3f}})")
```

## Training Details

- **Training Framework:** PyTorch + Transformers
- **Optimization:** Custom focal loss for class imbalance
- **Validation:** Domain adaptation on journal entries
- **Performance:** Optimized for personal/journal text emotion detection

## Intended Use

This model is specifically designed for emotion detection in personal journal entries and similar informal text. 
It may not perform optimally on formal text or other domains.

## Limitations

- Trained primarily on English text
- Optimized for informal, personal writing style
- May have biases present in the training data
"""
    
    with open(os.path.join(temp_dir, "README.md"), 'w') as f:
        f.write(model_card)
    print("  ✅ Created model card (README.md)")
    
    # Create requirements.txt for the model
    requirements = """torch>=1.9.0
transformers>=4.21.0
numpy>=1.21.0
"""
    
    with open(os.path.join(temp_dir, "requirements.txt"), 'w') as f:
        f.write(requirements)
    print("  ✅ Created requirements.txt")
    
    return {
        'emotion_labels': emotion_labels,
        'id2label': id2label,
        'label2id': label2id,
        'num_labels': len(emotion_labels)
    }

def upload_to_huggingface(temp_dir: str, model_info: Dict[str, Any]) -> str:
    """Upload model to HuggingFace Hub."""
    print(f"\n🚀 UPLOADING TO HUGGINGFACE HUB")
    print("=" * 40)
    
    # Get user info
    api = HfApi()
    user_info = api.whoami()
    username = user_info['name']
    
    # Create repository name
    repo_name = f"{username}/samo-dl-emotion-model"
    print(f"📦 Repository: {repo_name}")
    
    try:
        # Create repository
        create_repo(repo_name, exist_ok=True)
        print("✅ Repository created/confirmed")
        
        # Upload all files
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=repo_name,
            repo_type="model"
        )
        print("✅ Model uploaded successfully!")
        
        model_url = f"https://huggingface.co/{repo_name}"
        print(f"🔗 Model URL: {model_url}")
        
        return repo_name
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None

def update_deployment_config(repo_name: str, model_info: Dict[str, Any]):
    """Update deployment configurations to use the new model."""
    print(f"\n🔧 UPDATING DEPLOYMENT CONFIGURATIONS")
    print("=" * 40)
    
    # Update model_utils.py to use the new model
    model_utils_path = "deployment/cloud-run/model_utils.py"
    
    if os.path.exists(model_utils_path):
        with open(model_utils_path, 'r') as f:
            content = f.read()
        
        # Update model loading to use HuggingFace model
        updated_content = content.replace(
            "AutoTokenizer.from_pretrained('distilroberta-base')",
            f"AutoTokenizer.from_pretrained('{repo_name}')"
        ).replace(
            "AutoModelForSequenceClassification.from_pretrained(\n            'distilroberta-base',",
            f"AutoModelForSequenceClassification.from_pretrained(\n            '{repo_name}',"
        )
        
        with open(model_utils_path, 'w') as f:
            f.write(updated_content)
        
        print(f"✅ Updated {model_utils_path}")
    
    # Create a new deployment config file
    config_path = "deployment/custom_model_config.json"
    config = {
        "model_name": repo_name,
        "model_type": "custom_trained",
        "emotion_labels": model_info['emotion_labels'],
        "num_labels": model_info['num_labels'],
        "id2label": model_info['id2label'],
        "label2id": model_info['label2id'],
        "deployment_ready": True
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created {config_path}")
    print("\n📋 Next steps:")
    print("  1. Test the deployment locally")
    print("  2. Update environment variables if needed")
    print("  3. Deploy to production")

def main():
    """Main function."""
    print_banner()
    
    # Step 1: Find trained model
    model_path = find_best_trained_model()
    if not model_path:
        return False
    
    # Step 2: Setup authentication
    if not setup_huggingface_auth():
        return False
    
    # Step 3: Prepare model
    temp_dir = "./temp_model_upload"
    model_info = prepare_model_for_upload(model_path, temp_dir)
    
    # Step 4: Upload to HuggingFace
    repo_name = upload_to_huggingface(temp_dir, model_info)
    if not repo_name:
        return False
    
    # Step 5: Update deployment configs
    update_deployment_config(repo_name, model_info)
    
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print("🧹 Cleaned up temporary files")
    
    print("\n🎉 SUCCESS! Your custom model is now ready for deployment!")
    print(f"🔗 Model: https://huggingface.co/{repo_name}")
    print("\n📋 To use in deployment:")
    print(f"   MODEL_NAME={repo_name}")
    print("   Update your environment variables and redeploy")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)