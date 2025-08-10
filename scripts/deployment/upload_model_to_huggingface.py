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
    
    # Ensure primary model directory exists
    primary_model_dir = "/Users/minervae/Projects/SAMO--GENERAL/SAMO--DL/deployment/models"
    if not os.path.exists(primary_model_dir):
        print(f"📁 Creating model directory: {primary_model_dir}")
        try:
            os.makedirs(primary_model_dir, exist_ok=True)
            print(f"✅ Created directory: {primary_model_dir}")
        except Exception as e:
            print(f"⚠️ Could not create directory: {e}")
    
    print(f"🎯 PRIMARY SEARCH LOCATION: {primary_model_dir}")
    print("🔄 Also checking fallback locations...")
    
    # Priority order of model locations
    model_search_paths = [
        # PRIMARY: User's specified model directory (absolute path)
        "/Users/minervae/Projects/SAMO--GENERAL/SAMO--DL/deployment/models/best_domain_adapted_model.pth",
        "/Users/minervae/Projects/SAMO--GENERAL/SAMO--DL/deployment/models/comprehensive_emotion_model_final",
        "/Users/minervae/Projects/SAMO--GENERAL/SAMO--DL/deployment/models/emotion_model_ensemble_final",
        "/Users/minervae/Projects/SAMO--GENERAL/SAMO--DL/deployment/models/emotion_model_specialized_final",
        "/Users/minervae/Projects/SAMO--GENERAL/SAMO--DL/deployment/models/emotion_model_fixed_bulletproof_final",
        "/Users/minervae/Projects/SAMO--GENERAL/SAMO--DL/deployment/models/domain_adapted_model",
        "/Users/minervae/Projects/SAMO--GENERAL/SAMO--DL/deployment/models/emotion_model",
        "/Users/minervae/Projects/SAMO--GENERAL/SAMO--DL/deployment/models/best_simple_model.pth",
        "/Users/minervae/Projects/SAMO--GENERAL/SAMO--DL/deployment/models/best_focal_model.pth",
        
        # LOCAL: Relative path (in case absolute path doesn't work)
        "./deployment/models/best_domain_adapted_model.pth",
        "./deployment/models/comprehensive_emotion_model_final",
        "./deployment/models/emotion_model_ensemble_final",
        "./deployment/models/emotion_model_specialized_final",
        "./deployment/models/emotion_model_fixed_bulletproof_final",
        "./deployment/models/domain_adapted_model",
        "./deployment/models/emotion_model",
        "./deployment/models/best_simple_model.pth",
        "./deployment/models/best_focal_model.pth",
        
        # FALLBACK: Common locations (from Colab downloads)
        os.path.expanduser("~/Downloads/best_domain_adapted_model.pth"),
        os.path.expanduser("~/Downloads/comprehensive_emotion_model_final"),
        os.path.expanduser("~/Desktop/best_domain_adapted_model.pth"),
        os.path.expanduser("~/Desktop/comprehensive_emotion_model_final"),
        
        # From local training scripts
        "./models/checkpoints/focal_loss_best_model.pt",
        "./models/checkpoints/simple_working_model.pt",
        "./models/checkpoints/minimal_working_model.pt",
        
        # From notebook exports (relative to project root)
        "./emotion_model_ensemble_final",
        "./emotion_model_specialized_final",
        "./emotion_model_fixed_bulletproof_final",
        "./comprehensive_emotion_model_final",
        "./domain_adapted_model",
        "./emotion_model",
        
        # Individual files (project root)
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
        print(f"  2. Place it in: {primary_model_dir}")
        print("  3. Run this script again")
        print("\n📂 Expected model files:")
        print("   - best_domain_adapted_model.pth")
        print("   - comprehensive_emotion_model_final/ (directory)")
        print("   - emotion_model_ensemble_final/ (directory)")
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
        "deployment_ready": True,
        "deployment_options": {
            "serverless_api": {
                "url": f"https://api-inference.huggingface.co/models/{repo_name}",
                "cost": "free",
                "best_for": "development_testing",
                "cold_starts": True,
                "rate_limits": True
            },
            "inference_endpoints": {
                "setup_url": "https://ui.endpoints.huggingface.co/",
                "cost": "paid_per_usage",
                "best_for": "production",
                "cold_starts": False,
                "consistent_latency": True
            },
            "self_hosted": {
                "model_loading": f"AutoModelForSequenceClassification.from_pretrained('{repo_name}')",
                "cost": "infrastructure_costs",
                "best_for": "maximum_control",
                "requires": ["transformers", "torch"]
            }
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created {config_path}")
    
    # Create environment template files for different deployment strategies
    create_environment_templates(repo_name)
    
    print("\n📋 Next steps:")
    print("  1. Choose your deployment strategy:")
    print("     - Serverless API (free, for development)")
    print("     - Inference Endpoints (paid, for production)")  
    print("     - Self-hosted (your infrastructure)")
    print("  2. Test locally with the new model")
    print("  3. Deploy to your chosen environment")
    print("  4. Monitor usage and performance")

def create_environment_templates(repo_name: str):
    """Create environment configuration templates for different deployment strategies."""
    
    # Serverless API template
    serverless_env = f"""# HuggingFace Serverless API Configuration
# Best for: Development, testing, light usage
# Cost: Free with rate limits

HF_TOKEN=your_hf_token_here
MODEL_NAME={repo_name}
DEPLOYMENT_TYPE=serverless
API_URL=https://api-inference.huggingface.co/models/{repo_name}

# Optional settings
MAX_RETRIES=3
TIMEOUT_SECONDS=30
RATE_LIMIT_PAUSE=1
"""
    
    with open(".env.serverless.template", 'w') as f:
        f.write(serverless_env)
    print("✅ Created .env.serverless.template")
    
    # Inference Endpoints template  
    endpoints_env = f"""# HuggingFace Inference Endpoints Configuration
# Best for: Production, consistent latency, high throughput
# Cost: Paid per resource usage

HF_TOKEN=your_hf_token_here
MODEL_NAME={repo_name}
DEPLOYMENT_TYPE=endpoint
INFERENCE_ENDPOINT_URL=https://your-endpoint-id.us-east-1.aws.endpoints.huggingface.cloud

# Setup your endpoint at: https://ui.endpoints.huggingface.co/
# Choose instance type: CPU (cost-effective) or GPU (faster)

# Optional settings
MAX_RETRIES=3
TIMEOUT_SECONDS=10
"""
    
    with open(".env.endpoints.template", 'w') as f:
        f.write(endpoints_env)
    print("✅ Created .env.endpoints.template")
    
    # Self-hosted template
    selfhosted_env = f"""# Self-Hosted Configuration  
# Best for: Maximum control, custom requirements, data privacy
# Cost: Your infrastructure costs

MODEL_NAME={repo_name}
DEPLOYMENT_TYPE=local
DEVICE=cpu  # or 'cuda' if you have GPU

# Model loading will be done locally using transformers library
# Requires: pip install transformers torch

# Optional optimization settings
TORCH_NUM_THREADS=4
MODEL_CACHE_DIR=./model_cache
BATCH_SIZE=1
MAX_LENGTH=128
"""
    
    with open(".env.selfhosted.template", 'w') as f:
        f.write(selfhosted_env)
    print("✅ Created .env.selfhosted.template")

def setup_git_lfs():
    """Set up Git LFS for large model files."""
    print("\n🔧 SETTING UP GIT LFS FOR LARGE MODEL FILES")
    print("=" * 40)
    
    try:
        # Check if git lfs is available
        import subprocess
        result = subprocess.run(['git', 'lfs', 'version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("⚠️ Git LFS not available. Large model files will use regular git.")
            print("   Install with: git lfs install")
            return False
            
        # Track large model files
        lfs_patterns = [
            "*.bin",
            "*.safetensors", 
            "*.onnx",
            "*.pkl",
            "*.pth",
            "*.pt",
            "*.h5"
        ]
        
        for pattern in lfs_patterns:
            subprocess.run(['git', 'lfs', 'track', pattern], capture_output=True, text=True)
            print(f"✅ Tracking {pattern} with Git LFS")
        
        # Update .gitattributes if it exists
        gitattributes_path = ".gitattributes"
        if os.path.exists(gitattributes_path):
            with open(gitattributes_path, 'r') as f:
                content = f.read()
            
            # Add LFS tracking if not already present
            for pattern in lfs_patterns:
                lfs_line = f"{pattern} filter=lfs diff=lfs merge=lfs -text"
                if lfs_line not in content:
                    content += f"\n{lfs_line}"
            
            with open(gitattributes_path, 'w') as f:
                f.write(content)
            
            print("✅ Updated .gitattributes for Git LFS")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Git LFS setup failed: {e}")
        print("   Large model files will be uploaded directly")
        return False

def upload_to_huggingface(temp_dir: str, model_info: Dict[str, Any]) -> str:
    """Upload model to HuggingFace Hub."""
    print(f"\n🚀 UPLOADING TO HUGGINGFACE HUB")
    print("=" * 40)
    
    # Set up Git LFS before upload
    setup_git_lfs()
    
    # Get user info
    api = HfApi()
    user_info = api.whoami()
    username = user_info['name']
    
    # Create repository name
    repo_name = f"{username}/samo-dl-emotion-model"
    print(f"📦 Repository: {repo_name}")
    
    try:
        # Create repository (public by default for free hosting)
        create_repo(
            repo_name, 
            exist_ok=True,
            private=False,  # Public repos are free
            repo_type="model"
        )
        print("✅ Repository created/confirmed (public)")
        
        # Upload all files
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=repo_name,
            repo_type="model",
            commit_message="Upload custom emotion detection model"
        )
        print("✅ Model uploaded successfully!")
        
        model_url = f"https://huggingface.co/{repo_name}"
        print(f"🔗 Model URL: {model_url}")
        
        # Print deployment options
        print(f"\n🎯 DEPLOYMENT OPTIONS:")
        print(f"  🆓 Serverless API: https://api-inference.huggingface.co/models/{repo_name}")
        print(f"  🚀 Inference Endpoints: https://ui.endpoints.huggingface.co/ (create endpoint)")
        print(f"  🏠 Self-hosted: AutoModelForSequenceClassification.from_pretrained('{repo_name}')")
        
        return repo_name
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("\n🔍 Common issues:")
        print("  - Check your HF token has write permissions")
        print("  - Ensure you haven't exceeded storage quotas")
        print("  - Large files need Git LFS (we tried to set this up)")
        print("  - Check network connection and HF Hub status")
        return None

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
    
    print("\n📋 DEPLOYMENT STRATEGIES:")
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│ 🆓 SERVERLESS API (Recommended for Development)                    │")
    print("│   • Cost: Free with rate limits                                    │")
    print("│   • Setup: Use .env.serverless.template                            │")
    print("│   • Test: curl with HF_TOKEN authorization                         │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│ 🚀 INFERENCE ENDPOINTS (Recommended for Production)                │")
    print("│   • Cost: Paid per usage (~$0.06-1.20/hour)                       │")
    print("│   • Setup: https://ui.endpoints.huggingface.co/                    │")
    print("│   • Benefits: No cold starts, consistent latency                   │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│ 🏠 SELF-HOSTED (Maximum Control)                                   │")
    print("│   • Cost: Your infrastructure                                      │")
    print("│   • Setup: Use .env.selfhosted.template                            │")
    print("│   • Benefits: Complete control, data privacy                       │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    print("\n🚀 QUICK TEST (Serverless API):")
    print(f"   export HF_TOKEN='your_token_here'")
    print(f"   curl -X POST \\")
    print(f"     -H \"Authorization: Bearer $HF_TOKEN\" \\")
    print(f"     -H \"Content-Type: application/json\" \\")
    print(f"     -d '{{\"inputs\": \"I am feeling really happy today!\"}}' \\")
    print(f"     https://api-inference.huggingface.co/models/{repo_name}")
    
    print("\n📁 FILES CREATED:")
    print("   • deployment/custom_model_config.json (model metadata)")
    print("   • .env.serverless.template (for serverless API)")
    print("   • .env.endpoints.template (for inference endpoints)")
    print("   • .env.selfhosted.template (for self-hosting)")
    
    print("\n📖 NEXT STEPS:")
    print("   1. Choose deployment strategy (start with serverless for free)")
    print("   2. Copy appropriate .env template to .env")
    print("   3. Set your HF_TOKEN in the environment")
    print("   4. Test your model with the quick test above")
    print("   5. Integrate into your application")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)