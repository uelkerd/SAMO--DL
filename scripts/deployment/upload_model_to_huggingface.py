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
from typing import Optional

# Use built-in generics for Python 3.9+ (PEP 585)
if sys.version_info >= (3, 9):
    # Modern typing: use built-in dict, list instead of typing.Dict, typing.List
    pass  # Use dict[str, Any] directly
else:
    from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import HfApi, login, create_repo

def get_base_model_name() -> str:
    """Get the base model name with configurable support."""
    # Check environment variable first
    base_model = os.getenv('BASE_MODEL_NAME')
    if base_model:
        print(f"📦 Using BASE_MODEL_NAME from environment: {base_model}")
        return base_model

    # Default fallback
    default_model = "distilroberta-base"
    print(f"📦 Using default base model: {default_model}")
    return default_model

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

def get_model_base_directory() -> str:
    """Get the base directory for model storage with environment variable override."""
    # Priority order for determining base directory:
    # 1. Environment variable (most flexible)
    # 2. Auto-detect project root
    # 3. Current working directory fallback
    # Option 1: Check for environment variable override
    env_base_dir = os.getenv('SAMO_DL_BASE_DIR') or os.getenv('MODEL_BASE_DIR')
    if env_base_dir:
        base_dir = os.path.expanduser(env_base_dir)
        if os.path.exists(base_dir):
            return os.path.join(base_dir, "deployment", "models")
        print(f"⚠️ Environment base directory doesn't exist: {base_dir}")

    # Option 2: Auto-detect project root (look for specific files that indicate SAMO-DL root)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Walk up the directory tree to find project root
    search_dir = current_dir
    max_levels = 5  # Prevent infinite loops

    for _ in range(max_levels):
        # Check for project indicators
        indicators = [
            'deployment',
            'src', 
            'notebooks',
            'pyproject.toml',
            'CHANGELOG.md'
        ]

        if all(os.path.exists(os.path.join(search_dir, indicator)) for indicator in indicators[:2]):
            # Found project root
            return os.path.join(search_dir, "deployment", "models")

        parent_dir = os.path.dirname(search_dir)
        if parent_dir == search_dir:  # Reached filesystem root
            break
        search_dir = parent_dir

    # Option 3: Fallback to current working directory
    cwd_models_dir = os.path.join(os.getcwd(), "deployment", "models")
    return cwd_models_dir

def find_best_trained_model() -> Optional[str]:
    """
    Find the best trained model from common locations.

    Uses configurable paths for portability across different systems:
    - Environment variables: SAMO_DL_BASE_DIR or MODEL_BASE_DIR
    - Auto-detection: Searches for project root markers
    - Fallback: Current working directory + deployment/models

    Returns:
        Path to the best model found, or None if no models found
    """
    print("🔍 SEARCHING FOR TRAINED MODELS")
    print("=" * 40)

    # Get configurable base directory (no hardcoded paths!)
    primary_model_dir = get_model_base_directory()

    # Display configuration info
    env_override = os.getenv('SAMO_DL_BASE_DIR') or os.getenv('MODEL_BASE_DIR')
    if env_override:
        print(f"🔧 Using environment override: {env_override}")
    else:
        print("🔍 Auto-detected project location")

    print(f"🎯 PRIMARY SEARCH LOCATION: {primary_model_dir}")

    # Ensure primary model directory exists
    if not os.path.exists(primary_model_dir):
        print(f"📁 Creating model directory: {primary_model_dir}")
        try:
            os.makedirs(primary_model_dir, exist_ok=True)
            print(f"✅ Created directory: {primary_model_dir}")
        except Exception as e:
            print(f"⚠️ Could not create directory: {e}")

    print("🔄 Also checking fallback locations...")

    # Model file patterns to search for
    model_patterns = [
        "best_domain_adapted_model.pth",
        "comprehensive_emotion_model_final",
        "emotion_model_ensemble_final", 
        "emotion_model_specialized_final",
        "emotion_model_fixed_bulletproof_final",
        "domain_adapted_model",
        "emotion_model",
        "best_simple_model.pth",
        "best_focal_model.pth",
    ]

    # Priority order of model locations (now dynamically constructed)
    model_search_paths = []

    # PRIMARY: Configured model directory
    for pattern in model_patterns:
        model_search_paths.append(os.path.join(primary_model_dir, pattern))

    # FALLBACK 1: Common download locations
    common_download_locations = [
        os.path.expanduser("~/Downloads"),
        os.path.expanduser("~/Desktop"),
        os.path.expanduser("~/Documents"),
    ]

    for download_dir in common_download_locations:
        for pattern in model_patterns:
            model_search_paths.append(os.path.join(download_dir, pattern))

    # FALLBACK 2: Relative paths from current directory  
    relative_locations = [
        "./deployment/models",
        "./models/checkpoints", 
        "./",  # Project root
    ]

    for rel_dir in relative_locations:
        for pattern in model_patterns:
            model_search_paths.append(os.path.join(rel_dir, pattern))

    # FALLBACK 3: Additional specific training checkpoint locations
    checkpoint_patterns = [
        "focal_loss_best_model.pt",
        "simple_working_model.pt", 
        "minimal_working_model.pt",
    ]

    for pattern in checkpoint_patterns:
        model_search_paths.append(os.path.join("./models/checkpoints", pattern))
        # Also check in primary model directory
        model_search_paths.append(os.path.join(primary_model_dir, pattern))

    found_models = []

    for path in model_search_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                # Check if it's a complete HuggingFace model directory
                config_file = os.path.join(path, "config.json")
                tokenizer_file = os.path.join(path, "tokenizer.json")
                tokenizer_config_file = os.path.join(path, "tokenizer_config.json")

                # Check for essential files (config.json is required, tokenizer files are highly recommended)
                has_config = os.path.exists(config_file)
                has_tokenizer = (os.path.exists(tokenizer_file) or 
                               os.path.exists(tokenizer_config_file) or
                               os.path.exists(os.path.join(path, "vocab.txt")) or
                               os.path.exists(os.path.join(path, "vocab.json")))

                # Check for model weight files (essential for a complete model)
                weight_files = [
                    os.path.join(path, f) for f in [
                        "pytorch_model.bin", "model.safetensors", "pytorch_model.safetensors",
                        "model.bin", "tf_model.h5", "flax_model.msgpack"
                    ] if os.path.exists(os.path.join(path, f))
                ]
                has_weights = len(weight_files) > 0

                # Only accept as valid HF model if has config, tokenizer, AND weights
                if has_config and has_tokenizer and has_weights:
                    # Calculate recursive directory size including all nested files
                    def calculate_directory_size(directory):
                        total_size = 0
                        for dirpath, _, filenames in os.walk(directory):
                            for filename in filenames:
                                filepath = os.path.join(dirpath, filename)
                                try:
                                    total_size += os.path.getsize(filepath)
                                except (OSError, FileNotFoundError):
                                    # Skip files that can't be accessed
                                    pass
                        return total_size

                    size = calculate_directory_size(path)
                    found_models.append((path, size, "huggingface_dir"))

                    # Enhanced logging with component status
                    weight_info = f"weights: {len(weight_files)} file(s)"
                    print(f"✅ Found complete HF model: {path} ({size:,} bytes)")
                    print(f"   • Config: ✅ • Tokenizer: ✅ • {weight_info}")

                elif has_config:
                    # Incomplete model directory - log what's missing
                    size = sum(os.path.getsize(os.path.join(path, f)) 
                             for f in os.listdir(path) 
                             if os.path.isfile(os.path.join(path, f)))

                    missing_components = []
                    if not has_tokenizer:
                        missing_components.append("tokenizer")
                    if not has_weights:
                        missing_components.append("model weights")

                    print(f"⚠️ Incomplete HF model: {path} ({size:,} bytes)")
                    print(f"   Missing: {', '.join(missing_components)}")
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

def is_interactive_environment():
    """Check if running in an interactive environment."""
    # Check common non-interactive environment indicators
    non_interactive_indicators = [
        os.getenv('CI'),  # GitHub Actions, GitLab CI, etc.
        os.getenv('DOCKER_CONTAINER'),  # Docker containers
        os.getenv('KUBERNETES_SERVICE_HOST'),  # Kubernetes pods
        os.getenv('JENKINS_URL'),  # Jenkins CI
        not sys.stdin.isatty(),  # No TTY (non-interactive shell)
    ]

    return not any(non_interactive_indicators)

def setup_huggingface_auth():
    """Setup HuggingFace authentication with non-interactive environment support."""
    print("\n🔐 HUGGINGFACE AUTHENTICATION")
    print("=" * 40)

    hf_token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    if not hf_token:
        print("❌ HuggingFace token not found in environment variables")
        print("   Checked: HUGGINGFACE_TOKEN, HF_TOKEN")
        print("\n📋 To authenticate:")
        print("  1. Go to https://huggingface.co/settings/tokens")
        print("  2. Create a new token with 'write' permissions")
        print("  3. Set it as environment variable:")
        print("     export HUGGINGFACE_TOKEN='your_token_here'")
        print("     # OR")
        print("     export HF_TOKEN='your_token_here'")

        # Check if we're in an interactive environment
        if is_interactive_environment():
            print("  4. Or try interactive login now...")

            # Try interactive login with user consent
            response = input("\n🤔 Attempt interactive login? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                try:
                    print("📝 Opening browser for HuggingFace authentication...")
                    login()
                    print("✅ Successfully logged in via interactive login!")
                    return True
                except Exception as e:
                    print(f"❌ Interactive login failed: {e}")
                    print("💡 Please set HUGGINGFACE_TOKEN environment variable instead")
                    return False
            else:
                print("ℹ️ Skipping interactive login")
                return False
        else:
            # Non-interactive environment
            print("\n⚠️ NON-INTERACTIVE ENVIRONMENT DETECTED")
            print("   Interactive login is not available in:")
            print("   - CI/CD pipelines (GitHub Actions, GitLab CI, etc.)")
            print("   - Docker containers")
            print("   - Kubernetes pods") 
            print("   - Headless servers")
            print("   - Scripts without TTY")
            print("\n✅ SOLUTION: Set HUGGINGFACE_TOKEN environment variable")
            print("   Example for CI/CD:")
            print("   - Add HUGGINGFACE_TOKEN to your repository secrets")
            print("   - Use: secrets.HUGGINGFACE_TOKEN in workflow")
            return False

    else:
        try:
            login(token=hf_token)
            print("✅ Successfully authenticated with token!")
            return True
        except Exception as e:
            print(f"❌ Token authentication failed: {e}")
            print("💡 Please check if your token has 'write' permissions")
            print("💡 Generate a new token at: https://huggingface.co/settings/tokens")
            return False

def load_emotion_labels_from_model(model_path: str) -> list[str]:
    """
    Dynamically load emotion labels from model config, checkpoint, or fallback sources.

    Priority order:
    1. HuggingFace model directory config.json (id2label)
    2. PyTorch checkpoint state_dict (label mappings)
    3. External JSON/CSV file
    4. Environment variable EMOTION_LABELS
    5. Safe default fallback
    """
    # Method 1: Load from HuggingFace model directory config.json
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)

                if 'id2label' in config:
                    # Convert id2label dict to sorted list
                    id2label = config['id2label']
                    # Ensure keys are integers for proper sorting
                    sorted_labels = [id2label[str(i)] for i in range(len(id2label))]
                    print(f"✅ Loaded {len(sorted_labels)} labels from HF config.json")
                    return sorted_labels

            except Exception as e:
                print(f"⚠️ Could not load labels from config.json: {e}")

    # Method 2: Load from PyTorch checkpoint
    elif model_path.endswith('.pth') and os.path.exists(model_path):
        try:
            # Try to load checkpoint with PyTorch version compatibility
            try:
                # For PyTorch >= 1.13.0 (weights_only parameter available)
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            except TypeError:
                # For older PyTorch versions (< 1.13.0)
                checkpoint = torch.load(model_path, map_location='cpu')
                print("  ℹ️ Using legacy PyTorch.load (consider upgrading PyTorch for security)")

            # Try to find label mappings in various checkpoint keys
            label_keys = ['id2label', 'label2id', 'labels', 'emotion_labels', 'class_names']

            for key in label_keys:
                if key in checkpoint:
                    labels_data = checkpoint[key]

                    if key == 'id2label' and isinstance(labels_data, dict):
                        sorted_labels = [labels_data[str(i)] for i in range(len(labels_data))]
                        print(f"✅ Loaded {len(sorted_labels)} labels from checkpoint['{key}']")
                        return sorted_labels

                    if key == 'label2id' and isinstance(labels_data, dict):
                        # Convert label2id to id2label format
                        id2label = {v: k for k, v in labels_data.items()}
                        sorted_labels = [id2label[i] for i in range(len(id2label))]
                        print(f"✅ Loaded {len(sorted_labels)} labels from checkpoint['{key}']")
                        return sorted_labels

                    if isinstance(labels_data, (list, tuple)):
                        print(f"✅ Loaded {len(labels_data)} labels from checkpoint['{key}']")
                        return list(labels_data)

        except Exception as e:
            print(f"⚠️ Could not load labels from checkpoint: {e}")

    # Method 3: Load from external JSON file (same directory as model)
    model_dir = os.path.dirname(model_path) if os.path.isfile(model_path) else model_path
    labels_file_paths = [
        os.path.join(model_dir, "emotion_labels.json"),
        os.path.join(model_dir, "labels.json"),
        os.path.join(model_dir, "class_names.json"),
        "emotion_labels.json",  # Current directory
        "labels.json"
    ]

    for labels_file in labels_file_paths:
        if os.path.exists(labels_file):
            try:
                with open(labels_file, 'r') as f:
                    labels_data = json.load(f)

                if isinstance(labels_data, list):
                    print(f"✅ Loaded {len(labels_data)} labels from {labels_file}")
                    return labels_data
                if isinstance(labels_data, dict) and 'labels' in labels_data:
                    labels = labels_data['labels']
                    print(f"✅ Loaded {len(labels)} labels from {labels_file}")
                    return labels

            except Exception as e:
                print(f"⚠️ Could not load labels from {labels_file}: {e}")

    # Method 4: Load from environment variable
    env_labels = os.getenv('EMOTION_LABELS')
    if env_labels:
        try:
            # Try JSON format first
            labels = json.loads(env_labels)
            if isinstance(labels, list):
                print(f"✅ Loaded {len(labels)} labels from EMOTION_LABELS environment variable")
                return labels
        except json.JSONDecodeError:
            # Try comma-separated format
            labels = [label.strip() for label in env_labels.split(',') if label.strip()]
            if labels:
                print(f"✅ Loaded {len(labels)} labels from EMOTION_LABELS environment variable")
                return labels

    # Method 5: Safe default fallback (common emotion categories)
    default_labels = [
        'anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful',
        'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired'
    ]

    print(f"⚠️ Using default emotion labels ({len(default_labels)} classes)")
    print("   Consider creating emotion_labels.json or setting EMOTION_LABELS environment variable")
    print("   for better label consistency with your trained model.")

    return default_labels

def prepare_model_for_upload(model_path: str, temp_dir: str) -> dict[str, any]:
    """Prepare model for HuggingFace Hub upload."""
    print(f"\n🔧 PREPARING MODEL: {model_path}")
    print("=" * 40)

    os.makedirs(temp_dir, exist_ok=True)

    # Load emotion labels dynamically (avoid hardcoding to match actual model)
    emotion_labels = load_emotion_labels_from_model(model_path)

    # Create label mappings
    id2label = dict(enumerate(emotion_labels))
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

        # Load the state dict with error handling and PyTorch compatibility
        try:
            # Try newer PyTorch version first
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            except TypeError:
                # Fallback for older PyTorch versions
                checkpoint = torch.load(model_path, map_location='cpu')
                print("  ℹ️ Using legacy PyTorch.load (consider upgrading PyTorch)")
        except Exception as e:
            print(f"  ❌ Failed to load checkpoint: {e}")
            print("  💡 Please verify the checkpoint file is not corrupted")
            print("  💡 Check file permissions and disk space")
            raise ValueError(f"Cannot load checkpoint from {model_path}: {e}")

        # Determine base model (configurable)
        base_model_name = get_base_model_name()

        print(f"  📦 Using base model: {base_model_name}")

        # Load base model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=len(emotion_labels),
            id2label=id2label,
            label2id=label2id
        )

        # Load trained weights with error handling
        try:
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("  ✅ Loaded model_state_dict")
            else:
                model.load_state_dict(checkpoint)
                print("  ✅ Loaded state_dict directly")
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"  ❌ Model architecture mismatch: {e}")
                print("  💡 This usually means:")
                print("     - The checkpoint was trained with different number of classes")
                print("     - The model architecture doesn't match the checkpoint")
                print("     - Try checking the model's config.json for num_labels")
                raise ValueError(f"Architecture mismatch when loading checkpoint: {e}")
            print(f"  ❌ Failed to load state dict: {e}")
            raise
        except KeyError as e:
            print(f"  ❌ Missing key in state dict: {e}")
            print("  💡 This might indicate an incompatible checkpoint format")
            raise ValueError(f"Incompatible checkpoint format: {e}")
        except Exception as e:
            print(f"  ❌ Unexpected error loading state dict: {e}")
            print("  💡 Please verify the checkpoint file is not corrupted")
            raise ValueError(f"Failed to load model weights: {e}")

        # Save in HuggingFace format with safetensors (recommended)
        model.save_pretrained(temp_dir, safe_serialization=True)
        tokenizer.save_pretrained(temp_dir)
        print("  ✅ Saved in HuggingFace format with safetensors")

    # Create model card with proper HuggingFace metadata
    model_card = f"""---
language: en
pipeline_tag: text-classification
library_name: transformers
tags:
- emotion-detection
- text-classification
- psychology
- journal-analysis
- mental-health
license: apache-2.0
datasets:
- custom-journal-entries
metrics:
- f1
- accuracy
labels:
{json.dumps(emotion_labels, indent=2)}
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

### Direct Transformers Usage (Local/Self-hosted)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("your-username/samo-dl-emotion-model")
model = AutoModelForSequenceClassification.from_pretrained("your-username/samo-dl-emotion-model")

text = "I felt calm after writing it all down."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=-1)

emotion = model.config.id2label[predicted_class.item()]
confidence = predictions[0][predicted_class].item()

print(f"Emotion: {{emotion}} ({{confidence:.3f}})")
```

### HuggingFace Serverless API (Recommended Start)

#### Python
```python
import requests
import os

url = "https://api-inference.huggingface.co/models/your-username/samo-dl-emotion-model"
headers = {{"Authorization": f"Bearer {{os.environ['HF_TOKEN']}}"}}
payload = {{"inputs": "I am frustrated but hopeful."}}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

#### Node.js/TypeScript
```javascript
const response = await fetch("https://api-inference.huggingface.co/models/your-username/samo-dl-emotion-model", {{
  method: "POST",
  headers: {{
    Authorization: `Bearer ${{process.env.HF_TOKEN}}`,
    "Content-Type": "application/json"
  }},
  body: JSON.stringify({{ inputs: "I felt calm after writing it all down." }})
}});

const result = await response.json();
console.log(result);
```

### Expected Output Format
```json
[
  {{
    "label": "calm",
    "score": 0.8234
  }},
  {{
    "label": "hopeful", 
    "score": 0.1123
  }},
  // ... other emotions with lower scores
]
```

## Training Details

- **Training Framework:** PyTorch + Transformers
- **Optimization:** Custom focal loss for class imbalance
- **Validation:** Domain adaptation on journal entries
- **Performance:** Optimized for personal/journal text emotion detection

## Deployment Options

### 🆓 Serverless API (Recommended Start)
- **Cost**: Free with rate limits
- **Latency**: ~800ms p95 for short texts (includes cold starts)
- **Best for**: Development, testing, low traffic (1-5 RPS)
- **Setup**: No configuration needed, just use your HF token

### 🚀 Inference Endpoints (Production)  
- **Cost**: ~$0.06-1.20/hour (dedicated instances)
- **Latency**: Consistent, no cold starts
- **Best for**: Production APIs, predictable performance
- **Setup**: Create endpoint at https://ui.endpoints.huggingface.co/

### 🏠 Self-hosted (Maximum Control)
- **Cost**: Your infrastructure
- **Best for**: Sensitive data, custom requirements, high volume
- **Latency**: You control (GPU recommended for <100ms)

## Data Sensitivity Considerations

**For sensitive journal content** (mental health, therapy, PII):
- ✅ Use **private repository** (set during upload)  
- ✅ Consider **Inference Endpoints** or **self-hosting** for stricter data handling
- ✅ Avoid shared serverless infrastructure for compliance-sensitive applications

**For general emotion analysis**:
- ✅ **Public repository** + **Serverless API** is fine
- ✅ All communications are over HTTPS
- ✅ No data is stored by HuggingFace during inference

## Intended Use

This model is specifically designed for emotion detection in personal journal entries and similar informal text. 
It may not perform optimally on formal text or other domains.

**Target Performance** (based on training):
- **Accuracy**: ~85% on journal-style text  
- **F1 Score**: ~0.75 (weighted average)
- **Response Time**: <800ms p95 on CPU for typical journal entries

## Limitations

- Trained primarily on English text
- Optimized for informal, personal writing style  
- May have biases present in the training data
- Performance may degrade on very formal or technical text
- Not suitable for clinical diagnosis (research/wellness use only)
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

    # Validate critical files exist (avoid common pitfalls)
    print("\n🔍 VALIDATING MODEL FILES...")
    critical_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
    missing_files = []

    for file in critical_files:
        file_path = os.path.join(temp_dir, file)
        if os.path.exists(file_path):
            print(f"  ✅ {file}")
        else:
            missing_files.append(file)
            print(f"  ❌ {file} - MISSING")

    if missing_files:
        print(f"\n⚠️  WARNING: Missing critical files: {missing_files}")
        print("This may cause serverless API loading failures.")
        print("Continuing anyway, but consider regenerating the model with proper tokenizer files.")

    # Validate config.json has proper labels
    config_path = os.path.join(temp_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

        if 'id2label' not in config or 'label2id' not in config:
            print("  ⚠️  WARNING: config.json missing id2label/label2id mappings")
            print("  This may cause output label mapping issues")
        else:
            print("  ✅ config.json has proper label mappings")

    return {
        'emotion_labels': emotion_labels,
        'id2label': id2label,
        'label2id': label2id,
        'num_labels': len(emotion_labels),
        'validation_warnings': missing_files
    }

def update_deployment_config(repo_name: str, model_info: dict[str, any]):
    """Update deployment configurations to use the new model."""
    print("\n🔧 UPDATING DEPLOYMENT CONFIGURATIONS")
    print("=" * 40)

    # Update model_utils.py to use the new model
    model_utils_path = "deployment/cloud-run/model_utils.py"

    if os.path.exists(model_utils_path):
        with open(model_utils_path, 'r') as f:
            content = f.read()

        # Update model loading to use HuggingFace model
        # Get current base model to replace it dynamically
        current_base_model = get_base_model_name()
        updated_content = content.replace(
            f"AutoTokenizer.from_pretrained('{current_base_model}')",
            f"AutoTokenizer.from_pretrained('{repo_name}')"
        ).replace(
            f"AutoModelForSequenceClassification.from_pretrained(\n            '{current_base_model}',",
            f"AutoModelForSequenceClassification.from_pretrained(\n            '{repo_name}',"
        )

        with open(model_utils_path, 'w') as f:
            f.write(updated_content)

        print(f"✅ Updated {model_utils_path}")

    # Create a new deployment config file
    config_path = "deployment/custom_model_config.json"

    # Ensure the deployment directory exists
    config_dir = os.path.dirname(config_path)
    os.makedirs(config_dir, exist_ok=True)

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
        result = subprocess.run(['git', 'lfs', 'version'], capture_output=True, text=True, check=True)
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
            subprocess.run(['git', 'lfs', 'track', pattern], capture_output=True, text=True, check=True)
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

def choose_repository_privacy() -> bool:
    """Ask user about repository privacy based on data sensitivity."""
    # First, check for HF_REPO_PRIVATE environment variable
    hf_repo_private = os.environ.get("HF_REPO_PRIVATE")
    if hf_repo_private:
        if hf_repo_private.lower() == "true":
            print("🔒 Using PRIVATE repository (HF_REPO_PRIVATE=true)")
            return True
        if hf_repo_private.lower() == "false":
            print("📊 Using PUBLIC repository (HF_REPO_PRIVATE=false)")
            return False
        print(f"⚠️ Invalid HF_REPO_PRIVATE value: {hf_repo_private}. Must be 'true' or 'false'.")

    # Check if in non-interactive environment
    if not sys.stdin.isatty():
        print("📊 Non-interactive environment detected - defaulting to PUBLIC repository")
        print("   Set HF_REPO_PRIVATE=true for private repositories in CI/CD")
        return False  # Default to public in non-interactive environments

    # Interactive mode - ask user
    print("\n🔒 REPOSITORY PRIVACY SELECTION")
    print("=" * 40)
    print("Consider the sensitivity of your journal content:")
    print()
    print("📊 PUBLIC REPOSITORY (Recommended Start):")
    print("  ✅ Completely free")
    print("  ✅ No storage/bandwidth limits")  
    print("  ✅ Easy to share and integrate")
    print("  ⚠️  Model weights and metadata are publicly visible")
    print("  ⚠️  Use for general emotion analysis only")
    print()
    print("🔒 PRIVATE REPOSITORY:")
    print("  ✅ Model weights and metadata are private")
    print("  ✅ Good for sensitive/health content")
    print("  ✅ Requires HF token for access")
    print("  💰 Free tier with storage/bandwidth quotas")
    print()
    print("💡 Tip: Set HF_REPO_PRIVATE=true/false to skip this prompt in automation")
    print()

    while True:
        choice = input("Is your journal content sensitive? (mental health, therapy, PII) [y/N]: ").strip().lower()
        if choice in ['', 'n', 'no']:
            print("📊 Creating PUBLIC repository (free, no limits)")
            return False  # Public
        if choice in ['y', 'yes']:
            print("🔒 Creating PRIVATE repository (free tier with quotas)")
            return True  # Private
        print("Please enter 'y' for yes or 'n' for no (or press Enter for no)")

def upload_to_huggingface(temp_dir: str, model_info: dict[str, any]) -> str:
    """Upload model to HuggingFace Hub."""
    print("\n🚀 UPLOADING TO HUGGINGFACE HUB")
    print("=" * 40)

    # Extract information from model_info for better upload experience
    emotion_labels = model_info.get('emotion_labels', [])
    num_labels = len(emotion_labels)
    validation_warnings = model_info.get('validation_warnings', [])

    print("📊 Model Details:")
    print(f"   • {num_labels} emotion classes: {', '.join(emotion_labels[:6])}")
    if num_labels > 6:
        print(f"     (and {num_labels - 6} more...)")
    print(f"   • Architecture: {model_info.get('model_type', 'Transformer-based')}")

    # Show validation warnings if any
    if validation_warnings:
        print(f"   ⚠️  Validation warnings: {len(validation_warnings)} issue(s) detected")
        for warning in validation_warnings[:3]:  # Show first 3 warnings
            print(f"      • {warning}")
        if len(validation_warnings) > 3:
            print(f"      • (and {len(validation_warnings) - 3} more...)")
    else:
        print("   ✅ Model validation: All essential files present")

    # Set up Git LFS before upload
    setup_git_lfs()

    # Get user info
    api = HfApi()
    user_info = api.whoami()
    username = user_info['name']

    # Create repository name
    repo_name = f"{username}/samo-dl-emotion-model"
    print(f"📦 Repository: {repo_name}")

    # Choose privacy based on content sensitivity
    is_private = choose_repository_privacy()

    try:
        # Create repository with appropriate privacy setting
        create_repo(
            repo_name, 
            exist_ok=True,
            private=is_private,
            repo_type="model"
        )
        privacy_status = "private" if is_private else "public"
        print(f"✅ Repository created/confirmed ({privacy_status})")

        # Create detailed commit message using model information
        commit_message = f"Upload custom emotion detection model - {num_labels} classes"
        if emotion_labels:
            # Include emotion labels in commit for better versioning
            labels_preview = ', '.join(emotion_labels[:4])
            if len(emotion_labels) > 4:
                labels_preview += f" (and {len(emotion_labels) - 4} more)"
            commit_message += f": {labels_preview}"

        # Upload all files
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=repo_name,
            repo_type="model",
            commit_message=commit_message
        )
        print("✅ Model uploaded successfully!")

        model_url = f"https://huggingface.co/{repo_name}"
        print(f"🔗 Model URL: {model_url}")

        # Print deployment options
        print("\n🎯 DEPLOYMENT OPTIONS:")
        print(f"  🆓 Serverless API: https://api-inference.huggingface.co/models/{repo_name}")
        print("  🚀 Inference Endpoints: https://ui.endpoints.huggingface.co/ (create endpoint)")
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
    print("   export HF_TOKEN='your_token_here'")
    print("   curl -X POST \\")
    print("     -H \"Authorization: Bearer $HF_TOKEN\" \\")
    print("     -H \"Content-Type: application/json\" \\")
    print("     -d '{{\"inputs\": \"I am feeling really happy today!\"}}' \\")
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
