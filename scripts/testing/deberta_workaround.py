#!/usr/bin/env python3
"""
DeBERTa Workaround - Manual Safetensors Loading

This script manually downloads and loads the DeBERTa model using safetensors
to bypass the PyTorch vulnerability issue.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
import torch
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_file

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def download_deberta_files():
    """Download DeBERTa model files manually."""
    print("📥 Downloading DeBERTa files manually...")

    model_name = "duelker/samo-goemotions-deberta-v3-large"
    local_dir = Path("/tmp/deberta_manual")

    # Create directory
    local_dir.mkdir(exist_ok=True)

    # Files to download
    files_to_download = [
        "model.safetensors",
        "config.json",
        "tokenizer_config.json",
        "spm.model",
        "special_tokens_map.json",
        "added_tokens.json"
    ]

    downloaded_files = {}

    for filename in files_to_download:
        try:
            print(f"Downloading {filename}...")
            local_path = hf_hub_download(
                repo_id=model_name,
                filename=filename,
                local_dir=local_dir
            )
            downloaded_files[filename] = local_path
            print(f"✅ Downloaded {filename}")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            return None

    return downloaded_files

def load_deberta_manually():
    """Load DeBERTa model manually using safetensors."""
    print("🔧 Loading DeBERTa manually...")

    # Download files
    files = download_deberta_files()
    if not files:
        return None

    try:
        # Load config
        config_path = files["config.json"]
        config = AutoConfig.from_pretrained(config_path)

        # Load tokenizer
        tokenizer_path = str(Path(files["tokenizer_config.json"]).parent)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Load model weights using safetensors (bypasses PyTorch vulnerability)
        model_path = files["model.safetensors"]
        print("Loading safetensors file...")
        state_dict = load_file(model_path)

        # Create model architecture
        from transformers import DebertaForSequenceClassification
        model = DebertaForSequenceClassification(config)

        # Load state dict
        model.load_state_dict(state_dict)
        model.eval()

        print("✅ Model loaded successfully!")
        return model, tokenizer

    except Exception as e:
        print(f"❌ Manual loading failed: {e}")
        return None

def create_pipeline_from_components(model, tokenizer):
    """Create a simple pipeline from model and tokenizer."""
    def predict(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)

        # Get emotion labels from config
        id2label = getattr(model.config, 'id2label', {})
        predicted_emotion = id2label.get(predictions.item(), f"emotion_{predictions.item()}")
        confidence = probabilities[0][predictions.item()].item()

        return {
            'label': predicted_emotion,
            'score': confidence,
            'probabilities': probabilities[0].tolist()
        }

    return predict

def test_deberta_workaround():
    """Test the DeBERTa workaround."""
    print("🧪 Testing DeBERTa Workaround")
    print("=" * 40)

    # Load model manually
    result = load_deberta_manually()
    if not result:
        print("❌ Model loading failed")
        return

    model, tokenizer = result

    # Create prediction function
    predict_fn = create_pipeline_from_components(model, tokenizer)

    # Test predictions
    test_texts = [
        "I am so happy today!",
        "I'm feeling really sad.",
        "I'm frustrated and angry."
    ]

    print("\n🔬 Testing predictions...")
    for text in test_texts:
        result = predict_fn(text)
        print(f"Text: {text}")
        print(".3f")
        print()

    print("✅ DeBERTa workaround successful!")

def main():
    """Main function."""
    print("🔧 DeBERTa Manual Loading Workaround")
    print("=" * 40)
    print("Bypassing PyTorch vulnerability using safetensors")
    print()

    test_deberta_workaround()

if __name__ == "__main__":
    main()
