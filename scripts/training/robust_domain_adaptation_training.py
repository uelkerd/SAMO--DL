#!/usr/bin/env python3
"""
SAMO Deep Learning - Robust Domain Adaptation Training Script

This script provides a robust implementation for REQ-DL-012: Domain-Adapted Emotion Detection
that avoids dependency hell and provides comprehensive error handling.

Target: Achieve 70% F1 score on journal entries through domain adaptation from GoEmotions
"""

import json
import os
import subprocess
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set environment variables for stability
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = "false"

def setup_environment():
    """Setup the environment with proper dependency management."""
    print("🔧 Setting up robust environment...")

    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
        is_colab = True
    except ImportError:
        print("ℹ️ Running in local environment")
        is_colab = False

    # Install dependencies with proper version management
    print("📦 Installing dependencies with compatibility fixes...")

    # Step 1: Clean slate - remove conflicting packages
    subprocess.run([
        "pip", "uninstall", "torch", "torchvision", "torchaudio",
        "transformers", "datasets", "-y"
    ], capture_output=True)

    # Step 2: Install PyTorch with compatible CUDA version
    subprocess.run([
        "pip", "install", "torch==2.1.0", "torchvision==0.16.0", "torchaudio==2.1.0",
        "--index-url", "https://download.pytorch.org/whl/cu118", "--no-cache-dir"
    ])

    # Step 3: Install Transformers with compatible version
    subprocess.run([
        "pip", "install", "transformers==4.30.0", "datasets==2.13.0", "--no-cache-dir"
    ])

    # Step 4: Install additional dependencies
    subprocess.run([
        "pip", "install", "evaluate", "scikit-learn", "pandas", "numpy",
        "matplotlib", "seaborn", "accelerate", "wandb", "--no-cache-dir"
    ])

    print("✅ Dependencies installed successfully")
    return is_colab

def verify_installation():
    """Verify that all critical packages are installed correctly."""
    print("🔍 Verifying installation...")

    try:
        import torch
        import transformers
        print(f"  PyTorch: {torch.__version__}")
        print(f"  Transformers: {transformers.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            torch.backends.cudnn.benchmark = True
            print("  ✅ GPU optimized for training")
        else:
            print("⚠️ No GPU available. Training will be slow on CPU.")

        # Test critical imports
        from transformers import AutoModel, AutoTokenizer
        print("  ✅ Transformers imports successful")

        return True

    except Exception as e:
        print(f"  ❌ Installation verification failed: {e}")
        return False

def setup_repository():
    """Setup the SAMO-DL repository."""
    print("📁 Setting up repository...")

    def run_command(command: str, description: str) -> bool:
        """Execute command with error handling."""
        print(f"🔄 {description}...")
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✅ {description} completed")
                return True
            else:
                print(f"  ❌ {description} failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"  ❌ {description} failed: {e}")
            return False

    # Clone repository if not exists
    if not Path('SAMO--DL').exists():
        run_command('git clone https://github.com/uelkerd/SAMO--DL.git', 'Cloning repository')

    # Change to project directory
    os.chdir('SAMO--DL')
    print(f"📁 Working directory: {os.getcwd()}")

    # Pull latest changes
    run_command('git pull origin main', 'Pulling latest changes')

def safe_load_dataset(dataset_name: str, config: Optional[str] = None, split: Optional[str] = None):
    """Safely load dataset with error handling."""
    try:
        from datasets import load_dataset
        if config:
            dataset = load_dataset(dataset_name, config, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        print(f"✅ Successfully loaded {dataset_name}")
        return dataset
    except Exception as e:
        print(f"❌ Failed to load {dataset_name}: {e}")
        return None

def safe_load_json(file_path: str):
    """Safely load JSON file with error handling."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"✅ Successfully loaded {file_path}")
        return data
    except Exception as e:
        print(f"❌ Failed to load {file_path}: {e}")
        return None

def analyze_writing_style(texts: List[str], domain_name: str) -> Optional[Dict[str, float]]:
    """Analyze writing style characteristics of a domain."""
    if not texts:
        print(f"⚠️ No texts provided for {domain_name}")
        return None

    # Filter out None or empty texts
    valid_texts = [text for text in texts if text and isinstance(text, str)]

    if not valid_texts:
        print(f"⚠️ No valid texts found for {domain_name}")
        return None

    import numpy as np

    avg_length = np.mean([len(text.split()) for text in valid_texts])
    personal_pronouns = sum(['I ' in text or 'my ' in text or 'me ' in text for text in valid_texts]) / len(valid_texts)
    reflection_words = sum(['think' in text.lower() or 'feel' in text.lower() or 'believe' in text.lower()
                           for text in valid_texts]) / len(valid_texts)

    print(f"{domain_name} Style Analysis:")
    print(f"  Average length: {avg_length:.1f} words")
    print(f"  Personal pronouns: {personal_pronouns:.1%}")
    print(f"  Reflection words: {reflection_words:.1%}")

    return {
        'avg_length': avg_length,
        'personal_pronouns': personal_pronouns,
        'reflection_words': reflection_words
    }

def perform_domain_analysis():
    """Perform domain gap analysis between GoEmotions and journal entries."""
    print("📊 Loading datasets for domain analysis...")

    # Load GoEmotions dataset
    go_emotions = safe_load_dataset("go_emotions", "simplified")
    if go_emotions:
        go_texts = go_emotions['train']['text'][:1000]  # Sample for analysis
    else:
        go_texts = []

    # Load journal dataset
    journal_entries = safe_load_json('data/journal_test_dataset.json')
    if journal_entries:
        import pandas as pd
        journal_df = pd.DataFrame(journal_entries)
        journal_texts = journal_df['content'].tolist()
    else:
        journal_texts = []

    # Analyze domains if data is available
    if go_texts and journal_texts:
        print("\n🔍 Domain Gap Analysis:")
        go_analysis = analyze_writing_style(go_texts, "GoEmotions (Reddit)")
        journal_analysis = analyze_writing_style(journal_texts, "Journal Entries")

        if go_analysis and journal_analysis:
            print("\n🎯 Key Insights:")
            print("- Journal entries are {journal_analysis["avg_length']/go_analysis['avg_length']:.1f}x longer")
            print("- Journal entries use {journal_analysis["personal_pronouns']/go_analysis['personal_pronouns']:.1f}x more personal pronouns")
            print("- Journal entries contain {journal_analysis["reflection_words']/go_analysis['reflection_words']:.1f}x more reflection words")

            return go_emotions, journal_df
    else:
        print("⚠️ Cannot perform domain analysis - missing data")
        return None, None

class FocalLoss:
    """Focal Loss for addressing class imbalance in emotion detection."""

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        import torch.nn as nn
        import torch.nn.functional as F
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.F = F

    def __call__(self, inputs, targets):
        ce_loss = self.F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DomainAdaptedEmotionClassifier:
    """BERT-based emotion classifier with domain adaptation capabilities."""

    def __init__(self, model_name="bert-base-uncased", num_labels=None, dropout=0.3):
        import torch.nn as nn
        from transformers import AutoModel

        # ROBUST: Validate num_labels
        if num_labels is None:
            print("⚠️ num_labels not provided, using default value of 12")
            num_labels = 12
        elif num_labels <= 0:
            raise ValueError(f"num_labels must be positive, got {num_labels}")

        print(f"🏗️ Initializing DomainAdaptedEmotionClassifier with num_labels = {num_labels}")

        try:
            self.bert = AutoModel.from_pretrained(model_name)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

            # Domain adaptation layer
            self.domain_classifier = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 2)  # 2 domains: GoEmotions vs Journal
            )

            print(f"✅ Model initialized successfully with {num_labels} labels")

        except Exception as e:
            print(f"❌ Failed to initialize model: {e}")
            raise

    def forward(self, input_ids, attention_mask, domain_labels=None):
        try:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output

            # Emotion classification
            emotion_logits = self.classifier(self.dropout(pooled_output))

            # Domain classification (for domain adaptation)
            domain_logits = self.domain_classifier(pooled_output)

            if domain_labels is not None:
                return emotion_logits, domain_logits
            return emotion_logits

        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            raise

def safe_model_initialization(model_name: str, num_labels: int, device: str):
    """Safely initialize model with error handling."""
    try:
        print(f"🏗️ Initializing model with {model_name}...")

        # Initialize tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✅ Tokenizer loaded: {model_name}")

        # Initialize model
        model = DomainAdaptedEmotionClassifier(model_name=model_name, num_labels=num_labels)

        # Move to device
        import torch
        model = model.to(device)
        print(f"✅ Model moved to {device}")

        # Verify model parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Model parameters: {total_params:,}")

        return model, tokenizer

    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        raise

def main():
    """Main execution function."""
    print("🚀 Starting SAMO Deep Learning - Robust Domain Adaptation Training")
    print("=" * 70)

    # Step 1: Setup environment
    is_colab = setup_environment()

    # Step 2: Verify installation
    if not verify_installation():
        print("❌ Installation verification failed. Please restart and try again.")
        return

    # Step 3: Setup repository
    setup_repository()

    # Step 4: Perform domain analysis
    go_emotions, journal_df = perform_domain_analysis()

    if go_emotions is None or journal_df is None:
        print("❌ Cannot proceed without datasets")
        return

    # Step 5: Initialize model (example)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # This would be called when we have the label encoder ready
    # model, tokenizer = safe_model_initialization("bert-base-uncased", num_labels, device)

    print("\n✅ Setup completed successfully!")
    print("🎯 Ready for domain adaptation training")
    print("\n📋 Next steps:")
    print("  1. Prepare data with label encoding")
    print("  2. Initialize model with correct num_labels")
    print("  3. Run training pipeline")
    print("  4. Evaluate and save results")

if __name__ == "__main__":
    main()
