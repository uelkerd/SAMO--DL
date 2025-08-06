#!/usr/bin/env python3
"""
Pre-warm models for CI pipeline to avoid download delays during testing.
This script downloads and caches commonly used models for faster CI execution.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def pre_warm_models():
    """Pre-download and cache models for faster CI execution."""
    print("Pre-warming models for CI pipeline...")
    
    try:
        from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
        import torch
        
        # Pre-download BERT models
        print("Downloading BERT base...")
        AutoTokenizer.from_pretrained('bert-base-uncased')
        AutoModel.from_pretrained('bert-base-uncased')
        
        # Pre-download T5 models
        print("Downloading T5 small...")
        AutoTokenizer.from_pretrained('t5-small')
        AutoModelForSeq2SeqLM.from_pretrained('t5-small')
        
        print("Models pre-warmed successfully!")
        return True
        
    except Exception as e:
        print(f"Error pre-warming models: {e}")
        return False

if __name__ == "__main__":
    success = pre_warm_models()
    sys.exit(0 if success else 1) 