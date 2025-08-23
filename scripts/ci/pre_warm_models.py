#!/usr/bin/env python3
"""Pre-warm models for CI pipeline to avoid download delays during testing.

This script downloads and caches commonly used models for faster CI execution.
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def pre_warm_models():
    """Pre-download and cache models for faster CI execution."""
    print("Pre-warming models for CI pipeline...")

    try:
        import os

        from src.common.env import is_truthy

        # Respect offline mode in CI to avoid failing when network is unavailable.
        offline = is_truthy(os.getenv("HF_HUB_OFFLINE")) or is_truthy(
            os.getenv("TRANSFORMERS_OFFLINE")
        )
        if offline:
            print("Offline mode detected. Skipping pre-warm.")
            return True
        from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

        # Pre-download BERT models
        print("Downloading BERT base...")
        AutoTokenizer.from_pretrained("bert-base-uncased")
        AutoModel.from_pretrained("bert-base-uncased")

        # Pre-download T5 models
        print("Downloading T5 small...")
        AutoTokenizer.from_pretrained("t5-small")
        AutoModelForSeq2SeqLM.from_pretrained("t5-small")

        print("Models pre-warmed successfully!")
        return True

    except Exception as e:
        print(f"Error pre-warming models: {e}")
        return False


if __name__ == "__main__":
    success = pre_warm_models()
    sys.exit(0 if success else 1)
