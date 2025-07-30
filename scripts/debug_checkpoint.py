#!/usr/bin/env python3
"""
Debug Checkpoint Format
"""

import torch
from pathlib import Path


def debug_checkpoint():
    checkpoint_path = Path("test_checkpoints/best_model.pt")

    if not checkpoint_path.exists():
        print("❌ Checkpoint not found")
        return

    print("🔍 Debugging checkpoint format...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    print("Checkpoint type: {type(checkpoint)}")
    print("Checkpoint content: {checkpoint}")

    if isinstance(checkpoint, dict):
        print("\n📋 Dictionary keys:")
        for key in checkpoint:
            print("  - {key}: {type(checkpoint[key])}")
    elif isinstance(checkpoint, tuple):
        print("\n📋 Tuple length: {len(checkpoint)}")
        for i, item in enumerate(checkpoint):
            print("  - Item {i}: {type(item)}")
            if isinstance(item, dict):
                print("    Keys: {list(item.keys())}")


if __name__ == "__main__":
    debug_checkpoint()
