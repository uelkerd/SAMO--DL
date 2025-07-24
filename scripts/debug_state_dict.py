#!/usr/bin/env python3
"""
Debug Model State Dict Structure
"""

import torch
from pathlib import Path

def debug_state_dict():
    checkpoint_path = Path("test_checkpoints/best_model.pt")
    
    if not checkpoint_path.exists():
        print("❌ Checkpoint not found")
        return
    
    print("🔍 Debugging model_state_dict structure...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"Checkpoint type: {type(checkpoint)}")
    print(f"model_state_dict type: {type(checkpoint['model_state_dict'])}")
    
    state_dict = checkpoint['model_state_dict']
    
    if isinstance(state_dict, dict):
        print("✅ State dict is a dictionary")
        print(f"Number of keys: {len(state_dict.keys())}")
        print("First few keys:")
        for i, key in enumerate(list(state_dict.keys())[:5]):
            print(f"  {key}: {type(state_dict[key])}")
    elif isinstance(state_dict, tuple):
        print("❌ State dict is a tuple")
        print(f"Tuple length: {len(state_dict)}")
        print("Tuple contents:")
        for i, item in enumerate(state_dict):
            print(f"  [{i}]: {type(item)} - {item}")
    else:
        print(f"❌ Unexpected type: {type(state_dict)}")
        print(f"Content: {state_dict}")

if __name__ == "__main__":
    debug_state_dict() 