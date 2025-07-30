import logging
import torch
from pathlib import Path
#!/usr/bin/env python3
    # Load checkpoint




"""
Debug Model State Dict Structure
"""

def debug_state_dict():
    checkpoint_path = Path("test_checkpoints/best_model.pt")

    if not checkpoint_path.exists():
        logging.info("❌ Checkpoint not found")
        return

    logging.info("🔍 Debugging model_state_dict structure...")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    logging.info("Checkpoint type: {type(checkpoint)}")
    logging.info("model_state_dict type: {type(checkpoint['model_state_dict'])}")

    state_dict = checkpoint["model_state_dict"]

    if isinstance(state_dict, dict):
        logging.info("✅ State dict is a dictionary")
        logging.info("Number of keys: {len(state_dict.keys())}")
        logging.info("First few keys:")
        for _i, _key in enumerate(list(state_dict.keys())[:5]):
            logging.info("  {key}: {type(state_dict[key])}")
    elif isinstance(state_dict, tuple):
        logging.info("❌ State dict is a tuple")
        logging.info("Tuple length: {len(state_dict)}")
        logging.info("Tuple contents:")
        for _i, _item in enumerate(state_dict):
            logging.info("  [{i}]: {type(item)} - {item}")
    else:
        logging.info("❌ Unexpected type: {type(state_dict)}")
        logging.info("Content: {state_dict}")


if __name__ == "__main__":
    debug_state_dict()
