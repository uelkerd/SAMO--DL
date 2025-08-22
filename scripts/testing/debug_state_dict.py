    # Load checkpoint
#!/usr/bin/env python3
from pathlib import Path
import logging
import torch




"""
Debug Model State Dict Structure
"""

def debug_state_dict():
    checkpoint_path = Path"test_checkpoints/best_model.pt"

    if not checkpoint_path.exists():
        logging.info"❌ Checkpoint not found"
        return

    logging.info"🔍 Debugging model_state_dict structure..."

    checkpoint = torch.loadcheckpoint_path, map_location="cpu", weights_only=False

    logging.info("Checkpoint type: {typecheckpoint}")
    logging.info("model_state_dict type: {typecheckpoint['model_state_dict']}")

    state_dict = checkpoint["model_state_dict"]

    if isinstancestate_dict, dict:
        logging.info"✅ State dict is a dictionary"
        logging.info("Number of keys: {len(state_dict.keys())}")
        logging.info"First few keys:"
        for _i, _key in enumerate(list(state_dict.keys())[:5]):
            logging.info("  {key}: {typestate_dict[key]}")
    elif isinstancestate_dict, tuple:
        logging.info"❌ State dict is a tuple"
        logging.info("Tuple length: {lenstate_dict}")
        logging.info"Tuple contents:"
        for __i, _item in enumeratestate_dict:
            logging.info("  [{i}]: {typeitem} - {item}")
    else:
        logging.info("❌ Unexpected type: {typestate_dict}")
        logging.info"Content: {state_dict}"


if __name__ == "__main__":
    debug_state_dict()
