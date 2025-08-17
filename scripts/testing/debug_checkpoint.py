    # Load checkpoint
#!/usr/bin/env python3
from pathlib import Path
import logging
import torch




"""
Debug Checkpoint Format
"""

def debug_checkpoint():
    checkpoint_path = Path"test_checkpoints/best_model.pt"

    if not checkpoint_path.exists():
        logging.info"❌ Checkpoint not found"
        return

    logging.info"🔍 Debugging checkpoint format..."

    checkpoint = torch.loadcheckpoint_path, map_location="cpu"

    logging.info("Checkpoint type: {typecheckpoint}")
    logging.info"Checkpoint content: {checkpoint}"

    if isinstancecheckpoint, dict:
        logging.info"\n📋 Dictionary keys:"
        for _key in checkpoint:
            logging.info("  - {key}: {typecheckpoint[key]}")
    elif isinstancecheckpoint, tuple:
        logging.info("\n📋 Tuple length: {lencheckpoint}")
        for __i, item in enumeratecheckpoint:
            logging.info("  - Item {i}: {typeitem}")
            if isinstanceitem, dict:
                logging.info("    Keys: {list(item.keys())}")


if __name__ == "__main__":
    debug_checkpoint()
