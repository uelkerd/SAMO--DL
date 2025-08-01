    # Load checkpoint
#!/usr/bin/env python3
from pathlib import Path
import logging
import torch




"""
Debug Checkpoint Format
"""

def debug_checkpoint():
    checkpoint_path = Path("test_checkpoints/best_model.pt")

    if not checkpoint_path.exists():
        logging.info("❌ Checkpoint not found")
        return

    logging.info("🔍 Debugging checkpoint format...")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    logging.info("Checkpoint type: {type(checkpoint)}")
    logging.info("Checkpoint content: {checkpoint}")

    if isinstance(checkpoint, dict):
        logging.info("\n📋 Dictionary keys:")
        for _key in checkpoint:
            logging.info("  - {key}: {type(checkpoint[key])}")
    elif isinstance(checkpoint, tuple):
        logging.info("\n📋 Tuple length: {len(checkpoint)}")
        for __i, item in enumerate(checkpoint):
            logging.info("  - Item {i}: {type(item)}")
            if isinstance(item, dict):
                logging.info("    Keys: {list(item.keys())}")


if __name__ == "__main__":
    debug_checkpoint()
