#!/usr/bin/env python3
"""
Ensure local copy of the Hugging Face emotion model exists.

- Defaults:
  - repo_id: j-hartmann/emotion-english-distilroberta-base
  - target_dir: /models/emotion-english-distilroberta-base
  - token: from HF_TOKEN (optional)

- Behavior:
  - If required files are present, exit successfully (unless --force).
  - Otherwise, download using huggingface_hub.snapshot_download if available;
    fallback to transformers save_pretrained.
  - Logs to .logs/model_download.log
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List

# Add src to path to import constants
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from constants import EMOTION_MODEL_DIR

DEFAULT_REPO_ID = "j-hartmann/emotion-english-distilroberta-base"
DEFAULT_TARGET_DIR = EMOTION_MODEL_DIR
LOG_DIR = Path(".logs")
LOG_FILE = LOG_DIR / "model_download.log"

# Configure logging
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(LOG_FILE)),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ensure_local_emotion_model")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments and environment overrides."""
    parser = argparse.ArgumentParser(
        description="Ensure local HF emotion model is available"
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("EMOTION_MODEL_REPO", DEFAULT_REPO_ID),
        help="HF repo id",
    )
    parser.add_argument(
        "--target-dir",
        default=os.environ.get("EMOTION_MODEL_DIR", DEFAULT_TARGET_DIR),
        help="Local model directory",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", ""),
        help="HF access token (optional)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    return parser.parse_args()


def required_files_present(target_dir: Path) -> bool:
    """Return True if minimal set of model files exists in target_dir."""
    # Accept either tokenizer.json or (vocab.json + merges.txt)
    files_any = [
        ["tokenizer.json"],
        ["vocab.json", "merges.txt"],
    ]
    required = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
    ]
    have_all_required = all((target_dir / f).exists() for f in required)
    have_any_tokenizer = any(
        all((target_dir / f).exists() for f in group) for group in files_any
    )
    return have_all_required and have_any_tokenizer


def ensure_with_hf_hub(
    repo_id: str, target_dir: Path, token: str | None
) -> bool:
    """Download model snapshot via huggingface_hub if available."""
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as e:
        logger.warning("huggingface_hub not available: %s", e)
        return False

    logger.info(
        "Using huggingface_hub.snapshot_download for repo %s", repo_id
    )
    try:
        snapshot_download(
            repo_id=repo_id,
            token=token or None,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            allow_patterns=None,
            ignore_patterns=None,
        )
        return True
    except Exception as e:
        logger.error("snapshot_download failed: %s", e)
        return False


def ensure_with_transformers(
    repo_id: str, target_dir: Path, token: str | None
) -> bool:
    """Download and save model/tokenizer via transformers.*_pretrained APIs."""
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )  # type: ignore
    except Exception as e:
        logger.error("transformers not available: %s", e)
        return False

    logger.info("Using transformers save_pretrained for repo %s", repo_id)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id, token=token or None
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            repo_id, token=token or None
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(target_dir))
        model.save_pretrained(str(target_dir))
        return True
    except Exception as e:
        logger.error("transformers download/save failed: %s", e)
        return False


def main() -> int:
    """Entry point: ensure local model is materialized and valid."""
    args = parse_args()
    repo_id = args.repo_id
    target_dir = Path(args.target_dir)
    token = args.hf_token or None

    logger.info("Requested repo: %s", repo_id)
    logger.info("Target directory: %s", str(target_dir))

    if not args.force and required_files_present(target_dir):
        logger.info("Model already present at %s", str(target_dir))
        return 0

    logger.info("Ensuring model files... force=%s", args.force)

    ok = ensure_with_hf_hub(repo_id, target_dir, token)
    if not ok:
        logger.info(
            "Falling back to transformers save_pretrained approach"
        )
        ok = ensure_with_transformers(repo_id, target_dir, token)

    if not ok:
        logger.error(
            "Failed to materialize model to %s", str(target_dir)
        )
        return 2

    if required_files_present(target_dir):
        logger.info("✅ Model ready at %s", str(target_dir))
        return 0

    logger.error("Model files still incomplete at %s", str(target_dir))
    return 3


if __name__ == "__main__":
    sys.exit(main())
