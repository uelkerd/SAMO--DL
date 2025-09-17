#!/usr/bin/env python3
"""Prefetch models for Docker builds to improve startup time."""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def prefetch_emotion_model(cache_dir: str = "/app/models"):
    """Prefetch emotion detection model."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        model_name = 'duelker/samo-goemotions-deberta-v3-large'
        logger.info(f"Downloading emotion model {model_name}...")
        
        AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
        
        logger.info("✓ Emotion model downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to download emotion model: {e}")
        return False

def prefetch_t5_model(cache_dir: str = "/app/models"):
    """Prefetch T5 summarization model."""
    try:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        
        model_name = 't5-small'
        logger.info(f"Downloading T5 model {model_name}...")
        
        T5Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)
        
        logger.info("✓ T5 model downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to download T5 model: {e}")
        return False

def prefetch_whisper_model(cache_dir: str = "/app/models"):
    """Prefetch Whisper transcription model."""
    try:
        import whisper
        
        model_size = 'base'
        logger.info(f"Downloading Whisper model {model_size}...")
        
        whisper.load_model(model_size, download_root=cache_dir)
        
        logger.info("✓ Whisper model downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to download Whisper model: {e}")
        return False

def main():
    """Main prefetch function."""
    cache_dir = sys.argv[1] if len(sys.argv) > 1 else "/app/models"
    
    logger.info("Starting model prefetch...")
    
    # Create cache directory
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    success = True
    success &= prefetch_emotion_model(cache_dir)
    success &= prefetch_t5_model(cache_dir)
    success &= prefetch_whisper_model(cache_dir)
    
    if success:
        logger.info("All models downloaded successfully!")
        sys.exit(0)
    else:
        logger.error("Some models failed to download")
        sys.exit(1)

if __name__ == "__main__":
    main()
