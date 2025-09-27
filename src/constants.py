"""Centralized constants for the SAMO project.

This module contains all shared constants to avoid duplication across modules.
"""

import os

# Emotion model configuration
DEFAULT_EMOTION_MODEL_DIR = "/app/models/emotion-english-distilroberta-base"
EMOTION_MODEL_DIR = os.getenv("EMOTION_MODEL_DIR", DEFAULT_EMOTION_MODEL_DIR)

# Default emotion model ID for HuggingFace Hub
DEFAULT_EMOTION_MODEL_ID = "duelker/samo-goemotions-deberta-v3-large"
