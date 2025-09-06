"""
Centralized constants for the SAMO project.

This module contains all shared constants to avoid duplication across modules.
"""

import os

# Emotion model configuration
DEFAULT_EMOTION_MODEL_DIR = '/models/emotion-english-distilroberta-base'
EMOTION_MODEL_DIR = os.getenv('EMOTION_MODEL_DIR', DEFAULT_EMOTION_MODEL_DIR)
