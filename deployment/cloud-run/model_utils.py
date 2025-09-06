"""
Shared model utilities for Cloud Run deployment with Hugging Face emotion model.

This module provides common functionality for model loading, inference,
and error handling to eliminate code duplication between API servers.
"""

import logging
import os
import threading
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers.pipelines import TextClassificationPipeline

# Import centralized constants
from ...src.constants import EMOTION_MODEL_DIR

logger = logging.getLogger(__name__)

# Global variables for model management
emotion_pipeline = None
model_loaded = False
model_loading = False
model_lock = threading.Lock()
model_ready_event = threading.Event()

# Configuration
EMOTION_PROVIDER = os.getenv('EMOTION_PROVIDER', 'hf')
EMOTION_LOCAL_ONLY = os.getenv('EMOTION_LOCAL_ONLY', '1').lower() in (
    '1', 'true', 'yes'
)
MAX_TEXT_LENGTH = int(os.getenv('MAX_TEXT_LENGTH', '1000'))

# Emotion labels for the HF emotion model (6 classes)
EMOTION_LABELS = [
    'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'
]

# Runtime emotion labels
emotion_labels_runtime: List[str] = EMOTION_LABELS.copy()


def _create_emotion_pipeline(tokenizer, model) -> TextClassificationPipeline:
    """Create an emotion text-classification pipeline from tokenizer and model.

    Args:
        tokenizer: The tokenizer instance
        model: The model instance

    Returns:
        A configured Hugging Face text-classification pipeline.
    """
    return pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        device=0 if torch.cuda.is_available() else -1
    )


def _validate_and_prepare_texts(
    texts: List[str]
) -> Tuple[List[Optional[Dict[str, Any]]], List[str], List[int]]:
    """Validate input texts and prepare them for batch processing.

    Args:
        texts: List of input texts to validate

    Returns:
        Tuple of (results_list, valid_texts, valid_indices)
    """
    results = [None] * len(texts)
    valid_texts = []
    valid_indices = []

    for i, text in enumerate(texts):
        if not isinstance(text, str):
            results[i] = {
                'error': 'Text must be a non-empty string',
                'emotions': [],
                'confidence': 0.0
            }
        elif not text.strip():
            results[i] = {
                'error': 'Text must be a non-empty string',
                'emotions': [],
                'confidence': 0.0
            }
        elif len(text) > MAX_TEXT_LENGTH:
            results[i] = {
                'error': f'Text too long (max {MAX_TEXT_LENGTH} characters)',
                'emotions': [],
                'confidence': 0.0
            }
        else:
            valid_texts.append(text)
            valid_indices.append(i)

    return results, valid_texts, valid_indices


def ensure_model_loaded() -> bool:
    """Thread-safe emotion model loading with proper error handling.

    Returns:
        bool: True if model is loaded successfully, False otherwise
    """
    global emotion_pipeline, model_loaded, model_loading, emotion_labels_runtime

    with model_lock:
        if model_loaded:
            return True
        if model_loading:
            wait_for_loader = True
        else:
            model_loading = True
            model_ready_event.clear()
            wait_for_loader = False

    if wait_for_loader:
        model_ready_event.wait()
        return model_loaded

    try:
        logger.info("🔄 Loading emotion model from: %s", EMOTION_MODEL_DIR)

        # Check if local model directory exists
        if EMOTION_LOCAL_ONLY and os.path.isdir(EMOTION_MODEL_DIR):
            # Load from local directory
            logger.info("📁 Loading from local model directory: %s",
                        EMOTION_MODEL_DIR)
            tokenizer = AutoTokenizer.from_pretrained(
                EMOTION_MODEL_DIR, local_files_only=True
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                EMOTION_MODEL_DIR, local_files_only=True
            )
            emotion_pipeline = _create_emotion_pipeline(tokenizer, model)
            logger.info("✅ Emotion model loaded from local directory")
        else:
            # Load from Hugging Face Hub (with fallback to download if not cached)
            logger.info("🌐 Loading emotion model from Hugging Face Hub")
            try:
                emotion_pipeline = pipeline(
                    task="text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True,
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("✅ Emotion model loaded from Hugging Face Hub")
            except Exception as download_error:
                logger.warning("Failed to load from cache, downloading model: %s",
                               download_error)
                # Force download the model
                from huggingface_hub import snapshot_download
                model_path = snapshot_download(
                    repo_id="j-hartmann/emotion-english-distilroberta-base",
                    local_dir=EMOTION_MODEL_DIR,
                    local_dir_use_symlinks=False
                )
                logger.info("📥 Model downloaded to: %s", model_path)

                # Load from downloaded directory
                tokenizer = AutoTokenizer.from_pretrained(
                    EMOTION_MODEL_DIR, local_files_only=True
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    EMOTION_MODEL_DIR, local_files_only=True
                )
                emotion_pipeline = _create_emotion_pipeline(tokenizer, model)
                logger.info("✅ Emotion model loaded from downloaded files")

        # Update runtime labels from loaded model if available
        try:
            id2label = emotion_pipeline.model.config.id2label
            emotion_labels_runtime = [
                id2label[i]
                for i in range(len(id2label))
            ]
        except Exception as label_err:
            logger.debug("Unable to derive runtime labels from model config: %s", label_err)
        with model_lock:
            model_loaded = True
            model_loading = False
            model_ready_event.set()
        logger.info("🎉 Emotion model loading completed successfully")
        return True

    except Exception as e:
        logger.exception("❌ Failed to load emotion model: %s", e)
        with model_lock:
            model_loaded = False
            model_loading = False
            model_ready_event.set()
        return False


def predict_emotions(text: str) -> Dict[str, Any]:
    """
    Predict emotions for given text using the emotion model.

    Args:
        text (str): Input text to analyze

    Returns:
        Dict[str, Any]: Prediction results with emotions and confidence scores
    """
    # Validate input first
    ok, err = validate_text_input(text)
    if not ok:
        return {'error': err, 'emotions': [], 'confidence': 0.0}

    if not ensure_model_loaded():
        return {
            'error': 'Emotion model not available',
            'emotions': [],
            'confidence': 0.0
        }

    try:

        # Use the emotion pipeline for prediction
        results = emotion_pipeline(text)

        # Format results to match expected output
        emotions = []
        for result in results[0]:  # results is a list with one item for single text
            emotions.append({
                'emotion': result['label'],
                'confidence': result['score']
            })

        # Sort by confidence (highest first)
        emotions.sort(key=lambda x: x['confidence'], reverse=True)

        # Overall confidence is the highest confidence score
        overall_confidence = emotions[0]['confidence'] if emotions else 0.0

        return {
            'text': text,
            'emotions': emotions,
            'confidence': overall_confidence,
            'timestamp': time.time()
        }

    except Exception as e:
        logger.exception("❌ Emotion prediction failed: %s", e)
        return {
            'error': 'Emotion prediction failed',
            'emotions': [],
            'confidence': 0.0
        }


def get_model_status() -> Dict[str, Any]:
    """Get current emotion model status.

    Returns:
        Dict[str, Any]: Model status information
    """
    return {
        'model_loaded': model_loaded,
        'model_loading': model_loading,
        'model_dir': EMOTION_MODEL_DIR,
        'model_provider': EMOTION_PROVIDER,
        'local_only': EMOTION_LOCAL_ONLY,
        'max_text_length': MAX_TEXT_LENGTH,
        'emotion_labels': emotion_labels_runtime,
        'timestamp': time.time()
    }


def predict_emotions_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """Predict emotions for multiple texts using the emotion model.

    Args:
        texts (List[str]): List of input texts to analyze

    Returns:
        List[Dict[str, Any]]: List of prediction results for each text
    """
    if not ensure_model_loaded():
        return [{
            'error': 'Emotion model not available',
            'emotions': [],
            'confidence': 0.0
        } for _ in texts]

    try:
        # Validate and prepare texts for processing
        results, valid_texts_to_process, valid_indices = \
            _validate_and_prepare_texts(texts)

        # Only run pipeline if there are valid texts
        if valid_texts_to_process:
            # Process valid texts in a single batch
            batch_results = emotion_pipeline(valid_texts_to_process)

            # Place successful results back into the correctly ordered list
            for i, result in enumerate(batch_results):
                original_idx = valid_indices[i]
                text = valid_texts_to_process[i]

                # Convert emotion results to list comprehension
                emotions = [
                    {
                        'emotion': emotion_result['label'],
                        'confidence': emotion_result['score']
                    }
                    for emotion_result in result
                ]

                # Sort by confidence (highest first)
                emotions.sort(key=lambda x: x['confidence'], reverse=True)

                # Overall confidence is the highest confidence score
                overall_confidence = emotions[0]['confidence'] if emotions else 0.0

                results[original_idx] = {
                    'text': text,
                    'emotions': emotions,
                    'confidence': overall_confidence,
                    'timestamp': time.time()
                }

        return results

    except Exception as e:
        logger.exception("❌ Batch emotion prediction failed: %s", e)
        return [{
            'error': 'Batch emotion prediction failed',
            'emotions': [],
            'confidence': 0.0
        } for _ in texts]


def validate_text_input(text: str) -> Tuple[bool, str]:
    """
    Validate text input for prediction.

    Args:
        text (str): Text to validate

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not isinstance(text, str) or not text.strip():
        return False, 'Text must be a non-empty string'
    if len(text) > MAX_TEXT_LENGTH:
        return False, f'Text too long (max {MAX_TEXT_LENGTH} characters)'
    return True, ''
