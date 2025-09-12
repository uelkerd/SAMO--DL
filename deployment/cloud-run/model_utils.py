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

# Import centralized constants with fallback for non-package environments
try:
    from src.constants import EMOTION_MODEL_DIR  # single source of truth
except ImportError:
    EMOTION_MODEL_DIR = os.getenv(
        'EMOTION_MODEL_DIR',
        '/app/models/emotion-english-distilroberta-base'
    )

# Model configuration
USE_DEBERTA = os.getenv('USE_DEBERTA', 'false').lower() in ('true', '1', 'yes')
DEBERTA_MODEL_NAME = os.getenv('DEBERTA_MODEL_NAME', 'duelker/samo-goemotions-deberta-v3-large')
PRODUCTION_MODEL_NAME = os.getenv('PRODUCTION_MODEL_NAME', 'j-hartmann/emotion-english-distilroberta-base')

logger = logging.getLogger(__name__)

# Global variables for model management
emotion_pipeline = None
emotion_tokenizer = None  # For direct DeBERTa loading
emotion_model = None      # For direct DeBERTa loading
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
    # For DeBERTa, create pipeline manually due to tokenizer compatibility issues
    if USE_DEBERTA:
        class CustomPipeline(TextClassificationPipeline):
            def __init__(self, model, tokenizer, **kwargs):
                super().__init__(model=model, tokenizer=tokenizer, **kwargs)

            def __call__(self, inputs, **kwargs):
                # Override to handle DeBERTa tokenizer issues
                if isinstance(inputs, str):
                    inputs = [inputs]

                results = []
                for text in inputs:
                    # Manual tokenization and inference
                    encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
                    with torch.no_grad():
                        outputs = self.model(**encoded)
                        predictions = torch.sigmoid(outputs.logits).squeeze(0)

                    # Convert to expected format
                    emotions = []
                    emotion_labels = [
                        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
                        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
                        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
                        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
                        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
                    ]

                    for i, score in enumerate(predictions):
                        if score > 0.05:  # Only include significant emotions
                            emotions.append({
                                'label': emotion_labels[i] if i < len(emotion_labels) else f'LABEL_{i}',
                                'score': float(score)
                            })

                    emotions.sort(key=lambda x: x['score'], reverse=True)
                    results.append(emotions)

                return results

        return CustomPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
    # Use standard pipeline for production model
    return pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        device=0 if torch.cuda.is_available() else -1
    )


def _predict_emotions_deberta(text: str) -> List[Dict[str, Any]]:
    """Direct emotion prediction for DeBERTa model bypassing pipeline.

    Args:
        text: Input text to analyze

    Returns:
        List of emotion predictions with labels and scores
    """
    if emotion_tokenizer is None or emotion_model is None:
        raise RuntimeError("DeBERTa model not loaded")

    try:
        # Tokenize input
        encoded = emotion_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        )

        # Move to same device as model
        device = next(emotion_model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Get predictions
        with torch.no_grad():
            outputs = emotion_model(**encoded)
            # Apply sigmoid for multi-label classification
            predictions = torch.sigmoid(outputs.logits).squeeze(0)

        # DeBERTa emotion labels (28 emotions)
        emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]

        # Convert to expected format
        emotions = []
        for i, score in enumerate(predictions):
            score_val = float(score)
            if score_val > 0.05:  # Only include significant emotions
                emotions.append({
                    'label': emotion_labels[i] if i < len(emotion_labels) else f'LABEL_{i}',
                    'score': score_val
                })

        # Sort by confidence (highest first)
        emotions.sort(key=lambda x: x['score'], reverse=True)
        return emotions

    except Exception as e:
        logger.exception("DeBERTa prediction failed: %s", e)
        raise


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
    global emotion_pipeline, emotion_tokenizer, emotion_model, model_loaded, model_loading, emotion_labels_runtime

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
                EMOTION_MODEL_DIR, local_files_only=True, torch_dtype="float32"
            )
            emotion_pipeline = _create_emotion_pipeline(tokenizer, model)
            logger.info("✅ Emotion model loaded from local directory")
        else:
            # Choose model based on configuration
            if USE_DEBERTA:
                model_name = DEBERTA_MODEL_NAME
                model_kwargs = {
                    "torch_dtype": "float32",
                    "use_safetensors": True,
                    "ignore_mismatched_sizes": True
                }
                max_length = 256
                model_type = "DeBERTa (28 emotions)"

                # DIRECT LOADING FOR DeBERTa - bypass pipeline entirely
                logger.info(f"🔧 Loading {model_type} with direct approach (bypassing pipeline)")

                try:
                    # Load tokenizer and model directly with slow tokenizer
                    emotion_tokenizer = AutoTokenizer.from_pretrained(
                        model_name, use_fast=False
                    )
                    emotion_model = AutoModelForSequenceClassification.from_pretrained(
                        model_name, **model_kwargs
                    )
                    # Set pipeline to None for DeBERTa (we use direct inference)
                    emotion_pipeline = None
                    logger.info(f"✅ {model_type} loaded successfully with direct approach")

                except Exception as direct_load_error:
                    logger.warning(f"Direct loading failed, trying download approach: {direct_load_error}")

                    # Force download the model
                    from huggingface_hub import snapshot_download
                    download_dir = f"/tmp/{model_name.replace('/', '_')}"
                    model_path = snapshot_download(
                        repo_id=model_name,
                        local_dir=download_dir,
                        local_dir_use_symlinks=False
                    )
                    logger.info("📥 Model downloaded to: %s", model_path)

                    # Load from downloaded directory with direct approach
                    emotion_tokenizer = AutoTokenizer.from_pretrained(
                        download_dir, local_files_only=True, use_fast=False
                    )
                    emotion_model = AutoModelForSequenceClassification.from_pretrained(
                        download_dir, local_files_only=True, **model_kwargs
                    )
                    emotion_pipeline = None
                    logger.info(f"✅ {model_type} loaded from downloaded files (direct approach)")

            else:
                # PRODUCTION MODEL - use pipeline approach (works fine)
                model_name = PRODUCTION_MODEL_NAME
                model_kwargs = {"torch_dtype": "float32"}
                max_length = 512
                model_type = "Production (6 emotions)"

                # Load from Hugging Face Hub (with fallback to download if not cached)
                logger.info(f"🌐 Loading {model_type} from Hugging Face Hub")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_name, **model_kwargs
                    )
                    emotion_pipeline = _create_emotion_pipeline(tokenizer, model)
                    logger.info(f"✅ {model_type} loaded from Hugging Face Hub")
                except Exception as download_error:
                    logger.warning(f"Failed to load from cache, downloading {model_type}: {download_error}")

                    # Force download the model
                    from huggingface_hub import snapshot_download
                    download_dir = f"/tmp/{model_name.replace('/', '_')}"
                    model_path = snapshot_download(
                        repo_id=model_name,
                        local_dir=download_dir,
                        local_dir_use_symlinks=False
                    )
                    logger.info("📥 Model downloaded to: %s", model_path)

                    # Load from downloaded directory
                    tokenizer = AutoTokenizer.from_pretrained(
                        download_dir, local_files_only=True
                    )
                    model = AutoModelForSequenceClassification.from_pretrained(
                        download_dir, local_files_only=True, **model_kwargs
                    )
                    emotion_pipeline = _create_emotion_pipeline(tokenizer, model)
                    logger.info(f"✅ {model_type} loaded from downloaded files")

        # Update runtime labels from loaded model if available
        try:
            if USE_DEBERTA and emotion_model is not None:
                # For DeBERTa, use our predefined labels
                emotion_labels_runtime = [
                    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
                    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
                    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
                    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
                    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
                ]
            elif emotion_pipeline is not None:
                id2label = emotion_pipeline.model.config.id2label
                emotion_labels_runtime = [
                    id2label[i] for i in range(len(id2label))
                ]
        except Exception as label_err:
            logger.debug("Unable to derive runtime labels from model config: %s", label_err)

        with model_lock:
            model_loaded = True
            model_loading = False
            model_ready_event.set()

        model_type = "DeBERTa (28 emotions)" if USE_DEBERTA else "Production (6 emotions)"
        logger.info(f"🎉 {model_type} loading completed successfully")
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
        if USE_DEBERTA:
            # Use direct inference for DeBERTa (bypasses pipeline issues)
            emotion_results = _predict_emotions_deberta(text)

            # Format results to match expected output
            emotions = []
            for result in emotion_results:
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
        # Use the emotion pipeline for production model
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
        if USE_DEBERTA:
            # Use direct inference for DeBERTa (process one by one to avoid batch complexity)
            results = []
            for text in texts:
                ok, err = validate_text_input(text)
                if not ok:
                    results.append({'error': err, 'emotions': [], 'confidence': 0.0})
                else:
                    try:
                        emotion_results = _predict_emotions_deberta(text)

                        # Format results to match expected output
                        emotions = []
                        for result in emotion_results:
                            emotions.append({
                                'emotion': result['label'],
                                'confidence': result['score']
                            })

                        # Sort by confidence (highest first)
                        emotions.sort(key=lambda x: x['confidence'], reverse=True)

                        # Overall confidence is the highest confidence score
                        overall_confidence = emotions[0]['confidence'] if emotions else 0.0

                        results.append({
                            'text': text,
                            'emotions': emotions,
                            'confidence': overall_confidence,
                            'timestamp': time.time()
                        })
                    except Exception as e:
                        logger.exception("DeBERTa batch prediction failed for text: %s", e)
                        results.append({
                            'error': 'Emotion prediction failed',
                            'emotions': [],
                            'confidence': 0.0
                        })

            return results
        # Use pipeline for production model
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
