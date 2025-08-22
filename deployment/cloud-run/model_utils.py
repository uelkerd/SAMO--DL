"""
Shared model utilities for Cloud Run deployment.

This module provides common functionality for model loading, inference,
and error handling to eliminate code duplication between API servers.
"""

import logging
import os
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model management
model = None
tokenizer = None
model_loaded = False
model_loading = False
model_lock = threading.Lock()

# Configuration
MODEL_PATH = os.getenv('MODEL_PATH', '/app/model/best_simple_model.pth')
MAX_LENGTH = int(os.getenv('MAX_LENGTH', '128'))
MAX_TEXT_LENGTH = int(os.getenv('MAX_TEXT_LENGTH', '1000'))
PREDICTION_THRESHOLD = float(os.getenv('PREDICTION_THRESHOLD', '0.5'))

# Emotion labels (12 classes for DistilRoBERTa model)
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disgust', 'embarrassment'
]
# Runtime label list derived from model config if available; falls back to default
emotion_labels_runtime: List[str] = EMOTION_LABELS.copy()
# Whether the active model is multi-label (sigmoid) or single-label (softmax)
is_multi_label_runtime: bool = False


def _load_repo_id_from_config() -> Optional[str]:
    cfg_path = Path('deployment/custom_model_config.json')
    if cfg_path.exists():
        try:
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
            repo_id = cfg.get('model_name') or cfg.get('repo_id')
            if repo_id:
                return repo_id
        except Exception as e:
            logger.warning("Failed to read %s: %s", cfg_path, e)
    return None


def _resolve_model_repo_id() -> str:
    # Priority: config file -> BASE_MODEL_NAME env -> distilroberta-base
    repo_id = _load_repo_id_from_config()
    if repo_id:
        logger.info("Using model from config: %s", repo_id)
        return repo_id
    env_model = os.getenv('BASE_MODEL_NAME')
    if env_model:
        logger.info("Using model from BASE_MODEL_NAME env: %s", env_model)
        return env_model
    default_model = 'distilroberta-base'
    logger.info("Using default base model: %s", default_model)
    return default_model


def ensure_model_loaded() -> bool:
    """
    Thread-safe model loading with proper error handling.

    Returns:
        bool: True if model is loaded successfully, False otherwise
    """
    global model, tokenizer, model_loaded, model_loading

    with model_lock:
        if model_loaded:
            return True

        if model_loading:
            # Wait for another thread to finish loading
            while model_loading:
                time.sleep(0.1)
            return model_loaded

        model_loading = True

    try:
        repo_id = _resolve_model_repo_id()
        logger.info("🔄 Loading model and tokenizer: %s", repo_id)

        # Load tokenizer
        tokenizer_local = AutoTokenizer.from_pretrained(repo_id)

        # Load model
        model_local = AutoModelForSequenceClassification.from_pretrained(
            repo_id
        )

        # Load trained weights if available
        if Path(MODEL_PATH).exists():
            logger.info(f"📁 Loading trained weights from {MODEL_PATH}")
            try:
                state = torch.load(MODEL_PATH, map_location='cpu')
                model_local.load_state_dict(state, strict=True)
                logger.info("Applied local fine-tuned weights from %s", MODEL_PATH)
            except Exception as weight_err:
                logger.warning(
                               "Failed to apply local weights from %s; using HF pretrained weights: %s",
                               MODEL_PATH,
                               weight_err
                              )
        else:
            logger.warning(
                           f"⚠️ No trained weights found at {MODEL_PATH},
                           using base/pretrained weights"
                          )

        model_local.eval()

        # Derive labels from model config if available
        derived_labels: List[str] = EMOTION_LABELS
        labels_from_config = getattr(model_local.config, 'id2label', None)
        try:
            if isinstance(labels_from_config, dict) and labels_from_config:
                try:
                    sorted_labels = sorted(
                                           labels_from_config.items(),
                                           key=lambda item: int(item[0])
                                          )
                except (ValueError, TypeError) as sort_exc:
                    logger.warning(
                                   "Could not sort id2label by integer keys. Using insertion order: %s",
                                   sort_exc
                                  )
                    sorted_labels = list(labels_from_config.items())
                derived_labels = [str(v) for _, v in sorted_labels]
        except Exception as e:
            logger.warning(
                           "Failed to parse id2label mapping; falling back to defaults: %s",
                           e
                          )

        # Resolve multi-label mode: env override -> config -> default False
        ml_env = os.getenv('MULTI_LABEL', 'auto').lower()
        if ml_env in ('1', 'true', 'yes'):
            ml_flag = True
        elif ml_env in ('0', 'false', 'no'):
            ml_flag = False
        else:
            ml_flag = str(
                          getattr(model_local.config,
                          'problem_type',
                          '')).lower() == 'multi_label_classification'

        with model_lock:
            # Assign only after successful load to avoid races
            global model, tokenizer, emotion_labels_runtime, is_multi_label_runtime
            model = model_local
            tokenizer = tokenizer_local
            emotion_labels_runtime = derived_labels
            is_multi_label_runtime = ml_flag
            model_loaded = True
            model_loading = False

        logger.info("✅ Model loaded successfully!")
        logger.info(
                    "🎯 Active labels (%d): %s",
                    len(emotion_labels_runtime),
                    emotion_labels_runtime
                   )
        logger.info(
                    "🧮 Inference mode: %s",
                    'multi-label (sigmoid)' if is_multi_label_runtime else 'single-label (softmax)'
                   )
        logger.info("🔧 Prediction threshold: %.2f", PREDICTION_THRESHOLD)
        return True

    except Exception as e:
        with model_lock:
            model_loading = False

        logger.exception(f"❌ Failed to load model: {str(e)}")
        logger.error("Model loading failed - check model configuration")
        return False


def predict_emotions(text: str) -> Dict[str, Any]:
    """
    Predict emotions for given text.

    Args:
        text (str): Input text to analyze

    Returns:
        Dict[str, Any]: Prediction results with emotions and confidence scores
    """
    if not ensure_model_loaded():
        return {
            'error': 'Model not available',
            'emotions': [],
            'confidence': 0.0
        }

    try:
        # Validate input
        if not text or not text.strip():
            return {
                'error': 'Text field is required',
                'emotions': [],
                'confidence': 0.0
            }

        if len(text) > MAX_TEXT_LENGTH:
            return {
                'error': f'Text too long (max {MAX_TEXT_LENGTH} characters)',
                'emotions': [],
                'confidence': 0.0
            }

        # Tokenize input
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        emotions: List[Dict[str, Any]] = []
        overall_confidence: float = 0.0

        if is_multi_label_runtime:
            # Multi-label: sigmoid + threshold
            probabilities = torch.sigmoid(logits)[0]
            pairs = [
                (
                    emotion_labels_runtime[i] if 0 <= i < len(
                                                              emotion_labels_runtime) else f"LABEL_{i}",
                                                              
                    probabilities[i].item(),
                )
                for i in range(probabilities.shape[-1])
            ]
            # Filter by threshold, fall back to top-1 if none pass
            filtered = [(
                         lbl,
                         prob) for lbl,
                         prob in pairs if prob >= PREDICTION_THRESHOLD]
            if not filtered and pairs:
                # pick the best single label
                best_lbl, best_prob = max(pairs, key=lambda kv: kv[1])
                filtered = [(best_lbl, best_prob)]
            # Sort by confidence
            filtered.sort(key=lambda kv: kv[1], reverse=True)
            emotions = [{'emotion': lbl, 'confidence': prob} for lbl, prob in filtered]
            overall_confidence = emotions[0]['confidence'] if emotions else 0.0
        else:
            # Single-label: softmax + top-k
            probabilities = torch.softmax(logits, dim=1)
            k = min(3, probabilities.shape[-1])
            top_probs, top_indices = torch.topk(probabilities[0], k=k)
            for prob, idx in zip(top_probs, top_indices):
                emotions.append({
                    'emotion': emotion_labels_runtime[idx.item(
                                                               )] if 0 <= idx.item() < len(emotion_labels_runtime) else "UNKNOWN_EMOTION",
                                                               
                    'confidence': prob.item()
                })
            overall_confidence = top_probs[0].item() if len(top_probs) > 0 else 0.0

        return {
            'text': text,
            'emotions': emotions,
            'confidence': overall_confidence,
            'timestamp': time.time()
        }

    except Exception as e:
        logger.exception(f"❌ Prediction failed: {str(e)}")
        return {
            'error': 'Prediction failed',
            'emotions': [],
            'confidence': 0.0
        }


def get_model_status() -> Dict[str, Any]:
    """
    Get current model status.

    Returns:
        Dict[str, Any]: Model status information
    """
    return {
        'model_loaded': model_loaded,
        'model_loading': model_loading,
        'model_path': MODEL_PATH,
        'max_length': MAX_LENGTH,
        'max_text_length': MAX_TEXT_LENGTH,
        'emotion_labels': emotion_labels_runtime,
        'multi_label': is_multi_label_runtime,
        'prediction_threshold': PREDICTION_THRESHOLD,
        'timestamp': time.time()
    }


def validate_text_input(text: str) -> Tuple[bool, str]:
    """
    Validate text input for prediction.

    Args:
        text (str): Text to validate

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not text or not isinstance(text, str):
        return False, 'Text must be a non-empty string'
    if len(text) > MAX_TEXT_LENGTH:
        return False, f'Text too long (max {MAX_TEXT_LENGTH} characters)'
    return True, '' 
