"""
Shared model utilities for Cloud Run deployment.

This module provides common functionality for model loading, inference,
and error handling to eliminate code duplication between API servers.
"""

import logging
import os
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

# Emotion labels (12 classes for DistilRoBERTa model)
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disgust', 'embarrassment'
]


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
        logger.info("🔄 Loading DistilRoBERTa model and tokenizer...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            'distilroberta-base',
            num_labels=len(EMOTION_LABELS)
        )
        
        # Load trained weights if available
        if Path(MODEL_PATH).exists():
            logger.info(f"📁 Loading trained weights from {MODEL_PATH}")
            model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        else:
            logger.warning(f"⚠️ No trained weights found at {MODEL_PATH}, using base model")
        
        model.eval()
        
        with model_lock:
            model_loaded = True
            model_loading = False
        
        logger.info("✅ Model loaded successfully!")
        return True
        
    except Exception as e:
        with model_lock:
            model_loading = False
        
        # Log error without exposing sensitive path information
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
            probabilities = torch.softmax(outputs.logits, dim=1)
        
        # Get top emotions
        top_probs, top_indices = torch.topk(probabilities[0], k=3)
        
        emotions = []
        for prob, idx in zip(top_probs, top_indices):
            emotions.append({
                'emotion': EMOTION_LABELS[idx.item()],
                'confidence': prob.item()
            })
        
        # Calculate overall confidence
        overall_confidence = top_probs[0].item()
        
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
        'emotion_labels': EMOTION_LABELS,
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
    if not text or not text.strip():
        return False, 'Text field is required'
    
    if len(text) > MAX_TEXT_LENGTH:
        return False, f'Text too long (max {MAX_TEXT_LENGTH} characters)'
    
    return True, '' 