"""
Enhanced BERT-based Emotion Classifier for SAMO Deep Learning.

This module provides an enhanced BERT-based multi-label emotion classification model
with improved error handling, performance optimizations, and advanced features.
"""

import logging
import warnings
from typing import Optional, Union, List, Dict, Tuple, Any
from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .emotion_labels import GOEMOTIONS_EMOTIONS

# Configure logging
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


@dataclass
class EmotionPrediction:
    """Structured emotion prediction result."""
    emotions: Dict[str, float]
    primary_emotion: str
    confidence: float
    emotional_intensity: str
    top_k_emotions: List[Tuple[str, float]]
    prediction_metadata: Dict[str, Any]


class EnhancedBERTEmotionClassifier(nn.Module):
    """Enhanced BERT-based emotion classifier with advanced features.

    Features:
    - Robust error handling and recovery
    - Performance optimizations (mixed precision, caching)
    - Advanced emotion analysis (intensity, confidence scoring)
    - Model management utilities
    - Comprehensive logging and monitoring
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_emotions: int = 28,
        hidden_dropout_prob: float = 0.3,
        classifier_dropout_prob: float = 0.5,
        freeze_bert_layers: int = 0,
        temperature: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        use_mixed_precision: bool = True,
        cache_embeddings: bool = False,
        max_sequence_length: int = 512,
    ) -> None:
        """Initialize enhanced BERT emotion classifier.

        Args:
            model_name: Hugging Face model name
            num_emotions: Number of emotion categories
            hidden_dropout_prob: Dropout rate for BERT hidden layers
            classifier_dropout_prob: Dropout rate for classification head
            freeze_bert_layers: Number of BERT layers to freeze initially
            temperature: Temperature scaling parameter for calibration
            class_weights: Optional class weights for imbalanced data
            use_mixed_precision: Enable mixed precision training/inference
            cache_embeddings: Cache BERT embeddings for repeated inputs
            max_sequence_length: Maximum input sequence length
        """
        super().__init__()

        self.model_name = model_name
        self.num_emotions = num_emotions
        self.hidden_dropout_prob = hidden_dropout_prob
        self.classifier_dropout_prob = classifier_dropout_prob
        self.freeze_bert_layers = freeze_bert_layers
        self.temperature = temperature
        self.prediction_threshold = 0.6
        self.class_weights = class_weights
        self.emotion_labels = GOEMOTIONS_EMOTIONS[:num_emotions]
        self.use_mixed_precision = use_mixed_precision
        self.cache_embeddings = cache_embeddings
        self.max_sequence_length = max_sequence_length

        # Device setup with fallback
        self.device = self._setup_device()

        # Initialize model components
        self._initialize_bert_model()
        self._initialize_classifier()
        self._initialize_utilities()

        # Move to device
        self.to(self.device)

    @staticmethod
    def _setup_device() -> torch.device:
        """Setup device with fallback handling."""
        try:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("Using CUDA device: %s", torch.cuda.get_device_name())
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Using MPS device (Apple Silicon)")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU device")
            return device
        except Exception as e:
            logger.warning("Device setup failed, falling back to CPU: %s", e)
            return torch.device("cpu")

    def _initialize_bert_model(self) -> None:
        """Initialize BERT model with error handling."""
        try:
            config = AutoConfig.from_pretrained(self.model_name)
            config.hidden_dropout_prob = self.hidden_dropout_prob
            config.attention_probs_dropout_prob = self.hidden_dropout_prob

            self.bert = AutoModel.from_pretrained(self.model_name, config=config)
            self.bert_hidden_size = config.hidden_size

            logger.info("BERT model loaded: %s", self.model_name)
        except Exception as e:
            logger.error("Failed to load BERT model: %s", e)
            raise RuntimeError(f"BERT model initialization failed: {e}") from e

    def _initialize_classifier(self) -> None:
        """Initialize classification head."""
        self.classifier = nn.Sequential(
            nn.Dropout(self.classifier_dropout_prob),
            nn.Linear(self.bert_hidden_size, self.bert_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.classifier_dropout_prob),
            nn.Linear(self.bert_hidden_size, self.num_emotions),
        )

        self.temperature = nn.Parameter(torch.ones(1) * self.temperature)
        self._init_classification_layers()

        # Freeze BERT layers if specified
        if self.freeze_bert_layers > 0:
            self._freeze_bert_layers(self.freeze_bert_layers)

    def _initialize_utilities(self) -> None:
        """Initialize utility components."""
        # Embedding cache for repeated inputs
        if self.cache_embeddings:
            self._embedding_cache = {}

        # Performance tracking
        self._inference_count = 0
        self._total_inference_time = 0.0

        # Error tracking
        self._error_count = 0
        self._last_error = None

    def _init_classification_layers(self) -> None:
        """Initialize classification layers with proper weight initialization."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _freeze_bert_layers(self, num_layers: int) -> None:
        """Freeze the first num_layers of BERT."""
        if num_layers <= 0:
            return

        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        # Freeze specified number of encoder layers
        for i in range(min(num_layers, len(self.bert.encoder.layer))):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False

        logger.info("Froze %s BERT layers", num_layers)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass with error handling and optimizations."""
        try:
            # Get BERT outputs
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = bert_outputs.pooler_output

            # Classification head
            logits = self.classifier(pooled_output)

            # Apply temperature scaling
            logits = logits / self.temperature

            return logits

        except Exception as e:
            logger.exception("Forward pass failed")
            self._error_count += 1
            self._last_error = str(e)
            raise RuntimeError(f"Model forward pass failed: {e}") from e

    @contextmanager
    def inference_mode(self):
        """Context manager for inference mode with optimizations."""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        yield
                else:
                    yield
        finally:
            if was_training:
                self.train()

    def predict_emotions(
        self,
        texts: Union[str, List[str]],
        top_k: int = 5,
        return_metadata: bool = True,
        batch_size: int = 32
    ) -> Union[EmotionPrediction, List[EmotionPrediction]]:
        """Predict emotions for input text(s) with enhanced features.

        Args:
            texts: Input text or list of texts
            top_k: Number of top emotions to return
            return_metadata: Whether to return prediction metadata
            batch_size: Batch size for processing multiple texts

        Returns:
            EmotionPrediction or list of EmotionPrediction objects
        """
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if start_time:
            start_time.record()

        try:
            # Handle single text input
            if isinstance(texts, str):
                return self._predict_single_text(texts, top_k, return_metadata)

            # Handle multiple texts
            return self._predict_batch_texts(texts, top_k, return_metadata, batch_size)

        except Exception as e:
            logger.exception("Emotion prediction failed")
            self._error_count += 1
            self._last_error = str(e)
            raise RuntimeError(f"Emotion prediction failed: {e}") from e
        finally:
            if start_time:
                end_time = torch.cuda.Event(enable_timing=True)
                end_time.record()
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time) / 1000.0
                self._update_performance_metrics(inference_time)

    def _predict_single_text(
        self, 
        text: str, 
        top_k: int, 
        return_metadata: bool
    ) -> EmotionPrediction:
        """Predict emotions for a single text."""
        if not text or not text.strip():
            return self._create_empty_prediction(return_metadata)

        # Tokenize input
        tokenizer = self._get_tokenizer()
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_sequence_length,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Get predictions
        with self.inference_mode():
            logits = self.forward(input_ids, attention_mask)
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]

        # Process results
        return self._process_prediction_results(
            probabilities, top_k, return_metadata, text
        )

    def _predict_batch_texts(
        self,
        texts: List[str],
        top_k: int,
        return_metadata: bool,
        batch_size: int
    ) -> List[EmotionPrediction]:
        """Predict emotions for multiple texts in batches."""
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = self._process_batch(batch_texts, top_k, return_metadata)
            results.extend(batch_results)

        return results

    def _process_batch(
        self,
        texts: List[str],
        top_k: int,
        return_metadata: bool
    ) -> List[EmotionPrediction]:
        """Process a batch of texts."""
        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            return [self._create_empty_prediction(return_metadata) for _ in texts]

        # Tokenize batch
        tokenizer = self._get_tokenizer()
        inputs = tokenizer(
            valid_texts,
            truncation=True,
            padding=True,
            max_length=self.max_sequence_length,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Get predictions
        with self.inference_mode():
            logits = self.forward(input_ids, attention_mask)
            probabilities = torch.sigmoid(logits).cpu().numpy()

        # Process results
        results = []
        for i, prob in enumerate(probabilities):
            result = self._process_prediction_results(
                prob, top_k, return_metadata, valid_texts[i]
            )
            results.append(result)

        return results

    def _process_prediction_results(
        self,
        probabilities: np.ndarray,
        top_k: int,
        return_metadata: bool,
        text: str
    ) -> EmotionPrediction:
        """Process prediction results into structured format."""
        # Create emotion dictionary
        emotions = {
            self.emotion_labels[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }

        # Get top-k emotions
        top_k_indices = np.argsort(probabilities)[-top_k:][::-1]
        top_k_emotions = [
            (self.emotion_labels[i], float(probabilities[i]))
            for i in top_k_indices
        ]

        # Primary emotion and confidence
        primary_idx = np.argmax(probabilities)
        primary_emotion = self.emotion_labels[primary_idx]
        confidence = float(probabilities[primary_idx])

        # Emotional intensity
        emotional_intensity = self._calculate_emotional_intensity(probabilities)

        # Metadata
        metadata = {}
        if return_metadata:
            metadata = {
                "text_length": len(text),
                "prediction_threshold": self.prediction_threshold,
                "temperature": float(self.temperature.item()),
                "model_name": self.model_name,
                "num_emotions": self.num_emotions,
                "max_confidence": float(np.max(probabilities)),
                "confidence_std": float(np.std(probabilities)),
            }

        return EmotionPrediction(
            emotions=emotions,
            primary_emotion=primary_emotion,
            confidence=confidence,
            emotional_intensity=emotional_intensity,
            top_k_emotions=top_k_emotions,
            prediction_metadata=metadata
        )

    @staticmethod
    def _calculate_emotional_intensity(probabilities: np.ndarray) -> str:
        """Calculate emotional intensity based on prediction distribution."""
        max_prob = np.max(probabilities)
        prob_std = np.std(probabilities)

        if max_prob >= 0.8 and prob_std >= 0.3:
            return "very_high"
        if max_prob >= 0.7 and prob_std >= 0.2:
            return "high"
        if max_prob >= 0.5 and prob_std >= 0.1:
            return "moderate"
        if max_prob >= 0.3:
            return "low"
        return "very_low"

    def _create_empty_prediction(self, return_metadata: bool) -> EmotionPrediction:
        """Create empty prediction for invalid inputs."""
        emotions = {emotion: 0.0 for emotion in self.emotion_labels}
        metadata = {"error": "empty_input"} if return_metadata else {}

        return EmotionPrediction(
            emotions=emotions,
            primary_emotion="neutral",
            confidence=0.0,
            emotional_intensity="very_low",
            top_k_emotions=[("neutral", 1.0)],
            prediction_metadata=metadata
        )

    def _get_tokenizer(self):
        """Get or create tokenizer with caching."""
        if not hasattr(self, '_tokenizer'):
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                logger.info("Tokenizer loaded: %s", self.model_name)
            except Exception as e:
                logger.error("Failed to load tokenizer: %s", e)
                raise RuntimeError(f"Tokenizer loading failed: {e}") from e
        return self._tokenizer

    def _update_performance_metrics(self, inference_time: float) -> None:
        """Update performance tracking metrics."""
        self._inference_count += 1
        self._total_inference_time += inference_time

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        avg_inference_time = (
            self._total_inference_time / self._inference_count 
            if self._inference_count > 0 else 0.0
        )

        return {
            "total_inferences": self._inference_count,
            "total_inference_time": self._total_inference_time,
            "average_inference_time": avg_inference_time,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(self._inference_count, 1),
            "last_error": self._last_error,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_name": self.model_name,
            "num_emotions": self.num_emotions,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "device": str(self.device),
            "use_mixed_precision": self.use_mixed_precision,
            "max_sequence_length": self.max_sequence_length,
            "prediction_threshold": self.prediction_threshold,
            "temperature": float(self.temperature.item()),
        }

    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

    def count_frozen_parameters(self) -> int:
        """Count frozen parameters."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def save_model(self, path: str) -> None:
        """Save model with error handling."""
        try:
            torch.save({
                'model_state_dict': self.state_dict(),
                'model_config': {
                    'model_name': self.model_name,
                    'num_emotions': self.num_emotions,
                    'hidden_dropout_prob': self.hidden_dropout_prob,
                    'classifier_dropout_prob': self.classifier_dropout_prob,
                    'freeze_bert_layers': self.freeze_bert_layers,
                    'temperature': float(self.temperature.item()),
                }
            }, path)
            logger.info("Model saved to: %s", path)
        except Exception as e:
            logger.error("Failed to save model: %s", e)
            raise RuntimeError(f"Model saving failed: {e}") from e

    @classmethod
    def load_model(cls, path: str, device: Optional[str] = None) -> 'EnhancedBERTEmotionClassifier':
        """Load model with error handling."""
        try:
            checkpoint = torch.load(path, map_location=device or 'cpu')
            model_config = checkpoint['model_config']

            model = cls(**model_config)
            model.load_state_dict(checkpoint['model_state_dict'])

            logger.info("Model loaded from: %s", path)
            return model
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            raise RuntimeError(f"Model loading failed: {e}") from e
