#!/usr/bin/env python3
"""
Configuration module for SAMO Emotion Detection System

This module provides centralized configuration management for the emotion
detection system, ensuring consistency across training and evaluation.
"""

from typing import Dict, Any
from dataclasses import dataclass

# Default configuration values
DEFAULT_CONFIG = {
    "model": {
        "name": "bert-base-uncased",
        "num_emotions": 28,
        "hidden_dropout_prob": 0.3,
        "classifier_dropout_prob": 0.5,
        "freeze_bert_layers": 6,
        "temperature": 1.0,
    },
    "training": {
        "batch_size": 16,
        "learning_rate": 2e-5,
        "num_epochs": 10,
        "weight_decay": 0.01,
    },
    "evaluation": {
        "threshold": 0.2,  # Default evaluation threshold
        "top_k": 5,
    },
    "prediction": {
        "threshold": 0.6,  # Default prediction threshold
        "max_length": 512,
    }
}


def get_evaluation_threshold() -> float:
    """Get current evaluation threshold from global config."""
    return _config.evaluation_threshold


def get_prediction_threshold() -> float:
    """Get current prediction threshold from global config."""
    return _config.prediction_threshold


@dataclass
class EmotionDetectionConfig:
    """Configuration class for emotion detection system."""

    # Model configuration
    model_name: str = "bert-base-uncased"
    num_emotions: int = 28
    hidden_dropout_prob: float = 0.3
    classifier_dropout_prob: float = 0.5
    freeze_bert_layers: int = 6
    temperature: float = 1.0

    # Training configuration
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    weight_decay: float = 0.01

    # Evaluation configuration
    evaluation_threshold: float = 0.2
    top_k: int = 5

    # Prediction configuration
    prediction_threshold: float = 0.6
    max_length: int = 512

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EmotionDetectionConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_name": self.model_name,
            "num_emotions": self.num_emotions,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "classifier_dropout_prob": self.classifier_dropout_prob,
            "freeze_bert_layers": self.freeze_bert_layers,
            "temperature": self.temperature,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "weight_decay": self.weight_decay,
            "evaluation_threshold": self.evaluation_threshold,
            "top_k": self.top_k,
            "prediction_threshold": self.prediction_threshold,
            "max_length": self.max_length,
        }


def get_default_config() -> EmotionDetectionConfig:
    """Get default configuration."""
    return EmotionDetectionConfig()


def get_config_from_dict(config_dict: Dict[str, Any]) -> EmotionDetectionConfig:
    """Get configuration from dictionary with defaults."""
    default_config = get_default_config()

    # Update with provided values
    for key, value in config_dict.items():
        if hasattr(default_config, key):
            setattr(default_config, key, value)

    return default_config


# Global configuration instance
_config = get_default_config()


def get_config() -> EmotionDetectionConfig:
    """Get current configuration."""
    return _config


def update_config(config_dict: Dict[str, Any]) -> None:
    """Update global configuration."""
    global _config
    _config = get_config_from_dict(config_dict)


def reset_config() -> None:
    """Reset to default configuration."""
    global _config
    _config = get_default_config()
