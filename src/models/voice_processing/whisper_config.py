#!/usr/bin/env python3
"""
SAMO Whisper Configuration Module

This module handles configuration loading and management for the SAMO Whisper
transcription system, supporting both YAML file configuration and programmatic defaults.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

logger = logging.getLogger(__name__)


class SAMOWhisperConfig:
    """Configuration for SAMO Whisper transcription."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file or defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                self._load_from_dict(config_data)
        else:
            self._load_defaults()
    
    def _load_from_dict(self, config_data: Dict[str, Any]):
        """Load configuration from dictionary."""
        whisper_config = config_data.get('whisper', {})
        transcription_config = config_data.get('transcription', {})
        
        # Load from whisper section first, then transcription section as fallback
        self.model_size = whisper_config.get('model_size', 'base')
        self.language = whisper_config.get('language', None)
        self.device = whisper_config.get('device', None)
        
        # Load transcription parameters from both sections
        self.task = whisper_config.get('task', transcription_config.get('task', 'transcribe'))
        self.temperature = whisper_config.get('temperature', transcription_config.get('temperature', 0.0))
        self.beam_size = whisper_config.get('beam_size', transcription_config.get('beam_size', None))
        self.best_of = whisper_config.get('best_of', transcription_config.get('best_of', None))
        self.patience = whisper_config.get('patience', transcription_config.get('patience', None))
        self.length_penalty = whisper_config.get('length_penalty', transcription_config.get('length_penalty', None))
        self.suppress_tokens = whisper_config.get('suppress_tokens', transcription_config.get('suppress_tokens', '-1'))
        self.initial_prompt = whisper_config.get('initial_prompt', transcription_config.get('initial_prompt', None))
        self.condition_on_previous_text = whisper_config.get('condition_on_previous_text', transcription_config.get('condition_on_previous_text', True))
        self.fp16 = whisper_config.get('fp16', transcription_config.get('fp16', True))
        self.compression_ratio_threshold = whisper_config.get('compression_ratio_threshold', transcription_config.get('compression_ratio_threshold', 2.4))
        self.logprob_threshold = whisper_config.get('logprob_threshold', transcription_config.get('logprob_threshold', -1.0))
        self.no_speech_threshold = whisper_config.get('no_speech_threshold', transcription_config.get('no_speech_threshold', 0.6))
    
    def _load_defaults(self):
        """Load default configuration."""
        self.model_size = 'base'
        self.language = None
        self.task = 'transcribe'
        self.temperature = 0.0
        self.beam_size = None
        self.best_of = None
        self.patience = None
        self.length_penalty = None
        self.suppress_tokens = '-1'
        self.initial_prompt = None
        self.condition_on_previous_text = True
        self.fp16 = True
        self.compression_ratio_threshold = 2.4
        self.logprob_threshold = -1.0
        self.no_speech_threshold = 0.6
        self.device = None
    
    def get_transcription_options(self) -> Dict[str, Any]:
        """Get transcription options as a dictionary, filtering out None values."""
        options = {
            "language": self.language,
            "task": self.task,
            "temperature": self.temperature,
            "best_of": self.best_of,
            "beam_size": self.beam_size,
            "patience": self.patience,
            "length_penalty": self.length_penalty,
            "suppress_tokens": self.suppress_tokens,
            "initial_prompt": self.initial_prompt,
            "condition_on_previous_text": self.condition_on_previous_text,
            "fp16": self.fp16,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "logprob_threshold": self.logprob_threshold,
            "no_speech_threshold": self.no_speech_threshold,
        }
        return {k: v for k, v in options.items() if v is not None}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model_size": self.model_size,
            "language": self.language,
            "device": self.device,
            "task": self.task,
            "temperature": self.temperature,
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "patience": self.patience,
            "length_penalty": self.length_penalty,
            "suppress_tokens": self.suppress_tokens,
            "initial_prompt": self.initial_prompt,
            "condition_on_previous_text": self.condition_on_previous_text,
            "fp16": self.fp16,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "logprob_threshold": self.logprob_threshold,
            "no_speech_threshold": self.no_speech_threshold,
        }
