#!/usr/bin/env python3
"""
SAMO-Optimized T5 Text Summarization Model

This module provides a specialized T5 summarization model optimized for
journal entries and emotional text processing in the SAMO-DL system.

Key Features:
- T5-small model for efficient processing
- SAMO-specific parameter optimization
- Emotional keyword extraction
- Batch processing capabilities
- Comprehensive error handling
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Any
import yaml
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SAMOT5Summarizer:
    """
    SAMO-optimized T5 text summarization model.
    
    Optimized for journal entries with emotional context awareness
    and configurable summarization parameters.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the SAMO T5 summarizer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        self._load_model()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        default_config = {
            "model": {
                "name": "t5-small",
                "device": None
            },
            "generation": {
                "max_length": 100,
                "min_length": 20,
                "num_beams": 4,
                "early_stopping": True,
                "repetition_penalty": 1.2,
                "length_penalty": 1.0,
                "do_sample": False,
                "temperature": 1.0
            },
            "validation": {
                "min_words": 20,
                "max_words": 1000
            },
            "performance": {
                "batch_size": 4,
                "timeout_seconds": 30
            },
            "samo_optimizations": {
                "emotional_context": True,
                "preserve_tone": True,
                "journal_mode": True,
                "extract_key_emotions": True,
                "sanitize_input": True,
                "log_level": "INFO"
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                logger.warning("Failed to load config from %s: %s", config_path, e)
                
        return default_config
    
    def _get_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self) -> None:
        """Load the T5 model and tokenizer."""
        try:
            model_name = self.config["model"]["name"]
            logger.info("Loading T5 model: %s", model_name)
            
            # Load tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            
            # Load model
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("T5 model loaded successfully on %s", self.device)
            
        except Exception as e:
            logger.error("Failed to load T5 model: %s", e)
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _validate_input(self, text: str) -> Tuple[bool, str]:
        """
        Validate input text for summarization.
        
        Args:
            text: Input text to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(text, str):
            return False, "Input must be a string"
        
        if not text.strip():
            return False, "Input text cannot be empty"
        
        word_count = len(text.split())
        min_words = self.config["validation"]["min_words"]
        max_words = self.config["validation"]["max_words"]
        
        if word_count < min_words:
            return False, f"Text too short (minimum {min_words} words)"
        
        if word_count > max_words:
            return False, f"Text too long (maximum {max_words} words)"
        
        return True, ""
    
    def _extract_emotional_keywords(self, text: str) -> List[str]:
        """
        Extract emotional keywords from text for SAMO optimization.
        
        Args:
            text: Input text
            
        Returns:
            List of emotional keywords
        """
        # Simple emotional keyword extraction
        emotional_keywords = [
            'happy', 'sad', 'angry', 'excited', 'worried', 'grateful',
            'anxious', 'proud', 'confident', 'overwhelmed', 'peaceful',
            'frustrated', 'hopeful', 'disappointed', 'relieved', 'nervous'
        ]
        
        text_lower = text.lower()
        return [kw for kw in emotional_keywords if kw in text_lower]
    
    def generate_summary(self, text: str) -> Dict[str, Any]:
        """
        Generate a summary for the given text.
        
        Args:
            text: Input text to summarize
            
        Returns:
            Dictionary containing summary and metadata
        """
        start_time = time.time()
        
        # Validate input
        is_valid, error_msg = self._validate_input(text)
        if not is_valid:
            return {
                "summary": "",
                "error": error_msg,
                "success": False,
                "processing_time": 0.0
            }
        
        try:
            # Extract emotional keywords for SAMO optimization
            emotional_keywords = self._extract_emotional_keywords(text)
            
            # Prepare input for T5
            input_text = f"summarize: {text}"
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate summary
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=self.config["generation"]["max_length"],
                    min_length=self.config["generation"]["min_length"],
                    num_beams=self.config["generation"]["num_beams"],
                    early_stopping=self.config["generation"]["early_stopping"],
                    repetition_penalty=self.config["generation"]["repetition_penalty"],
                    length_penalty=self.config["generation"]["length_penalty"],
                    do_sample=self.config["generation"]["do_sample"],
                    temperature=self.config["generation"]["temperature"]
                )
            
            # Decode output
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove "summarize:" prefix if present
            if summary.startswith("summarize:"):
                summary = summary[10:].strip()
            
            # Calculate metrics
            original_length = len(text.split())
            summary_length = len(summary.split())
            compression_ratio = summary_length / original_length if original_length > 0 else 0
            processing_time = time.time() - start_time
            
            return {
                "summary": summary,
                "original_length": original_length,
                "summary_length": summary_length,
                "compression_ratio": compression_ratio,
                "emotional_keywords": emotional_keywords,
                "processing_time": processing_time,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            logger.error("Summarization failed: %s", e)
            return {
                "summary": "",
                "error": str(e),
                "success": False,
                "processing_time": time.time() - start_time
            }
    
    def generate_batch_summaries(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Generate summaries for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of summary dictionaries
        """
        results = []
        
        for text in texts:
            result = self.generate_summary(text)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.config["model"]["name"],
            "device": self.device,
            "config": self.config,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None
        }


def create_samo_t5_summarizer(config_path: Optional[str] = None) -> SAMOT5Summarizer:
    """
    Factory function to create a SAMO T5 summarizer instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        SAMOT5Summarizer instance
    """
    return SAMOT5Summarizer(config_path)


# Example usage
if __name__ == "__main__":
    # Test the summarizer
    summarizer = create_samo_t5_summarizer()
    
    test_text = """
    Today I had an amazing experience at the conference. I learned so much about AI and machine learning.
    The speakers were incredibly knowledgeable and the networking opportunities were fantastic. I met
    several people who share my passion for deep learning and we exchanged contact information. I'm
    feeling really excited about the future possibilities and can't wait to implement some of the
    techniques I learned. This has been one of the most productive days I've had in months.
    """
    
    result = summarizer.generate_summary(test_text)
    print("Summary:", result["summary"])
    print("Compression ratio:", result["compression_ratio"])
    print("Emotional keywords:", result["emotional_keywords"])
