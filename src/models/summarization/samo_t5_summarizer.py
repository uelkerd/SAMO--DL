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

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

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
        self.device = self._get_device(self.config)
        self._load_model()
        
    @staticmethod
    def _load_config(config_path: Optional[str]) -> Dict[str, Any]:
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
                "log_level": "INFO",
                "emotional_keywords": [
                    'happy', 'sad', 'angry', 'excited', 'worried', 'grateful',
                    'anxious', 'proud', 'confident', 'overwhelmed', 'peaceful',
                    'frustrated', 'hopeful', 'disappointed', 'relieved', 'nervous'
                ]
            }
        }
        
        def recursive_merge_dicts(default, override):
            """Recursively merge two dictionaries, with values from override taking precedence."""
            for key, value in default.items():
                if key not in override:
                    override[key] = value
                elif isinstance(value, dict) and isinstance(override[key], dict):
                    override[key] = recursive_merge_dicts(value, override[key])
            return override

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                # Deep merge user config into default config
                for key, value in user_config.items():
                    if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
            except Exception as e:
                logger.warning("Failed to load config from %s: %s. Using default config.", config_path, e)
                
        return default_config
    
    @staticmethod
    def _get_device(config: Dict[str, Any]) -> str:
        """Get the best available device, respecting user-specified device override."""
        # Check if user specified a device in config
        user_device = config.get("model", {}).get("device")
        if user_device:
            logger.info("Using user-specified device: %s", user_device)
            return user_device
        
        # Auto-detect best available device
        if torch.cuda.is_available():
            return "cuda"
        
        if getattr(torch.backends, "mps", None) is not None and getattr(torch.backends.mps, "is_available", lambda: False)():
            return "mps"
        
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
    
    @staticmethod
    def _extract_emotional_keywords(text: str, config: Dict[str, Any]) -> List[str]:
        """
        Extract emotional keywords from text for SAMO optimization.
        
        Args:
            text: Input text
            config: Configuration dictionary
            
        Returns:
            List of emotional keywords
        """
        import re
        
        # Get configurable emotional keywords
        emotional_keywords = config["samo_optimizations"]["emotional_keywords"]
        
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in emotional_keywords:
            # Use word boundary matching to avoid false positives
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _sanitize_input(self, text: str) -> str:
        """
        Sanitize input text for SAMO optimization.
        
        Args:
            text: Input text to sanitize
            
        Returns:
            Sanitized text
        """
        # Basic sanitization - remove excessive whitespace and normalize
        import re
        # Remove multiple spaces and normalize line breaks
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    def _prepare_samo_input(self, text: str, emotional_keywords: List[str]) -> str:
        """
        Prepare input text with SAMO-specific optimizations.
        
        Args:
            text: Input text
            emotional_keywords: List of detected emotional keywords
            
        Returns:
            Optimized input text for T5
        """
        # Base summarization prompt
        input_text = f"summarize: {text}"
        
        # Add emotional context if enabled
        if self.config["samo_optimizations"]["emotional_context"] and emotional_keywords:
            emotion_context = f" [emotions: {', '.join(emotional_keywords)}]"
            input_text = f"summarize{emotion_context}: {text}"
        
        # Add journal mode context if enabled
        if self.config["samo_optimizations"]["journal_mode"]:
            input_text = f"summarize journal entry: {text}"
        
        # Add tone preservation instruction if enabled
        if self.config["samo_optimizations"]["preserve_tone"] and emotional_keywords:
            tone_instruction = f" [preserve emotional tone: {', '.join(emotional_keywords[:3])}]"
            input_text += tone_instruction
        
        return input_text
    
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
            # Apply SAMO optimizations
            processed_text = text
            
            # Sanitize input if enabled
            if self.config["samo_optimizations"]["sanitize_input"]:
                processed_text = self._sanitize_input(processed_text)
            
            # Extract emotional keywords for SAMO optimization
            emotional_keywords = []
            if self.config["samo_optimizations"]["extract_key_emotions"]:
                emotional_keywords = self._extract_emotional_keywords(processed_text, self.config)
            
            # Prepare input for T5 with SAMO optimizations
            input_text = self._prepare_samo_input(processed_text, emotional_keywords)
            
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
            
            # Clean up any potential prefixes or artifacts
            summary = summary.strip()
            
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
        Generate summaries for multiple texts using true batch processing.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of summary dictionaries
        """
        start_time = time.time()
        results = []
        
        # Validate all inputs first
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            is_valid, error_msg = self._validate_input(text)
            if is_valid:
                valid_texts.append(text)
                valid_indices.append(i)
            else:
                results.append({
                    "summary": "",
                    "error": error_msg,
                    "success": False,
                    "processing_time": 0.0
                })
        
        if not valid_texts:
            return results
        
        # Process in batches for efficiency
        batch_size = self.config["performance"]["batch_size"]
        
        for batch_start in range(0, len(valid_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_texts))
            batch_texts = valid_texts[batch_start:batch_end]
            batch_indices = valid_indices[batch_start:batch_end]
            
            try:
                # Apply SAMO optimizations to each text in batch
                processed_texts = []
                batch_emotional_keywords = []
                
                for text in batch_texts:
                    # Apply SAMO optimizations
                    processed_text = text
                    if self.config["samo_optimizations"]["sanitize_input"]:
                        processed_text = self._sanitize_input(processed_text)
                    
                    # Extract emotional keywords
                    emotional_keywords = []
                    if self.config["samo_optimizations"]["extract_key_emotions"]:
                        emotional_keywords = self._extract_emotional_keywords(processed_text, self.config)
                    
                    batch_emotional_keywords.append(emotional_keywords)
                    
                    # Prepare SAMO input
                    input_text = self._prepare_samo_input(processed_text, emotional_keywords)
                    processed_texts.append(input_text)
                
                # Tokenize entire batch at once
                inputs = self.tokenizer(
                    processed_texts,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                # Generate summaries for entire batch
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=self.config["generation"]["max_length"],
                        min_length=self.config["generation"]["min_length"],
                        num_beams=self.config["generation"]["num_beams"],
                        early_stopping=self.config["generation"]["early_stopping"],
                        repetition_penalty=self.config["generation"]["repetition_penalty"],
                        length_penalty=self.config["generation"]["length_penalty"],
                        do_sample=self.config["generation"]["do_sample"],
                        temperature=self.config["generation"]["temperature"]
                    )
                
                # Decode all outputs at once
                summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # Process each output in the batch
                for i, (summary, original_text, emotional_keywords) in enumerate(zip(summaries, batch_texts, batch_emotional_keywords)):
                    # Clean up any potential prefixes or artifacts
                    summary = summary.strip()
                    
                    # Calculate metrics
                    original_length = len(original_text.split())
                    summary_length = len(summary.split())
                    compression_ratio = summary_length / original_length if original_length > 0 else 0
                    
                    # Insert result at correct index
                    result_index = batch_indices[i]
                    while len(results) <= result_index:
                        results.append(None)
                    
                    results[result_index] = {
                        "summary": summary,
                        "original_length": original_length,
                        "summary_length": summary_length,
                        "compression_ratio": compression_ratio,
                        "emotional_keywords": emotional_keywords,
                        "processing_time": time.time() - start_time,
                        "success": True,
                        "error": None
                    }
                    
            except Exception as e:
                logger.error("Batch processing failed: %s", e)
                # Add error results for this batch
                for i, idx in enumerate(batch_indices):
                    while len(results) <= idx:
                        results.append(None)
                    results[idx] = {
                        "summary": "",
                        "error": str(e),
                        "success": False,
                        "processing_time": time.time() - start_time
                    }
        
        # Fill in any missing results with errors
        for i in range(len(texts)):
            if i >= len(results) or results[i] is None:
                results.append({
                    "summary": "",
                    "error": "Processing failed",
                    "success": False,
                    "processing_time": 0.0
                })
        
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
