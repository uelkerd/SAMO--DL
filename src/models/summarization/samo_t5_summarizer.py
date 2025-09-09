#!/usr/bin/env python3
"""
SAMO-Optimized T5 Summarization Wrapper.

This module provides a SAMO-specific wrapper around the T5 summarization
with optimized parameters for journal entries and emotional text analysis.
"""

import logging
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

from .t5_summarization import create_t5_summarizer, T5Summarizer, SummarizationConfig

logger = logging.getLogger(__name__)

class SAMOT5Summarizer:
    """SAMO-optimized T5 summarization wrapper."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize SAMO T5 summarizer with optimized configuration.
        
        Args:
            config_path: Path to SAMO T5 configuration file
        """
        self.config_path = config_path or "configs/samo_t5_config.yaml"
        self.config = self._load_config()
        self.summarizer = None
        self._initialize_summarizer()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load SAMO T5 configuration."""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.warning("Config file not found, using defaults: %s", self.config_path)
                return self._get_default_config()
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info("Loaded SAMO T5 config from: %s", self.config_path)
            return config
            
        except Exception as e:
            logger.error("Failed to load config: %s", e)
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default SAMO configuration."""
        return {
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
    
    def _initialize_summarizer(self):
        """Initialize the underlying T5 summarizer."""
        try:
            model_config = self.config["model"]
            generation_config = self.config["generation"]
            
            # Create T5 configuration
            t5_config = SummarizationConfig(
                model_name=model_config["name"],
                max_length=generation_config["max_length"],
                min_length=generation_config["min_length"],
                num_beams=generation_config["num_beams"],
                early_stopping=generation_config["early_stopping"],
                device=model_config["device"],
                do_sample=generation_config["do_sample"],
                temperature=generation_config["temperature"],
                repetition_penalty=generation_config["repetition_penalty"],
                length_penalty=generation_config["length_penalty"]
            )
            
            # Create summarizer
            self.summarizer = T5Summarizer(t5_config)
            logger.info("✅ SAMO T5 summarizer initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize SAMO T5 summarizer: %s", e)
            raise RuntimeError(f"SAMO T5 initialization failed: {e}")
    
    def summarize_journal_entry(
        self, 
        text: str, 
        extract_emotions: bool = True
    ) -> Dict[str, Any]:
        """Summarize a journal entry with SAMO-specific optimizations.
        
        Args:
            text: Journal entry text to summarize
            extract_emotions: Whether to extract emotional context
            
        Returns:
            Dictionary with summary and SAMO-specific metadata
        """
        # Validate input
        word_count = len(text.split()) if text else 0
        validation = self.config["validation"]
        
        if word_count < validation["min_words"]:
            return {
                "summary": text.strip() if text and word_count >= 10 else "",
                "confidence": 0.0,
                "input_length": word_count,
                "summary_length": word_count if text and word_count >= 10 else 0,
                "processing_time": 0.0,
                "scores": {},
                "note": "Input too short for meaningful summarization",
                "samo_metadata": {
                    "journal_mode": True,
                    "emotional_context": False,
                    "extraction_successful": False
                }
            }
        
        if word_count > validation["max_words"]:
            logger.warning("Text too long (%d words), truncating to %d", 
                         word_count, validation["max_words"])
            text = " ".join(text.split()[:validation["max_words"]])
        
        # Generate summary using underlying T5 model
        result = self.summarizer.summarize(text)
        
        # Add SAMO-specific metadata
        result["samo_metadata"] = {
            "journal_mode": True,
            "emotional_context": extract_emotions,
            "extraction_successful": True,
            "config_used": {
                "max_length": self.config["generation"]["max_length"],
                "min_length": self.config["generation"]["min_length"],
                "num_beams": self.config["generation"]["num_beams"],
                "repetition_penalty": self.config["generation"]["repetition_penalty"]
            }
        }
        
        # Extract emotional context if requested
        if extract_emotions and self.config["samo_optimizations"]["extract_key_emotions"]:
            result["emotional_keywords"] = self._extract_emotional_keywords(text)
        
        return result
    
    def summarize_batch_journal_entries(
        self, 
        texts: list, 
        batch_size: Optional[int] = None
    ) -> list:
        """Summarize multiple journal entries in batches.
        
        Args:
            texts: List of journal entry texts
            batch_size: Batch size for processing
            
        Returns:
            List of summarization results
        """
        batch_size = batch_size or self.config["performance"]["batch_size"]
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = [
                self.summarize_journal_entry(text) 
                for text in batch
            ]
            results.extend(batch_results)
            logger.info("Processed SAMO batch %d: %d entries", 
                       i//batch_size + 1, len(batch))
        
        return results
    
    def _extract_emotional_keywords(self, text: str) -> list:
        """Extract emotional keywords from text (simple implementation).
        
        Args:
            text: Input text
            
        Returns:
            List of emotional keywords found
        """
        # Simple emotional keyword extraction
        emotional_keywords = [
            "happy", "sad", "angry", "excited", "worried", "anxious", 
            "proud", "disappointed", "grateful", "frustrated", "hopeful",
            "overwhelmed", "confident", "nervous", "relieved", "stressed",
            "joyful", "depressed", "optimistic", "pessimistic", "calm",
            "energetic", "tired", "motivated", "discouraged", "peaceful"
        ]
        
        text_lower = text.lower()
        found_keywords = [
            keyword for keyword in emotional_keywords 
            if keyword in text_lower
        ]
        
        return found_keywords[:10]  # Return top 10 emotional keywords
    
    def get_samo_config(self) -> Dict[str, Any]:
        """Get current SAMO configuration."""
        return self.config
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information including SAMO-specific details."""
        if not self.summarizer:
            return {"error": "Summarizer not initialized"}
        
        base_info = self.summarizer.get_model_info()
        base_info["samo_optimizations"] = self.config["samo_optimizations"]
        base_info["config_source"] = self.config_path
        
        return base_info

def create_samo_t5_summarizer(config_path: Optional[str] = None) -> SAMOT5Summarizer:
    """Create SAMO-optimized T5 summarizer.
    
    Args:
        config_path: Path to SAMO T5 configuration file
        
    Returns:
        Configured SAMOT5Summarizer instance
    """
    return SAMOT5Summarizer(config_path)

def test_samo_t5_summarizer():
    """Test SAMO T5 summarizer with sample journal entries."""
    logger.info("Testing SAMO T5 summarizer...")
    
    summarizer = create_samo_t5_summarizer()
    
    # Sample journal entries
    test_entries = [
        """Today was such a rollercoaster of emotions. I started the morning feeling anxious about my job interview, but I tried to stay positive. The interview actually went really well - I felt confident and articulate. The interviewer seemed impressed with my experience. After that, I met up with Sarah for coffee and we talked about everything that's been going on in our lives. She's been struggling with her relationship, and I tried to be supportive. By evening, I was exhausted but also proud of myself for handling a stressful day so well. I'm learning to trust myself more and not overthink everything.""",
        
        """Had a difficult conversation with mom today about dad's health. The doctors want to run more tests, and we're all worried. I hate feeling so helpless when someone I love is suffering. But I'm grateful that our family is pulling together during this time. My sister and I are planning to visit next weekend to help out. Sometimes I wonder if I'm strong enough to handle these kinds of challenges, but I know I have to be there for the people who matter most. Love really is everything."""
    ]
    
    for i, entry in enumerate(test_entries, 1):
        logger.info("\n--- Journal Entry %d ---", i)
        result = summarizer.summarize_journal_entry(entry)
        
        logger.info("Summary: %s", result["summary"])
        logger.info("Confidence: %.3f", result["confidence"])
        logger.info("Emotional keywords: %s", result.get("emotional_keywords", []))
        logger.info("SAMO metadata: %s", result["samo_metadata"])
    
    # Test batch processing
    logger.info("\n--- Batch Processing Test ---")
    batch_results = summarizer.summarize_batch_journal_entries(test_entries)
    logger.info("Processed %d entries in batch", len(batch_results))
    
    # Show configuration
    config = summarizer.get_samo_config()
    logger.info("SAMO Configuration: %s", config["samo_optimizations"])
    
    logger.info("✅ SAMO T5 summarizer test complete!")

if __name__ == "__main__":
    test_samo_t5_summarizer()
