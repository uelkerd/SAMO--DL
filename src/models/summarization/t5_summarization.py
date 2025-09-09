#!/usr/bin/env python3
"""
T5 Summarization Module for SAMO-DL.

This module implements T5-based text summarization using Hugging Face transformers.
Provides extractive and abstractive summarization capabilities with confidence scoring.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import logging
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re

logger = logging.getLogger(__name__)

@dataclass
class SummarizationConfig:
    """Configuration for T5 summarization."""
    model_name: str = "t5-small"
    max_length: int = 512
    min_length: int = 50
    num_beams: int = 4
    early_stopping: bool = True
    device: Optional[str] = None
    do_sample: bool = False
    temperature: float = 1.0
    repetition_penalty: float = 1.2
    length_penalty: float = 1.0

class T5Summarizer:
    """T5-based text summarizer."""

    def __init__(self, config: Optional[SummarizationConfig] = None):
        """Initialize T5 summarizer."""
        self.config = config or SummarizationConfig()

        if self.config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        logger.info("Loading T5 model: %s", self.config.model_name)

        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype=(
                    torch.float16 if self.device.type == "cuda" else torch.float32
                )
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info("✅ T5 model loaded successfully on %s", self.device)
        except Exception as e:
            logger.error("❌ Failed to load T5 model: %s", e)
            raise RuntimeError(f"T5 model loading failed: {e}")

    def summarize(
        self, 
        text: str, 
        max_length: Optional[int] = None, 
        min_length: Optional[int] = None,
        num_beams: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate summary for input text using T5.

        Args:
            text: Input text to summarize
            max_length: Maximum summary length (overrides config)
            min_length: Minimum summary length (overrides config)
            num_beams: Number of beams for generation (overrides config)

        Returns:
            Dictionary containing summary, scores, and metadata
        """
        # Enhanced input validation
        word_count = len(text.split()) if text else 0
        if not text or word_count < 20:
            return {
                "summary": text.strip() if text and word_count >= 10 else "",
                "confidence": 0.0,
                "input_length": word_count,
                "summary_length": word_count if text and word_count >= 10 else 0,
                "processing_time": 0.0,
                "scores": {},
                "note": "Input too short for meaningful summarization" if text and word_count < 20 else "Empty input"
            }

        start_time = (
            torch.cuda.Event(enable_timing=True)
            if self.device.type == "cuda" else None
        )
        end_time = (
            torch.cuda.Event(enable_timing=True)
            if self.device.type == "cuda" else None
        )

        if start_time:
            start_time.record()

        # Preprocess text
        input_text = self._preprocess_text(text)
        input_ids = self.tokenizer.encode(
            f"summarize: {input_text}",
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True
        ).to(self.device)

        # Generation parameters
        gen_max_length = max_length or self.config.max_length
        gen_min_length = min_length or self.config.min_length
        gen_num_beams = num_beams or self.config.num_beams

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                max_length=gen_max_length,
                min_length=gen_min_length,
                num_beams=gen_num_beams,
                early_stopping=self.config.early_stopping,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                repetition_penalty=self.config.repetition_penalty,
                length_penalty=self.config.length_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode summary - T5 generates the full sequence, not just the new part
        # We need to extract only the summary part after "summarize:"
        full_output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Extract summary by finding the part after "summarize:"
        if "summarize:" in full_output:
            summary = full_output.split("summarize:")[-1].strip()
        else:
            # Fallback: if no "summarize:" prefix, use the full output
            summary = full_output.strip()

        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            processing_time = (
                start_time.elapsed_time(end_time) / 1000.0
            )  # ms to seconds
        else:
            processing_time = 0.0

        # Calculate confidence/quality scores
        scores = self._calculate_summary_scores(input_ids, generated_ids)

        result = {
            "summary": summary.strip(),
            "confidence": scores.get("confidence", 0.0),
            "input_length": len(text.split()),
            "summary_length": len(summary.split()),
            "processing_time": processing_time,
            "scores": scores,
            "input_text": (
                input_text[:200] + "..."
                if len(input_text) > 200 else input_text
            )
        }

        logger.info(
            "Summarization complete: %s → %s words",
            result['input_length'], result['summary_length']
        )
        return result

    def batch_summarize(
        self, 
        texts: List[str], 
        batch_size: int = 4
    ) -> List[Dict[str, Any]]:
        """Summarize multiple texts in batches."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = [self.summarize(text) for text in batch]
            results.extend(batch_results)
            logger.info("Processed batch %s: %s texts", i//batch_size + 1, len(batch))
        return results

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """Preprocess text for T5 summarization."""
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        return text

    def _calculate_summary_scores(self, input_ids, generated_ids) -> Dict[str, float]:
        """Calculate quality scores for generated summary."""
        with torch.no_grad():
            outputs = self.model(
                input_ids, 
                labels=generated_ids,
                return_dict=True
            )

            loss = outputs.loss.item() if outputs.loss is not None else float('inf')
            # Convert negative log likelihood to confidence (simplified)
            confidence = max(
                0.0, 1.0 - (loss / 5.0)
            )  # Normalize roughly

            # Calculate perplexity
            perplexity = (
                torch.exp(outputs.loss).item()
                if outputs.loss is not None else float('inf')
            )

        return {
            "confidence": confidence,
            "perplexity": perplexity,
            "loss": loss
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.config.model_name,
            "device": str(self.device),
            "max_length": self.config.max_length,
            "min_length": self.config.min_length,
            "num_beams": self.config.num_beams,
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature
        }

def create_t5_summarizer(
    model_name: str = "t5-small", 
    device: Optional[str] = None
) -> T5Summarizer:
    """Create T5 summarizer with specified configuration."""
    config = SummarizationConfig(model_name=model_name, device=device)
    summarizer = T5Summarizer(config)
    logger.info("Created T5 summarizer: %s", model_name)
    return summarizer

def test_t5_summarizer() -> None:
    """Test T5 summarizer."""
    logger.info("Testing T5 summarizer...")

    sample_text = """
    Artificial intelligence is transforming industries worldwide. Machine learning algorithms
    are being used in healthcare for diagnostics, in finance for fraud detection,
    and in
    transportation for autonomous vehicles. The rapid advancement of AI technology presents
    both opportunities and challenges for society as we navigate the ethical
    implications
    and workforce transformations that accompany this digital revolution.
    """.strip()

    summarizer = create_t5_summarizer()

    result = summarizer.summarize(sample_text)

    logger.info("✅ T5 summarizer test complete!")
    logger.info("Summary: %s", result['summary'])
    logger.info("Confidence: %.2f", result['confidence'])
    logger.info("Model info: %s", summarizer.get_model_info())

if __name__ == "__main__":
    test_t5_summarizer()
