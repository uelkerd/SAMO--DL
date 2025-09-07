#!/usr/bin/env python3
"""
T5-based Text Summarization for SAMO Deep Learning.

This module provides T5-based text summarization capabilities for
journal entries and other text content.
"""

import logging
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

# Configure logging
# G004: Logging f-strings temporarily allowed for development
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress tokenizer warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, module="transformers.tokenization_utils_base"
)


@dataclass
class SummarizationConfig:
    """Configuration for T5/BART summarization model."""

    model_name: str = "t5-small"  # Start with small model for development
    max_source_length: int = 512  # Input text length
    max_target_length: int = 128  # Summary length
    min_target_length: int = 30  # Minimum summary length
    num_beams: int = 4  # Beam search for quality
    length_penalty: float = 0.8  # Encourage shorter summaries
    early_stopping: bool = True  # Stop when all beams finished
    no_repeat_ngram_size: int = 2  # Avoid repetition
    temperature: float = 1.0  # Sampling temperature
    top_p: float = 0.9  # Nucleus sampling
    device: Optional[str] = None  # Auto-detect if None


class SummarizationDataset(Dataset):
    """Dataset for journal entry summarization."""

    def __init__(
        self,
        texts: List[str],
        summaries: List[str],
        tokenizer,
        max_source_length: int = 512,
        max_target_length: int = 128,
    ) -> None:
        """Initialize summarization dataset.

        Args:
            texts: List of input texts (journal entries)
            summaries: List of target summaries
            tokenizer: Tokenizer for the model
            max_source_length: Maximum input sequence length
            max_target_length: Maximum summary sequence length
        """
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        assert len(texts) == len(summaries), "Texts and summaries must have same length"
        logger.info(
            "Initialized SummarizationDataset with {len(texts)} examples",
            extra={"format_args": True},
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example."""
        text = self.texts[idx]
        summary = self.summaries[idx]

        if "t5" in self.tokenizer.name_or_path.lower():
            text = f"summarize: {text}"

        source_encoding = self.tokenizer(
            text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_encoding = self.tokenizer(
            summary,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
        }


class T5SummarizationModel(nn.Module):
    """T5/BART Summarization Model for emotional journal analysis."""

    def __init__(
        self, config: SummarizationConfig = None, model_name: Optional[str] = None
    ) -> None:
        """Initialize T5/BART summarization model.

        Args:
            config: Model configuration
            model_name: Override model name from config
        """
        super().__init__()

        self.config = config or SummarizationConfig()
        if model_name:
            self.config.model_name = model_name

        self.model_name = self.config.model_name

        if self.config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        logger.info(
            "Initializing {self.model_name} summarization model...", extra={"format_args": True}
        )

        # Use cache directory from environment
        cache_dir = os.environ.get('HF_HOME', '/app/models')

        if "bart" in self.model_name.lower():
            self.tokenizer = BartTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir)
            self.model = BartForConditionalGeneration.from_pretrained(self.model_name, cache_dir=cache_dir)
        elif "t5" in self.model_name.lower():
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, cache_dir=cache_dir)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, cache_dir=cache_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, cache_dir=cache_dir)

        self.model.to(self.device)

        self.num_parameters = self.model.num_parameters()
        logger.info(
            "Loaded {self.model_name} with {self.num_parameters:,} parameters",
            extra={"format_args": True},
        )
        logger.info("Model device: {self.device}", extra={"format_args": True})

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        return {
            "loss": outputs.loss if labels is not None else None,
            "logits": outputs.logits,
            "hidden_states": outputs.decoder_hidden_states
            if hasattr(outputs, "decoder_hidden_states")
            else None,
        }

    def generate_summary(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        length_penalty: Optional[float] = None,
        early_stopping: Optional[bool] = None,
        no_repeat_ngram_size: Optional[int] = None,
    ) -> str:
        """Generate summary for a single text.

        Args:
            text: Input text to summarize
            max_length: Override max summary length
            min_length: Override min summary length
            num_beams: Override beam search size
            length_penalty: Override length penalty
            early_stopping: Override early stopping
            no_repeat_ngram_size: Override n-gram repetition prevention

        Returns:
            Generated summary text
        """
        # Treat API max/min as new token targets for speed and stability on CPU
        max_length = max_length or self.config.max_target_length
        min_length = min_length or self.config.min_target_length
        # Reduce beams for larger models to avoid long runtimes on CPU
        default_beams = (
            2
            if (
                "base" in self.model_name.lower()
                or "large" in self.model_name.lower()
            )
            else self.config.num_beams
        )
        num_beams = num_beams or default_beams
        length_penalty = length_penalty or self.config.length_penalty
        early_stopping = (
            early_stopping if early_stopping is not None else self.config.early_stopping
        )
        no_repeat_ngram_size = no_repeat_ngram_size or self.config.no_repeat_ngram_size

        if "t5" in self.model_name.lower():
            text = f"summarize: {text}"

        inputs = self.tokenizer(
            text,
            max_length=self.config.max_source_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            summary_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_length,
                min_new_tokens=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                no_repeat_ngram_size=no_repeat_ngram_size,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=False,  # Use beam search, not sampling
            )

        summary = self.tokenizer.decode(
            summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        return summary.strip()

    def generate_batch_summaries(
        self, texts: List[str], batch_size: int = 4, **generation_kwargs
    ) -> List[str]:
        """Generate summaries for a batch of texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            **generation_kwargs: Additional generation arguments

        Returns:
            List of generated summaries
        """
        summaries = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            if "t5" in self.model_name.lower():
                batch_texts = [f"summarize: {text}" for text in batch_texts]

            inputs = self.tokenizer(
                batch_texts,
                max_length=self.config.max_source_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            # Reduce beams for larger models to avoid long runtimes on CPU
            default_beams = (
                2
                if (
                    "base" in self.model_name.lower()
                    or "large" in self.model_name.lower()
                )
                else self.config.num_beams
            )

            self.model.eval()
            with torch.no_grad():
                summary_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=generation_kwargs.get(
                        "max_length", self.config.max_target_length
                    ),
                    min_new_tokens=generation_kwargs.get(
                        "min_length", self.config.min_target_length
                    ),
                    num_beams=generation_kwargs.get(
                        "num_beams", default_beams
                    ),
                    length_penalty=generation_kwargs.get(
                        "length_penalty", self.config.length_penalty
                    ),
                    early_stopping=generation_kwargs.get(
                        "early_stopping", self.config.early_stopping
                    ),
                    no_repeat_ngram_size=generation_kwargs.get(
                        "no_repeat_ngram_size", self.config.no_repeat_ngram_size
                    ),
                    do_sample=False,
                )

            batch_summaries = self.tokenizer.batch_decode(
                summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            summaries.extend([s.strip() for s in batch_summaries])

        return summaries

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "total_parameters": self.num_parameters,
            "trainable_parameters": self.count_parameters(),
            "device": str(self.device),
            "max_source_length": self.config.max_source_length,
            "max_target_length": self.config.max_target_length,
            "min_target_length": self.config.min_target_length,
        }


def create_t5_summarizer(
    model_name: str = "t5-small",
    max_source_length: int = 512,
    max_target_length: int = 128,
    min_target_length: int = 30,
    device: Optional[str] = None,
) -> T5SummarizationModel:
    """Create T5/BART summarization model with specified configuration.

    Args:
        model_name: Model name (t5-small, t5-base, facebook/bart-base, etc.)
        max_source_length: Maximum input text length
        max_target_length: Maximum summary length
        min_target_length: Minimum summary length
        device: Device for model ('cuda', 'cpu', or None for auto)

    Returns:
        Configured T5SummarizationModel instance
    """
    config = SummarizationConfig(
        model_name=model_name,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        min_target_length=min_target_length,
        device=device,
    )

    model = T5SummarizationModel(config)
    logger.info("Created {model_name} summarization model", extra={"format_args": True})

    return model


def test_summarization_model() -> None:
    """Test the summarization model with sample journal entries."""
    logger.info("Testing T5 summarization model...")

    model = create_t5_summarizer("t5-small")

    test_texts = [
        """Today was such a rollercoaster of emotions. I started the morning feeling anxious about my job interview, but I tried to stay positive. The interview actually went really well - I felt confident and articulate. The interviewer seemed impressed with my experience. After that, I met up with Sarah for coffee and we talked about everything that's been going on in our lives. She's been struggling with her relationship, and I tried to be supportive. By evening, I was exhausted but also proud of myself for handling a stressful day so well. I'm learning to trust myself more and not overthink everything.""",
        """Had a difficult conversation with mom today about dad's health. The doctors want to run more tests, and we're all worried. I hate feeling so helpless when someone I love is suffering. But I'm grateful that our family is pulling together during this time. My sister and I are planning to visit next weekend to help out. Sometimes I wonder if I'm strong enough to handle these kinds of challenges, but I know I have to be there for the people who matter most. Love really is everything.""",
        """Work has been incredibly stressful lately. My boss keeps piling on more projects, and I'm starting to feel overwhelmed. I've been staying late almost every night this week. On the positive side, I finally finished that big presentation I've been working on for months. It felt amazing to see it come together. I think I need to have a conversation with my manager about workload balance. I love my job, but I also need to take care of my mental health. Maybe it's time to set some boundaries.""",
    ]

    logger.info(
        "Generating summaries for {len(test_texts)} journal entries...", extra={"format_args": True}
    )

    for _i, text in enumerate(test_texts, 1):
        model.generate_summary(text)

        logger.info("\n--- Journal Entry {i} ---", extra={"format_args": True})
        logger.info("Original ({len(text)} chars): {text[:100]}...", extra={"format_args": True})
        logger.info("Summary ({len(summary)} chars): {summary}", extra={"format_args": True})

    logger.info("\nTesting batch summarization...")
    batch_summaries = model.generate_batch_summaries(test_texts, batch_size=2)

    for _i, _summary in enumerate(batch_summaries, 1):
        logger.info("Batch Summary {i}: {summary}", extra={"format_args": True})

    model.get_model_info()
    logger.info("\nModel Info: {info}", extra={"format_args": True})

    logger.info("✅ T5 summarization model test complete!")


if __name__ == "__main__":
    test_summarization_model()
