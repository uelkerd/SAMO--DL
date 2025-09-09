#!/usr/bin/env python3
"""SAMO Deep Learning - Text Summarization Module.

This module implements T5/BART-based summarization for extracting emotional core
from journal conversations and providing intelligent summaries for users.

Key Components:
- T5SummarizationModel: Core T5/BART implementation
- SummarizationDataset: Dataset processing for journal entries
- SummarizationTrainer: End-to-end training pipeline
- SummarizationAPI: FastAPI endpoints for Web Dev integration

Performance Targets:
- Summarization Quality: >4.0/5.0 human evaluation score
- Response Latency: <500ms for P95 requests
- ROUGE Score: >0.4 for extractive quality
"""

# Only import what's actually needed for the API
try:
    from .t5_summarizer import create_t5_summarizer
except ImportError:
    create_t5_summarizer = None

# Optional imports for training/development
try:
    from .dataset_loader import SummarizationDataset, create_summarization_loader
except ImportError:
    SummarizationDataset = None
    create_summarization_loader = None

try:
    from .training_pipeline import SummarizationTrainer, train_summarization_model
except ImportError:
    SummarizationTrainer = None
    train_summarization_model = None

__version__ = "0.1.0"
__author__ = "SAMO Deep Learning Team"

__all__ = [
    "create_t5_summarizer",
]
