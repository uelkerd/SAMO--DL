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

from .dataset_loader import SummarizationDataset, create_summarization_loader
from .t5_summarizer import T5SummarizationModel, create_t5_summarizer
from .training_pipeline import SummarizationTrainer, train_summarization_model

__version__ = "0.1.0"
__author__ = "SAMO Deep Learning Team"

__all__ = [
    "SummarizationDataset",
    "SummarizationTrainer",
    "T5SummarizationModel",
    "create_summarization_loader",
    "create_t5_summarizer",
    "train_summarization_model"
]
