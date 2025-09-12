"""SAMO Deep Learning - Emotion Detection Module.

This module implements the core emotion detection pipeline using BERT fine-tuned
on the GoEmotions dataset for 27-category emotion classification.

Core Components:
- dataset_loader: GoEmotions data loading and preprocessing
- bert_classifier: BERT-based emotion classification model
- training_pipeline: Model training orchestration
- evaluation_metrics: Emotion-specific evaluation metrics
"""

__version__ = "0.1.0"
