from __future__ import annotations

import os
import logging
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_MODEL_DIR = "/models/emotion-english-distilroberta-base"


class EmotionService:
    """Abstract emotion classification service interface."""

    def classify(self, texts: Union[str, List[str]]) -> List[List[Dict[str, Any]]]:
        """Classify one or many texts into emotion score distributions."""
        raise NotImplementedError


class HFEmotionService(EmotionService):
    """Hugging Face transformers pipeline-backed emotion classifier.

    Supports local-only loading via environment variables to avoid network access.
    """

    def __init__(
        self,
        model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        hf_token_env: str = "HF_TOKEN",
        model_dir_env: str = "EMOTION_MODEL_DIR",
        local_only_env: str = "EMOTION_LOCAL_ONLY",
    ) -> None:
        self.model_name = model_name
        self.hf_token_env = hf_token_env
        self.model_dir_env = model_dir_env
        self.local_only_env = local_only_env
        self._pipeline = None
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        """Load tokenizer/model pipeline (preferring local dir when configured)."""
        if self._pipeline is not None:
            return
        try:
            from transformers import (
                pipeline,
                AutoTokenizer,
                AutoModelForSequenceClassification,
            )  # type: ignore
        except Exception as e:  # pragma: no cover
            logger.error("Failed to import transformers components: %s", e)
            raise

        model_dir = os.environ.get(self.model_dir_env)
        local_only = os.environ.get(
            self.local_only_env, "1"
        ).strip() not in {"", "0", "false", "False"}

        if not model_dir and local_only:
            model_dir = DEFAULT_LOCAL_MODEL_DIR

        if model_dir and os.path.isdir(model_dir):
            # Load strictly from local directory
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir, local_files_only=True
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_dir, local_files_only=True
            )
            self._pipeline = pipeline(
                task="text-classification",
                model=model,
                tokenizer=tokenizer,
                return_all_scores=True,
            )
            logger.info("HFEmotionService loaded local model dir: %s", model_dir)
            return

        if local_only:
            # Local-only requested but directory missing → fail fast
            raise RuntimeError(
                "Local-only mode enabled but local model directory not found. "
                f"Expected at: {model_dir or DEFAULT_LOCAL_MODEL_DIR}. "
                f"Please place the model files locally or set {self.model_dir_env}."
            )

        # Fallback to remote model (dev only). Token optional.
        kwargs: Dict[str, Any] = {
            "task": "text-classification",
            "model": self.model_name,
            "return_all_scores": True,
        }
        if (token := os.environ.get(self.hf_token_env)):
            kwargs["token"] = token
        self._pipeline = pipeline(**kwargs)
        logger.info("HFEmotionService loaded remote model: %s", self.model_name)

    def classify(self, texts: Union[str, List[str]]) -> List[List[Dict[str, Any]]]:
        """Return list of per-text distributions [{label, score}, ...]."""
        inputs = [texts] if isinstance(texts, str) else texts
        if self._pipeline is None:
            self._ensure_loaded()
        cleaned = [t if isinstance(t, str) else str(t) for t in inputs]
        return self._pipeline(cleaned, truncation=True)  # type: ignore
