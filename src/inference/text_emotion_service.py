from __future__ import annotations

import os
import logging
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class EmotionService:
    """Abstract emotion classification service interface."""

    def classify(self, texts: Union[str, List[str]]) -> List[List[Dict[str, Any]]]:
        raise NotImplementedError


class HFEmotionService(EmotionService):
    """Hugging Face transformers pipeline-backed emotion classifier."""

    def __init__(
        self,
        model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        hf_token_env: str = "HF_TOKEN",
    ) -> None:
        self.model_name = model_name
        self.hf_token_env = hf_token_env
        self._pipeline = None
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline  # type: ignore
        except Exception as e:  # pragma: no cover
            logger.error("Failed to import transformers.pipeline: %s", e)
            raise

        token = os.environ.get(self.hf_token_env)
        kwargs: Dict[str, Any] = {
            "task": "text-classification",
            "model": self.model_name,
            "return_all_scores": True,
        }
        if token:
            kwargs["token"] = token
        # Truncation is passed during call; pipeline loads lazily.
        self._pipeline = pipeline(**kwargs)
        logger.info("HFEmotionService loaded model: %s", self.model_name)

    def classify(self, texts: Union[str, List[str]]) -> List[List[Dict[str, Any]]]:
        if isinstance(texts, str):
            inputs = [texts]
        else:
            inputs = texts
        if self._pipeline is None:
            self._ensure_loaded()
        # Ensure non-empty strings and minimal sanitization at this layer.
        cleaned = [t if isinstance(t, str) else str(t) for t in inputs]
        results = self._pipeline(cleaned, truncation=True)  # type: ignore
        # The HF pipeline returns a list[ list[ {label, score} ] ] for return_all_scores=True
        return results  # type: ignore