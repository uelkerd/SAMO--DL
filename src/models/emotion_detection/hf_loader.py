#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class HFEmotionDetector:
    model: AutoModelForSequenceClassification
    tokenizer: AutoTokenizer
    id2label: Dict[int, str]
    multi_label: bool

    def predict(self, text: str, threshold: float = 0.5) -> Dict:
        if not text:
            return {}
        self.model.eval()
        with torch.no_grad():
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding=False,
                max_length=512,
                return_tensors="pt",
            )
            outputs = self.model(**encoded)
            logits = outputs.logits
            if self.multi_label:
                probs = torch.sigmoid(logits)[0].cpu().tolist()
            else:
                probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        # Map to labels
        emotions = {self.id2label.get(i, str(i)): float(p) for i, p in enumerate(probs)}
        if emotions:
            primary_label, confidence = max(emotions.items(), key=lambda kv: kv[1])
        else:
            primary_label, confidence = "neutral", 1.0
        if confidence >= 0.75:
            intensity = "high"
        elif confidence >= 0.4:
            intensity = "moderate"
        else:
            intensity = "low"
        return {
            "emotions": emotions,
            "primary_emotion": primary_label,
            "confidence": float(confidence),
            "emotional_intensity": intensity,
        }


def load_hf_emotion_model(model_id: str, token: Optional[str] = None, force_multi_label: Optional[bool] = None) -> HFEmotionDetector:
    cfg = AutoConfig.from_pretrained(model_id, token=token)
    tok = AutoTokenizer.from_pretrained(model_id, token=token, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id, token=token)

    id2label = getattr(cfg, "id2label", None) or {i: str(i) for i in range(cfg.num_labels)}
    # Determine multi-label vs multi-class
    if force_multi_label is not None:
        multi_label = bool(force_multi_label)
    else:
        problem_type = getattr(cfg, "problem_type", None)
        multi_label = problem_type == "multi_label_classification"
    return HFEmotionDetector(model=mdl, tokenizer=tok, id2label=id2label, multi_label=multi_label)