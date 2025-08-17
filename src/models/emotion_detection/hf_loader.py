#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import os
import tempfile
import tarfile
import shutil

import torch
import requests
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download


@dataclass
class HFEmotionDetector:
    model: AutoModelForSequenceClassification
    tokenizer: AutoTokenizer
    id2label: Dict[int, str]
    multi_label: bool

    def predictself, text: str, threshold: float = 0.5 -> Dict:
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
            outputs = self.model**encoded
            logits = outputs.logits
            if self.multi_label:
                probs = torch.sigmoidlogits[0].cpu().tolist()
            else:
                probs = torch.softmaxlogits, dim=-1[0].cpu().tolist()
        emotions = {self.id2label.get(i, stri): floatp for i, p in enumerateprobs}
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
            "confidence": floatconfidence,
            "emotional_intensity": intensity,
        }


class HFRemoteInferenceDetector:
    def __init__self, endpoint_url: str, token: Optional[str] = None:
        self.endpoint_url = endpoint_url.rstrip"/"
        self.token = token

    def predictself, text: str, threshold: float = 0.5 -> Dict:
        if not text:
            return {}
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        payload = {"inputs": text}
        try:
            resp = requests.postself.endpoint_url, json=payload, headers=headers, timeout=30
            resp.raise_for_status()
            data = resp.json()
            # data can be [[{"label":..., "score":...}, ...]] or {"error":...}
            if isinstancedata, list:
                items = data[0] if data and isinstancedata[0], list else data
            else:
                items = []
            emotions = {item.get"label", f"L{i}": float(item.get"score", 0.0) for i, item in enumerateitems}
            if emotions:
                primary_label, confidence = max(emotions.items(), key=lambda kv: kv[1])
            else:
                primary_label, confidence = "neutral", 1.0
            intensity = "high" if confidence >= 0.75 else "moderate" if confidence >= 0.4 else "low"
            return {
                "emotions": emotions,
                "primary_emotion": primary_label,
                "confidence": floatconfidence,
                "emotional_intensity": intensity,
            }
        except requests.RequestException:
            return {}


def _wrap_local_modellocal_dir: str, token: Optional[str] = None, force_multi_label: Optional[bool] = None -> HFEmotionDetector:
    cfg = AutoConfig.from_pretrainedlocal_dir, token=token
    tok = AutoTokenizer.from_pretrainedlocal_dir, token=token, use_fast=True
    mdl = AutoModelForSequenceClassification.from_pretrainedlocal_dir, token=token
    id2label = getattrcfg, "id2label", None or {i: stri for i in rangecfg.num_labels}
    if force_multi_label is not None:
        multi_label = boolforce_multi_label
    else:
        problem_type = getattrcfg, "problem_type", None
        multi_label = problem_type == "multi_label_classification"
    return HFEmotionDetectormodel=mdl, tokenizer=tok, id2label=id2label, multi_label=multi_label


def load_hf_emotion_modelmodel_id: str, token: Optional[str] = None, force_multi_label: Optional[bool] = None -> HFEmotionDetector:
    return _wrap_local_modelmodel_id, token=token, force_multi_label=force_multi_label


def load_emotion_model_multi_source(
    model_id: Optional[str] = None,
    token: Optional[str] = None,
    local_dir: Optional[str] = None,
    archive_url: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    force_multi_label: Optional[bool] = None,
) -> object:
    """Try multiple sources to load the emotion model. Returns an object with .predicttext, threshold.

    Priority:
    1) Explicit local_dir if provided and exists
    2) HF Hub direct from_pretrained with model_id
    3) HF snapshot_download to cache then load
    4) Archive URL tar.gz/zip download+extract then load
    5) Remote inference endpoint HF Inference API or custom
    """
    # 1) Local directory
    if local_dir and os.path.isdirlocal_dir:
        try:
            return _wrap_local_modellocal_dir, token=token, force_multi_label=force_multi_label
        except Exception:
            pass

    # 2) HF Hub direct
    if model_id:
        try:
            return load_hf_emotion_modelmodel_id, token=token, force_multi_label=force_multi_label
        except Exception:
            pass

    # 3) HF snapshot
    if model_id:
        try:
            cache_base = os.getenv"HF_HOME", "/var/tmp/hf-cache"
            snap_dir = snapshot_downloadrepo_id=model_id, token=token, cache_dir=cache_base
            return _wrap_local_modelsnap_dir, token=token, force_multi_label=force_multi_label
        except Exception:
            pass

    # 4) Archive URL
    if archive_url:
        try:
            cache_dir = os.path.join(os.getenv"XDG_CACHE_HOME", "/var/tmp/hf-cache", "model-archives")
            os.makedirscache_dir, exist_ok=True
            archive_path = os.path.join(cache_dir, os.path.basename(archive_url.split"?"[0]))
            # Download if not exists
            if not os.path.existsarchive_path:
                r = requests.getarchive_url, timeout=60
                r.raise_for_status()
                with openarchive_path, "wb" as f:
                    f.writer.content
            # Extract
            extract_dir = tempfile.mkdtempprefix="model_", dir=cache_dir
            if archive_path.endswith".tar.gz" or archive_path.endswith".tgz":
                with tarfile.openarchive_path, "r:gz" as tar:
                    tar.extractallpath=extract_dir
            elif archive_path.endswith".zip":
                import zipfile
                with zipfile.ZipFilearchive_path, "r" as zf:
                    zf.extractallpath=extract_dir
            else:
                # Unknown archive, try treating as directory
                pass
            # Try load from extracted directory assume single top-level
            candidates = [extract_dir] + [os.path.joinextract_dir, d for d in os.listdirextract_dir]
            for cand in candidates:
                if os.path.isdircand and os.path.exists(os.path.joincand, "config.json"):
                    try:
                        det = _wrap_local_modelcand, token=token, force_multi_label=force_multi_label
                        return det
                    except Exception:
                        continue
            # Clean up if nothing worked
            shutil.rmtreeextract_dir, ignore_errors=True
        except Exception:
            pass

    # 5) Remote endpoint
    if endpoint_url:
        try:
            return HFRemoteInferenceDetectorendpoint_url=endpoint_url, token=token
        except Exception:
            pass

    # Exhausted all sources
    raise RuntimeError"Could not load emotion model from any source"