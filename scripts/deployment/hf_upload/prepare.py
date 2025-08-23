import json
import logging
import os
import shutil
from string import Template
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .discovery import get_base_model_name


def _render_template(path: str, context: Dict[str, Any]) -> str:
    with open(path) as f:
        raw = f.read()
    # Simple $var substitution
    return Template(raw).safe_substitute(**context)


def load_emotion_labels_from_model(model_path: str) -> List[str]:
    # Method 1: HF model dir
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    config = json.load(f)
                if "id2label" in config:
                    id2label = config["id2label"]
                    sorted_labels = [id2label[str(i)] for i in range(len(id2label))]
                    logging.info(
                        "Loaded %d labels from HF config.json", len(sorted_labels)
                    )
                    return sorted_labels
            except Exception as e:
                logging.warning("Could not load labels from config.json: %s", e)
    # Method 2: checkpoint
    elif model_path.endswith(".pth") and os.path.exists(model_path):
        try:
            try:
                checkpoint = torch.load(
                    model_path, map_location="cpu", weights_only=False
                )
            except TypeError:
                checkpoint = torch.load(model_path, map_location="cpu")
                logging.info("Using legacy torch.load; consider upgrading PyTorch")
            for key in [
                "id2label",
                "label2id",
                "labels",
                "emotion_labels",
                "class_names",
            ]:
                if key in checkpoint:
                    labels_data = checkpoint[key]
                    if key == "id2label" and isinstance(labels_data, dict):
                        sorted_labels = [
                            labels_data[str(i)] for i in range(len(labels_data))
                        ]
                        logging.info(
                            "Loaded %d labels from checkpoint['%s']",
                            len(sorted_labels),
                            key,
                        )
                        return sorted_labels
                    if key == "label2id" and isinstance(labels_data, dict):
                        id2label = {v: k for k, v in labels_data.items()}
                        sorted_labels = [id2label[i] for i in range(len(id2label))]
                        logging.info(
                            "Loaded %d labels from checkpoint['%s']",
                            len(sorted_labels),
                            key,
                        )
                        return sorted_labels
                    if isinstance(labels_data, (list, tuple)):
                        logging.info(
                            "Loaded %d labels from checkpoint['%s']",
                            len(labels_data),
                            key,
                        )
                        return list(labels_data)
        except Exception as e:
            logging.warning("Could not load labels from checkpoint: %s", e)
    # Method 3: external JSON
    model_dir = (
        os.path.dirname(model_path) if os.path.isfile(model_path) else model_path
    )
    for name in ["emotion_labels.json", "labels.json", "class_names.json"]:
        labels_path = os.path.join(model_dir, name)
        if os.path.exists(labels_path):
            try:
                with open(labels_path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    logging.info("Loaded %d labels from %s", len(data), labels_path)
                    return data
                if isinstance(data, dict) and "labels" in data:
                    labels = data["labels"]
                    logging.info("Loaded %d labels from %s", len(labels), labels_path)
                    return labels
            except Exception as e:
                logging.warning("Could not load labels from %s: %s", labels_path, e)
    # Method 4: env
    env_labels = os.getenv("EMOTION_LABELS")
    if env_labels:
        try:
            labels = json.loads(env_labels)
            if isinstance(labels, list):
                logging.info("Loaded %d labels from EMOTION_LABELS", len(labels))
                return labels
        except json.JSONDecodeError:
            labels = [s.strip() for s in env_labels.split(",") if s.strip()]
            if labels:
                logging.info("Loaded %d labels from EMOTION_LABELS (csv)", len(labels))
                return labels
    # Default
    default_labels = [
        "anxious",
        "calm",
        "content",
        "excited",
        "frustrated",
        "grateful",
        "happy",
        "hopeful",
        "overwhelmed",
        "proud",
        "sad",
        "tired",
    ]
    logging.warning("Using default emotion labels (%d)", len(default_labels))
    return default_labels


def prepare_model_for_upload(
    model_path: str,
    temp_dir: str,
    templates_dir: str,
    allow_missing: bool = False,
    base_model_override: Optional[str] = None,
    repo_id: Optional[str] = None,
) -> Dict[str, Any]:
    logging.info("Preparing model: %s", model_path)
    os.makedirs(temp_dir, exist_ok=True)

    emotion_labels = load_emotion_labels_from_model(model_path)
    id2label = dict(enumerate(emotion_labels))
    label2id = {label: i for i, label in enumerate(emotion_labels)}

    if os.path.isdir(model_path):
        logging.info("Processing HuggingFace model directory...")
        for file in os.listdir(model_path):
            src = os.path.join(model_path, file)
            dst = os.path.join(temp_dir, file)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
    else:
        logging.info("Converting .pth checkpoint to HuggingFace format...")
        try:
            try:
                checkpoint = torch.load(
                    model_path, map_location="cpu", weights_only=False
                )
            except TypeError:
                checkpoint = torch.load(model_path, map_location="cpu")
                logging.info("Using legacy torch.load; consider upgrading PyTorch")
        except Exception as e:
            raise ValueError(f"Cannot load checkpoint from {model_path}: {e}")

        base_model_name = get_base_model_name(base_model_override)
        logging.info("Using base model: %s", base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=len(emotion_labels),
            id2label=id2label,
            label2id=label2id,
        )
        try:
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            raise ValueError(f"Failed to load model weights: {e}")
        model.save_pretrained(temp_dir, safe_serialization=True)
        tokenizer.save_pretrained(temp_dir)

    # Render model card from template
    model_card_path = os.path.join(templates_dir, "model_card.md.tmpl")
    model_card = _render_template(
        model_card_path,
        {
            "labels_json": json.dumps(emotion_labels, indent=2),
            "labels_joined": ", ".join(emotion_labels),
            "num_labels": str(len(emotion_labels)),
            "repo_id": repo_id or "your-username/samo-dl-emotion-model",
        },
    )
    with open(os.path.join(temp_dir, "README.md"), "w") as f:
        f.write(model_card)

    # requirements.txt from template
    req_path = os.path.join(templates_dir, "requirements_model.txt.tmpl")
    with open(req_path) as f:
        requirements = f.read()
    with open(os.path.join(temp_dir, "requirements.txt"), "w") as f:
        f.write(requirements)

    # Validate critical files
    critical_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    missing_files: List[str] = []
    for name in critical_files:
        if not os.path.exists(os.path.join(temp_dir, name)):
            missing_files.append(name)
    if missing_files and not allow_missing:
        raise RuntimeError(
            "Missing required files for HuggingFace model upload: "
            + ", ".join(missing_files)
        )

    # Validate label mappings present in config if exists
    config_json = os.path.join(temp_dir, "config.json")
    if os.path.exists(config_json):
        try:
            with open(config_json) as f:
                cfg = json.load(f)
            if "id2label" not in cfg or "label2id" not in cfg:
                logging.warning("config.json missing id2label/label2id mappings")
        except Exception as e:
            logging.warning("Could not read config.json: %s", e)

    return {
        "emotion_labels": emotion_labels,
        "id2label": id2label,
        "label2id": label2id,
        "num_labels": len(emotion_labels),
        "validation_warnings": missing_files,
    }
