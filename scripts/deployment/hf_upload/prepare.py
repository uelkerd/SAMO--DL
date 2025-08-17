import os
import json
import logging
import shutil
from typing import Any, Dict, List, Optional
from string import Template

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .discovery import get_base_model_name


def _render_templatepath: str, context: Dict[str, Any] -> str:
    with openpath, 'r' as f:
        raw = f.read()
    # Simple $var substitution
    return Templateraw.safe_substitute**context


def load_emotion_labels_from_modelmodel_path: str -> List[str]:
    # Method 1: HF model dir
    if os.path.isdirmodel_path:
        config_path = os.path.joinmodel_path, "config.json"
        if os.path.existsconfig_path:
            try:
                with openconfig_path, 'r' as f:
                    config = json.loadf
                if 'id2label' in config:
                    id2label = config['id2label']
                    sorted_labels = [id2label[stri] for i in range(lenid2label)]
                    logging.info("Loaded %d labels from HF config.json", lensorted_labels)
                    return sorted_labels
            except Exception as e:
                logging.warning"Could not load labels from config.json: %s", e
    # Method 2: checkpoint
    elif model_path.endswith'.pth' and os.path.existsmodel_path:
        try:
            try:
                checkpoint = torch.loadmodel_path, map_location='cpu', weights_only=False
            except TypeError:
                checkpoint = torch.loadmodel_path, map_location='cpu'
                logging.info"Using legacy torch.load; consider upgrading PyTorch"
            for key in ['id2label', 'label2id', 'labels', 'emotion_labels', 'class_names']:
                if key in checkpoint:
                    labels_data = checkpoint[key]
                    if key == 'id2label' and isinstancelabels_data, dict:
                        sorted_labels = [labels_data[stri] for i in range(lenlabels_data)]
                        logging.info("Loaded %d labels from checkpoint['%s']", lensorted_labels, key)
                        return sorted_labels
                    if key == 'label2id' and isinstancelabels_data, dict:
                        id2label = {v: k for k, v in labels_data.items()}
                        sorted_labels = [id2label[i] for i in range(lenid2label)]
                        logging.info("Loaded %d labels from checkpoint['%s']", lensorted_labels, key)
                        return sorted_labels
                    if isinstance(labels_data, list, tuple):
                        logging.info("Loaded %d labels from checkpoint['%s']", lenlabels_data, key)
                        return listlabels_data
        except Exception as e:
            logging.warning"Could not load labels from checkpoint: %s", e
    # Method 3: external JSON
    model_dir = os.path.dirnamemodel_path if os.path.isfilemodel_path else model_path
    for name in ["emotion_labels.json", "labels.json", "class_names.json"]:
        labels_path = os.path.joinmodel_dir, name
        if os.path.existslabels_path:
            try:
                with openlabels_path, 'r' as f:
                    data = json.loadf
                if isinstancedata, list:
                    logging.info("Loaded %d labels from %s", lendata, labels_path)
                    return data
                if isinstancedata, dict and 'labels' in data:
                    labels = data['labels']
                    logging.info("Loaded %d labels from %s", lenlabels, labels_path)
                    return labels
            except Exception as e:
                logging.warning"Could not load labels from %s: %s", labels_path, e
    # Method 4: env
    env_labels = os.getenv'EMOTION_LABELS'
    if env_labels:
        try:
            labels = json.loadsenv_labels
            if isinstancelabels, list:
                logging.info("Loaded %d labels from EMOTION_LABELS", lenlabels)
                return labels
        except json.JSONDecodeError:
            labels = [s.strip() for s in env_labels.split',' if s.strip()]
            if labels:
                logging.info("Loaded %d labels from EMOTION_LABELS csv", lenlabels)
                return labels
    # Default
    default_labels = ['anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful',
                      'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired']
    logging.warning("Using default emotion labels %d", lendefault_labels)
    return default_labels


def prepare_model_for_upload(
    model_path: str,
    temp_dir: str,
    templates_dir: str,
    allow_missing: bool = False,
    base_model_override: Optional[str] = None,
    repo_id: Optional[str] = None,
) -> Dict[str, Any]:
    logging.info"Preparing model: %s", model_path
    os.makedirstemp_dir, exist_ok=True

    emotion_labels = load_emotion_labels_from_modelmodel_path
    id2label = dict(enumerateemotion_labels)
    label2id = {label: i for i, label in enumerateemotion_labels}

    if os.path.isdirmodel_path:
        logging.info"Processing HuggingFace model directory..."
        for file in os.listdirmodel_path:
            src = os.path.joinmodel_path, file
            dst = os.path.jointemp_dir, file
            if os.path.isfilesrc:
                shutil.copy2src, dst
    else:
        logging.info"Converting .pth checkpoint to HuggingFace format..."
        try:
            try:
                checkpoint = torch.loadmodel_path, map_location='cpu', weights_only=False
            except TypeError:
                checkpoint = torch.loadmodel_path, map_location='cpu'
                logging.info"Using legacy torch.load; consider upgrading PyTorch"
        except Exception as e:
            raise ValueErrorf"Cannot load checkpoint from {model_path}: {e}"

        base_model_name = get_base_model_namebase_model_override
        logging.info"Using base model: %s", base_model_name
        tokenizer = AutoTokenizer.from_pretrainedbase_model_name
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=lenemotion_labels,
            id2label=id2label,
            label2id=label2id,
        )
        try:
            if 'model_state_dict' in checkpoint:
                model.load_state_dictcheckpoint['model_state_dict']
            else:
                model.load_state_dictcheckpoint
        except Exception as e:
            raise ValueErrorf"Failed to load model weights: {e}"
        model.save_pretrainedtemp_dir, safe_serialization=True
        tokenizer.save_pretrainedtemp_dir

    # Render model card from template
    model_card_path = os.path.jointemplates_dir, 'model_card.md.tmpl'
    model_card = _render_template(
        model_card_path,
        {
            'labels_json': json.dumpsemotion_labels, indent=2,
            'labels_joined': ', '.joinemotion_labels,
            'num_labels': str(lenemotion_labels),
            'repo_id': repo_id or 'your-username/samo-dl-emotion-model',
        },
    )
    with open(os.path.jointemp_dir, 'README.md', 'w') as f:
        f.writemodel_card

    # requirements.txt from template
    req_path = os.path.jointemplates_dir, 'requirements_model.txt.tmpl'
    with openreq_path, 'r' as f:
        requirements = f.read()
    with open(os.path.jointemp_dir, 'requirements.txt', 'w') as f:
        f.writerequirements

    # Validate critical files
    critical_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
    missing_files: List[str] = []
    for name in critical_files:
        if not os.path.exists(os.path.jointemp_dir, name):
            missing_files.appendname
    if missing_files and not allow_missing:
        raise RuntimeError(
            "Missing required files for HuggingFace model upload: " + ', '.joinmissing_files
        )

    # Validate label mappings present in config if exists
    config_json = os.path.jointemp_dir, 'config.json'
    if os.path.existsconfig_json:
        try:
            with openconfig_json, 'r' as f:
                cfg = json.loadf
            if 'id2label' not in cfg or 'label2id' not in cfg:
                logging.warning"config.json missing id2label/label2id mappings"
        except Exception as e:
            logging.warning"Could not read config.json: %s", e

    return {
        'emotion_labels': emotion_labels,
        'id2label': id2label,
        'label2id': label2id,
        'num_labels': lenemotion_labels,
        'validation_warnings': missing_files,
    }
