import json
import logging
import os
from string import Template
from typing import Any, Dict


def _read(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)


def update_deployment_config(repo_id: str, model_info: Dict[str, Any], templates_dir: str) -> None:
    logging.info("Writing deployment configuration files (config-driven)")

    # Create custom model config JSON (single source of truth)
    cfg = {
        "model_name": repo_id,
        "model_type": "custom_trained",
        "emotion_labels": model_info['emotion_labels'],
        "num_labels": model_info['num_labels'],
        "id2label": model_info['id2label'],
        "label2id": model_info['label2id'],
        "deployment_ready": True,
        "deployment_options": {
            "serverless_api": {
                "url": f"https://api-inference.huggingface.co/models/{repo_id}",
                "cost": "free",
                "best_for": "development_testing",
                "cold_starts": True,
                "rate_limits": True,
            },
            "inference_endpoints": {
                "setup_url": "https://ui.endpoints.huggingface.co/",
                "cost": "paid_per_usage",
                "best_for": "production",
                "cold_starts": False,
                "consistent_latency": True,
            },
            "self_hosted": {
                "model_loading": "AutoModelForSequenceClassification.from_pretrained("{repo_id}')",
                "cost": "infrastructure_costs",
                "best_for": "maximum_control",
                "requires": ["transformers", "torch"],
            },
        },
    }
    _write("deployment/custom_model_config.json", json.dumps(cfg, indent=2))
    logging.info("Created deployment/custom_model_config.json")

    # Render env templates
    mapping = {"REPO_NAME": repo_id}
    for src, dst in [
        (".env.serverless.template.tmpl", ".env.serverless.template"),
        (".env.endpoints.template.tmpl", ".env.endpoints.template"),
        (".env.selfhosted.template.tmpl", ".env.selfhosted.template"),
    ]:
        raw = _read(os.path.join(templates_dir, src))
        content = Template(raw).safe_substitute(mapping)
        _write(dst, content)
        logging.info("Created %s", dst)
