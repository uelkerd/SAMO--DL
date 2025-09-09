import os
import sys
import logging
from typing import Optional, List, Tuple
import contextlib


def get_base_model_name(override: Optional[str] = None) -> str:
    if override:
        logging.info("Using base model from CLI: %s", override)
        return override
    base_model = os.getenv('BASE_MODEL_NAME')
    if base_model:
        logging.info("Using BASE_MODEL_NAME from environment: %s", base_model)
        return base_model
    default_model = "distilroberta-base"
    logging.info("Using default base model: %s", default_model)
    return default_model


def get_model_base_directory() -> str:
    env_base_dir = os.getenv('SAMO_DL_BASE_DIR') or os.getenv('MODEL_BASE_DIR')
    if env_base_dir:
        base_dir = os.path.expanduser(env_base_dir)
        if os.path.exists(base_dir):
            return os.path.join(base_dir, "deployment", "models")
        logging.warning("Environment base directory doesn't exist: %s", base_dir)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    search_dir = current_dir
    max_levels = 5

    for _ in range(max_levels):
        indicators = ['deployment', 'src']
        if all(os.path.exists(os.path.join(search_dir, indicator)) for indicator in indicators):
            return os.path.join(search_dir, "deployment", "models")
        parent_dir = os.path.dirname(search_dir)
        if parent_dir == search_dir:
            break
        search_dir = parent_dir

    return os.path.join(os.getcwd(), "deployment", "models")


def is_interactive_environment() -> bool:
    non_interactive_indicators = [
        os.getenv('CI'),
        os.getenv('DOCKER_CONTAINER'),
        os.getenv('KUBERNETES_SERVICE_HOST'),
        os.getenv('JENKINS_URL'),
        not sys.stdin.isatty(),
    ]
    return not any(non_interactive_indicators)


def _calculate_directory_size(directory: str) -> int:
    total_size = 0
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            with contextlib.suppress(OSError):
                total_size += os.path.getsize(filepath)
    return total_size


def find_best_trained_model() -> Optional[str]:
    logging.info("Searching for trained models")
    primary_model_dir = get_model_base_directory()
    env_override = os.getenv('SAMO_DL_BASE_DIR') or os.getenv('MODEL_BASE_DIR')
    if env_override:
        logging.info("Using environment override: %s", env_override)
    logging.info("Primary search location: %s", primary_model_dir)

    if not os.path.exists(primary_model_dir):
        try:
            os.makedirs(primary_model_dir, exist_ok=True)
            logging.info("Created model directory: %s", primary_model_dir)
        except Exception as e:
            logging.warning("Could not create model directory: %s", e)

    model_patterns = [
        "best_domain_adapted_model.pth",
        "comprehensive_emotion_model_final",
        "emotion_model_ensemble_final",
        "emotion_model_specialized_final",
        "emotion_model_fixed_bulletproof_final",
        "domain_adapted_model",
        "emotion_model",
        "best_simple_model.pth",
        "best_focal_model.pth",
    ]

    model_search_paths: List[str] = []
    for pattern in model_patterns:
        model_search_paths.append(os.path.join(primary_model_dir, pattern))

    common_download_locations = [
        os.path.expanduser("~/Downloads"),
        os.path.expanduser("~/Desktop"),
        os.path.expanduser("~/Documents"),
    ]
    for download_dir in common_download_locations:
        for pattern in model_patterns:
            model_search_paths.append(os.path.join(download_dir, pattern))

    relative_locations = [
        "./deployment/models",
        "./models/checkpoints",
        "./",
    ]
    for rel_dir in relative_locations:
        for pattern in model_patterns:
            model_search_paths.append(os.path.join(rel_dir, pattern))

    checkpoint_patterns = [
        "focal_loss_best_model.pt",
        "simple_working_model.pt",
        "minimal_working_model.pt",
    ]
    for pattern in checkpoint_patterns:
        model_search_paths.append(os.path.join("./models/checkpoints", pattern))
        model_search_paths.append(os.path.join(primary_model_dir, pattern))

    found_models: List[Tuple[str, int, str]] = []

    for path in model_search_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                config_file = os.path.join(path, "config.json")
                tokenizer_candidates = [
                    os.path.join(path, "tokenizer.json"),
                    os.path.join(path, "tokenizer_config.json"),
                    os.path.join(path, "vocab.txt"),
                    os.path.join(path, "vocab.json"),
                ]
                has_config = os.path.exists(config_file)
                has_tokenizer = any(os.path.exists(p) for p in tokenizer_candidates)
                weight_files = [
                    os.path.join(path, f) for f in [
                        "pytorch_model.bin", "model.safetensors", "pytorch_model.safetensors",
                        "model.bin", "tf_model.h5", "flax_model.msgpack"
                    ] if os.path.exists(os.path.join(path, f))
                ]
                has_weights = len(weight_files) > 0
                if has_config and has_tokenizer and has_weights:
                    size = _calculate_directory_size(path)
                    found_models.append((path, size, "huggingface_dir"))
                    logging.info("Found complete HF model: %s (%s bytes)", path, f"{size:,}")
                elif has_config:
                    size = sum(
                        os.path.getsize(os.path.join(path, f))
                        for f in os.listdir(path)
                        if os.path.isfile(os.path.join(path, f))
                    )
                    missing_components = []
                    if not has_tokenizer:
                        missing_components.append("tokenizer")
                    if not has_weights:
                        missing_components.append("model weights")
                    logging.warning("Incomplete HF model: %s (%s bytes) missing: %s", path, f"{size:,}", ', '.join(missing_components))
            else:
                size = os.path.getsize(path)
                found_models.append((path, size, "model_file"))
                logging.info("Found model file: %s (%s bytes)", path, f"{size:,}")

    if not found_models:
        logging.error("No trained models found. Place your model in: %s", primary_model_dir)
        return None

    best_model = max(found_models, key=lambda x: x[1])
    logging.info("Selected best model: %s (%s bytes)", best_model[0], f"{best_model[1]:,}")
    return best_model[0]
