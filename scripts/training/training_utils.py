#!/usr/bin/env python3
"""
Simple Training Helper Utilities for SAMO Deep Learning

This module provides minimal, focused utilities for training scripts.
Keeps scope small and focused on common training tasks.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json


def setup_training_logging(log_file: str = "training.log") -> logging.Logger:
    """Setup basic training logging configuration.

    Args:
        log_file: Path to log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("training")

    # Return early if logger is already configured (idempotent)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Ensure log directory exists
    import os
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # File handler with explicit encoding
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Prevent duplicate console output via root handlers
    logger.propagate = False

    return logger


def save_training_config(config: Dict[str, Any], output_dir: str) -> Path:
    """Save training configuration to JSON file.

    Args:
        config: Training configuration dictionary
        output_dir: Directory to save config

    Returns:
        Path to saved config file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config_file = output_path / "training_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, default=str)

    return config_file


def load_training_config(config_file: str) -> Dict[str, Any]:
    """Load training configuration from JSON file.

    Args:
        config_file: Path to config file

    Returns:
        Training configuration dictionary

    Raises:
        ValueError: If config_file path is not safe
        FileNotFoundError: If config file doesn't exist
    """
    # Security: Validate file path to prevent path traversal attacks
    config_path = Path(config_file)

    # Ensure path is safe (no parent directory traversal)
    if ".." in str(config_path) or config_path.is_absolute():
        raise ValueError(f"Invalid config file path: {config_file}")

    # Ensure file exists and is a file
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not config_path.is_file():
        raise ValueError(f"Path is not a file: {config_file}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def create_output_dirs(base_dir: str, experiment_name: str) -> Dict[str, Path]:
    """Create standard output directory structure for training.

    Args:
        base_dir: Base directory for outputs
        experiment_name: Name of the experiment

    Returns:
        Dictionary of created directory paths
    """
    base_path = Path(base_dir) / experiment_name

    dirs = {
        "checkpoints": base_path / "checkpoints",
        "logs": base_path / "logs",
        "models": base_path / "models",
        "results": base_path / "results",
        "configs": base_path / "configs"
    }

    # Create directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def get_gpu_info() -> Dict[str, Any]:
    """Get basic GPU information for training setup.

    Returns:
        Dictionary with GPU information
    """
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "available": True,
                "count": torch.cuda.device_count(),
                "current": torch.cuda.current_device(),
                "name": torch.cuda.get_device_name(0),
                "memory_total": torch.cuda.get_device_properties(0).total_memory / 1e9
            }
        return {"available": False}
    except ImportError:
        return {"available": False, "error": "PyTorch not available"}


def validate_training_data(
    data_path: str, 
    expected_columns: list, 
    logger: Optional[logging.Logger] = None
) -> bool:
    """Basic validation of training data structure.

    Args:
        data_path: Path to training data file
        expected_columns: List of expected column names
        logger: Optional logger instance, falls back to root logger if not provided
    Returns:
        True if validation passes, False otherwise

    Raises:
        ValueError: If data_path is not safe
    """
    # Use provided logger or fall back to root logger
    logger = logger or logging.getLogger(__name__)

    # Security: Validate file path to prevent path traversal attacks
    data_path_obj = Path(data_path)

    # Ensure path is safe (no parent directory traversal)
    if ".." in str(data_path_obj) or data_path_obj.is_absolute():
        raise ValueError(f"Invalid data file path: {data_path}")

    try:
        import pandas as pd
        df = pd.read_csv(data_path_obj)

        # Check if all expected columns exist
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            logger.warning("Missing columns: %s", missing_columns)
            return False

        # Check if data is not empty
        if len(df) == 0:
            logger.warning("Training data is empty")
            return False

        return True

    except Exception as e:
        logger.error("Data validation failed: %s", e)
        return False
