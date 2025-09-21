#!/usr/bin/env python3
"""Command-line interface for SAMO training operations."""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _is_relative_to(path: Path, other: Path) -> bool:
    """Check if path is relative to other (Python 3.8 compatible)."""
    try:
        path.resolve().relative_to(other.resolve())
        return True
    except ValueError:
        return False


def main():
    """Main CLI entry point for training operations."""
    parser = argparse.ArgumentParser(
        description="SAMO Deep Learning Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  samo-train --help                    # Show this help message
  samo-train emotion                  # Train emotion detection model
  samo-train summarization            # Train text summarization model
  samo-train voice                    # Train voice processing model
  samo-train all                      # Train all models
        """,
    )

    parser.add_argument(
        "command",
        choices=["emotion", "summarization", "voice", "all"],
        help="Training command to execute",
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to training configuration file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Output directory for trained models (default: ./models)",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for training if available",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Starting SAMO training: {args.command}")

    try:
        if args.command == "emotion":
            train_emotion_model(args)
        elif args.command == "summarization":
            train_summarization_model(args)
        elif args.command == "voice":
            train_voice_model(args)
        elif args.command == "all":
            train_all_models(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


def train_emotion_model(args):
    """Train emotion detection model."""
    logger.info("Training emotion detection model...")

    try:
        # Import and run emotion training
        from src.models.emotion_detection.training_pipeline import run_emotion_training

        config = {
            "output_dir": args.output_dir,
            "use_gpu": args.gpu,
            "config_file": args.config,
        }

        run_emotion_training(config)
        logger.info("Emotion model training completed successfully")

    except ImportError:
        logger.warning("Emotion training pipeline not available, using fallback")
        # Fallback to a simple training script
        import subprocess

        script_path = (
            Path(__file__).parent.parent.parent
            / "scripts"
            / "training"
            / "minimal_working_training.py"
        )
        # Resolve to absolute path and validate it's within the project directory
        script_path = script_path.resolve()
        project_root = Path(__file__).parent.parent.parent.resolve()

        if (
            script_path.exists()
            and script_path.is_file()
            and _is_relative_to(script_path, project_root)
        ):
            cmd = [sys.executable, str(script_path)]
            if args.gpu:
                cmd.append("--gpu")
            subprocess.run(cmd, check=True)
        else:
            logger.error("No emotion training script found")
            sys.exit(1)


def train_summarization_model(args):
    """Train text summarization model."""
    logger.info("Training text summarization model...")

    try:
        # Import and run summarization training
        from src.models.summarization.training_pipeline import (
            run_summarization_training,
        )

        config = {
            "output_dir": args.output_dir,
            "use_gpu": args.gpu,
            "config_file": args.config,
        }

        run_summarization_training(config)
        logger.info("Summarization model training completed successfully")

    except ImportError:
        logger.warning("Summarization training pipeline not available")
        logger.info("Summarization training not implemented yet")


def train_voice_model(args):
    """Train voice processing model."""
    logger.info("Training voice processing model...")

    try:
        # Import and run voice training
        from src.models.voice_processing.training_pipeline import run_voice_training

        config = {
            "output_dir": args.output_dir,
            "use_gpu": args.gpu,
            "config_file": args.config,
        }

        run_voice_training(config)
        logger.info("Voice model training completed successfully")

    except ImportError:
        logger.warning("Voice training pipeline not available")
        logger.info("Voice training not implemented yet")


def train_all_models(args):
    """Train all available models."""
    logger.info("Training all models...")

    train_emotion_model(args)
    train_summarization_model(args)
    train_voice_model(args)

    logger.info("All model training completed")


if __name__ == "__main__":
    main()
