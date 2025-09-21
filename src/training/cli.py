#!/usr/bin/env python3
"""Training CLI stub for SAMO - Training moved to separate repository."""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Training CLI entry point - redirects to separate training repository."""
    parser = argparse.ArgumentParser(
        description="SAMO Deep Learning Training CLI (Deprecated)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Note: Training functionality has been moved to a separate repository.
This CLI is kept for compatibility but no longer performs training.

For training operations, please use the dedicated training repository.
        """,
    )

    parser.add_argument(
        "command",
        choices=["emotion", "summarization", "voice", "all"],
        help="Training command (deprecated - no longer functional)",
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to training configuration file (deprecated)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Output directory for trained models (deprecated)",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for training (deprecated)",
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

    logger.warning(
        "Training functionality has been moved to a separate repository. "
        "This CLI no longer performs actual training operations."
    )
    logger.info(f"Requested command: {args.command}")
    logger.info(
        "Please use the dedicated training repository for actual training operations."
    )

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
