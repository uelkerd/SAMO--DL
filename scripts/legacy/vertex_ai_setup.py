#!/usr/bin/env python3
"""Vertex AI Setup Script for SAMO Emotion Detection Model"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging
import os
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from google.cloud import aiplatform
    from google.cloud import storage
    import subprocess
except ImportError as e:
    logger.error("Missing required dependencies: %s", e)
    logger.info("Please install: pip install google-cloud-aiplatform google-cloud-storage")
    sys.exit(1)


def setup_vertex_ai_environment(project_id: str, region: str = "us-central1") -> Dict[str, Any]:
    """Setup Vertex AI environment for emotion detection model training.

    Args:
        project_id: GCP project ID
        region: GCP region for Vertex AI

    Returns:
        Dictionary with setup status and configuration
    """
    try:
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=region)
        logger.info("✅ Vertex AI initialized successfully")

        # Verify project access
        storage_client = storage.Client(project=project_id)
        buckets = list(storage_client.list_buckets())
        logger.info("✅ GCP project access verified")

        return {
            "status": "success",
            "project_id": project_id,
            "region": region,
            "buckets_count": len(buckets)
        }

    except Exception as e:
        logger.error("❌ Vertex AI setup failed: %s", e)
        return {
            "status": "error",
            "error": str(e)
        }


def main():
    """Main function to setup Vertex AI environment."""
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        logger.error("❌ GOOGLE_CLOUD_PROJECT environment variable not set")
        sys.exit(1)

    logger.info("🚀 Setting up Vertex AI for project: %s", project_id)
    result = setup_vertex_ai_environment(project_id)

    if result["status"] == "success":
        logger.info("✅ Vertex AI setup completed successfully")
        logger.info("   Project: %s", result["project_id"])
        logger.info("   Region: %s", result["region"])
        logger.info("   Buckets: %s", result["buckets_count"])
    else:
        logger.error("❌ Vertex AI setup failed: %s", result["error"])
        sys.exit(1)


if __name__ == "__main__":
    main()
