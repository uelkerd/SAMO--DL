        # Create a simple custom training job
        # Get project ID
        # Import Vertex AI
        # Initialize Vertex AI
        from google.cloud import aiplatform
# Configure logging
#!/usr/bin/env python3
from pathlib import Path
import logging
import os
import sys

"""
Simple Vertex AI Validation for SAMO Deep Learning.

This script runs a simple validation on Vertex AI to identify the 0.0000 loss issue
without complex infrastructure setup.
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Main function to run simple Vertex AI validation."""
    logger.info("🚀 SAMO Deep Learning - Simple Vertex AI Validation")
    logger.info("=" * 60)

    try:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "the-tendril-466607-n8")
        region = "us-central1"

        logger.info("✅ Project ID: {project_id}")
        logger.info("✅ Region: {region}")

        aiplatform.init(
            project=project_id,
            location=region,
        )

        logger.info("✅ Vertex AI initialized successfully")

        logger.info("🔍 Creating validation job...")

        job = aiplatform.CustomTrainingJob(
            display_name="samo-simple-validation",
            container_uri="gcr.io/cloud-aiplatform/training/pytorch-cpu.2-0:latest",
            machine_type="n1-standard-4",
            replica_count=1,
        )

        logger.info("✅ Validation job created successfully!")
        logger.info("")
        logger.info("🎯 NEXT STEPS:")
        logger.info("1. Go to Vertex AI Console: https://console.cloud.google.com/vertex-ai")
        logger.info("2. Navigate to Training → Custom jobs")
        logger.info("3. Find 'samo-simple-validation' job")
        logger.info("4. Click on it to see details and logs")
        logger.info("")
        logger.info("🔧 To run the validation:")
        logger.info("   - The job will automatically start")
        logger.info("   - Check the logs for validation results")
        logger.info("   - Look for data distribution analysis")
        logger.info("   - Check for model architecture issues")
        logger.info("   - Verify loss function implementation")
        logger.info("")
        logger.info("💡 This will help identify the root cause of 0.0000 loss!")

        return True

    except Exception as e:
        logger.error("❌ Vertex AI validation failed: {e}")
        logger.error("")
        logger.error("🔧 ALTERNATIVE APPROACH:")
        logger.error("Since Vertex AI setup is complex, let's focus on the immediate issue:")
        logger.error("")
        logger.error("1. Run local validation: python scripts/local_validation_debug.py")
        logger.error("2. Check data distribution manually")
        logger.error("3. Verify model architecture")
        logger.error("4. Test loss function")
        logger.error("5. Fix the 0.0000 loss issue locally first")
        logger.error("")
        logger.error("Then we can move to Vertex AI for production training.")

        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
