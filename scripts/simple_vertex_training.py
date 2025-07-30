import os
import sys
#!/usr/bin/env python3
import logging
import pandas as pd
from google.cloud import aiplatform
# Configure logging
        # Initialize Vertex AI
        # Load and check data
        # Load data
        # Check target column
        # Create dataset
        # Start training



"""
Simple Vertex AI Training Script
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) != 3:
        logging.info("Usage: python simple_vertex_training.py <project_id> <bucket_name>")
        sys.exit(1)

    project_id = sys.argv[1]
    sys.argv[2]

    logging.info("🚀 Starting simple Vertex AI training...")
    logging.info("📊 Project: {project_id}")
    logging.info("📦 Bucket: {bucket_name}")

    try:
        aiplatform.init(project=project_id, location="us-central1")
        logging.info("✅ Vertex AI initialized")

        train_path = "vertex_ai_data/train_data.csv"
        test_path = "vertex_ai_data/test_data.csv"

        if not os.path.exists(train_path):
            logging.info("❌ Training data not found: {train_path}")
            sys.exit(1)

        train_df = pd.read_csv(train_path)
        pd.read_csv(test_path)

        logging.info("✅ Training data loaded: {len(train_df)} samples")
        logging.info("✅ Test data loaded: {len(test_df)} samples")
        logging.info("📋 Columns: {train_df.columns.tolist()}")
        logging.info("📋 First row: {train_df.iloc[0].to_dict()}")

        target_column = "emotions"
        if target_column not in train_df.columns:
            print(
                "❌ Target column '{target_column}' not found in columns: {train_df.columns.tolist()}"
            )
            sys.exit(1)

        logging.info("✅ Target column '{target_column}' found")

        dataset = aiplatform.TextDataset.create(
            display_name="samo-emotions-dataset",
            gcs_source="gs://{bucket_name}/vertex_ai_data/train_data.csv",
            import_schema_uri=aiplatform.schema.dataset.ioformat.text.multi_label_classification,
        )

        logging.info("✅ Dataset created: {dataset.name}")

        job = aiplatform.AutoMLTextTrainingJob(
            display_name="samo-emotions-model", prediction_type="classification", multi_label=True
        )

        job.run(
            dataset=dataset,
            target_column=target_column,
            training_fraction_split=0.8,
            validation_fraction_split=0.1,
            test_fraction_split=0.1,
            model_display_name="samo-emotions-automl",
        )

        logging.info("✅ Training started: {model.name}")
        logging.info("🎉 Model training initiated successfully!")

    except Exception as _:
        logging.info("❌ Training failed: {e}")
        logger.error("Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
