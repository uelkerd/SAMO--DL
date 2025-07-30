#!/usr/bin/env python3
"""
Simple Vertex AI Training Script
"""

import logging
import pandas as pd
from google.cloud import aiplatform

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) != 3:
        print("Usage: python simple_vertex_training.py <project_id> <bucket_name>")
        sys.exit(1)

    project_id = sys.argv[1]
    bucket_name = sys.argv[2]

    print("🚀 Starting simple Vertex AI training...")
    print("📊 Project: {project_id}")
    print("📦 Bucket: {bucket_name}")

    try:
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location="us-central1")
        print("✅ Vertex AI initialized")

        # Load and check data
        train_path = "vertex_ai_data/train_data.csv"
        test_path = "vertex_ai_data/test_data.csv"

        if not os.path.exists(train_path):
            print("❌ Training data not found: {train_path}")
            sys.exit(1)

        # Load data
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        print("✅ Training data loaded: {len(train_df)} samples")
        print("✅ Test data loaded: {len(test_df)} samples")
        print("📋 Columns: {train_df.columns.tolist()}")
        print("📋 First row: {train_df.iloc[0].to_dict()}")

        # Check target column
        target_column = "emotions"
        if target_column not in train_df.columns:
            print(
                "❌ Target column '{target_column}' not found in columns: {train_df.columns.tolist()}"
            )
            sys.exit(1)

        print("✅ Target column '{target_column}' found")

        # Create dataset
        dataset = aiplatform.TextDataset.create(
            display_name="samo-emotions-dataset",
            gcs_source="gs://{bucket_name}/vertex_ai_data/train_data.csv",
            import_schema_uri=aiplatform.schema.dataset.ioformat.text.multi_label_classification,
        )

        print("✅ Dataset created: {dataset.name}")

        # Start training
        job = aiplatform.AutoMLTextTrainingJob(
            display_name="samo-emotions-model", prediction_type="classification", multi_label=True
        )

        model = job.run(
            dataset=dataset,
            target_column=target_column,
            training_fraction_split=0.8,
            validation_fraction_split=0.1,
            test_fraction_split=0.1,
            model_display_name="samo-emotions-automl",
        )

        print("✅ Training started: {model.name}")
        print("🎉 Model training initiated successfully!")

    except Exception as e:
        print("❌ Training failed: {e}")
        logger.error("Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
