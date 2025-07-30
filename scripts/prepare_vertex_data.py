                # Convert to comma-separated string
                # Find similar samples to base synthetic data on
                # Oversample to reach target
            # Analyze current state
            # Convert to DataFrame
            # Convert to Vertex AI format
            # Find samples with this emotion
            # Format: "text","emotion1,emotion2,emotion3"
            # Split data
            # Write to files
        # Analyze emotion distribution
        # Balance dataset
        # Cleanup
        # Convert to Vertex AI format
        # Create temporary files
        # Data rows
        # Display analysis
        # Get target counts for each emotion (aim for at least 2% representation)
        # GoEmotions emotion labels (27 emotions)
        # Header
        # Identify class imbalance issues
        # Identify issues
        # Initialize data preparation
        # Load and analyze data
        # Re-analyze balanced dataset
        # Save locally
        # Save metadata
        # Strategy 1: Oversample minority classes
        # Strategy 2: Add synthetic samples for very rare emotions
        # Upload test file
        # Upload to GCS
        # Upload to GCS
        # Upload train file
    # Check if data file exists
# Configure logging
#!/usr/bin/env python3
from collections import Counter
from google.cloud import storage
from sklearn.model_selection import train_test_split
from typing import Any
import json
import logging
import os
import pandas as pd
import sys
import tempfile



"""
SAMO GoEmotions Data Preparation for Vertex AI AutoML
Converts your current dataset to Vertex AI format and addresses F1 score issues
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SAMOVertexDataPreparation:
    """Handles data preparation for Vertex AI AutoML training"""

    def __init__(self, project_id: str, bucket_name: str):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)

        self.emotion_labels = [
            "admiration",
            "amusement",
            "anger",
            "annoyance",
            "approval",
            "caring",
            "confusion",
            "curiosity",
            "desire",
            "disappointment",
            "disapproval",
            "disgust",
            "embarrassment",
            "excitement",
            "fear",
            "gratitude",
            "grie",
            "joy",
            "love",
            "nervousness",
            "optimism",
            "pride",
            "realization",
            "relie",
            "remorse",
            "sadness",
            "surprise",
            "neutral",
        ]

        logger.info("Initialized SAMO Vertex Data Preparation")
        logger.info("Project: {project_id}")
        logger.info("Bucket: {bucket_name}")
        logger.info("Emotion labels: {len(self.emotion_labels)} emotions")

    def load_and_analyze_data(self, data_path: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Load and analyze the current dataset"""
        logger.info("Loading data from: {data_path}")

        try:
            with open(data_path, encoding="utf-8") as f:
                data = json.load(f)

            logger.info("Loaded {len(data)} entries")

            df = pd.DataFrame(data)

            analysis = self._analyze_dataset(df)

            return df, analysis

        except Exception:
            logger.error("Error loading data: {e}")
            raise

    def _analyze_dataset(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze dataset characteristics and identify issues"""
        logger.info("Analyzing dataset...")

        analysis = {
            "total_samples": len(df),
            "emotion_distribution": {},
            "class_imbalance": {},
            "avg_emotions_per_sample": 0,
            "issues": [],
        }

        emotion_counts = Counter()
        total_emotions = 0

        for _, row in df.iterrows():
            emotions = row.get("emotions", [])
            if isinstance(emotions, list):
                for emotion in emotions:
                    emotion_counts[emotion] += 1
                total_emotions += len(emotions)

        analysis["emotion_distribution"] = dict(emotion_counts)
        analysis["avg_emotions_per_sample"] = total_emotions / len(df) if df.shape[0] > 0 else 0

        total_samples = len(df)
        for emotion, count in emotion_counts.items():
            percentage = (count / total_samples) * 100
            analysis["class_imbalance"][emotion] = {
                "count": count,
                "percentage": percentage,
                "severely_imbalanced": percentage < 1.0,  # Less than 1% is severely imbalanced
            }

        severely_imbalanced = [
            e for e, data in analysis["class_imbalance"].items() if data["severely_imbalanced"]
        ]

        if severely_imbalanced:
            analysis["issues"].append("Severely imbalanced classes: {severely_imbalanced}")

        if analysis["avg_emotions_per_sample"] < 1.0:
            analysis["issues"].append("Low emotion density per sample")

        logger.info("Analysis complete:")
        logger.info("- Total samples: {analysis['total_samples']}")
        logger.info("- Average emotions per sample: {analysis['avg_emotions_per_sample']:.2f}")
        logger.info("- Severely imbalanced classes: {len(severely_imbalanced)}")

        return analysis

    def balance_dataset(self, df: pd.DataFrame, analysis: dict[str, Any]) -> pd.DataFrame:
        """Balance the dataset to improve F1 scores"""
        logger.info("Balancing dataset...")

        balanced_samples = []

        target_percentage = 2.0
        target_count = max(50, int(len(df) * target_percentage / 100))

        for emotion in self.emotion_labels:
            emotion_samples = []
            for _, row in df.iterrows():
                emotions = row.get("emotions", [])
                if isinstance(emotions, list) and emotion in emotions:
                    emotion_samples.append(row.to_dict())

            current_count = len(emotion_samples)
            logger.info("Emotion '{emotion}': {current_count} samples")

            if current_count < target_count and current_count > 0:
                oversample_factor = target_count // current_count + 1
                for _ in range(oversample_factor):
                    balanced_samples.extend(emotion_samples)
                logger.info(
                    "  -> Oversampled to {len(emotion_samples) * oversample_factor} samples"
                )
            else:
                balanced_samples.extend(emotion_samples)

        for emotion in self.emotion_labels:
            emotion_samples = [s for s in balanced_samples if emotion in s.get("emotions", [])]

            if len(emotion_samples) < 10:  # Very rare emotion
                logger.info("Creating synthetic samples for '{emotion}'")

                similar_samples = [
                    s
                    for s in df.to_dict("records")
                    if any(e in s.get("emotions", []) for e in ["joy", "sadness", "anger"])
                ]

                if similar_samples:
                    for i in range(10 - len(emotion_samples)):
                        base_sample = similar_samples[i % len(similar_samples)].copy()
                        base_sample["emotions"] = [emotion]
                        base_sample["text"] = "[SYNTHETIC] {base_sample.get('text', '')}"
                        balanced_samples.append(base_sample)

        balanced_df = pd.DataFrame(balanced_samples)

        self._analyze_dataset(balanced_df)

        logger.info("Balancing complete:")
        logger.info("- Original samples: {len(df)}")
        logger.info("- Balanced samples: {len(balanced_df)}")
        logger.info("- Improvement: {len(balanced_df) - len(df)} additional samples")

        return balanced_df

    def convert_to_vertex_format(self, df: pd.DataFrame) -> tuple[str, str]:
        """Convert dataset to Vertex AI AutoML format"""
        logger.info("Converting to Vertex AI format...")

        train_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        test_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)

        try:
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=None)

            train_data = self._convert_to_vertex_format(train_df, "train")
            test_data = self._convert_to_vertex_format(test_df, "test")

            train_file.write(train_data)
            test_file.write(test_data)

            train_file.close()
            test_file.close()

            logger.info("Conversion complete:")
            logger.info("- Train samples: {len(train_df)}")
            logger.info("- Test samples: {len(test_df)}")

            return train_file.name, test_file.name

        except Exception:
            logger.error("Error in conversion: {e}")
            raise

    def _convert_to_vertex_format(self, df: pd.DataFrame, split_name: str) -> str:
        """Convert DataFrame to Vertex AI CSV format"""
        lines = []

        header = "text,emotions\n"
        lines.append(header)

        for _, row in df.iterrows():
            row.get("text", "").replace('"', '""')  # Escape quotes
            emotions = row.get("emotions", [])

            ",".join(emotions) if isinstance(emotions, list) else str(emotions)

            line = '"{text}","{emotion_str}"\n'
            lines.append(line)

        return "".join(lines)

    def upload_to_gcs(self, train_file: str, test_file: str) -> tuple[str, str]:
        """Upload prepared data to Google Cloud Storage"""
        logger.info("Uploading to Google Cloud Storage...")

        train_blob_name = "vertex_ai_data/train_data.csv"
        train_blob = self.bucket.blob(train_blob_name)
        train_blob.upload_from_filename(train_file)
        train_gcs_uri = "gs://{self.bucket_name}/{train_blob_name}"

        test_blob_name = "vertex_ai_data/test_data.csv"
        test_blob = self.bucket.blob(test_blob_name)
        test_blob.upload_from_filename(test_file)
        test_gcs_uri = "gs://{self.bucket_name}/{test_blob_name}"

        logger.info("Upload complete:")
        logger.info("- Train data: {train_gcs_uri}")
        logger.info("- Test data: {test_gcs_uri}")

        return train_gcs_uri, test_gcs_uri

    def save_metadata(self, analysis: dict[str, Any], train_uri: str, test_uri: str) -> str:
        """Save metadata for the training pipeline"""
        metadata = {
            "project_id": self.project_id,
            "bucket_name": self.bucket_name,
            "train_data_uri": train_uri,
            "test_data_uri": test_uri,
            "emotion_labels": self.emotion_labels,
            "analysis": analysis,
            "preparation_timestamp": pd.Timestamp.now().isoformat(),
            "vertex_ai_format": "multi-label-classification",
        }

        metadata_file = "vertex_ai_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        metadata_blob_name = "vertex_ai_data/metadata.json"
        metadata_blob = self.bucket.blob(metadata_blob_name)
        metadata_blob.upload_from_filename(metadata_file)

        logger.info("Metadata saved: {metadata_file}")
        logger.info("Metadata uploaded: gs://{self.bucket_name}/{metadata_blob_name}")

        return metadata_file

    def cleanup_temp_files(self, train_file: str, test_file: str):
        """Clean up temporary files"""
        try:
            os.unlink(train_file)
            os.unlink(test_file)
            logger.info("Temporary files cleaned up")
        except Exception:
            logger.warning("Error cleaning up temp files: {e}")


def main():
    """Main execution function"""
    if len(sys.argv) != 4:
        logging.info("Usage: python prepare_vertex_data.py <project_id> <bucket_name> <data_path>")
        sys.exit(1)

    project_id = sys.argv[1]
    bucket_name = sys.argv[2]
    data_path = sys.argv[3]

    logging.info("🚀 Starting SAMO Vertex AI Data Preparation...")
    logging.info("📊 Project: {project_id}")
    logging.info("📦 Bucket: {bucket_name}")
    logging.info("📁 Data: {data_path}")

    if not Path(data_path):
        logging.info("❌ Data file not found: {data_path}")
        sys.exit(1)

    try:
        preparer = SAMOVertexDataPreparation(project_id, bucket_name)

        df, analysis = preparer.load_and_analyze_data(data_path)

        logging.info("\n📈 Dataset Analysis:")
        logging.info("- Total samples: {analysis['total_samples']}")
        logging.info("- Average emotions per sample: {analysis['avg_emotions_per_sample']:.2f}")
        logging.info("- Issues found: {len(analysis['issues'])}")

        if analysis["issues"]:
            logging.info("- Issues: {', '.join(analysis['issues'])}")

        balanced_df = preparer.balance_dataset(df, analysis)

        train_file, test_file = preparer.convert_to_vertex_format(balanced_df)

        train_uri, test_uri = preparer.upload_to_gcs(train_file, test_file)

        preparer.save_metadata(analysis, train_uri, test_uri)

        preparer.cleanup_temp_files(train_file, test_file)

        logging.info("\n🎉 Data preparation complete!")
        logging.info("✅ Train data: {train_uri}")
        logging.info("✅ Test data: {test_uri}")
        logging.info("✅ Metadata: {metadata_file}")
        logging.info("\n🚀 Ready for Vertex AI AutoML training!")

    except Exception:
        logging.info("❌ Error during data preparation: {e}")
        logger.error("Data preparation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
