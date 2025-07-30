#!/usr/bin/env python3
"""
SAMO GoEmotions Data Preparation for Vertex AI AutoML
Converts your current dataset to Vertex AI format and addresses F1 score issues
"""

import pandas as pd
from collections import Counter
import logging
from google.cloud import storage
import tempfile
from typing import Any
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SAMOVertexDataPreparation:
    """Handles data preparation for Vertex AI AutoML training"""

    def __init__(self, project_id: str, bucket_name: str):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)

        # GoEmotions emotion labels (27 emotions)
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

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Analyze current state
            analysis = self._analyze_dataset(df)

            return df, analysis

        except Exception as e:
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

        # Analyze emotion distribution
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

        # Identify class imbalance issues
        total_samples = len(df)
        for emotion, count in emotion_counts.items():
            percentage = (count / total_samples) * 100
            analysis["class_imbalance"][emotion] = {
                "count": count,
                "percentage": percentage,
                "severely_imbalanced": percentage < 1.0,  # Less than 1% is severely imbalanced
            }

        # Identify issues
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

        # Strategy 1: Oversample minority classes
        balanced_samples = []

        # Get target counts for each emotion (aim for at least 2% representation)
        target_percentage = 2.0
        target_count = max(50, int(len(df) * target_percentage / 100))

        for emotion in self.emotion_labels:
            # Find samples with this emotion
            emotion_samples = []
            for _, row in df.iterrows():
                emotions = row.get("emotions", [])
                if isinstance(emotions, list) and emotion in emotions:
                    emotion_samples.append(row.to_dict())

            current_count = len(emotion_samples)
            logger.info("Emotion '{emotion}': {current_count} samples")

            if current_count < target_count and current_count > 0:
                # Oversample to reach target
                oversample_factor = target_count // current_count + 1
                for _ in range(oversample_factor):
                    balanced_samples.extend(emotion_samples)
                logger.info(
                    "  -> Oversampled to {len(emotion_samples) * oversample_factor} samples"
                )
            else:
                balanced_samples.extend(emotion_samples)

        # Strategy 2: Add synthetic samples for very rare emotions
        for emotion in self.emotion_labels:
            emotion_samples = [s for s in balanced_samples if emotion in s.get("emotions", [])]

            if len(emotion_samples) < 10:  # Very rare emotion
                logger.info("Creating synthetic samples for '{emotion}'")

                # Find similar samples to base synthetic data on
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

        # Re-analyze balanced dataset
        self._analyze_dataset(balanced_df)

        logger.info("Balancing complete:")
        logger.info("- Original samples: {len(df)}")
        logger.info("- Balanced samples: {len(balanced_df)}")
        logger.info("- Improvement: {len(balanced_df) - len(df)} additional samples")

        return balanced_df

    def convert_to_vertex_format(self, df: pd.DataFrame) -> tuple[str, str]:
        """Convert dataset to Vertex AI AutoML format"""
        logger.info("Converting to Vertex AI format...")

        # Create temporary files
        train_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        test_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)

        try:
            # Split data
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=None)

            # Convert to Vertex AI format
            train_data = self._convert_to_vertex_format(train_df, "train")
            test_data = self._convert_to_vertex_format(test_df, "test")

            # Write to files
            train_file.write(train_data)
            test_file.write(test_data)

            train_file.close()
            test_file.close()

            logger.info("Conversion complete:")
            logger.info("- Train samples: {len(train_df)}")
            logger.info("- Test samples: {len(test_df)}")

            return train_file.name, test_file.name

        except Exception as e:
            logger.error("Error in conversion: {e}")
            raise

    def _convert_to_vertex_format(self, df: pd.DataFrame, split_name: str) -> str:
        """Convert DataFrame to Vertex AI CSV format"""
        lines = []

        # Header
        header = "text,emotions\n"
        lines.append(header)

        # Data rows
        for _, row in df.iterrows():
            text = row.get("text", "").replace('"', '""')  # Escape quotes
            emotions = row.get("emotions", [])

            if isinstance(emotions, list):
                # Convert to comma-separated string
                emotion_str = ",".join(emotions)
            else:
                emotion_str = str(emotions)

            # Format: "text","emotion1,emotion2,emotion3"
            line = '"{text}","{emotion_str}"\n'
            lines.append(line)

        return "".join(lines)

    def upload_to_gcs(self, train_file: str, test_file: str) -> tuple[str, str]:
        """Upload prepared data to Google Cloud Storage"""
        logger.info("Uploading to Google Cloud Storage...")

        # Upload train file
        train_blob_name = "vertex_ai_data/train_data.csv"
        train_blob = self.bucket.blob(train_blob_name)
        train_blob.upload_from_filename(train_file)
        train_gcs_uri = "gs://{self.bucket_name}/{train_blob_name}"

        # Upload test file
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

        # Save locally
        metadata_file = "vertex_ai_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Upload to GCS
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
        except Exception as e:
            logger.warning("Error cleaning up temp files: {e}")


def main():
    """Main execution function"""
    if len(sys.argv) != 4:
        print("Usage: python prepare_vertex_data.py <project_id> <bucket_name> <data_path>")
        sys.exit(1)

    project_id = sys.argv[1]
    bucket_name = sys.argv[2]
    data_path = sys.argv[3]

    print("🚀 Starting SAMO Vertex AI Data Preparation...")
    print("📊 Project: {project_id}")
    print("📦 Bucket: {bucket_name}")
    print("📁 Data: {data_path}")

    # Check if data file exists
    if not os.path.exists(data_path):
        print("❌ Data file not found: {data_path}")
        sys.exit(1)

    try:
        # Initialize data preparation
        preparer = SAMOVertexDataPreparation(project_id, bucket_name)

        # Load and analyze data
        df, analysis = preparer.load_and_analyze_data(data_path)

        # Display analysis
        print("\n📈 Dataset Analysis:")
        print("- Total samples: {analysis['total_samples']}")
        print("- Average emotions per sample: {analysis['avg_emotions_per_sample']:.2f}")
        print("- Issues found: {len(analysis['issues'])}")

        if analysis["issues"]:
            print("- Issues: {', '.join(analysis['issues'])}")

        # Balance dataset
        balanced_df = preparer.balance_dataset(df, analysis)

        # Convert to Vertex AI format
        train_file, test_file = preparer.convert_to_vertex_format(balanced_df)

        # Upload to GCS
        train_uri, test_uri = preparer.upload_to_gcs(train_file, test_file)

        # Save metadata
        metadata_file = preparer.save_metadata(analysis, train_uri, test_uri)

        # Cleanup
        preparer.cleanup_temp_files(train_file, test_file)

        print("\n🎉 Data preparation complete!")
        print("✅ Train data: {train_uri}")
        print("✅ Test data: {test_uri}")
        print("✅ Metadata: {metadata_file}")
        print("\n🚀 Ready for Vertex AI AutoML training!")

    except Exception as e:
        print("❌ Error during data preparation: {e}")
        logger.error("Data preparation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
