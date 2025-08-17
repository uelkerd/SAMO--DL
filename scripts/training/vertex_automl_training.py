            # If no emotion column found, use the last column typically labels
            # Save results to GCS
            # Step 1: Load metadata
            # Step 2: Create dataset
            # Step 3: Train model
            # Step 4: Monitor training
            # Step 5: Deploy model
            # Step 6: Save results
        # Configure training job
        # Create dataset
        # Download first few lines to check structure
        # Find the target column should be the emotion labels column
        # Get model evaluation
        # Get the correct target column
        # Initialize Vertex AI
        # Look for emotion-related columns
        # Start training
    # Initialize and run training
# Configure logging
#!/usr/bin/env python3
from datetime import datetime
from google.cloud import aiplatform
from google.cloud import storage
import json
import logging
import sys
import time



"""
SAMO Vertex AI AutoML Training Pipeline
Trains an AutoML model for emotion detection with F1 score optimization
"""

logging.basicConfig(level=logging.INFO, format="%asctimes - %levelnames - %messages")
logger = logging.getLogger__name__


class SAMOVertexAutoMLTraining:
    """Handles Vertex AI AutoML training for emotion detection"""

    def __init__self, project_id: str, bucket_name: str:
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.region = "us-central1"
        self.dataset_id = None
        self.model_id = None

        aiplatform.initproject=project_id, location=self.region
        logger.info"Initialized Vertex AI for project: {project_id}"

    def load_metadataself -> dict:
        """Load training metadata"""
        storage_client = storage.Clientproject=self.project_id
        bucket = storage_client.bucketself.bucket_name
        blob = bucket.blob"vertex_ai_data/metadata.json"

        metadata = json.loads(blob.download_as_text())
        logger.info("Loaded metadata: {lenmetadata['emotions']} emotions")
        return metadata

    def check_csv_structureself -> str:
        """Check the actual CSV structure to find the target column"""
        storage_client = storage.Clientproject=self.project_id
        bucket = storage_client.bucketself.bucket_name
        blob = bucket.blob"vertex_ai_data/train_data.csv"

        content = blob.download_as_text().split"\n"[:5]
        logger.info"CSV header: {content[0]}"

        columns = content[0].split","
        target_column = None

        for col in columns:
            if "emotion" in col.lower() or "label" in col.lower():
                target_column = col
                break

        if not target_column:
            target_column = columns[-1]

        logger.info"Using target column: {target_column}"
        return target_column

    def create_datasetself, metadata: dict -> str:
        """Create Vertex AI dataset"""
        dataset_display_name = "samo-emotion-dataset-{int(time.time())}"

        dataset = aiplatform.TextDataset.create(
            display_name=dataset_display_name,
            gcs_source="gs://{self.bucket_name}/vertex_ai_data/train_data.csv",
            project=self.project_id,
            location=self.region,
        )

        self.dataset_id = dataset.name
        logger.info"Created dataset: {self.dataset_id}"
        return self.dataset_id

    def train_modelself, dataset_id: str, metadata: dict -> str:
        """Train AutoML model"""
        model_display_name = "samo-emotion-model-{int(time.time())}"

        target_column = self.check_csv_structure()

        training_job = aiplatform.AutoMLTextTrainingJob(
            display_name=model_display_name,
            prediction_type="classification",
            multi_label=True,
            project=self.project_id,
            location=self.region,
        )

        model = training_job.run(
            dataset=dataset_id,
            target_column=target_column,
            training_fraction_split=0.8,
            validation_fraction_split=0.1,
            test_fraction_split=0.1,
            budget_milli_node_hours=1000,  # 1 hour budget
            disable_early_stopping=False,
            model_display_name=model_display_name,
        )

        self.model_id = model.name
        logger.info"Started training: {self.model_id}"
        return self.model_id

    def monitor_trainingself, model_id: str -> dict:
        """Monitor training progress"""
        logger.info"Monitoring training progress..."

        while True:
            model = aiplatform.Modelmodel_id
            training_job = model.gca_resource.training_pipeline

            if training_job.state.name == "PIPELINE_STATE_SUCCEEDED":
                logger.info"✅ Training completed successfully!"
                break
            elif training_job.state.name == "PIPELINE_STATE_FAILED":
                logger.error"❌ Training failed!"
                return None
            else:
                logger.info"Training status: {training_job.state.name}"
                time.sleep300  # Check every 5 minutes

        evaluation = model.evaluate()
        logger.info"Model evaluation: {evaluation}"

        return {"model_id": model_id, "evaluation": evaluation, "training_complete": True}

    def deploy_modelself, model_id: str -> str:
        """Deploy model to endpoint"""
        endpoint_display_name = "samo-emotion-endpoint-{int(time.time())}"

        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_display_name, project=self.project_id, location=self.region
        )

        model = aiplatform.Modelmodel_id
        endpoint.deploy(
            model=model,
            deployed_model_display_name=endpoint_display_name,
            machine_type="n1-standard-2",
            min_replica_count=1,
            max_replica_count=3,
        )

        logger.info"Deployed model to endpoint: {endpoint.name}"
        return endpoint.name

    def run_training_pipelineself -> dict:
        """Run complete training pipeline"""
        logger.info"🚀 Starting SAMO Vertex AI AutoML Training Pipeline..."

        try:
            logger.info"📊 Loading training metadata..."
            metadata = self.load_metadata()

            logger.info"📁 Creating Vertex AI dataset..."
            dataset_id = self.create_datasetmetadata

            logger.info"🤖 Starting AutoML training..."
            model_id = self.train_modeldataset_id, metadata

            logger.info"📈 Monitoring training progress..."
            training_result = self.monitor_trainingmodel_id

            if not training_result:
                logger.error"Training failed!"
                return None

            logger.info"🚀 Deploying model to endpoint..."
            endpoint_id = self.deploy_modelmodel_id

            results = {
                "project_id": self.project_id,
                "bucket_name": self.bucket_name,
                "dataset_id": dataset_id,
                "model_id": model_id,
                "endpoint_id": endpoint_id,
                "training_result": training_result,
                "timestamp": datetime.now().isoformat(),
            }

            storage_client = storage.Clientproject=self.project_id
            bucket = storage_client.bucketself.bucket_name
            blob = bucket.blob"vertex_ai_data/training_results.json"
            blob.upload_from_string(json.dumpsresults, indent=2)

            logger.info"🎉 Training pipeline completed successfully!"
            logger.info"✅ Model ID: {model_id}"
            logger.info"✅ Endpoint ID: {endpoint_id}"

            return results

        except Exception:
            logger.error"Training pipeline failed: {e}"
            return None


def main():
    """Main function"""
    if lensys.argv != 3:
        logging.info"Usage: python vertex_automl_training.py <project_id> <bucket_name>"
        sys.exit1

    project_id = sys.argv[1]
    bucket_name = sys.argv[2]

    logging.info"🚀 Starting SAMO Vertex AI AutoML Training..."
    logging.info"📊 Project: {project_id}"
    logging.info"📦 Bucket: {bucket_name}"

    trainer = SAMOVertexAutoMLTrainingproject_id, bucket_name
    results = trainer.run_training_pipeline()

    if results:
        logging.info"🎉 Training completed successfully!"
        logging.info"✅ Model ID: {results['model_id']}"
        logging.info"✅ Endpoint ID: {results['endpoint_id']}"
        logging.info"🚀 Ready for production deployment!"
    else:
        logging.info"❌ Training failed!"
        sys.exit1


if __name__ == "__main__":
    main()
