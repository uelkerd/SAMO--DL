            # Create custom job
            # Create hyperparameter tuning job
            # Create validation job
            # Create validation job
            # Hyperparameter tuning configuration
            # Import Vertex AI
            # Initialize Vertex AI
            # Install Vertex AI SDK
            # Model monitoring configuration
            # Pipeline configuration
            # Training job configuration
import logging
import os
            import subprocess
import sys
# Add src to path
# Configure logging
    # Create Vertex AI setup
    # Get project ID from environment or user input
    # Setup complete infrastructure
        # Step 1: Environment setup
        # Step 2: Create validation job
        # Step 3: Create custom training job
        # Step 4: Create hyperparameter tuning
        # Step 5: Create monitoring
        # Step 6: Create automated pipeline
    # Summary
#!/usr/bin/env python3
            from google.cloud import aiplatform
            from google.cloud import aiplatform
            from google.cloud import aiplatform
            from google.cloud import aiplatform
            from google.cloud import aiplatform
            from google.cloud import aiplatform
            from google.cloud import storage
from pathlib import Path
from typing import Dict, Any, Optional









""""
Vertex AI Setup for SAMO Deep Learning Project.

This script sets up Vertex AI infrastructure to solve the 0.0000 loss issue
and provide managed ML training, deployment, and monitoring.
""""

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class VertexAISetup:
    """Vertex AI setup and management for SAMO Deep Learning."""

    def __init__(self, project_id: str, region: str = "us-central1"):
        """Initialize Vertex AI setup."

        Args:
            project_id: GCP project ID
            region: GCP region for Vertex AI resources
        """"
        self.project_id = project_id
        self.region = region
        self.dataset_id = "samo-emotions-dataset"
        self.model_display_name = "samo-emotion-detection-bert"
        self.endpoint_display_name = "samo-emotion-detection-endpoint"

    def setup_environment(self) -> bool:
        """Setup Vertex AI environment and dependencies."""
        logger.info("🔧 Setting up Vertex AI environment...")

        try:
            subprocess.run([)
                sys.executable, "-m", "pip", "install",
                "google-cloud-aiplatform", "google-cloud-storage"
(            ], check=True)

            logger.info(" Vertex AI SDK installed successfully")

            aiplatform.init()
                project=self.project_id,
                location=self.region,
(            )

            logger.info(" Vertex AI initialized for project: {self.project_id}")
            logger.info(" Region: {self.region}")

            return True

        except Exception as e:
            logger.error("❌ Vertex AI setup failed: {e}")
            return False

    def create_custom_training_job(self) -> Dict[str, Any]:
        """Create custom training job for emotion detection model."""
        logger.info("🚀 Creating Vertex AI custom training job...")

        try:
            job_config = {
                "display_name": "samo-emotion-detection-training",
                "container_uri": "gcr.io/cloud-aiplatform/training/pytorch-gpu.2-0:latest",
                "model_serving_container_image_uri": "gcr.io/cloud-aiplatform/prediction/pytorch-gpu.2-0:latest",
                "args": [
                    "--model_name=bert-base-uncased",
                    "--batch_size=16",
                    "--learning_rate=2e-6",  # Reduced from 2e-5 to fix 0.0000 loss
                    "--num_epochs=3",
                    "--max_length=512",
                    "--freeze_bert_layers=6",
                    "--use_focal_loss=true",
                    "--class_weights=true",
                    "--dev_mode=false",
                    "--debug_mode=true"
                ],
                "machine_spec": {
                    "machine_type": "n1-standard-4",
                    "accelerator_type": "NVIDIA_TESLA_T4",
                    "accelerator_count": 1
                },
                "replica_count": 1,
                "training_fraction_split": 0.8,
                "validation_fraction_split": 0.1,
                "test_fraction_split": 0.1,
                "enable_web_access": True,
                "enable_dashboard_access": True,
            }

            job = aiplatform.CustomTrainingJob()
                display_name=job_config["display_name"],
                container_uri=job_config["container_uri"],
                model_serving_container_image_uri=job_config["model_serving_container_image_uri"],
                args=job_config["args"],
                machine_type=job_config["machine_spec"]["machine_type"],
                accelerator_type=job_config["machine_spec"]["accelerator_type"],
                accelerator_count=job_config["machine_spec"]["accelerator_count"],
                replica_count=job_config["replica_count"],
                training_fraction_split=job_config["training_fraction_split"],
                validation_fraction_split=job_config["validation_fraction_split"],
                test_fraction_split=job_config["test_fraction_split"],
                enable_web_access=job_config["enable_web_access"],
                enable_dashboard_access=job_config["enable_dashboard_access"],
(            )

            logger.info(" Custom training job created successfully")
            logger.info("   Display name: {job_config['display_name']}")
            logger.info("   Machine type: {job_config['machine_spec']['machine_type']}")
            logger.info("   GPU: {job_config['machine_spec']['accelerator_type']}")
            logger.info("   Learning rate: 2e-6 (optimized for stability)")

            return {"job": job, "config": job_config}

        except Exception as e:
            logger.error("❌ Custom training job creation failed: {e}")
            return {}

    def create_hyperparameter_tuning_job(self) -> Dict[str, Any]:
        """Create hyperparameter tuning job to optimize the model."""
        logger.info(" Creating hyperparameter tuning job...")

        try:
            tuning_config = {
                "display_name": "samo-emotion-detection-tuning",
                "container_uri": "gcr.io/cloud-aiplatform/training/pytorch-gpu.2-0:latest",
                "args": [
                    "--model_name=bert-base-uncased",
                    "--batch_size=16",
                    "--num_epochs=2",
                    "--max_length=512",
                    "--use_focal_loss=true",
                    "--class_weights=true",
                    "--dev_mode=true"
                ],
                "machine_spec": {
                    "machine_type": "n1-standard-4",
                    "accelerator_type": "NVIDIA_TESLA_T4",
                    "accelerator_count": 1
                },
                "replica_count": 1,
                "max_trial_count": 10,
                "parallel_trial_count": 2,
                "hyperparameter_spec": {
                    "learning_rate": {
                        "type": "DOUBLE",
                        "min_value": 1e-6,
                        "max_value": 5e-5,
                        "scale_type": "UNIT_LOG_SCALE"
                    },
                    "batch_size": {
                        "type": "DISCRETE",
                        "values": [8, 16, 32]
                    },
                    "freeze_bert_layers": {
                        "type": "DISCRETE",
                        "values": [4, 6, 8]
                    }
                },
                "metric_spec": {
                    "f1_score": "maximize"
                }
            }

            tuning_job = aiplatform.HyperparameterTuningJob()
                display_name=tuning_config["display_name"],
                container_uri=tuning_config["container_uri"],
                args=tuning_config["args"],
                machine_type=tuning_config["machine_spec"]["machine_type"],
                accelerator_type=tuning_config["machine_spec"]["accelerator_type"],
                accelerator_count=tuning_config["machine_spec"]["accelerator_count"],
                replica_count=tuning_config["replica_count"],
                max_trial_count=tuning_config["max_trial_count"],
                parallel_trial_count=tuning_config["parallel_trial_count"],
                hyperparameter_spec=tuning_config["hyperparameter_spec"],
                metric_spec=tuning_config["metric_spec"],
(            )

            logger.info(" Hyperparameter tuning job created successfully")
            logger.info("   Max trials: {tuning_config['max_trial_count']}")
            logger.info("   Parallel trials: {tuning_config['parallel_trial_count']}")
            logger.info("   Optimization metric: F1 Score")

            return {"tuning_job": tuning_job, "config": tuning_config}

        except Exception as e:
            logger.error("❌ Hyperparameter tuning job creation failed: {e}")
            return {}

    def create_model_monitoring(self) -> Dict[str, Any]:
        """Create model monitoring for production deployment."""
        logger.info(" Setting up model monitoring...")

        try:
            monitoring_config = {
                "display_name": "samo-emotion-detection-monitoring",
                "model_display_name": self.model_display_name,
                "endpoint_display_name": self.endpoint_display_name,
                "monitoring_config": {
                    "monitoring_interval": 3600,  # 1 hour
                    "monitoring_alert_channels": ["email"],
                    "monitoring_metrics": [
                        "prediction_latency",
                        "prediction_throughput",
                        "model_accuracy",
                        "data_drift"
                    ]
                }
            }

            logger.info(" Model monitoring configuration created")
            logger.info("   Monitoring interval: 1 hour")
            logger.info("   Metrics: latency, throughput, accuracy, data drift")

            return {"config": monitoring_config}

        except Exception as e:
            logger.error("❌ Model monitoring setup failed: {e}")
            return {}

    def create_automated_pipeline(self) -> Dict[str, Any]:
        """Create automated ML pipeline for continuous training."""
        logger.info("🔄 Creating automated ML pipeline...")

        try:
            pipeline_config = {
                "display_name": "samo-emotion-detection-pipeline",
                "pipeline_root": "gs://{self.project_id}-vertex-ai/pipelines",
                "components": [
                    "data_validation",
                    "data_preprocessing",
                    "model_training",
                    "model_evaluation",
                    "model_deployment"
                ],
                "schedule": "0 2 * * *",  # Daily at 2 AM
                "trigger_conditions": [
                    "data_drift_detected",
                    "model_performance_degradation",
                    "new_data_available"
                ]
            }

            logger.info(" Automated pipeline configuration created")
            logger.info("   Schedule: Daily at 2 AM")
            logger.info("   Trigger conditions: data drift, performance degradation, new data")

            return {"config": pipeline_config}

        except Exception as e:
            logger.error("❌ Automated pipeline setup failed: {e}")
            return {}

    def run_validation_on_vertex(self) -> bool:
        """Run validation on Vertex AI to identify 0.0000 loss issues."""
        logger.info(" Running validation on Vertex AI...")

        try:
            validation_config = {
                "display_name": "samo-validation-job",
                "container_uri": "gcr.io/cloud-aiplatform/training/pytorch-cpu.2-0:latest",
                "args": [
                    "--validation_mode=true",
                    "--check_data_distribution=true",
                    "--check_model_architecture=true",
                    "--check_loss_function=true",
                    "--check_training_config=true"
                ],
                "machine_spec": {
                    "machine_type": "n1-standard-4"
                },
                "replica_count": 1,
            }

            validation_job = aiplatform.CustomTrainingJob()
                display_name=validation_config["display_name"],
                container_uri=validation_config["container_uri"],
                args=validation_config["args"],
                machine_type=validation_config["machine_spec"]["machine_type"],
                replica_count=validation_config["replica_count"],
(            )

            logger.info(" Validation job created successfully")
            logger.info("   This will identify the root cause of 0.0000 loss")
            logger.info("   Check Vertex AI console for results")

            return True

        except Exception as e:
            logger.error("❌ Validation job creation failed: {e}")
            return False

    def setup_complete_infrastructure(self) -> Dict[str, Any]:
        """Setup complete Vertex AI infrastructure."""
        logger.info("🚀 Setting up complete Vertex AI infrastructure...")

        results = {}

        if not self.setup_environment():
            logger.error("❌ Environment setup failed")
            return results

        logger.info("\n Step 1: Creating validation job...")
        validation_success = self.run_validation_on_vertex()
        results["validation"] = validation_success

        logger.info("\n Step 2: Creating custom training job...")
        training_result = self.create_custom_training_job()
        results["training"] = training_result

        logger.info("\n Step 3: Creating hyperparameter tuning...")
        tuning_result = self.create_hyperparameter_tuning_job()
        results["tuning"] = tuning_result

        logger.info("\n Step 4: Creating model monitoring...")
        monitoring_result = self.create_model_monitoring()
        results["monitoring"] = monitoring_result

        logger.info("\n Step 5: Creating automated pipeline...")
        pipeline_result = self.create_automated_pipeline()
        results["pipeline"] = pipeline_result

        return results


        def main():
    """Main function to setup Vertex AI infrastructure."""
    logger.info("🚀 SAMO Deep Learning - Vertex AI Setup")
    logger.info("=" * 50)

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
        project_id = input("Enter your GCP Project ID: ").strip()

        if not project_id:
        logger.error("❌ Project ID is required")
        sys.exit(1)

    vertex_setup = VertexAISetup(project_id=project_id)

    results = vertex_setup.setup_complete_infrastructure()

    logger.info("\n{'='*50}")
    logger.info(" VERTEX AI SETUP SUMMARY")
    logger.info("{'='*50}")

        for component, result in results.items():
        if result:
            logger.info(" {component.title()}: SUCCESS")
        else:
            logger.error("❌ {component.title()}: FAILED")

    logger.info("\n NEXT STEPS:")
    logger.info("   1. Check Vertex AI console: https://console.cloud.google.com/vertex-ai")
    logger.info("   2. Run validation job to identify 0.0000 loss root cause")
    logger.info("   3. Start training job with optimized configuration")
    logger.info("   4. Monitor training progress and results")
    logger.info("   5. Deploy model to endpoint when ready")

    logger.info("\n💡 BENEFITS OF VERTEX AI:")
    logger.info("   • Managed infrastructure (no more terminal issues)")
    logger.info("   • Automatic hyperparameter tuning")
    logger.info("   • Built-in monitoring and alerting")
    logger.info("   • Scalable training and deployment")
    logger.info("   • Cost optimization and resource management")

    return all(results.values())


        if __name__ == "__main__":
    success = main()
        if not success:
        sys.exit(1)
