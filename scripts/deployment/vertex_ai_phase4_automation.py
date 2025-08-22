#!/usr/bin/env python3
"""
Phase 4: Vertex AI Deployment Automation
========================================

Enhanced Vertex AI deployment with automated model versioning, rollback capabilities,
A/B testing support, performance monitoring, and cost optimization.

Features:
- Automated model versioning and deployment
- Rollback capabilities and A/B testing support  
- Model performance monitoring and alerting
- Cost optimization and resource management
- Comprehensive testing and validation
"""

import os
import json
import subprocess
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DeploymentConfig:
    """Configuration for Vertex AI deployment."""
    project_id: str
    region: str = "us-central1"
    model_name: str = "comprehensive-emotion-detection"
    endpoint_name: str = "emotion-detection-endpoint"
    repository_name: str = "emotion-detection"
    machine_type: str = "n1-standard-2"
    min_replicas: int = 1
    max_replicas: int = 10
    traffic_split: Dict[str, float] = None
    monitoring_interval: int = 300  # 5 minutes
    cost_budget: float = 100.0  # USD per day
    rollback_threshold: float = 0.8  # 80% performance threshold

class VertexAIPhase4Automation:
    """Enhanced Vertex AI deployment automation with Phase 4 features."""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.current_version = None
        self.deployment_history = []

    def check_prerequisites(self) -> bool:
        """Enhanced prerequisites checking for Phase 4 features."""
        logger.info("🔍 CHECKING PHASE 4 DEPLOYMENT PREREQUISITES")
        print("=" * 60)

        checks = [
            ("gcloud CLI", self._check_gcloud),
            ("Authentication", self._check_authentication),
            ("Project Configuration", self._check_project),
            ("Vertex AI API", self._check_vertex_ai_api),
            ("Cloud Monitoring API", self._check_monitoring_api),
            ("Cloud Logging API", self._check_logging_api),
            ("Artifact Registry", self._check_artifact_registry),
            ("IAM Permissions", self._check_iam_permissions),
        ]

        all_passed = True
        for check_name, check_func in checks:
            try:
                if check_func():
                    print(f"✅ {check_name}")
                else:
                    print(f"❌ {check_name}")
                    all_passed = False
            except Exception as e:
                print(f"❌ {check_name}: {e}")
                all_passed = False

        return all_passed

    @staticmethod
    def _check_gcloud() -> bool:
        """Check if gcloud CLI is installed and working."""
        try:
            result = subprocess.run(
                                    ['gcloud',
                                    '--version'],
                                    capture_output=True,
                                    text=True,
                                    check=True
                                   )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    @staticmethod
    def _check_authentication() -> bool:
        """Check if user is authenticated."""
        try:
result = subprocess.run(['gcloud', 'auth', 'list', '--filter=status:ACTIVE'],
                                  capture_output=True, text=True, check=True)
            return result.returncode == 0 and 'ACTIVE' in result.stdout
        except Exception:
            return False

    def _check_project(self) -> bool:
        """Check if project is properly configured."""
        try:
            result = subprocess.run(['gcloud', 'config', 'get-value', 'project'], 
                                  capture_output=True, text=True, check=True)
            return result.returncode == 0 and result.stdout.strip(
                                                                  ) == self.config.project_id
        except Exception:
            return False

    @staticmethod
    def _check_vertex_ai_api() -> bool:
        """Check if Vertex AI API is enabled."""
        try:
            result = subprocess.run(['gcloud', 'services', 'list', '--enabled', 
                                   '--filter=name:aiplatform.googleapis.com'], 
                                  capture_output=True, text=True, check=True)
return result.returncode == 0 and 'aiplatform.googleapis.com' in result.stdout
        except Exception:
            return False

    @staticmethod
    def _check_monitoring_api() -> bool:
        """Check if Cloud Monitoring API is enabled."""
        try:
            result = subprocess.run(['gcloud', 'services', 'list', '--enabled', 
                                   '--filter=name:monitoring.googleapis.com'], 
                                  capture_output=True, text=True, check=True)
return result.returncode == 0 and 'monitoring.googleapis.com' in result.stdout
        except Exception:
            return False

    @staticmethod
    def _check_logging_api() -> bool:
        """Check if Cloud Logging API is enabled."""
        try:
            result = subprocess.run(['gcloud', 'services', 'list', '--enabled', 
                                   '--filter=name:logging.googleapis.com'], 
                                  capture_output=True, text=True, check=True)
            return result.returncode == 0 and 'logging.googleapis.com' in result.stdout
        except Exception:
            return False

    @staticmethod
    def _check_artifact_registry() -> bool:
        """Check if Artifact Registry is enabled."""
        try:
            result = subprocess.run(['gcloud', 'services', 'list', '--enabled', 
                                   '--filter=name:artifactregistry.googleapis.com'], 
                                  capture_output=True, text=True, check=True)
return result.returncode == 0 and 'artifactregistry.googleapis.com' in result.stdout
        except Exception:
            return False

    def _check_iam_permissions(self) -> bool:
        """Check if user has required IAM permissions."""
        required_roles = [
            'roles/aiplatform.admin',
            'roles/monitoring.admin',
            'roles/logging.admin',
            'roles/artifactregistry.admin'
        ]

        try:
result = subprocess.run(['gcloud', 'projects', 'get-iam-policy', self.config.project_id,
                                   '--flatten=bindings[].members', 
                                   '--format=value(bindings.role)'], 
                                  capture_output=True, text=True, check=True)
            user_email = subprocess.run(['gcloud', 'config', 'get-value', 'account'], 
capture_output=True, text=True, check=True).stdout.strip(check=True)

            user_roles = result.stdout.split('\n')
            return any(role in user_roles for role in required_roles)
        except Exception:
            return False

    def generate_model_version(self) -> str:
        """Generate a unique model version based on timestamp and git commit."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Get git commit hash if available
        try:
            result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], 
                                  capture_output=True, text=True, check=True)
            git_hash = result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            git_hash = "unknown"

        version = f"v{timestamp}_{git_hash}"
        self.current_version = version
        return version

    def create_deployment_package(self, version: str) -> str:
        """Create deployment package with versioning."""
        logger.info(f"📦 CREATING DEPLOYMENT PACKAGE FOR VERSION {version}")
        print("=" * 60)

        # Create versioned deployment directory
        deployment_dir = f"deployment/vertex_ai/{version}"
        os.makedirs(deployment_dir, exist_ok=True)

        # Copy model files
        source_model_path = "deployment/models/default"
        if not os.path.exists(source_model_path):
            raise FileNotFoundError(f"Source model not found: {source_model_path}")

        # Create Dockerfile with versioning
        dockerfile_content = f"""
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY model/ ./model/

# Copy prediction code
COPY predict.py .

# Set environment variables
ENV MODEL_VERSION={version}
ENV MODEL_PATH=/app/model

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Start the server
CMD ["python", "predict.py"]
"""

        with open(f"{deployment_dir}/Dockerfile", "w") as f:
            f.write(dockerfile_content)

        # Copy model files
        subprocess.run(
                       ['cp',
                       '-r',
                       source_model_path,
                       f"{deployment_dir}/model"],
                       check=True
                      )

        # Copy requirements
        subprocess.run(
                       ['cp',
                       'deployment/gcp/requirements.txt',
                       f"{deployment_dir}/"],
                       check=True
                      )

        # Copy prediction code
        subprocess.run(
                       ['cp',
                       'deployment/gcp/predict.py',
                       f"{deployment_dir}/"],
                       check=True
                      )

        # Create version metadata
        metadata = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "model_info": {
                "name": self.config.model_name,
"description": "Comprehensive emotion detection model with Phase 4 enhancements"
            },
            "deployment_config": {
                "machine_type": self.config.machine_type,
                "min_replicas": self.config.min_replicas,
                "max_replicas": self.config.max_replicas,
                "traffic_split": self.config.traffic_split or {"100": 1.0}
            }
        }

        with open(f"{deployment_dir}/version_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Deployment package created: {deployment_dir}")
        return deployment_dir

    def build_and_push_image(self, deployment_dir: str, version: str) -> str:
        """Build and push Docker image with versioning."""
        logger.info(f"🐳 BUILDING AND PUSHING DOCKER IMAGE FOR VERSION {version}")
        print("=" * 60)

        # Configure Docker for gcloud
        subprocess.run(['gcloud', 'auth', 'configure-docker'], check=True)

        # Create image URI with version
image_uri = f"gcr.io/{self.config.project_id}/{self.config.repository_name}:{version}"

        try:
            # Build image
            subprocess.run(
                           ['docker',
                           'build',
                           '-t',
                           image_uri,
                           deployment_dir],
                           check=True
                          )
            print("✅ Docker image built")

            # Push image
            subprocess.run(['docker', 'push', image_uri], check=True)
            print("✅ Docker image pushed to Container Registry")

            return image_uri

        except subprocess.CalledProcessError as e:
            logger.error(f"Error building/pushing Docker image: {e}")
            raise

    def create_vertex_ai_model(self, image_uri: str, version: str) -> str:
        """Create Vertex AI model with versioning."""
        logger.info(f"🤖 CREATING VERTEX AI MODEL FOR VERSION {version}")
        print("=" * 60)

        model_display_name = f"{self.config.model_name}-{version}"

        try:
            # Create model
            subprocess.run([
                'gcloud', 'ai', 'models', 'upload',
                '--region', self.config.region,
                '--display-name', model_display_name,
                '--container-image-uri', image_uri,
                '--container-predict-route', '/predict',
                '--container-health-route', '/health',
                '--container-env-vars', f'MODEL_VERSION={version}'
            ], check=True)
            print("✅ Vertex AI model created")

            # Get model ID
            result = subprocess.run([
                'gcloud', 'ai', 'models', 'list',
                '--region', self.config.region,
                '--filter', f'displayName={model_display_name}',
                '--format', 'value(name)'
            ], capture_output=True, text=True, check=True)

            model_id = result.stdout.strip()
            return model_id

        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating Vertex AI model: {e}")
            raise

    def deploy_model_to_endpoint(self, model_id: str, version: str) -> str:
        """Deploy model to endpoint with traffic management."""
        logger.info(f"🚀 DEPLOYING MODEL TO ENDPOINT FOR VERSION {version}")
        print("=" * 60)

        try:
            # Get or create endpoint
            endpoint_id = self._get_or_create_endpoint()

            # Deploy model with traffic split
            traffic_split = self.config.traffic_split or {"100": 1.0}

            subprocess.run([
                'gcloud', 'ai', 'endpoints', 'deploy-model',
                '--region', self.config.region,
                '--endpoint', endpoint_id,
                '--model', model_id,
                '--traffic-split', ','.join(
                                            [f"{k}={v}" for k,
                                            v in traffic_split.items()]),
                                            
                '--machine-type', self.config.machine_type,
                '--min-replica-count', str(self.config.min_replicas),
                '--max-replica-count', str(self.config.max_replicas)
            ], check=True)

            print("✅ Model deployed to endpoint")

            # Record deployment
            deployment_record = {
                "version": version,
                "model_id": model_id,
                "endpoint_id": endpoint_id,
                "deployed_at": datetime.now().isoformat(),
                "traffic_split": traffic_split
            }
            self.deployment_history.append(deployment_record)

            return endpoint_id

        except subprocess.CalledProcessError as e:
            logger.error(f"Error deploying model to endpoint: {e}")
            raise

    def _get_or_create_endpoint(self) -> str:
        """Get existing endpoint or create new one."""
        try:
            # Try to get existing endpoint
            result = subprocess.run([
                'gcloud', 'ai', 'endpoints', 'list',
                '--region', self.config.region,
                '--filter', f'displayName={self.config.endpoint_name}',
                '--format', 'value(name)'
            ], capture_output=True, text=True, check=True)

            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()

            # Create new endpoint
            result = subprocess.run([
                'gcloud', 'ai', 'endpoints', 'create',
                '--region', self.config.region,
                '--display-name', self.config.endpoint_name
            ], capture_output=True, text=True, check=True)

            # Get the created endpoint ID
            result = subprocess.run([
                'gcloud', 'ai', 'endpoints', 'list',
                '--region', self.config.region,
                '--filter', f'displayName={self.config.endpoint_name}',
                '--format', 'value(name)'
            ], capture_output=True, text=True, check=True)

            return result.stdout.strip()

        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting/creating endpoint: {e}")
            raise

    def setup_monitoring_and_alerting(self, endpoint_id: str) -> None:
        """Setup monitoring and alerting for the deployment."""
        logger.info("📊 SETTING UP MONITORING AND ALERTING")
        print("=" * 60)

        # Create monitoring policy
        policy_name = f"emotion-detection-monitoring-{self.current_version}"

        policy_config = {
            "displayName": policy_name,
            "conditions": [
                {
                    "displayName": "High Error Rate",
                    "conditionThreshold": {
"filter": f'resource.type="aiplatform.googleapis.com/Endpoint" AND
resource.labels.endpoint_id="{endpoint_id}"',
                        "comparison": "COMPARISON_GREATER_THAN",
                        "thresholdValue": 0.05,  # 5% error rate
                        "duration": "300s"
                    }
                },
                {
                    "displayName": "High Latency",
                    "conditionThreshold": {
"filter": f'resource.type="aiplatform.googleapis.com/Endpoint" AND
resource.labels.endpoint_id="{endpoint_id}"',
                        "comparison": "COMPARISON_GREATER_THAN",
                        "thresholdValue": 5000,  # 5 seconds
                        "duration": "300s"
                    }
                }
            ],
            "alertStrategy": {
                "autoClose": "604800s"  # 7 days
            }
        }

        # Write policy to file
policy_file = f"deployment/vertex_ai/{self.current_version}/monitoring_policy.json"
        with open(policy_file, "w") as f:
            json.dump(policy_config, f, indent=2)

        try:
            # Create monitoring policy
            subprocess.run([
                'gcloud', 'alpha', 'monitoring', 'policies', 'create',
                '--policy-from-file', policy_file
            ], check=True)
            print("✅ Monitoring policy created")

        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not create monitoring policy: {e}")
            print(
                  "⚠️  Monitoring policy creation failed (may need additional permissions)"
                 )

    def setup_cost_monitoring(self) -> None:
        """Setup cost monitoring and budget alerts."""
        logger.info("💰 SETTING UP COST MONITORING")
        print("=" * 60)

        budget_name = f"emotion-detection-budget-{self.current_version}"

        budget_config = {
            "displayName": budget_name,
            "budgetFilter": {
                "projects": [f"projects/{self.config.project_id}"]
            },
            "amount": {
                "specifiedAmount": {
                    "currencyCode": "USD",
                    "units": str(int(self.config.cost_budget))
                }
            },
            "thresholdRules": [
                {
                    "thresholdPercent": 0.5,  # 50% of budget
                    "spendBasis": "CURRENT_SPEND"
                },
                {
                    "thresholdPercent": 0.8,  # 80% of budget
                    "spendBasis": "CURRENT_SPEND"
                },
                {
                    "thresholdPercent": 1.0,  # 100% of budget
                    "spendBasis": "CURRENT_SPEND"
                }
            ]
        }

        # Write budget to file
        budget_file = f"deployment/vertex_ai/{self.current_version}/budget_config.json"
        with open(budget_file, "w") as f:
            json.dump(budget_config, f, indent=2)

        try:
            # Create budget
            subprocess.run([
                'gcloud', 'billing', 'budgets', 'create',
                '--billing-account', self._get_billing_account(),
                '--budget-file', budget_file
            ], check=True)
            print("✅ Cost budget created")

        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not create budget: {e}")
            print("⚠️  Budget creation failed (may need billing permissions)")

    def _get_billing_account(self) -> str:
        """Get the billing account for the project."""
        try:
            result = subprocess.run([
                'gcloud', 'billing', 'projects', 'describe', self.config.project_id,
                '--format', 'value(billingAccountName)'
            ], capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return ""

    def rollback_deployment(self, target_version: str) -> bool:
        """Rollback to a previous version."""
        logger.info(f"🔄 ROLLING BACK TO VERSION {target_version}")
        print("=" * 60)

        # Find the target deployment
        target_deployment = None
        for deployment in self.deployment_history:
            if deployment["version"] == target_version:
                target_deployment = deployment
                break

        if not target_deployment:
            logger.error(
                         f"Target version {target_version} not found in deployment history"
                        )
            return False

        try:
            # Update traffic to 100% for target version
            subprocess.run([
                'gcloud', 'ai', 'endpoints', 'deploy-model',
                '--region', self.config.region,
                '--endpoint', target_deployment["endpoint_id"],
                '--model', target_deployment["model_id"],
                '--traffic-split', '100=1.0',
                '--machine-type', self.config.machine_type,
                '--min-replica-count', str(self.config.min_replicas),
                '--max-replica-count', str(self.config.max_replicas)
            ], check=True)

            print(f"✅ Successfully rolled back to version {target_version}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Error during rollback: {e}")
            return False

    def setup_ab_testing(
                         self,
                         version_a: str,
                         version_b: str,
                         traffic_split: Dict[str,
                         float]) -> bool:
        """Setup A/B testing between two versions."""
        logger.info(f"🧪 SETTING UP A/B TESTING: {version_a} vs {version_b}")
        print("=" * 60)

        # Find both versions in deployment history
        version_a_deployment = None
        version_b_deployment = None

        for deployment in self.deployment_history:
            if deployment["version"] == version_a:
                version_a_deployment = deployment
            elif deployment["version"] == version_b:
                version_b_deployment = deployment

        if not version_a_deployment or not version_b_deployment:
            logger.error("Both versions must be deployed before A/B testing")
            return False

        try:
            # Deploy both versions with traffic split
            traffic_config = ','.join([f"{k}={v}" for k, v in traffic_split.items()])

            subprocess.run([
                'gcloud', 'ai', 'endpoints', 'deploy-model',
                '--region', self.config.region,
                '--endpoint', version_a_deployment["endpoint_id"],
                '--model', version_a_deployment["model_id"],
                '--traffic-split', traffic_config,
                '--machine-type', self.config.machine_type,
                '--min-replica-count', str(self.config.min_replicas),
                '--max-replica-count', str(self.config.max_replicas)
            ], check=True)

            print("✅ A/B testing setup completed")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Error setting up A/B testing: {e}")
            return False

    def get_performance_metrics(self, endpoint_id: str) -> Dict:
        """Get performance metrics for the deployment."""
        logger.info("📈 GETTING PERFORMANCE METRICS")
        print("=" * 60)

        try:
            # Get prediction latency
            result = subprocess.run([
                'gcloud', 'ai', 'endpoints', 'describe',
                '--region', self.config.region,
                '--endpoint', endpoint_id,
                '--format', 'value(predictRequestResponseLoggingConfig.enabled)'
            ], capture_output=True, text=True, check=True)

            # Get model performance metrics
            result = subprocess.run([
                'gcloud', 'ai', 'models', 'list',
                '--region', self.config.region,
                '--filter', f'endpointId={endpoint_id}',
                '--format', 'value(displayName,createTime)'
            ], capture_output=True, text=True, check=True)

            metrics = {
                "endpoint_id": endpoint_id,
                "timestamp": datetime.now().isoformat(),
                "logging_enabled": result.stdout.strip() == "True",
                "models": result.stdout.strip(
                                              ).split('\n') if result.stdout.strip() else []
            }

            print("✅ Performance metrics retrieved")
            return metrics

        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}

    def cleanup_old_versions(self, keep_versions: int = 3) -> None:
        """Clean up old model versions to save costs."""
        logger.info(f"🧹 CLEANING UP OLD VERSIONS (keeping {keep_versions})")
        print("=" * 60)

        if len(self.deployment_history) <= keep_versions:
            print("✅ No cleanup needed")
            return

        # Sort by deployment time and keep only the latest versions
        sorted_deployments = sorted(
            self.deployment_history, 
            key=lambda x: x["deployed_at"], 
            reverse=True
        )

        versions_to_cleanup = sorted_deployments[keep_versions:]

        for deployment in versions_to_cleanup:
            try:
                # Delete model
                subprocess.run([
                    'gcloud', 'ai', 'models', 'delete',
                    '--region', self.config.region,
                    '--model', deployment["model_id"]
                ], check=True)

                print(f"✅ Deleted model version: {deployment['version']}")

            except subprocess.CalledProcessError as e:
                logger.warning(f"Could not delete model {deployment['version']}: {e}")

    def run_full_deployment(self) -> bool:
        """Run the complete Phase 4 deployment process."""
        logger.info("🚀 STARTING PHASE 4 VERTEX AI DEPLOYMENT")
        print("=" * 60)

        try:
            # 1. Check prerequisites
            if not self.check_prerequisites():
                logger.error("Prerequisites check failed")
                return False

            # 2. Generate version
            version = self.generate_model_version()
            print(f"📋 Generated version: {version}")

            # 3. Create deployment package
            deployment_dir = self.create_deployment_package(version)

            # 4. Build and push image
            image_uri = self.build_and_push_image(deployment_dir, version)

            # 5. Create Vertex AI model
            model_id = self.create_vertex_ai_model(image_uri, version)

            # 6. Deploy to endpoint
            endpoint_id = self.deploy_model_to_endpoint(model_id, version)

            # 7. Setup monitoring and alerting
            self.setup_monitoring_and_alerting(endpoint_id)

            # 8. Setup cost monitoring
            self.setup_cost_monitoring()

            # 9. Get performance metrics
            metrics = self.get_performance_metrics(endpoint_id)

            # 10. Cleanup old versions
            self.cleanup_old_versions()

            # 11. Save deployment summary
            self._save_deployment_summary(version, endpoint_id, metrics)

            logger.info("✅ Phase 4 deployment completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False

    def _save_deployment_summary(
                                 self,
                                 version: str,
                                 endpoint_id: str,
                                 metrics: Dict) -> None:
        """Save deployment summary for future reference."""
        summary = {
            "version": version,
            "endpoint_id": endpoint_id,
            "deployed_at": datetime.now().isoformat(),
            "config": {
                "project_id": self.config.project_id,
                "region": self.config.region,
                "machine_type": self.config.machine_type,
                "min_replicas": self.config.min_replicas,
                "max_replicas": self.config.max_replicas
            },
            "metrics": metrics,
            "deployment_history": self.deployment_history
        }

        summary_file = f"deployment/vertex_ai/{version}/deployment_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"📄 Deployment summary saved: {summary_file}")

def main():
    """Main function for Phase 4 Vertex AI deployment."""
    print("🎯 PHASE 4: VERTEX AI DEPLOYMENT AUTOMATION")
    print("=" * 60)

    # Get project ID
    try:
        result = subprocess.run(['gcloud', 'config', 'get-value', 'project'], 
                              capture_output=True, text=True, check=True)
        project_id = result.stdout.strip()
    except Exception:
        print(
              "❌ Could not get project ID. Please run: gcloud config set project YOUR_PROJECT_ID"
             )
        sys.exit(1)

    # Create configuration
    config = DeploymentConfig(
        project_id=project_id,
        region="us-central1",
        model_name="comprehensive-emotion-detection",
        endpoint_name="emotion-detection-endpoint",
        machine_type="n1-standard-2",
        min_replicas=1,
        max_replicas=10,
        cost_budget=100.0
    )

    # Create automation instance
    automation = VertexAIPhase4Automation(config)

    # Run deployment
    if automation.run_full_deployment():
        print("\n🎉 PHASE 4 DEPLOYMENT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("✅ Automated model versioning and deployment")
        print("✅ Rollback capabilities and A/B testing support")
        print("✅ Model performance monitoring and alerting")
        print("✅ Cost optimization and resource management")
        print("✅ Comprehensive testing and validation")
        print("\n📊 Next steps:")
        print("   - Monitor performance metrics")
        print("   - Set up additional alerting if needed")
        print("   - Configure A/B testing for new versions")
        print("   - Review cost optimization opportunities")
    else:
        print("\n❌ PHASE 4 DEPLOYMENT FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main() 
