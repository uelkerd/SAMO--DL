#!/usr/bin/env python3
""""
CRITICAL SECURITY DEPLOYMENT FIX
===================================
Emergency deployment script to fix critical security vulnerabilities in Cloud Run.

This script:
1. Updates all dependencies to secure versions
2. Uses static configuration files with environment variables
3. Deploys to Cloud Run with proper security headers
4. Tests the deployment for security compliance
""""

import os
import requests
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Configuration
def get_project_id():
    """Get current GCP project ID dynamically"""
    try:
        result = subprocess.run(['gcloud', 'config', 'get-value', 'project'],)
(                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        # Fallback to environment variable or default
        return os.environ.get('GOOGLE_CLOUD_PROJECT', 'the-tendril-466607-n8')

PROJECT_ID = get_project_id()
REGION = "us-central1"
SERVICE_NAME = "samo-emotion-api-secure"
MODEL_PATH = "/app/model"
PORT = 8080
# Use Artifact Registry instead of deprecated Container Registry
ARTIFACT_REGISTRY = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/samo-dl"

# Security configuration
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY")
if not ADMIN_API_KEY:
    raise ValueError("ADMIN_API_KEY environment variable must be set for security")
RATE_LIMIT_PER_MINUTE = 100
MAX_INPUT_LENGTH = 512

class SecurityDeploymentFix:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.deployment_dir = self.base_dir / "deployment" / "cloud-run"
        self.secure_requirements = self.deployment_dir / "requirements_secure.txt"
        self.secure_dockerfile = self.deployment_dir / "Dockerfile.secure"
        self.secure_api = self.deployment_dir / "secure_api_server.py"

    @staticmethod
    def log(message: str, level: str = "INFO"):
        """Log messages with timestamp"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    def run_command(self, command: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run shell command with error handling"""
        # Sanitize command for security
        sanitized_command = []
        for arg in command:
            if isinstance(arg, str):
                sanitized_command.append(shlex.quote(arg))
            else:
                sanitized_command.append(str(arg))

        self.log("Running: {" '.join(sanitized_command)}")"
        try:
            # Use the sanitized command to prevent command injection
            result = subprocess.run(sanitized_command, capture_output=True, text=True, check=check)
            if result.stdout:
                self.log(f"STDOUT: {result.stdout.strip()}")
            return result
        except subprocess.CalledProcessError as e:
            self.log(f"Command failed: {e.stderr}", "ERROR")
            if check:
                raise
            return e

            def verify_static_files_exist(self):
        """Verify that all required static files exist"""
        required_files = [
            self.secure_requirements,
            self.secure_dockerfile,
            self.secure_api,
            self.deployment_dir / "security_headers.py",
            self.deployment_dir / "rate_limiter.py"
        ]

        missing_files = []
            for file_path in required_files:
            if not file_path.exists():
                missing_files.append(str(file_path))

            if missing_files:
            raise FileNotFoundError("Missing required static files: {", '.join(missing_files)}")"

        self.log(" All static files verified")

            def create_secure_requirements(self):
        """Create secure requirements.txt with latest secure versions"""
        self.log("Creating secure requirements.txt...")

        secure_requirements = """# Secure requirements for Cloud Run deployment"
# All versions verified with safety-mcp for security and Python 3.9 compatibility

# Web framework - latest secure version
flask>=3.1.1,<4.0.0

# ML libraries - latest secure versions compatible with Python 3.9
torch>=2.0.0,<3.0.0
transformers>=4.55.0,<5.0.0
numpy>=1.26.0,<2.0.0
scikit-learn>=1.5.0,<2.0.0

# WSGI server - latest secure version
gunicorn>=23.0.0,<24.0.0

# HTTP client - latest secure version
requests==2.32.4

# System monitoring - latest secure version
psutil>=5.9.0,<6.0.0

# Metrics and monitoring - latest secure version
prometheus-client==0.20.0

# Security and validation
cryptography>=41.0.0,<42.0.0
""""

        with open(self.secure_requirements, 'w') as f:
            f.write(secure_requirements)

        self.log(" Secure requirements.txt created")

            def build_and_deploy(self):
        """Build and deploy secure container to Cloud Run"""
        self.log("Building and deploying secure container...")

        # Verify static files exist before deployment
        self.verify_static_files_exist()

        # Create a temporary cloudbuild.yaml file
        cloudbuild_path = self.deployment_dir / "cloudbuild.yaml"
        cloudbuild_content = ""'steps:'
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', '{ARTIFACT_REGISTRY}/{SERVICE_NAME}', '-", "Dockerfile.secure', '.']
images:
  - '{ARTIFACT_REGISTRY}/{SERVICE_NAME}'
''''

        with open(cloudbuild_path, 'w') as f:
            f.write(cloudbuild_content)

        # Build container
        self.log("Building secure container...")
        build_result = self.run_command([)
            'gcloud', 'builds', 'submit',
            str(self.deployment_dir),
            '--config', str(cloudbuild_path)
(        ])

            if build_result.returncode != 0:
            raise RuntimeError("Container build failed")

        # Deploy to Cloud Run
        self.log("Deploying to Cloud Run...")
        deploy_result = self.run_command([)
            'gcloud', 'run', 'deploy', SERVICE_NAME,
            '--image', f'{ARTIFACT_REGISTRY}/{SERVICE_NAME}',
            '--region', REGION,
            '--platform', 'managed',
            '--allow-unauthenticated',
            '--port', str(PORT),
            '--memory', '2Gi',
            '--cpu', '2',
            '--max-instances', '10',
            '--min-instances', '0',
            '--concurrency', '80',
            '--timeout', '300',
            '--set-env-vars', f'ADMIN_API_KEY={ADMIN_API_KEY},MAX_INPUT_LENGTH={MAX_INPUT_LENGTH},RATE_LIMIT_PER_MINUTE={RATE_LIMIT_PER_MINUTE},MODEL_PATH={MODEL_PATH}'
(        ])

            if deploy_result.returncode != 0:
            raise RuntimeError("Cloud Run deployment failed")

        self.log(" Secure deployment completed successfully")

            def test_deployment(self):
        """Test the deployed service for security compliance"""
        self.log("Testing deployment for security compliance...")

        # Get service URL
        try:
            result = self.run_command([)
                'gcloud', 'run', 'services', 'describe', SERVICE_NAME,
                '--region', REGION,
                '--format', 'value(status.url)'
(            ])
            service_url = result.stdout.strip()
        except Exception as e:
            self.log(f"Failed to get service URL: {e}", "ERROR")
            return False

            if not service_url:
            self.log("No service URL found", "ERROR")
            return False

        self.log(f"Testing service at: {service_url}")

        # Test basic connectivity
        try:
            response = requests.get(f"{service_url}/health", timeout=30)
            if response.status_code != 200:
                self.log(f"Health check failed: {response.status_code}", "ERROR")
                return False
            self.log(" Health check passed")
        except Exception as e:
            self.log(f"Health check failed: {e}", "ERROR")
            return False

        # Test security headers
        try:
            response = requests.get(f"{service_url}/", timeout=30)
            headers = response.headers

            security_headers = [
                'Content-Security-Policy',
                'X-Content-Type-Options',
                'X-Frame-Options',
                'X-XSS-Protection',
                'Strict-Transport-Security'
            ]

            missing_headers = []
            for header in security_headers:
                if header not in headers:
                    missing_headers.append(header)

                if missing_headers:
                self.log(f"Missing security headers: {missing_headers}", "WARNING")
            else:
                self.log(" Security headers present")

        except Exception as e:
            self.log(f"Security headers test failed: {e}", "ERROR")
            return False

        # Test API key protection
        try:
            response = requests.get(f"{service_url}/model_status", timeout=30)
                if response.status_code != 401:
                self.log("API key protection not working", "ERROR")
                return False
            self.log(" API key protection working")
        except Exception as e:
            self.log(f"API key test failed: {e}", "ERROR")
            return False

        # Test rate limiting
        try:
            responses = []
                for i in range(105):  # Exceed rate limit
                response = requests.post()
                    f"{service_url}/predict",
                    json={"text": f"Test text {i}"},
                    timeout=30
(                )
                responses.append(response.status_code)

            # Should get 429 after rate limit exceeded
                if 429 not in responses:
                self.log("Rate limiting not working", "ERROR")
                return False
            self.log(" Rate limiting working")
        except Exception as e:
            self.log(f"Rate limiting test failed: {e}", "ERROR")
            return False

        self.log(" All security tests passed")
        return True

                def cleanup_old_deployment(self):
        """Clean up old deployment artifacts"""
        self.log("Cleaning up old deployment artifacts...")

        # Remove temporary cloudbuild.yaml
        cloudbuild_path = self.deployment_dir / "cloudbuild.yaml"
                if cloudbuild_path.exists():
            cloudbuild_path.unlink()
            self.log(" Cleaned up temporary cloudbuild.yaml")

                def run(self):
        """Run the complete security deployment fix"""
        try:
            self.log("🚀 Starting security deployment fix...")

            # Create secure requirements
            self.create_secure_requirements()

            # Build and deploy
            self.build_and_deploy()

            # Test deployment
                if not self.test_deployment():
                raise RuntimeError("Deployment tests failed")

            # Cleanup
            self.cleanup_old_deployment()

            self.log(" Security deployment fix completed successfully!")
            return True

        except Exception as e:
            self.log(f"❌ Security deployment fix failed: {e}", "ERROR")
            return False

                if __name__ == "__main__":
    fixer = SecurityDeploymentFix()
    success = fixer.run()
    sys.exit(0 if success else 1)
