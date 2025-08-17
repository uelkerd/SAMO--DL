#!/usr/bin/env python3
""""
Deploy to GCP/Vertex AI
=======================

This script deploys the comprehensive emotion detection model to GCP/Vertex AI
for production use.
""""

import json
import os
import subprocess
import sys
from datetime import datetime

def check_prerequisites():
    """Check if all prerequisites are met for GCP deployment."""
    print(" CHECKING DEPLOYMENT PREREQUISITES")
    print("=" * 50)

    # Check if gcloud is installed
    try:
        result = subprocess.run(['gcloud', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(" gcloud CLI is installed")
        else:
            print("❌ gcloud CLI is not installed or not working")
            return False
    except FileNotFoundError:
        print("❌ gcloud CLI is not installed")
        print("   Install from: https://cloud.google.com/sdk/docs/install")
        return False

    # Check if user is authenticated
    try:
        result = subprocess.run(['gcloud', 'auth', 'list', '--filter=status:ACTIVE'], capture_output=True, text=True)
        if result.returncode == 0 and 'ACTIVE' in result.stdout:
            print(" User is authenticated with gcloud")
        else:
            print("❌ User is not authenticated with gcloud")
            print("   Run: gcloud auth login")
            return False
    except Exception as e:
        print(f"❌ Error checking authentication: {e}")
        return False

    # Check if project is set
    try:
        result = subprocess.run(['gcloud', 'config', 'get-value', 'project'], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            project_id = result.stdout.strip()
            print(f" Project is set: {project_id}")
        else:
            print("❌ No project is set")
            print("   Run: gcloud config set project YOUR_PROJECT_ID")
            return False
    except Exception as e:
        print(f"❌ Error checking project: {e}")
        return False

    # Check if Vertex AI API is enabled
    try:
        result = subprocess.run(['gcloud', 'services', 'list', '--enabled', '--filter=name:aiplatform.googleapis.com'], capture_output=True, text=True)
        if result.returncode == 0 and 'aiplatform.googleapis.com' in result.stdout:
            print(" Vertex AI API is enabled")
        else:
            print("❌ Vertex AI API is not enabled")
            print("   Run: gcloud services enable aiplatform.googleapis.com")
            return False
    except Exception as e:
        print(f"❌ Error checking Vertex AI API: {e}")
        return False

    print(" All prerequisites are met!")
    return True

        def prepare_model_for_deployment():
    """Prepare the model for deployment."""
    print("\n📦 PREPARING MODEL FOR DEPLOYMENT")
    print("=" * 50)

    # Check if default model exists
    default_model_path = "deployment/models/default"
        if not os.path.exists(default_model_path):
        print(f"❌ Default model not found at: {default_model_path}")
        return False

    # Check model files
    required_files = ['config.json', 'model.safetensors', 'tokenizer.json', 'vocab.json']
    missing_files = []

        for file in required_files:
        if not os.path.exists(os.path.join(default_model_path, file)):
            missing_files.append(file)

        if missing_files:
        print(f"❌ Missing model files: {missing_files}")
        return False

    print(" Model files are complete")

    # Read model metadata
    metadata_path = os.path.join(default_model_path, "model_metadata.json")
        if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(" Model metadata: {metadata.get("version', 'Unknown')}")"
        print("   Performance: {metadata.get("performance', {}).get('test_accuracy', 'Unknown')}")"
    else:
        print("⚠️ No model metadata found")

    return True

        def create_deployment_package():
    """Create a deployment package for Vertex AI."""
    print("\n📦 CREATING DEPLOYMENT PACKAGE")
    print("=" * 50)

    # Create deployment directory
    deployment_dir = "gcp_deployment"
        if os.path.exists(deployment_dir):
        import shutil
        shutil.rmtree(deployment_dir)
    os.makedirs(deployment_dir)

    # Copy model files
    model_source = "deployment/models/default"
    model_dest = os.path.join(deployment_dir, "model")

    import shutil
    shutil.copytree(model_source, model_dest)
    print(f" Model copied to: {model_dest}")

    # Create prediction script
    prediction_script = '''#!/usr/bin/env python3'
""""
Vertex AI Prediction Script
==========================

This script handles predictions for the emotion detection model on Vertex AI.
""""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

        class EmotionDetectionModel:
        def __init__(self):
        """Initialize the model."""
        self.model_path = os.path.join(os.getcwd(), "model")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')

        self.emotions = ['anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful', 'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired']

        def predict(self, text):
        """Make a prediction."""
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_label = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_label].item()

            # Get all probabilities
            all_probs = probabilities[0].cpu().numpy()

        # Get predicted emotion
        if predicted_label in self.model.config.id2label:
            predicted_emotion = self.model.config.id2label[predicted_label]
        elif str(predicted_label) in self.model.config.id2label:
            predicted_emotion = self.model.config.id2label[str(predicted_label)]
        else:
            predicted_emotion = f"unknown_{predicted_label}"

        # Create response
        response = {
            'text': text,
            'predicted_emotion': predicted_emotion,
            'confidence': float(confidence),
            'probabilities': {
                emotion: float(prob) for emotion, prob in zip(self.emotions, all_probs)
            },
            'model_version': '2.0',
            'model_type': 'comprehensive_emotion_detection'
        }

        return response

# Initialize model
model = EmotionDetectionModel()

        def predict(request):
    """Vertex AI prediction function."""
    try:
        # Parse request
        if isinstance(request, str):
            request_json = json.loads(request)
        else:
            request_json = request

        # Get text from request
        text = request_json.get('text', '')
        if not text:
            return json.dumps({'error': 'No text provided'})

        # Make prediction
        result = model.predict(text)

        return json.dumps(result)

    except Exception as e:
        return json.dumps({'error': str(e)})
''''

    with open(os.path.join(deployment_dir, "predict.py"), 'w') as f:
        f.write(prediction_script)
    print(" Prediction script created")

    # Create requirements.txt
    requirements = '''torch>=2.0.0'
transformers>=4.30.0
numpy>=1.21.0
''''

    with open(os.path.join(deployment_dir, "requirements.txt"), 'w') as f:
        f.write(requirements)
    print(" Requirements file created")

    # Create Dockerfile
    dockerfile = '''FROM python:3.9-slim'

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and prediction script
COPY model/ ./model/
COPY predict.py .

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/model

# Expose port
EXPOSE 8080

# Run the prediction service
CMD ["python", "predict.py"]
''''

    with open(os.path.join(deployment_dir, "Dockerfile"), 'w') as f:
        f.write(dockerfile)
    print(" Dockerfile created")

    # Create deployment configuration
    deployment_config = {
        'model_info': {
            'name': 'comprehensive_emotion_detection',
            'version': '2.0',
            'description': 'Comprehensive emotion detection model with focal loss, class weighting, and advanced data augmentation',
            'performance': {
                'basic_accuracy': '100.00%',
                'real_world_accuracy': '93.75%',
                'average_confidence': '83.9%'
            }
        },
        'deployment_info': {
            'created_at': datetime.now().isoformat(),
            'model_path': model_source,
            'deployment_package': deployment_dir
        }
    }

    with open(os.path.join(deployment_dir, "deployment_config.json"), 'w') as f:
        json.dump(deployment_config, f, indent=2)
    print(" Deployment configuration created")

    print(f" Deployment package created at: {deployment_dir}")
    return deployment_dir

        def deploy_to_vertex_ai(deployment_dir):
    """Deploy the model to Vertex AI."""
    print("\n🚀 DEPLOYING TO VERTEX AI")
    print("=" * 50)

    # Get project ID
    result = subprocess.run(['gcloud', 'config', 'get-value', 'project'], capture_output=True, text=True)
    project_id = result.stdout.strip()

    # Set region
    region = "us-central1"  # You can change this

    # Create model name
    model_name = "comprehensive-emotion-detection"
    endpoint_name = "emotion-detection-endpoint"

    print(" Deployment Configuration:")
    print(f"   Project ID: {project_id}")
    print(f"   Region: {region}")
    print(f"   Model Name: {model_name}")
    print(f"   Endpoint Name: {endpoint_name}")
    print()

    # Build and push Docker image
    print("🐳 Building and pushing Docker image...")

    # Create repository name
    repository_name = "emotion-detection"

    # Configure Docker for gcloud
    subprocess.run(['gcloud', 'auth', 'configure-docker'], check=True)

    # Build and push image
    image_uri = f"gcr.io/{project_id}/{repository_name}:latest"

    try:
        # Build image
        subprocess.run([)
            'docker', 'build', '-t', image_uri, deployment_dir
(        ], check=True)
        print(" Docker image built")

        # Push image
        subprocess.run(['docker', 'push', image_uri], check=True)
        print(" Docker image pushed to Container Registry")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error building/pushing Docker image: {e}")
        return False

    # Create Vertex AI model
    print("\n🤖 Creating Vertex AI model...")

    try:
        # Create model
        subprocess.run([)
            'gcloud', 'ai', 'models', 'upload',
            '--region', region,
            '--display-name', model_name,
            '--container-image-uri', image_uri,
            '--container-predict-route', '/predict',
            '--container-health-route', '/health'
(        ], check=True)
        print(" Vertex AI model created")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating Vertex AI model: {e}")
        return False

    # Create endpoint
    print("\n🌐 Creating endpoint...")

    try:
        subprocess.run([)
            'gcloud', 'ai', 'endpoints', 'create',
            '--region', region,
            '--display-name', endpoint_name
(        ], check=True)
        print(" Endpoint created")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating endpoint: {e}")
        return False

    # Deploy model to endpoint
    print("\n🚀 Deploying model to endpoint...")

    try:
        # Get model ID
        result = subprocess.run([)
            'gcloud', 'ai', 'models', 'list',
            '--region', region,
            '--filter', f'displayName={model_name}',
            '--format', 'value(name)'
(        ], capture_output=True, text=True, check=True)

        model_id = result.stdout.strip()

        # Get endpoint ID
        result = subprocess.run([)
            'gcloud', 'ai', 'endpoints', 'list',
            '--region', region,
            '--filter', f'displayName={endpoint_name}',
            '--format', 'value(name)'
(        ], capture_output=True, text=True, check=True)

        endpoint_id = result.stdout.strip()

        # Deploy model
        subprocess.run([)
            'gcloud', 'ai', 'endpoints', 'deploy-model', endpoint_id,
            '--region', region,
            '--model', model_id,
            '--display-name', f'{model_name}-deployment',
            '--machine-type', 'n1-standard-2',
            '--min-replica-count', '1',
            '--max-replica-count', '10'
(        ], check=True)
        print(" Model deployed to endpoint")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error deploying model: {e}")
        return False

    print("\n DEPLOYMENT COMPLETE!")
    print(f" Endpoint ID: {endpoint_id}")
    print(f"🌐 Region: {region}")
    print(f"🤖 Model: {model_name}")

    # Create deployment summary
    deployment_summary = {
        'status': 'success',
        'timestamp': datetime.now().isoformat(),
        'project_id': project_id,
        'region': region,
        'model_name': model_name,
        'endpoint_id': endpoint_id,
        'image_uri': image_uri,
        'deployment_dir': deployment_dir
    }

    with open(os.path.join(deployment_dir, "deployment_summary.json"), 'w') as f:
        json.dump(deployment_summary, f, indent=2)

    print(f"\n📁 Deployment summary saved to: {deployment_dir}/deployment_summary.json")

    return True

        def main():
    """Main deployment function."""
    print("🚀 GCP/VERTEX AI DEPLOYMENT")
    print("=" * 60)
    print("⏰ Started at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S')}")"
    print()

    # Check prerequisites
        if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return False

    # Prepare model
        if not prepare_model_for_deployment():
        print("\n❌ Model preparation failed.")
        return False

    # Create deployment package
    deployment_dir = create_deployment_package()
        if not deployment_dir:
        print("\n❌ Failed to create deployment package.")
        return False

    # Deploy to Vertex AI
        if not deploy_to_vertex_ai(deployment_dir):
        print("\n❌ Deployment to Vertex AI failed.")
        return False

    print("\n DEPLOYMENT SUCCESSFUL!")
    print("=" * 60)
    print("Your comprehensive emotion detection model is now deployed on GCP/Vertex AI!")
    print("You can now make predictions using the Vertex AI endpoint.")

    return True

        if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
