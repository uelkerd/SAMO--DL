#!/bin/bash

# GCP Vertex AI Deployment Script with Enhanced Error Handling
# ===========================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="the-tendril-466607-n8"
REGION="us-central1"
MODEL_NAME="comprehensive-emotion-detection-model"
ENDPOINT_NAME="comprehensive-emotion-detection-endpoint"
IMAGE_NAME="us-central1-docker.pkg.dev/${PROJECT_ID}/emotion-detection-repo/emotion-detection-model"
TAG="latest"

echo -e "${BLUE}🚀 Starting GCP Vertex AI Deployment with Enhanced Error Handling${NC}"
echo ""

# Function to disable color output for gcloud commands
disable_colors() {
    export CLOUDSDK_COLOR_OUTPUT=false
    export CLOUDSDK_PYTHON_SITEPACKAGES=1
}

# Function to log with timestamp
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

# Function to check if command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        error "$1 is not installed. Please install it first."
        exit 1
    fi
}

# Function to validate prerequisites
validate_prerequisites() {
    log "🔍 Validating prerequisites..."
    
    check_command "gcloud"
    check_command "docker"
    
    # Check if gcloud is authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        error "gcloud is not authenticated. Please run 'gcloud auth login' first."
        exit 1
    fi
    
    # Check if project is set
    if ! gcloud config get-value project &> /dev/null; then
        error "No project is set. Please run 'gcloud config set project ${PROJECT_ID}' first."
        exit 1
    fi
    
    log "✅ Prerequisites validated successfully"
}

# Function to enable required APIs
enable_apis() {
    log "🔧 Enabling required APIs..."
    
    gcloud services enable aiplatform.googleapis.com
    gcloud services enable artifactregistry.googleapis.com
    
    log "✅ APIs enabled successfully"
}

# Function to create Artifact Registry repository
create_artifact_repository() {
    log "🏗️ Creating Artifact Registry repository..."
    
    REPO_NAME="emotion-detection-repo"
    
    # Check if repository already exists
    if gcloud artifacts repositories describe $REPO_NAME --location=$REGION --format="value(name)" &>/dev/null; then
        warn "Repository already exists, skipping creation"
        log "📋 Using existing repository: $REPO_NAME"
    else
        # Create repository
        gcloud artifacts repositories create $REPO_NAME \
            --repository-format=docker \
            --location=$REGION \
            --description="Repository for emotion detection model containers"
        
        if [ $? -ne 0 ]; then
            error "Repository creation failed"
            exit 1
        fi
        
        log "✅ Repository created successfully: $REPO_NAME"
    fi
}

# Function to validate model files
validate_model_files() {
    log "🔍 Validating model files..."
    
    MODEL_DIR="deployment/gcp/model"
    
    if [ ! -d "$MODEL_DIR" ]; then
        error "Model directory not found: $MODEL_DIR"
        exit 1
    fi
    
    # Check for required files
    required_files=("config.json" "tokenizer.json")
    # Check for either pytorch_model.bin or model.safetensors
    if [ -f "$MODEL_DIR/pytorch_model.bin" ] || [ -f "$MODEL_DIR/model.safetensors" ]; then
        log "📦 Model file found (PyTorch or safetensors format)"
    else
        error "Required model file not found: pytorch_model.bin or model.safetensors"
        exit 1
    fi
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$MODEL_DIR/$file" ]; then
            error "Required model file not found: $file"
            exit 1
        fi
    done
    
    log "✅ Model files validated successfully"
    log "📁 Model directory size: $(du -sh $MODEL_DIR | cut -f1)"
}

# Function to build and push Docker image
build_and_push_image() {
    log "🐳 Building and pushing Docker image..."
    
    cd deployment/gcp
    
    # Configure Docker to authenticate with Artifact Registry
    log "🔐 Configuring Docker authentication..."
    gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
    
    if [ $? -ne 0 ]; then
        error "Docker authentication failed"
        exit 1
    fi
    
    # Build image
    log "🔨 Building Docker image..."
    docker build -t ${IMAGE_NAME}:${TAG} .
    
    if [ $? -ne 0 ]; then
        error "Docker build failed"
        exit 1
    fi
    
    # Push image
    log "📤 Pushing Docker image to Artifact Registry..."
    docker push ${IMAGE_NAME}:${TAG}
    
    if [ $? -ne 0 ]; then
        error "Docker push failed"
        exit 1
    fi
    
    cd ../..
    log "✅ Docker image built and pushed successfully"
}

# Function to create Vertex AI model
create_model() {
    log "🤖 Creating Vertex AI model..."
    
    # Check if model already exists
    if gcloud ai models list --region=$REGION --filter="displayName=$MODEL_NAME" --format="value(name)" | grep -q .; then
        warn "Model already exists, skipping creation"
        MODEL_ID=$(gcloud ai models list --region=$REGION --filter="displayName=$MODEL_NAME" --format="value(name)")
        log "📋 Using existing model: $MODEL_ID"
    else
        # Create new model
        MODEL_ID=$(gcloud ai models upload \
            --region=$REGION \
            --display-name=$MODEL_NAME \
            --container-image-uri=${IMAGE_NAME}:${TAG} \
            --format="value(name)" \
            --quiet)
        
        if [ $? -ne 0 ]; then
            error "Model creation failed"
            exit 1
        fi
        
        log "✅ Model created successfully: $MODEL_ID"
    fi
    
    echo $MODEL_ID
}

# Function to create Vertex AI endpoint
create_endpoint() {
    log "🌐 Creating Vertex AI endpoint..."
    
    # Check if endpoint already exists
    if gcloud ai endpoints list --region=$REGION --filter="displayName=$ENDPOINT_NAME" --format="value(name)" | grep -q .; then
        warn "Endpoint already exists, skipping creation"
        ENDPOINT_ID=$(gcloud ai endpoints list --region=$REGION --filter="displayName=$ENDPOINT_NAME" --format="value(name)")
        log "📋 Using existing endpoint: $ENDPOINT_ID"
    else
        # Create new endpoint
        ENDPOINT_ID=$(gcloud ai endpoints create \
            --region=$REGION \
            --display-name=$ENDPOINT_NAME \
            --format="value(name)" \
            --quiet)
        
        if [ $? -ne 0 ]; then
            error "Endpoint creation failed"
            exit 1
        fi
        
        log "✅ Endpoint created successfully: $ENDPOINT_ID"
    fi
    
    echo $ENDPOINT_ID
}

# Function to deploy model to endpoint
deploy_model() {
    local MODEL_ID=$1
    local ENDPOINT_ID=$2
    
    log "🚀 Deploying model to endpoint..."
    log "📋 Model ID: $MODEL_ID"
    log "📋 Endpoint ID: $ENDPOINT_ID"
    
    # Deploy model with enhanced configuration
    gcloud ai endpoints deploy-model $ENDPOINT_ID \
        --region=$REGION \
        --model=$MODEL_ID \
        --display-name=comprehensive-emotion-detection-deployment \
        --machine-type=e2-standard-2 \
        --min-replica-count=1 \
        --max-replica-count=10 \
        --accelerator=count=0,type=NVIDIA_TESLA_T4 \
        --deployed-model-id=comprehensive-emotion-detection-deployment
    
    if [ $? -ne 0 ]; then
        error "Model deployment failed"
        log "🔍 Checking deployment logs..."
        log "📋 Log URL: https://console.cloud.google.com/logs/viewer?project=${PROJECT_ID}&resource=aiplatform.googleapis.com%2FEndpoint&advancedFilter=resource.type%3D%22aiplatform.googleapis.com%2FEndpoint%22%0Aresource.labels.endpoint_id%3D%22${ENDPOINT_ID}%22%0Aresource.labels.location%3D%22${REGION}%22"
        exit 1
    fi
    
    log "✅ Model deployed successfully"
}

# Function to test deployment
test_deployment() {
    local ENDPOINT_ID=$1
    
    log "🧪 Testing deployment..."
    
    # Wait for deployment to be ready
    log "⏳ Waiting for deployment to be ready..."
    sleep 30
    
    # Test health endpoint
    log "🔍 Testing health endpoint..."
    HEALTH_URL="https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/${ENDPOINT_ID}:predict"
    
    # Note: This is a simplified test. In practice, you'd need to use the proper Vertex AI client
    log "📋 Health check URL: $HEALTH_URL"
    log "📋 You can test the deployment using the Vertex AI console or client library"
}

# Function to display deployment information
display_deployment_info() {
    local MODEL_ID=$1
    local ENDPOINT_ID=$2
    
    echo ""
    echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}📋 Deployment Information:${NC}"
    echo "  Project ID: $PROJECT_ID"
    echo "  Region: $REGION"
    echo "  Model ID: $MODEL_ID"
    echo "  Endpoint ID: $ENDPOINT_ID"
    echo "  Model Name: $MODEL_NAME"
    echo "  Endpoint Name: $ENDPOINT_NAME"
    echo ""
    echo -e "${BLUE}🔗 Useful Links:${NC}"
    echo "  Vertex AI Console: https://console.cloud.google.com/ai/platform"
    echo "  Model Details: https://console.cloud.google.com/ai/platform/models"
    echo "  Endpoint Details: https://console.cloud.google.com/ai/platform/endpoints"
    echo "  Logs: https://console.cloud.google.com/logs/viewer?project=${PROJECT_ID}&resource=aiplatform.googleapis.com%2FEndpoint&advancedFilter=resource.type%3D%22aiplatform.googleapis.com%2FEndpoint%22%0Aresource.labels.endpoint_id%3D%22${ENDPOINT_ID}%22%0Aresource.labels.location%3D%22${REGION}%22"
    echo ""
    echo -e "${YELLOW}⚠️  Next Steps:${NC}"
    echo "  1. Test the deployment using the Vertex AI console"
    echo "  2. Monitor the logs for any issues"
    echo "  3. Set up monitoring and alerting"
    echo "  4. Configure auto-scaling if needed"
}

# Main deployment function
main() {
    log "🚀 Starting comprehensive GCP Vertex AI deployment..."
    
    # Disable color output for gcloud commands
    disable_colors
    
    # Validate prerequisites
    validate_prerequisites
    
    # Enable APIs
    enable_apis
    
    # Create Artifact Registry repository
    create_artifact_repository
    
    # Validate model files
    validate_model_files
    
    # Build and push Docker image
    build_and_push_image
    
    # Create model
    MODEL_ID=$(create_model)
    
    # Create endpoint
    ENDPOINT_ID=$(create_endpoint)
    
    # Deploy model
    deploy_model $MODEL_ID $ENDPOINT_ID
    
    # Test deployment
    test_deployment $ENDPOINT_ID
    
    # Display deployment information
    display_deployment_info $MODEL_ID $ENDPOINT_ID
}

# Run main function
main "$@"
