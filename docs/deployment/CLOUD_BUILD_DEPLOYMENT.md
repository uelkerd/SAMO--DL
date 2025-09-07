# Cloud Build Deployment Guide

## 🚀 Automated Build and Deploy Pipeline

This guide covers the complete automated deployment pipeline using Google Cloud Build for the SAMO Emotion Detection API.

## 🚀 Features

- **Automated Docker Image Builds**: Build optimized Docker images for Cloud Run with build caching for faster subsequent builds.
- **Dynamic Image Tagging**: Use `$BUILD_ID` and `latest` tags to prevent overwrites and enable easy rollbacks.
- **Artifact Registry Integration**: Store images in Google Artifact Registry for better security, management, and performance.
- **Vulnerability Scanning**: Automatically scan built images for security vulnerabilities before deployment.
- **Automated Cloud Run Deployment**: Deploy the built image directly to Cloud Run as part of the build process.
- **Secure API Key Management**: Integrate with Google Secret Manager to handle `ADMIN_API_KEY` securely, avoiding hardcoded secrets in your repository or build logs.
- **Build Caching**: Leverage Docker layer caching to significantly speed up subsequent builds.
- **Comprehensive Logging**: Cloud Build logs provide detailed insights into the build and deployment process.
- **Machine Type Optimization**: Use `E2_HIGHCPU_8` for faster builds.

## 📋 Prerequisites

1. **Google Cloud Project** with billing enabled
2. **gcloud CLI** installed and authenticated
3. **Required APIs** enabled:
   - Cloud Build API
   - Cloud Run API
   - Secret Manager API
   - Artifact Registry API

## 🔧 Setup Instructions

### 1. Enable Required APIs

```bash
gcloud services enable secretmanager.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

### 2. Set Up Artifact Registry (Required)

The enhanced pipeline uses Google Artifact Registry for better image management and security:

```bash
# Run the setup script
./scripts/deployment/setup-artifact-registry.sh

# Or manually:
gcloud artifacts repositories create samo-dl-repo \
  --repository-format=docker \
  --location=us-central1 \
  --description="SAMO-DL Docker images repository"
```

### 3. Set Up Secret Manager (Required)

For production deployments, use Google Secret Manager to store the API key securely:

```bash
# Run the setup script
./scripts/deployment/setup-secret-manager.sh

# Or manually:
gcloud secrets create admin-api-key --replication-policy="automatic"
echo -n "your-secure-api-key-here" | gcloud secrets versions add admin-api-key --data-file=-  # skipcq: SCT-A000 - This is a placeholder, not a real secret
```

### 4. Grant Cloud Build Permissions

```bash
# Get your project number
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")

# Grant Cloud Build access to Secret Manager
gcloud secrets add-iam-policy-binding admin-api-key \
  --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

## 🚀 Deployment Options

### Single Configuration Approach

The deployment uses a consolidated `cloudbuild.yaml` that provides:

- ✅ **Secure by Default**: Uses Google Secret Manager for API key storage
- ✅ **Fully Parameterized**: All values configurable via substitutions
- ✅ **Production Ready**: Comprehensive logging and monitoring
- ✅ **Easy Customization**: Copy `cloudbuild.example.yaml` for custom configurations

### Deploy with Default Settings

```bash
gcloud builds submit --config cloudbuild.yaml
```

### Deploy with Custom Configuration

```bash
# Copy example configuration
cp cloudbuild.example.yaml cloudbuild.yaml

# Edit cloudbuild.yaml with your custom values
# Then deploy
gcloud builds submit --config cloudbuild.yaml
```

## 📊 Build Configuration Details

### Dynamic Image Tagging

Both configurations use `$BUILD_ID` for unique image tags:

```yaml
# Creates unique tags for each build
- 'us-central1-docker.pkg.dev/$PROJECT_ID/${_ARTIFACT_REPO}/emotion-detection-api:$BUILD_ID'
- 'us-central1-docker.pkg.dev/$PROJECT_ID/${_ARTIFACT_REPO}/emotion-detection-api:latest'
```

**Benefits:**
- ✅ No image overwrites
- ✅ Easy rollbacks
- ✅ Build history tracking

### Automated Cloud Run Deployment

The pipeline automatically deploys to Cloud Run with parameterized configuration:

```yaml
- name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
  entrypoint: 'gcloud'
  args: [
    'run', 'deploy', '${_SERVICE_NAME}',
    '--image', 'us-central1-docker.pkg.dev/$PROJECT_ID/${_ARTIFACT_REPO}/emotion-detection-api:$BUILD_ID',
    '--region', '${_REGION}',
    '--platform', 'managed',
    '--allow-unauthenticated',
    '--port', '${_PORT}',
    '--memory', '${_MEMORY}',
    '--cpu', '${_CPU}',
    '--max-instances', '${_MAX_INSTANCES}'
  ]
```

### Configurable Parameters

All deployment settings are parameterized and can be customized:

| Parameter | Default | Description | Options |
|-----------|---------|-------------|---------|
| `_SERVICE_NAME` | `emotion-detection-api` | Cloud Run service name | Any valid service name |
| `_REGION` | `us-central1` | Deployment region | Any valid GCP region |
| `_PORT` | `8080` | Service port | Any valid port number |
| `_MEMORY` | `2Gi` | Memory allocation | 128Mi, 256Mi, 512Mi, 1Gi, 2Gi, 4Gi, 8Gi |
| `_CPU` | `2` | CPU allocation | 1, 2, 4, 6, 8 |
| `_MAX_INSTANCES` | `10` | Maximum instances | Any positive integer |
| `_MACHINE_TYPE` | `E2_HIGHCPU_8` | Build machine type | E2_STANDARD_2, E2_HIGHCPU_4, E2_HIGHCPU_8 |
| `_DISK_SIZE` | `100` | Build disk size (GB) | Any positive integer |

## 🔒 Security Features

### Secret Manager Integration

```yaml
availableSecrets:
  secretManager:
    - versionName: projects/$PROJECT_ID/secrets/admin-api-key/versions/latest
      env: 'ADMIN_API_KEY'
```

### Secure API Key Management

The API key is securely retrieved from Google Secret Manager:

```yaml
availableSecrets:
  secretManager:
    - versionName: projects/$PROJECT_ID/secrets/admin-api-key/versions/latest
      env: 'ADMIN_API_KEY'
```

**Benefits:**
- ✅ No API keys in build logs
- ✅ Centralized secret management
- ✅ Automatic key rotation support
- ✅ Audit trail for secret access

## 📈 Build Performance

### Resource Configuration

```yaml
options:
  machineType: '${_MACHINE_TYPE}'  # Configurable: E2_STANDARD_2, E2_HIGHCPU_4, E2_HIGHCPU_8
  diskSizeGb: ${_DISK_SIZE}        # Configurable disk size in GB
  logging: CLOUD_LOGGING_ONLY      # Cloud Logging only
```

**Default Performance:**
- **Build Time**: ~5-8 minutes (E2_HIGHCPU_8)
- **Image Size**: ~3.8GB (optimized)
- **Memory**: 32GB RAM
- **CPU**: 8 vCPUs

**Customization Options:**
- **E2_STANDARD_2**: 2 vCPUs, 8GB RAM (~10-15 min build time)
- **E2_HIGHCPU_4**: 4 vCPUs, 16GB RAM (~7-10 min build time)  
- **E2_HIGHCPU_8**: 8 vCPUs, 32GB RAM (~5-8 min build time)

## 🧪 Testing the Deployment

### 1. Check Build Status

You can monitor the build progress in the Google Cloud Console:
`https://console.cloud.google.com/cloud-build/builds?project=YOUR_PROJECT_ID`

### 2. Verify Deployment

After a successful deployment, get the service URL:

```bash
SERVICE_URL=$(gcloud run services describe emotion-detection-api --region us-central1 --format='value(status.url)')
API_KEY=$(gcloud secrets versions access latest --secret="admin-api-key" --project="$PROJECT_ID")

# Test Health Endpoint
curl -s -H "X-API-Key: $API_KEY" "$SERVICE_URL/api/health" | jq .

# Test Prediction Endpoint
curl -s -X POST "$SERVICE_URL/api/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"text": "I am so happy with this new deployment!"}' | jq .
```

## ↩️ Rollback Procedures

Since we are using dynamic image tagging (`$BUILD_ID`), each successful build creates a uniquely tagged Docker image in Artifact Registry. This allows for easy rollbacks.

1.  **List Revisions**: Identify the previous stable revision in Cloud Run.
    ```bash
    gcloud run revisions list --service emotion-detection-api --region us-central1
    ```
2.  **Rollback**: Deploy a previous revision.
    ```bash
    gcloud run services update emotion-detection-api --region us-central1 --revision OLD_REVISION_NAME
    ```
    Alternatively, you can deploy a specific image tag:
    ```bash
    gcloud run deploy emotion-detection-api --image us-central1-docker.pkg.dev/$PROJECT_ID/${_ARTIFACT_REPO}/emotion-detection-api:OLD_BUILD_ID --region us-central1
    ```

## ⚠️ Troubleshooting

-   **`Unauthorized - Invalid API key`**: Ensure the `ADMIN_API_KEY` environment variable is correctly set in your Cloud Run service and that the `X-API-Key` header matches. If using Secret Manager, verify permissions.
-   **Build Failures**: Check Cloud Build logs for detailed error messages. Common issues include dependency conflicts, Dockerfile errors, or insufficient permissions.
-   **Deployment Failures**: Check Cloud Run service logs for application startup errors. Ensure the container is listening on port `8080`.
-   **`GH008: Your push referenced at least X unknown Git LFS objects`**: If you encounter this during `git push`, run `git lfs push --all origin` before pushing your branch again.
-   **`SCT-A000` Security Warnings**: If you encounter false positives for hardcoded secrets, add `# skipcq: SCT-A000 - This is a placeholder, not a real secret` to the relevant line.

## 📈 Monitoring

-   **Cloud Run Logs**: Access logs for your service in the Cloud Run console or via `gcloud run services logs read emotion-detection-api --region us-central1`.
-   **Cloud Monitoring**: Set up dashboards and alerts for request latency, error rates, and instance count.

---

**This guide ensures a secure, automated, and robust deployment process for your SAMO Emotion Detection API.**
