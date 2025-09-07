# Cloud Build Deployment Guide

## 🚀 Automated Build and Deploy Pipeline

This guide covers the complete automated deployment pipeline using Google Cloud Build for the SAMO Emotion Detection API.

## 📋 Prerequisites

1. **Google Cloud Project** with billing enabled
2. **gcloud CLI** installed and authenticated
3. **Required APIs** enabled:
   - Cloud Build API
   - Cloud Run API
   - Secret Manager API
   - Container Registry API

## 🔧 Setup Instructions

### 1. Enable Required APIs

```bash
gcloud services enable secretmanager.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable container.googleapis.com
```

### 2. Set Up Secret Manager (Recommended)

For production deployments, use Google Secret Manager to store the API key securely:

```bash
# Run the setup script
./scripts/deployment/setup-secret-manager.sh

# Or manually:
gcloud secrets create admin-api-key --replication-policy="automatic"
echo -n "your-secure-api-key-here" | gcloud secrets versions add admin-api-key --data-file=-  # skipcq: SCT-A000 - This is a placeholder, not a real secret
```

### 3. Grant Cloud Build Permissions

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
- 'gcr.io/$PROJECT_ID/emotion-detection-api:$BUILD_ID'
- 'gcr.io/$PROJECT_ID/emotion-detection-api:latest'
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
    '--image', 'gcr.io/$PROJECT_ID/emotion-detection-api:$BUILD_ID',
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

```bash
gcloud builds list --limit=5
```

### 2. Test the Deployed Service

```bash
# Health check
curl https://emotion-detection-api-71517823771.us-central1.run.app/api/health

# Emotion detection
curl -X POST https://emotion-detection-api-71517823771.us-central1.run.app/api/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"text": "I feel excited about this deployment!"}'
```

## 🔄 Rollback Procedures

### Rollback to Previous Version

```bash
# List available revisions
gcloud run revisions list --service=emotion-detection-api --region=us-central1

# Rollback to specific revision
gcloud run services update-traffic emotion-detection-api \
  --to-revisions=emotion-detection-api-00002-abc=100 \
  --region=us-central1
```

### Rollback to Previous Image

```bash
# Deploy previous image
gcloud run deploy emotion-detection-api \
  --image=gcr.io/$PROJECT_ID/emotion-detection-api:previous-build-id \
  --region=us-central1
```

## 📊 Monitoring and Logs

### View Build Logs

```bash
gcloud builds log [BUILD_ID]
```

### View Service Logs

```bash
gcloud run logs read emotion-detection-api --region=us-central1
```

### Monitor Performance

```bash
gcloud run services describe emotion-detection-api --region=us-central1
```

## 🚨 Troubleshooting

### Common Issues

1. **Build Fails with "No space left on device"**
   ```bash
   # Increase disk size in cloudbuild.yaml
   diskSizeGb: 200
   ```

2. **Secret Manager Access Denied**
   ```bash
   # Grant proper permissions
   gcloud secrets add-iam-policy-binding admin-api-key \
     --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
     --role="roles/secretmanager.secretAccessor"
   ```

3. **Cloud Run Deployment Fails**
   ```bash
   # Check service account permissions
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
     --role="roles/run.admin"
   ```

## 📚 Additional Resources

- [Cloud Build Documentation](https://cloud.google.com/build/docs)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Secret Manager Documentation](https://cloud.google.com/secret-manager/docs)
- [Container Registry Documentation](https://cloud.google.com/container-registry/docs)

---

## 🎯 Summary

This deployment pipeline provides:

- ✅ **Automated builds** with dynamic tagging
- ✅ **Secure API key management** via Secret Manager
- ✅ **One-command deployment** to Cloud Run
- ✅ **Easy rollbacks** with build history
- ✅ **Production-ready** security and monitoring

The pipeline is optimized for the SAMO voice-first mental health journaling app with real-time emotion detection capabilities.
