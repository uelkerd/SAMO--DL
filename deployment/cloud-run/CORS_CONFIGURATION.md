# CORS Configuration Guide

## Overview

The SAMO-DL API uses a secure CORS (Cross-Origin Resource Sharing) configuration that prevents the unsafe combination of `allow_origins=["*"]` with `allow_credentials=True`.

## Security Issue Fixed

**Before**: The API used `allow_origins=["*"]` with `allow_credentials=True`, which is:
- Invalid for browsers (CORS spec violation)
- Unsafe (allows any origin to send credentials)

**After**: The API now uses:
- Explicit allowed origins from environment configuration
- `allow_credentials=True` only when using explicit origins (not wildcard)
- `allow_credentials=False` when using wildcard origins for security

## Environment Variables

### CORS_ORIGINS
- **Description**: Comma-separated list of allowed origins
- **Default (Production)**: `https://samo-dl-demo.web.app,https://samo-dl-demo.firebaseapp.com`
- **Default (Staging)**: `https://samo-dl-demo-staging.web.app,https://samo-dl-demo-staging.firebaseapp.com`
- **Default (Development)**: `*` (wildcard - credentials disabled)

### Examples

#### Production Environment
```bash
export CORS_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"
```

#### Staging Environment
```bash
export CORS_ORIGINS="https://staging.yourdomain.com,https://staging-app.yourdomain.com"
```

#### Development Environment
```bash
# Uses wildcard by default - credentials automatically disabled
export CORS_ORIGINS="*"
```

## Configuration Logic

The API automatically determines whether to allow credentials based on the origins:

1. **Explicit Origins**: If `CORS_ORIGINS` contains specific domains (not `*`), then `allow_credentials=True`
2. **Wildcard Origins**: If `CORS_ORIGINS` contains `*`, then `allow_credentials=False` for security

## Deployment Configuration

### Cloud Run Environment Variables

Add the following to your Cloud Run service environment variables:

```yaml
# Production
CORS_ORIGINS: "https://samo-dl-demo.web.app,https://samo-dl-demo.firebaseapp.com"

# Staging  
CORS_ORIGINS: "https://samo-dl-demo-staging.web.app,https://samo-dl-demo-staging.firebaseapp.com"
```

### Cloud Build Configuration

Update your `cloudbuild.yaml` to include the CORS_ORIGINS environment variable:

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
    '--max-instances', '${_MAX_INSTANCES}',
    '--set-env-vars', 'CORS_ORIGINS=https://samo-dl-demo.web.app,https://samo-dl-demo.firebaseapp.com'
  ]
```

## Security Benefits

1. **Prevents CORS Violations**: No more browser CORS errors from invalid configurations
2. **Credential Security**: Credentials only sent to explicitly allowed origins
3. **Environment-Specific**: Different origins for dev/staging/production
4. **Configurable**: Easy to update allowed origins without code changes

## Testing CORS Configuration

### Check CORS Headers
```bash
curl -H "Origin: https://yourdomain.com" \
     -H "Access-Control-Request-Method: POST" \
     -H "Access-Control-Request-Headers: Content-Type" \
     -X OPTIONS \
     https://your-api-url.com/analyze/emotion
```

### Expected Response Headers
```
Access-Control-Allow-Origin: https://yourdomain.com
Access-Control-Allow-Credentials: true
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: *
```

## Troubleshooting

### Common Issues

1. **CORS Error with Credentials**: Check that your origin is in the `CORS_ORIGINS` list
2. **Wildcard with Credentials**: The API automatically disables credentials when using `*`
3. **Missing Environment Variable**: Falls back to production defaults

### Debug Mode

Set `LOG_LEVEL=debug` to see CORS configuration in logs:

```bash
export LOG_LEVEL=debug
```

## Migration Guide

### From Old Configuration

**Before**:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,  # ❌ Unsafe with wildcard
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**After**:
```python
# Automatically configured from environment
# allow_credentials=True only with explicit origins
# allow_credentials=False with wildcard origins
```

### Environment Setup

1. Set `CORS_ORIGINS` environment variable
2. Deploy with updated configuration
3. Test CORS headers with your frontend domain
4. Update frontend to handle new CORS behavior if needed
