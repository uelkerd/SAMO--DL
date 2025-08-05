# Vertex AI Deployment Guide

## Overview

This guide covers the deployment of the SAMO-DL emotion detection model to Google Cloud Platform (GCP) Vertex AI. The deployment uses a custom container approach for maximum flexibility and control.

## Architecture

The deployment consists of:
- **Custom Docker Container**: Flask-based HTTP server running on port 8080
- **Vertex AI Model**: Containerized model with comprehensive error handling
- **Vertex AI Endpoint**: Scalable endpoint for serving predictions
- **Health Monitoring**: Built-in health checks and logging

## Prerequisites

1. **GCP Account**: Active Google Cloud Platform account
2. **gcloud CLI**: Google Cloud SDK installed and authenticated
3. **Docker**: Docker installed and running locally
4. **Project Setup**: GCP project with billing enabled

## Quick Start

### 1. Local Testing (Recommended)

Before deploying to GCP, test the container locally:

```bash
# Test container locally
python scripts/deployment/test_container_locally.py
```

This will:
- Build the Docker image
- Start the container locally
- Test health and prediction endpoints
- Clean up test resources

### 2. Automated Deployment

Use the enhanced deployment script:

```bash
# Make script executable
chmod +x scripts/deployment/gcp_deploy_automation.sh

# Run deployment
./scripts/deployment/gcp_deploy_automation.sh
```

## Manual Deployment Steps

### 1. Prepare Model Files

Ensure your model files are in the correct location:

```bash
deployment/gcp/model/
├── config.json
├── model.safetensors (or pytorch_model.bin)
├── tokenizer.json
└── merges.txt (if applicable)
```

### 2. Build and Push Docker Image

```bash
cd deployment/gcp

# Build image
docker build -t gcr.io/YOUR_PROJECT_ID/emotion-detection-model:latest .

# Push to Container Registry
docker push gcr.io/YOUR_PROJECT_ID/emotion-detection-model:latest
```

### 3. Create Vertex AI Model

```bash
gcloud ai models upload \
    --region=us-central1 \
    --display-name=comprehensive-emotion-detection-model \
    --container-image-uri=gcr.io/YOUR_PROJECT_ID/emotion-detection-model:latest
```

### 4. Create Vertex AI Endpoint

```bash
gcloud ai endpoints create \
    --region=us-central1 \
    --display-name=comprehensive-emotion-detection-endpoint
```

### 5. Deploy Model to Endpoint

```bash
gcloud ai endpoints deploy-model ENDPOINT_ID \
    --region=us-central1 \
    --model=MODEL_ID \
    --display-name=comprehensive-emotion-detection-deployment \
    --machine-type=n1-standard-2 \
    --min-replica-count=1 \
    --max-replica-count=10
```

## Enhanced Error Handling

### Container Startup Issues

The enhanced `predict.py` includes comprehensive error handling:

1. **Model Path Resolution**: Uses environment variables with fallbacks
2. **File Validation**: Checks for required model files
3. **Retry Logic**: Attempts model loading up to 3 times
4. **Detailed Logging**: Comprehensive logging to `/tmp/vertex_ai_server.log`
5. **Health Checks**: Robust health check endpoint with model validation

### Common Issues and Solutions

#### 1. "Model server exited unexpectedly"

**Symptoms**: Container starts but crashes during model initialization

**Root Causes**:
- Missing model files
- Memory constraints
- Missing dependencies
- Incorrect model path

**Solutions**:
1. Check container logs in GCP Console
2. Validate model files are present
3. Increase container memory allocation
4. Test locally first using `test_container_locally.py`

#### 2. Model Loading Failures

**Symptoms**: Model fails to load with specific errors

**Solutions**:
1. Check model file integrity
2. Verify all required dependencies in `requirements.txt`
3. Ensure model path is correct
4. Check for memory issues

#### 3. Health Check Failures

**Symptoms**: Health endpoint returns 503 or fails

**Solutions**:
1. Check model initialization logs
2. Verify model can make predictions
3. Check for GPU/CPU compatibility issues

## Monitoring and Debugging

### Container Logs

Access container logs via GCP Console:
```
https://console.cloud.google.com/logs/viewer?project=YOUR_PROJECT_ID&resource=aiplatform.googleapis.com%2FEndpoint
```

### Health Monitoring

The container provides comprehensive health information:

```bash
# Health check
curl http://ENDPOINT_URL/health

# Response includes:
{
  "status": "healthy",
  "model_version": "2.0",
  "model_type": "comprehensive_emotion_detection",
  "model_loaded": true,
  "gpu_available": true,
  "test_prediction": "happy"
}
```

### Performance Monitoring

Monitor endpoint performance in GCP Console:
- Request latency
- Error rates
- Resource utilization
- Scaling metrics

## Configuration Options

### Environment Variables

- `MODEL_PATH`: Path to model directory (default: `/app/model`)
- `PYTHONPATH`: Python path (default: `/app`)
- `PYTHONUNBUFFERED`: Python output buffering (default: `1`)
- `FLASK_ENV`: Flask environment (default: `production`)

### Machine Types

Recommended configurations:
- **Development**: `n1-standard-2` (2 vCPU, 7.5 GB RAM)
- **Production**: `n1-standard-4` (4 vCPU, 15 GB RAM)
- **High Performance**: `n1-standard-8` (8 vCPU, 30 GB RAM)

### Scaling Configuration

- **Min Replicas**: 1 (for cost optimization)
- **Max Replicas**: 10 (for scalability)
- **Auto-scaling**: Based on CPU utilization

## Testing the Deployment

### Using Python Client

```python
from google.cloud import aiplatform

# Initialize client
aiplatform.init(project='YOUR_PROJECT_ID', location='us-central1')

# Get endpoint
endpoint = aiplatform.Endpoint('ENDPOINT_ID')

# Make prediction
response = endpoint.predict({
    'text': 'I am feeling happy today!'
})
print(response)
```

### Using curl

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling happy today!"}' \
  https://ENDPOINT_URL/predict
```

## Cost Optimization

### Resource Management

1. **Use appropriate machine types**: Start with `n1-standard-2`
2. **Set minimum replicas to 1**: Reduces idle costs
3. **Monitor usage**: Use GCP Console to track costs
4. **Scale down during off-hours**: Use scheduled scaling

### Billing Alerts

Set up billing alerts in GCP Console:
1. Go to Billing > Budgets & alerts
2. Create budget with alerts
3. Set threshold (e.g., $50/month)
4. Configure notification channels

## Troubleshooting Guide

### Deployment Failures

1. **Check prerequisites**: Ensure all tools are installed
2. **Validate model files**: Verify all required files are present
3. **Test locally**: Use `test_container_locally.py`
4. **Check logs**: Review container logs in GCP Console
5. **Verify permissions**: Ensure proper IAM roles

### Runtime Issues

1. **Model loading errors**: Check model file integrity
2. **Memory issues**: Increase machine type or optimize model
3. **Network issues**: Check firewall rules and VPC configuration
4. **Scaling problems**: Review auto-scaling configuration

### Performance Issues

1. **High latency**: Consider GPU acceleration
2. **Memory pressure**: Increase machine type
3. **Cold starts**: Increase minimum replicas
4. **Bottlenecks**: Profile model performance

## Security Considerations

### Container Security

1. **Use minimal base image**: Python slim image
2. **Scan for vulnerabilities**: Regular security scans
3. **Update dependencies**: Keep packages updated
4. **Limit permissions**: Use least privilege principle

### Network Security

1. **VPC configuration**: Use private VPC if needed
2. **Firewall rules**: Restrict access as needed
3. **TLS encryption**: Enable HTTPS endpoints
4. **Authentication**: Use proper authentication methods

## Best Practices

### Development Workflow

1. **Test locally first**: Always test container locally
2. **Use version tags**: Tag Docker images with versions
3. **Monitor deployments**: Set up proper monitoring
4. **Document changes**: Keep deployment documentation updated

### Production Deployment

1. **Use staging environment**: Test in staging first
2. **Blue-green deployment**: Use blue-green deployment strategy
3. **Rollback plan**: Have rollback procedures ready
4. **Monitoring**: Set up comprehensive monitoring

### Maintenance

1. **Regular updates**: Update dependencies regularly
2. **Security patches**: Apply security patches promptly
3. **Performance monitoring**: Monitor and optimize performance
4. **Cost monitoring**: Track and optimize costs

## Support and Resources

### Documentation

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Custom Container Guide](https://cloud.google.com/vertex-ai/docs/predictions/custom-container)
- [Docker Documentation](https://docs.docker.com/)

### Community Support

- [Google Cloud Community](https://cloud.google.com/community)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/google-cloud-platform)
- [GitHub Issues](https://github.com/your-repo/issues)

### Professional Support

- [Google Cloud Support](https://cloud.google.com/support)
- [Professional Services](https://cloud.google.com/professional-services)

## Conclusion

This enhanced deployment guide provides comprehensive coverage of deploying the SAMO-DL emotion detection model to Vertex AI. The key improvements include:

1. **Robust error handling**: Comprehensive error handling and logging
2. **Local testing**: Test container locally before deployment
3. **Automated deployment**: Streamlined deployment process
4. **Monitoring**: Built-in health checks and monitoring
5. **Troubleshooting**: Detailed troubleshooting guide

Follow this guide to successfully deploy your model to Vertex AI with confidence. 