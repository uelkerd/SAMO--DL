# Dockerfile Security Guide

## Overview

This document explains the security considerations and design decisions for different Dockerfile configurations in the SAMO project. Each Dockerfile is designed for a specific deployment environment with appropriate security measures.

## Dockerfile Configurations

### 1. Main Production Dockerfile (`/Dockerfile`)

**Purpose**: Production deployment with maximum security
**Server**: Gunicorn with Uvicorn workers
**Security Features**:
- ✅ Non-root user execution
- ✅ Pinned package versions (OS packages pinned; Python deps pinned in `requirements-api.txt` and enforced with `constraints.txt`)
- ✅ Minimal attack surface
- ✅ Health checks
- ✅ Environment variable configuration

**CMD**: 
```dockerfile
CMD ["sh", "-c", "gunicorn --bind ${HOST}:${PORT} --workers 2 --worker-class uvicorn.workers.UvicornWorker --access-logfile - --error-logfile - src.unified_ai_api:app"]
```

**Why Gunicorn?**
- Process management and monitoring
- Worker process isolation
- Better security posture
- Production-grade reliability

### 2. Cloud Run Dockerfile (`/deployment/cloud-run/Dockerfile`)

**Purpose**: Google Cloud Run deployment
**Server**: Uvicorn directly
**Security Features**:
- ✅ Non-root user execution
- ✅ Pinned package versions
  - OS packages pinned to Debian bookworm security releases
  - Python dependencies pinned in `requirements-api.txt` and additionally constrained with `constraints.txt` during install
- ✅ Health checks
- ✅ Environment variable configuration

**CMD**:
```dockerfile
CMD ["sh", "-c", "exec uvicorn src.unified_ai_api:app --host 0.0.0.0 --port ${PORT}"]
```

**Why Uvicorn for Cloud Run?**
- Cloud Run manages process lifecycle
- No need for Gunicorn process management
- Lighter weight for serverless environment
- Cloud Run provides security isolation

### 3. Unified API Dockerfile (`/deployment/cloud-run/Dockerfile.unified`)

**Purpose**: Unified API service for Cloud Run
**Server**: Uvicorn directly
**Security Features**:
- ✅ Non-root user execution
- ✅ Pinned package versions
  - OS packages pinned to Debian bookworm security releases
  - Python dependencies pinned via `requirements_unified.txt` and additionally constrained with `constraints.txt`
- ✅ Health checks
- ✅ Model pre-bundling for security

**CMD**:
```dockerfile
CMD ["sh", "-c", "exec uvicorn src.unified_ai_api:app --host 0.0.0.0 --port ${PORT}"]
```

## Security Analysis (Concise)

Some scanners may raise false positives in these Dockerfiles:
- Import path flagged as a key: `src.unified_ai_api:app` is a Python module path, not a secret.
- Subprocess usage in tests: arguments are static lists without `shell=True`, minimizing risk.
- Bindings/security headers: `0.0.0.0` exposure is intentional for containers; actual binding is controlled via environment variables and platform ingress.

These are documented to avoid unnecessary policy exceptions while keeping configurations secure and clear.

## Security Best Practices Implemented

### 1. User Management
- All Dockerfiles create non-root users
- Proper ownership of application files
- Minimal privileges for runtime

### 2. Package Security
- Pinned OS package versions
- Python packages pinned in requirements and enforced with constraints to ensure reproducibility
- Regular security updates
- Vulnerability scanning in CI/CD

### 3. Network Security
- Environment variable configuration
- No hardcoded bindings
- Proper EXPOSE directives

### 4. Process Security
- Health checks with timeouts
- Proper signal handling
- Resource limits where applicable

## Deployment Environment Considerations

### Production (Main Dockerfile)
- **Use Case**: Traditional server deployment
- **Server**: Gunicorn + Uvicorn workers
- **Security**: Maximum isolation and monitoring
- **Monitoring**: Process-level health checks

### Cloud Run (Cloud Run Dockerfiles)
- **Use Case**: Serverless deployment
- **Server**: Uvicorn directly
- **Security**: Platform-provided isolation
- **Monitoring**: Cloud Run health checks

## Security Recommendations

### 1. For Production Deployments
- Use the main Dockerfile with Gunicorn
- Implement proper logging and monitoring
- Use environment variables for configuration
- Regular security updates

### 2. For Cloud Run Deployments
- Use the cloud-run specific Dockerfiles
- Leverage Cloud Run security features
- Use Secret Manager for sensitive data
- Monitor Cloud Run logs and metrics

### 3. General Security
- Never commit secrets to version control
- Use environment variables for configuration
- Regular vulnerability scanning
- Keep dependencies updated

## Appendix: False Positive References

- Generic API key detection: the string `src.unified_ai_api:app` is an import path (FastAPI app instance), not a credential.
- Subprocess warnings: tests use argument lists with no `shell=True`, and file paths are programmatically controlled.
- Hardcoded bindings: `0.0.0.0` is a container best practice for network ingress; actual external exposure is managed by the orchestrator (e.g., Cloud Run).
