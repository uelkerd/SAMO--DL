# Consolidated Dockerfile Usage Guide

This consolidated Dockerfile replaces multiple separate Dockerfiles with a single, flexible solution that can build different variants using build arguments.

## **Build Arguments**

### **BUILD_TYPE** (default: `minimal`)
- **`minimal`** - Lightweight API server without ML dependencies
- **`unified`** - Full API with ML models (T5, Whisper)
- **`secure`** - Security-focused version with enhanced permissions
- **`production`** - Production-optimized version

### **INCLUDE_ML** (default: `false`)
- **`true`** - Includes ML dependencies (PyTorch, transformers, etc.)
- **`false`** - Excludes ML dependencies for smaller images

### **INCLUDE_SECURITY** (default: `false`)
- **`true`** - Enhanced security features (strict permissions, etc.)
- **`false`** - Standard security configuration

## **Build Commands**

### **Minimal Version (Default)**
```bash
# Build from project root
cd /Users/minervae/Projects/SAMO--GENERAL/SAMO--DL
docker build -f deployment/cloud-run/Dockerfile.consolidated -t samo-dl-minimal .
```

### **Unified Version (with ML)**
```bash
docker build \
  --build-arg BUILD_TYPE=unified \
  --build-arg INCLUDE_ML=true \
  -f deployment/cloud-run/Dockerfile.consolidated \
  -t samo-dl-unified .
```

### **Secure Version**
```bash
docker build \
  --build-arg BUILD_TYPE=secure \
  --build-arg INCLUDE_SECURITY=true \
  -f deployment/cloud-run/Dockerfile.consolidated \
  -t samo-dl-secure .
```

### **Production Version**
```bash
docker build \
  --build-arg BUILD_TYPE=production \
  --build-arg INCLUDE_ML=true \
  --build-arg INCLUDE_SECURITY=true \
  -f deployment/cloud-run/Dockerfile.consolidated \
  -t samo-dl-production .
```

## **Multi-Architecture Builds**

### **ARM64 (Apple Silicon)**
```bash
docker build \
  --platform linux/arm64 \
  --build-arg BUILD_TYPE=unified \
  --build-arg INCLUDE_ML=true \
  -f deployment/cloud-run/Dockerfile.consolidated \
  -t samo-dl-unified-arm64 .
```

### **x86_64 (Intel/AMD)**
```bash
docker build \
  --platform linux/amd64 \
  --build-arg BUILD_TYPE=unified \
  --build-arg INCLUDE_ML=true \
  -f deployment/cloud-run/Dockerfile.consolidated \
  -t samo-dl-unified-amd64 .
```

## **Image Characteristics**

### **Minimal Version**
- **Size**: ~200-300MB
- **Dependencies**: Basic API functionality only
- **Use case**: Simple deployments, testing, CI/CD

### **Unified Version**
- **Size**: ~2-4GB (includes ML models)
- **Dependencies**: Full ML stack (PyTorch, transformers, Whisper)
- **Use case**: Production ML inference, full API functionality

### **Secure Version**
- **Size**: Similar to minimal
- **Dependencies**: Enhanced security features
- **Use case**: Production deployments with security requirements

### **Production Version**
- **Size**: Similar to unified
- **Dependencies**: Full ML stack + security features
- **Use case**: Production ML deployments with security requirements

## **Requirements File Mapping**

The Dockerfile automatically selects the appropriate requirements file:
- `BUILD_TYPE=minimal` → `requirements_minimal.txt`
- `BUILD_TYPE=unified` → `requirements_unified.txt`
- `BUILD_TYPE=secure` → `requirements_secure.txt`
- `BUILD_TYPE=production` → `requirements_production.txt`

## **Testing the Builds**

### **Test Minimal Version**
```bash
docker run --rm -p 8080:8080 samo-dl-minimal
curl http://localhost:8080/health
```

### **Test Unified Version**
```bash
docker run --rm -p 8080:8080 samo-dl-unified
curl http://localhost:8080/health
# Should show ML models as available
```

### **Test Secure Version**
```bash
docker run --rm -p 8080:8080 samo-dl-secure
curl http://localhost:8080/health
```

## **Migration from Old Dockerfiles**

### **Before (Multiple Files)**
```bash
# Had to remember which Dockerfile to use
docker build -f deployment/cloud-run/Dockerfile -t samo-dl .
docker build -f deployment/cloud-run/Dockerfile.unified -t samo-dl-unified .
docker build -f deployment/cloud-run/Dockerfile.minimal -t samo-dl-minimal .
docker build -f deployment/cloud-run/Dockerfile.secure -t samo-dl-secure .
```

### **After (Single File)**
```bash
# One Dockerfile, multiple variants
docker build --build-arg BUILD_TYPE=minimal -f Dockerfile.consolidated -t samo-dl-minimal .
docker build --build-arg BUILD_TYPE=unified --build-arg INCLUDE_ML=true -f Dockerfile.consolidated -t samo-dl-unified .
docker build --build-arg BUILD_TYPE=secure --build-arg INCLUDE_SECURITY=true -f Dockerfile.consolidated -t samo-dl-secure .
```

## **Benefits**

✅ **Single source of truth** - one Dockerfile to maintain
✅ **Consistent behavior** - same base image, same patterns
✅ **Easy to update** - change once, affects all variants
✅ **Clear documentation** - obvious what each build arg does
✅ **Reduced duplication** - no repeated code
✅ **Flexible builds** - mix and match features as needed

## **Next Steps**

1. **Test all build variants** to ensure they work correctly
2. **Update CI/CD pipelines** to use the new consolidated approach
3. **Remove old Dockerfiles** once migration is complete
4. **Update deployment scripts** to use build arguments
