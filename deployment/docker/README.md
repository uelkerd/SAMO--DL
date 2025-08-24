# Docker images for SAMO-DL

This directory contains the maintained Dockerfiles:

- `Dockerfile.app`: Application image for API servers
  - Build args:
    - `BUILD_TYPE` = minimal|unified|secure|production (default: minimal)
    - `INCLUDE_ML` = true|false (default: false)
    - `INCLUDE_SECURITY` = true|false (default: false)
    - `PIP_CONSTRAINT` = optional path to a constraints file
- `Dockerfile.train`: Training image for Vertex AI jobs
  - Build args:
    - `PIP_CONSTRAINT` = optional path to a constraints file

Examples:

```bash
# Minimal app
docker build -f deployment/docker/Dockerfile.app -t samo-dl-app .

# Unified ML app with constraints
docker build \
  --build-arg BUILD_TYPE=unified \
  --build-arg INCLUDE_ML=true \
  --build-arg PIP_CONSTRAINT=dependencies/constraints.txt \
  -f deployment/docker/Dockerfile.app \
  -t samo-dl-app-unified .

# Training image with constraints
docker build \
  --build-arg PIP_CONSTRAINT=dependencies/constraints.txt \
  -f deployment/docker/Dockerfile.train \
  -t samo-dl-train .
```