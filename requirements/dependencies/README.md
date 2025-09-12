# Dependencies guide

Canonical requirement sets:

- `requirements-api.txt`: API/runtime dependencies (matches pyproject base/prod)
- `requirements-ml.txt`: ML stack (matches pyproject ml extra)
- `requirements-dev.txt`: development and testing utilities (dev/test extras)
- `constraints.txt`: pinning/compatibility guard used across installs

Redundant/legacy sets scheduled for removal:

- `requirements_deployment.txt`, `requirements_gcp.txt`, `requirements_minimal.txt`, `requirements_unified.txt`, `requirements_onnx.txt`, `requirements_production.txt`, `requirements_secure.txt`

Use patterns:

```bash
pip install -c dependencies/constraints.txt -r dependencies/requirements-api.txt
pip install -c dependencies/constraints.txt -r dependencies/requirements-dev.txt
pip install -c dependencies/constraints.txt -r dependencies/requirements-ml.txt
```

Rationale: fewer, clearer files reduce drift and simplify security scanning and upgrades.

