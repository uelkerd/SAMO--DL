# Requirements Management Guide

## Overview

This directory contains a hierarchical requirements structure designed to reduce duplication and ensure consistency across all SAMO deployment configurations.

## File Structure

```
deployment/
├── requirements-core.txt          # Core shared dependencies
├── requirements.txt               # Main deployment (extends core)
├── cloud-run/
│   ├── requirements_minimal.txt   # Minimal deployment (extends core)
│   ├── requirements_secure.txt    # Secure deployment (extends core)
│   └── requirements_onnx.txt      # ONNX-specific dependencies
└── README-requirements.md         # This file
```

## Core Requirements (`requirements-core.txt`)

The core requirements file contains all shared dependencies used across multiple deployment configurations:

- **Web Framework**: Flask, Gunicorn
- **ML Libraries**: PyTorch, Transformers, NumPy, scikit-learn, pandas
- **HTTP Client**: Requests
- **Monitoring**: psutil, prometheus-client
- **Security**: cryptography
- **Build Tools**: setuptools

## Extending Core Requirements

To create a new deployment-specific requirements file:

1. **Start with core requirements**:
   ```txt
   # My Deployment Requirements
   -r ../requirements-core.txt
   ```

2. **Add deployment-specific dependencies**:
   ```txt
   # Additional dependencies for this deployment
   flask-restx==1.3.0
   ```

3. **Override core versions if needed**:
   ```txt
   # Override core version for compatibility
   flask==2.3.3  # Override from core
   ```

## Benefits

### Reduced Duplication
- Single source of truth for common dependencies
- Easier maintenance and updates
- Consistent versioning across deployments

### Better Security
- Centralized security updates
- Easier vulnerability scanning
- Consistent security posture

### Improved Maintainability
- Update core dependencies in one place
- Clear separation of concerns
- Easier dependency audits

## Usage Examples

### Local Development
```bash
# Install core requirements
pip install -r deployment/requirements-core.txt

# Install deployment-specific requirements
pip install -r deployment/cloud-run/requirements_minimal.txt
```

### Docker Builds
```dockerfile
# Copy core requirements first
COPY requirements-core.txt .
RUN pip install -r requirements-core.txt

# Copy deployment-specific requirements
COPY cloud-run/requirements_secure.txt .
RUN pip install -r requirements_secure.txt
```

### CI/CD Pipelines
```yaml
# Install core dependencies
- name: Install core dependencies
  run: pip install -r deployment/requirements-core.txt

# Install deployment-specific dependencies
- name: Install deployment dependencies
  run: pip install -r deployment/cloud-run/requirements_secure.txt
```

## Best Practices

1. **Always extend core requirements** instead of duplicating
2. **Use version overrides sparingly** and document why
3. **Keep core requirements minimal** - only truly shared dependencies
4. **Test all combinations** when updating core requirements
5. **Document breaking changes** in core requirements

## Security Considerations

- **Pin versions** in core requirements for security
- **Regular updates** to address vulnerabilities
- **Audit dependencies** using safety-mcp
- **Monitor for CVEs** in all dependencies

## Maintenance

### Updating Core Requirements
1. Update `requirements-core.txt` with new versions
2. Test all deployment configurations
3. Update documentation if needed
4. Commit changes with clear commit message

### Adding New Dependencies
1. Determine if dependency belongs in core or specific deployment
2. Add to appropriate requirements file
3. Update documentation
4. Test deployment configurations

### Removing Dependencies
1. Check all deployment configurations for usage
2. Remove from appropriate requirements file
3. Update documentation
4. Test deployment configurations

## Troubleshooting

### Common Issues

**Import errors after updating core requirements**
- Check if dependency was removed from core
- Verify deployment-specific requirements include needed dependencies

**Version conflicts**
- Check for conflicting version specifications
- Use version overrides in deployment-specific files

**Missing dependencies**
- Ensure core requirements are installed first
- Check deployment-specific requirements for missing dependencies

### Debug Commands

```bash
# Check installed packages
pip list

# Check dependency tree
pip show <package-name>

# Verify requirements installation
pip check

# Check for conflicts
pip install --dry-run -r requirements.txt
```

## Future Improvements

- [ ] Automated dependency updates
- [ ] Dependency vulnerability scanning in CI/CD
- [ ] Requirements validation tests
- [ ] Automated compatibility testing
- [ ] Dependency usage analytics
