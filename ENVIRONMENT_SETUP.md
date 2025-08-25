# Environment Setup Guide

The SAMO Deep Learning project provides different conda environment files for different use cases:

## Environment Files

### `environment.yml` - Base Runtime Environment
- **Use case**: Production deployment, minimal API runtime
- **Contains**: Core API dependencies, database drivers, basic utilities
- **Python**: 3.12
- **Size**: Minimal (fast setup)

```bash
conda env create -f environment.yml
conda activate samo-dl-stable
```

### `environment.dev.yml` - Development Environment  
- **Use case**: Development, testing, code quality tools
- **Contains**: Base runtime + development tools (pytest, ruff, black, mypy, pre-commit, etc.)
- **Python**: 3.12
- **Size**: Medium (includes testing and linting tools)

```bash
conda env create -f environment.dev.yml
conda activate samo-dl-dev
```

### `environment.ml.yml` - Machine Learning Environment
- **Use case**: ML training, inference, data science work
- **Contains**: Base runtime + ML libraries (PyTorch, transformers, datasets, librosa, etc.)
- **Python**: 3.12  
- **Size**: Large (includes GPU-ready ML libraries)

```bash
conda env create -f environment.ml.yml
conda activate samo-dl-ml
```

## Quick Start

1. **For API development**: Use `environment.dev.yml`
2. **For ML/training work**: Use `environment.ml.yml`
3. **For production deployment**: Use `environment.yml`

## Environment Management

### Creating/Updating Environments

```bash
# Create new environment
conda env create -f environment.dev.yml

# Update existing environment
conda env update -f environment.dev.yml --prune

# Remove environment
conda env remove -n samo-dl-dev
```

### Switching Between Environments

```bash
# Activate development environment
conda activate samo-dl-dev

# Activate ML environment  
conda activate samo-dl-ml

# Deactivate current environment
conda deactivate
```

## CI/CD Considerations

- CI pipeline uses Python 3.12 to match all environment files
- Constraints from `dependencies/constraints.txt` are applied during pip installations
- All environments pin Python to 3.12 for consistency

## Troubleshooting

If you encounter issues:

1. **Environment conflicts**: Remove and recreate the environment
2. **Missing dependencies**: Check if you're using the right environment file for your use case
3. **Version mismatches**: Ensure constraints.txt is being applied during pip installs