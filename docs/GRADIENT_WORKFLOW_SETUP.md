# Paperspace Gradient Workflow Setup Guide

## Overview

This guide explains how to set up and use the Paperspace Gradient workflow infrastructure for automated training pipelines.

## Prerequisites

- Paperspace account with API access
- Paperspace CLI installed (`pip install gradient`)
- Python 3.8+ environment

## Installation

### 1. Install Paperspace CLI

```bash
pip install gradient
```

### 2. Authenticate with Paperspace

```bash
gradient apiKey YOUR_API_KEY
```

## Project Structure

```
.gradient/
├── workflows/
│   ├── training-pipeline.yaml      # Main training workflow
│   └── workflow.yaml               # Sample workflow
├── requirements-workflow-prod.txt   # Production dependencies (pinned versions)
├── requirements-workflow-dev.txt    # Development dependencies
└── requirements-workflow.txt        # Legacy requirements (deprecated)
```

## Requirements Management

### Production Dependencies (`requirements-workflow-prod.txt`)
- **Pinned versions** for reproducible builds
- **Core ML/DL libraries**: PyTorch, Transformers, etc.
- **Data processing**: NumPy, Pandas, Scikit-learn
- **No development tools** to minimize environment size

### Development Dependencies (`requirements-workflow-dev.txt`)
- **Testing tools**: pytest, black, flake8
- **Code quality**: pylint, mypy, pre-commit
- **Documentation**: Sphinx, Jupyter

## Workflow Configuration

### Training Pipeline Workflow

The main workflow (`training-pipeline.yaml`) consists of 5 stages:

1. **Setup Environment** - Install dependencies, verify GPU
2. **Data Preparation** - Prepare training data
3. **Model Training** - Train the model (uses V100 GPU)
4. **Model Evaluation** - Evaluate model performance
5. **Save Artifacts** - Save final model and results

### Input/Output Paths

**Important**: Gradient mounts dataset inputs/outputs at specific paths:

- **Inputs**: `/inputs/<input_name>` (not dataset reference)
- **Outputs**: `/outputs/<output_name>` (must match job output name)

**Example**:
```yaml
inputs:
  trained-model:  # This becomes /inputs/trained-model
    type: dataset
    with:
      ref: trained-model-output
```

## Usage

### Interactive Setup

```bash
python scripts/setup_gradient_workflow.py
```

### Non-Interactive Setup

```bash
export GRADIENT_NON_INTERACTIVE=1
export GRADIENT_PROJECT_ID=your_project_id
python scripts/setup_gradient_workflow.py
```

### Command-Line Interface

```bash
# Setup environment
python scripts/setup_gradient_workflow.py setup

# Run workflow
python scripts/setup_gradient_workflow.py run [project_id]

# Validate workflow file
python scripts/setup_gradient_workflow.py validate

# Create datasets
python scripts/setup_gradient_workflow.py datasets

# List projects
python scripts/setup_gradient_workflow.py projects
```

## Troubleshooting

### Common Issues

1. **Path Mismatches**: Ensure input/output paths match dataset names
2. **Missing Dependencies**: Use production requirements for workflows
3. **Authentication**: Verify API key with `gradient projects list`

### Validation

The setup script automatically validates:
- Workflow file structure
- Required datasets existence
- CLI availability and authentication

## Best Practices

1. **Use pinned versions** in production for reproducibility
2. **Separate dev/prod dependencies** to minimize environment size
3. **Validate paths** before running workflows
4. **Test locally** before deploying to Gradient
5. **Monitor resource usage** during training

## Security Notes

- API keys are stored in environment variables
- Subprocess calls use timeouts and proper error handling
- Input validation prevents command injection

## Support

For issues or questions:
1. Check the workflow validation output
2. Verify dataset paths and names
3. Ensure all dependencies are available
4. Check Paperspace CLI authentication
