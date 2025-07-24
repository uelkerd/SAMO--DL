# SAMO Deep Learning - Code Standards & Style Guide

## 📋 Overview

This document establishes consistent coding standards and style guidelines for the SAMO Deep Learning project. Following these standards ensures code maintainability, readability, and collaboration efficiency across the team.

## 🏗️ Project Structure

### Directory Organization

```
samo-dl/
├── configs/               # Configuration files
│   ├── development.yaml   # Development environment config
│   ├── production.yaml    # Production environment config
│   └── testing.yaml       # Testing environment config
├── data/                  # Data-related files
│   ├── cache/             # Cached datasets
│   ├── external/          # External data sources
│   ├── processed/         # Processed datasets
│   └── raw/               # Raw data files
├── docs/                  # Documentation
├── models/                # Trained models
│   ├── checkpoints/       # Model checkpoints
│   ├── emotion_detection/ # Emotion detection models
│   ├── summarization/     # Text summarization models
│   └── voice_processing/  # Voice processing models
├── notebooks/             # Jupyter notebooks
├── scripts/               # Utility scripts
│   ├── database/          # Database scripts
│   ├── maintenance/       # Maintenance scripts
│   └── ci/                # CI/CD scripts
├── src/                   # Source code
│   ├── data/              # Data processing modules
│   ├── evaluation/        # Model evaluation code
│   ├── inference/         # Inference code
│   ├── models/            # Model definitions
│   └── training/          # Training code
└── tests/                 # Tests
    ├── unit/              # Unit tests
    ├── integration/       # Integration tests
    └── e2e/               # End-to-end tests
```

### File Organization

Each Python module should follow this structure:

```python
"""
Module docstring describing the purpose of the module.
"""

# Standard library imports
import os
import sys
import json

# Third-party imports
import numpy as np
import torch
import transformers

# Local application imports
from src.data import preprocessing
from src.models import architectures

# Constants
MAX_SEQUENCE_LENGTH = 512
DEFAULT_BATCH_SIZE = 32

# Classes and functions
class ModelClass:
    """Class docstring."""
    pass

def utility_function():
    """Function docstring."""
    pass

# Main execution (if applicable)
if __name__ == "__main__":
    # Code to execute when run as script
    pass
```

## 🔤 Naming Conventions

### Python Naming Conventions

| Type | Convention | Examples |
|------|------------|----------|
| Modules | `lowercase_with_underscores.py` | `data_preprocessing.py`, `bert_classifier.py` |
| Packages | `lowercase` | `models`, `data`, `evaluation` |
| Classes | `CapitalizedWords` | `BERTEmotionClassifier`, `DataProcessor` |
| Functions | `lowercase_with_underscores()` | `process_data()`, `train_model()` |
| Variables | `lowercase_with_underscores` | `batch_size`, `learning_rate` |
| Constants | `UPPERCASE_WITH_UNDERSCORES` | `MAX_SEQUENCE_LENGTH`, `DEFAULT_BATCH_SIZE` |
| Private members | `_leading_underscore` | `_private_method()`, `_internal_state` |
| Protected members | `__double_underscore` | `__very_private_method()` |
| Magic methods | `__double_underscores__` | `__init__()`, `__call__()` |

### Model Naming Conventions

Models should follow this naming pattern:

```
{architecture}_{task}_{variant}_{version}
```

Examples:
- `bert_emotion_classifier_v1.0`
- `t5_summarizer_concise_v2.1`
- `whisper_transcriber_large_v1.2`

### File Naming Conventions

| File Type | Convention | Examples |
|-----------|------------|----------|
| Python modules | `lowercase_with_underscores.py` | `data_loader.py`, `model_training.py` |
| Configuration files | `lowercase_with_underscores.yaml` | `development.yaml`, `model_config.yaml` |
| Jupyter notebooks | `CamelCase.ipynb` or `descriptive_name.ipynb` | `DataExploration.ipynb`, `model_training_experiments.ipynb` |
| Test files | `test_*.py` | `test_bert_classifier.py`, `test_data_loader.py` |
| Documentation | `lowercase_with_underscores.md` | `api_specification.md`, `model_training_playbook.md` |

## 📏 Code Formatting

### Python Formatting

We use [Ruff](https://github.com/charliermarsh/ruff) for Python code formatting and linting. Configuration is in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 88
target-version = "py39"
select = ["E", "F", "I", "N", "W", "D", "UP", "ANN", "B", "C4", "SIM", "ERA"]
ignore = ["D203", "D213", "ANN101", "ANN102"]
```

Key formatting rules:
- Maximum line length: 88 characters
- Indentation: 4 spaces (no tabs)
- String quotes: Double quotes preferred
- Docstrings: Google style
- Import order: Standard library → Third-party → Local

### Configuration Files

For YAML configuration files:
- Use 2-space indentation
- Use lowercase keys with underscores
- Group related settings together with comments

Example:
```yaml
# Model configuration
model:
  name: bert-base-uncased
  num_labels: 28
  dropout_rate: 0.1

# Training parameters
training:
  batch_size: 32
  learning_rate: 2e-5
  num_epochs: 3
  early_stopping_patience: 2
```

## 📝 Documentation Standards

### Docstrings

We follow Google-style docstrings:

```python
def calculate_f1_score(y_true, y_pred, threshold=0.5):
    """Calculate F1 score for multi-label classification.

    Args:
        y_true: Array-like of shape (n_samples, n_labels) with ground truth labels.
        y_pred: Array-like of shape (n_samples, n_labels) with predicted probabilities.
        threshold: Float between 0 and 1, threshold for converting probabilities to binary.

    Returns:
        tuple: (micro_f1, macro_f1) scores.

    Example:
        >>> y_true = [[1, 0, 1], [0, 1, 0]]
        >>> y_pred = [[0.9, 0.2, 0.8], [0.1, 0.9, 0.3]]
        >>> calculate_f1_score(y_true, y_pred)
        (0.8, 0.78)
    """
```

### Class Docstrings

```python
class BERTEmotionClassifier(nn.Module):
    """BERT-based classifier for emotion detection.

    This model fine-tunes a pre-trained BERT model for multi-label
    emotion classification using the GoEmotions taxonomy (28 emotions).

    Attributes:
        bert: Pre-trained BERT model.
        dropout: Dropout layer for regularization.
        classifier: Linear layer for classification.
        model_name: Name of the pre-trained model.
        num_emotions: Number of emotion categories.
        temperature: Temperature parameter for calibration.
        prediction_threshold: Threshold for positive predictions.
    """
```

### Module Docstrings

```python
"""
BERT Emotion Classifier Module

This module implements a BERT-based emotion classifier for multi-label
emotion detection using the GoEmotions taxonomy. It includes the model
architecture, training pipeline, and inference utilities.

Classes:
    BERTEmotionClassifier: BERT-based classifier for emotion detection.
    EmotionDataset: PyTorch dataset for emotion classification data.

Functions:
    create_bert_emotion_classifier: Factory function to create a classifier.
    evaluate_emotion_classifier: Evaluate model performance.
"""
```

### Comments

- Use comments sparingly and only when necessary
- Focus on explaining "why", not "what" or "how"
- Keep comments up-to-date with code changes
- Use TODO, FIXME, and NOTE tags for special comments

```python
# Good comment - explains why
adjusted_threshold = base_threshold * 0.8  # Lower threshold for rare emotions

# Bad comment - just repeats code
batch_size = 32  # Set batch size to 32
```

## 🧪 Testing Standards

### Test File Organization

```python
"""Tests for the BERTEmotionClassifier."""

import pytest
import torch
from unittest.mock import MagicMock, patch

from src.models.emotion_detection.bert_classifier import BERTEmotionClassifier

class TestBertEmotionClassifier:
    """Test suite for BERTEmotionClassifier."""

    @pytest.fixture
    def model(self):
        """Create a test model instance."""
        return BERTEmotionClassifier(
            model_name="bert-base-uncased",
            num_emotions=28
        )

    def test_forward_pass(self, model):
        """Test forward pass through the model."""
        # Test implementation

    def test_predict_emotions(self, model):
        """Test emotion prediction with threshold."""
        # Test implementation
```

### Test Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`
- Test fixtures: Descriptive names of what they provide

### Test Best Practices

- Each test should be independent and isolated
- Use pytest fixtures for common setup
- Mock external dependencies
- Test edge cases and error conditions
- Include both positive and negative test cases
- Keep tests focused and small
- Use descriptive test names that explain the expected behavior

## 🔄 Git Workflow

### Branch Naming

```
{type}/{description}
```

Types:
- `feature`: New functionality
- `bugfix`: Bug fixes
- `hotfix`: Critical fixes for production
- `refactor`: Code refactoring without changing functionality
- `docs`: Documentation changes
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks

Examples:
- `feature/bert-emotion-classifier`
- `bugfix/fix-f1-score-calculation`
- `refactor/optimize-data-loading`

### Commit Message Format

```
{type}: {subject}

{body}

{footer}
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Formatting, missing semicolons, etc. (no code change)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat: Add temperature scaling for model calibration

Implement temperature scaling to calibrate confidence scores
in multi-label classification. This improves F1 scores by
properly scaling logits before applying the sigmoid function.

Resolves: #123
```

```
fix: Correct threshold in emotion prediction

Lower threshold from 0.5 to 0.2 for multi-label classification
to improve recall without significantly impacting precision.

Resolves: #456
```

### Pull Request Guidelines

- Keep PRs focused on a single feature or fix
- Include comprehensive description of changes
- Reference related issues
- Ensure all tests pass
- Update documentation as needed
- Request reviews from appropriate team members
- Follow the PR template

## 📊 Code Examples

### Model Class Example

```python
class BERTEmotionClassifier(nn.Module):
    """BERT-based classifier for emotion detection.

    This model fine-tunes a pre-trained BERT model for multi-label
    emotion classification using the GoEmotions taxonomy (28 emotions).

    Attributes:
        bert: Pre-trained BERT model.
        dropout: Dropout layer for regularization.
        classifier: Linear layer for classification.
        model_name: Name of the pre-trained model.
        num_emotions: Number of emotion categories.
        temperature: Temperature parameter for calibration.
        prediction_threshold: Threshold for positive predictions.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_emotions: int = 28,
        dropout_rate: float = 0.1,
    ):
        """Initialize the BERT emotion classifier.

        Args:
            model_name: Name of the pre-trained BERT model.
            num_emotions: Number of emotion categories.
            dropout_rate: Dropout probability for regularization.
        """
        super().__init__()
        self.model_name = model_name
        self.num_emotions = num_emotions

        # Initialize BERT model
        self.bert = AutoModel.from_pretrained(model_name)

        # Classifier head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_emotions)

        # Calibration parameters
        self.temperature = nn.Parameter(torch.ones(1))
        self.prediction_threshold = 0.6

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length).
            attention_mask: Attention mask of shape (batch_size, sequence_length).
            token_type_ids: Optional token type IDs.

        Returns:
            Dictionary containing logits, probabilities, and calibrated logits.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # Apply temperature scaling for calibration
        calibrated_logits = logits / self.temperature

        # Calculate probabilities
        probabilities = torch.sigmoid(calibrated_logits)

        return {
            "logits": logits,
            "calibrated_logits": calibrated_logits,
            "probabilities": probabilities,
        }

    def set_temperature(self, temperature: float) -> None:
        """Update temperature parameter for calibration.

        Args:
            temperature: New temperature value (must be positive).

        Raises:
            ValueError: If temperature is not positive.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")

        with torch.no_grad():
            self.temperature.fill_(temperature)

    def predict_emotions(
        self,
        texts: Union[str, List[str]],
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Dict:
        """Predict emotions from input texts.

        Args:
            texts: Input text or list of texts.
            threshold: Optional threshold override (default: self.prediction_threshold).
            top_k: Optional number of top emotions to return.

        Returns:
            Dictionary with predicted emotions and probabilities.
        """
        # Implementation details...
```

### Configuration Management Example

```python
def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If configuration file doesn't exist.
        ValueError: If configuration file is invalid.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Validate required fields
        required_fields = ["model", "training", "data"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in config")

        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")
```

### Error Handling Example

```python
def safe_model_load(model_path: str) -> nn.Module:
    """Safely load a PyTorch model with error handling.

    Args:
        model_path: Path to model checkpoint.

    Returns:
        Loaded PyTorch model.

    Raises:
        ModelLoadError: If model loading fails.
    """
    try:
        # Check if file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")

        # Create model
        model, _ = create_bert_emotion_classifier()

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")

        return model
    except (FileNotFoundError, ValueError) as e:
        # Re-raise with custom exception
        raise ModelLoadError(f"Failed to load model: {e}")
    except Exception as e:
        # Catch-all for unexpected errors
        raise ModelLoadError(f"Unexpected error loading model: {e}")
```

## 🚀 Performance Best Practices

### Memory Management

- Use context managers for large operations
- Release memory explicitly when possible
- Use generators for large data processing
- Implement lazy loading for large resources

```python
# Good practice - context manager for GPU operations
with torch.cuda.device(device):
    # GPU operations here
    pass  # Memory released when context exits

# Good practice - generator for large data processing
def process_large_dataset(file_path):
    with open(file_path, "r") as f:
        for line in f:
            # Process line
            yield processed_line
```

### Computation Efficiency

- Use vectorized operations when possible
- Avoid unnecessary computation in loops
- Use appropriate batch sizes
- Implement early stopping for training
- Use development mode for rapid iteration

```python
# Good practice - vectorized operations
embeddings = model(input_ids, attention_mask)

# Bad practice - unnecessary loop
results = []
for i in range(len(input_ids)):
    result = model(input_ids[i:i+1], attention_mask[i:i+1])
    results.append(result)
```

## 🔒 Security Best Practices

### API Keys and Secrets

- Never hardcode API keys or secrets
- Use environment variables for sensitive information
- Use .env files for local development (add to .gitignore)
- Use secrets management for production

```python
# Good practice - load from environment
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Bad practice - hardcoded secret
api_key = "sk-1234567890abcdef1234567890abcdef"
```

### Input Validation

- Validate all user inputs
- Implement proper error handling
- Use type hints and enforce types
- Sanitize inputs before processing

```python
def analyze_text(text: str, max_length: int = 5000) -> Dict:
    """Analyze text for emotions.

    Args:
        text: Input text to analyze.
        max_length: Maximum allowed text length.

    Returns:
        Dictionary with analysis results.

    Raises:
        ValueError: If text is invalid or too long.
    """
    # Validate input
    if not text or not isinstance(text, str):
        raise ValueError("Text must be a non-empty string")

    if len(text) > max_length:
        raise ValueError(f"Text exceeds maximum length of {max_length} characters")

    # Sanitize input
    text = text.strip()

    # Process text
    # ...
```

## 🧠 AI/ML Specific Standards

### Model Versioning

Models should be versioned using semantic versioning:

```
{major}.{minor}.{patch}
```

- Major: Breaking changes in model behavior or API
- Minor: Non-breaking improvements or enhancements
- Patch: Bug fixes and minor improvements

Example: `bert_emotion_classifier_v1.2.3`

### Experiment Tracking

All experiments should be tracked with:
- Unique experiment ID
- Complete configuration
- Performance metrics
- Timestamps
- Git commit hash
- Dataset version

```python
experiment_metadata = {
    "id": str(uuid.uuid4()),
    "name": "bert_emotion_classifier_temperature_tuning",
    "description": "Tuning temperature parameter for calibration",
    "config": config,
    "git_commit": get_git_commit_hash(),
    "dataset_version": "goemotions_v1.0",
    "timestamp": datetime.now().isoformat(),
    "metrics": {
        "micro_f1": 0.82,
        "macro_f1": 0.76,
        "precision": 0.85,
        "recall": 0.79
    }
}
```

### Model Saving

Models should be saved with comprehensive metadata:

```python
def save_model(model, path, metadata=None):
    """Save model with metadata.

    Args:
        model: PyTorch model to save.
        path: Path to save the model.
        metadata: Optional dictionary of metadata.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Prepare checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "model_name": model.model_name,
            "num_emotions": model.num_emotions,
            "temperature": model.temperature.item(),
            "prediction_threshold": model.prediction_threshold
        },
        "metadata": metadata or {},
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

    # Save checkpoint
    torch.save(checkpoint, path)
```

## 🔍 Code Review Guidelines

### Code Review Checklist

- Code follows style guide and conventions
- Documentation is complete and accurate
- Tests are comprehensive and pass
- Error handling is appropriate
- Performance considerations are addressed
- Security best practices are followed
- No unnecessary dependencies
- No duplicate code
- Code is maintainable and readable

### Code Review Process

1. **Author**: Create PR with comprehensive description
2. **Reviewers**: Review code within 24 hours
3. **Discussion**: Address feedback in comments
4. **Updates**: Author makes requested changes
5. **Final Review**: Reviewers approve changes
6. **Merge**: Author merges PR after approval

### Code Review Comments

- Be specific and actionable
- Explain reasoning behind suggestions
- Provide examples when helpful
- Be constructive and respectful
- Distinguish between required changes and suggestions
- Focus on the code, not the person

## 📚 Additional Resources

- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [PyTorch Best Practices](https://pytorch.org/docs/stable/notes/best_practices.html)
- [Ruff Documentation](https://github.com/charliermarsh/ruff)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)

## 🔄 Keeping This Document Updated

This document is a living guide that should evolve with the project. When making significant changes to coding standards or practices:

1. Create a PR with proposed changes
2. Discuss with the team
3. Update this document after approval
4. Communicate changes to the team

Last updated: July 25, 2025
