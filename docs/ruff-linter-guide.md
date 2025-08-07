# SAMO Deep Learning - Ruff Linter Guide

## Overview

Ruff is configured as the primary linting and code quality tool for the SAMO Deep Learning project. It provides fast, comprehensive analysis optimized for ML/Data Science workflows with automatic fixing capabilities.

## Quick Start

### Installation

Ruff is included in your conda environment:

```bash
conda activate samo-dl  # Already installed
```

### Basic Usage

```bash
# Run comprehensive check (recommended)
./scripts/lint.sh

# Quick lint check only
./scripts/lint.sh check

# Auto-fix issues
./scripts/lint.sh fix

# Show statistics
./scripts/lint.sh stats

# Get help
./scripts/lint.sh help
```

## What Ruff Checks

### Core Quality Rules (Always Enforced)

- **Syntax Errors**: Python syntax issues
- **Import Organization**: Proper import sorting and structure
- **Code Style**: PEP 8 compliance with 88-character line length
- **Security Issues**: Common security vulnerabilities (Bandit integration)
- **Bug Detection**: Common Python bugs and anti-patterns
- **Performance**: Inefficient code patterns

### ML/Data Science Specific Rules

- **Pandas Best Practices**: Efficient DataFrame operations
- **NumPy Conventions**: Proper array handling
- **Scientific Libraries**: Correct usage of sklearn, torch, transformers
- **Data Validation**: Type checking and data integrity
- **Memory Management**: Resource cleanup and optimization

### Development Quality

- **Documentation**: Google-style docstrings
- **Testing**: Pytest best practices
- **Error Handling**: Proper exception management
- **Logging**: Structured logging patterns

## Configuration Highlights

### Project Structure Awareness

```toml
# Different rules for different directories
"tests/**/*.py" = [
    "S101",    # Asserts expected in tests
    "ANN",     # Type annotations less critical
    "D",       # Documentation less critical
]

"notebooks/**/*.ipynb" = [
    "T201",    # Print statements expected
    "F401",    # Unused imports (exploration)
    "ANN",     # Type annotations optional
]

"src/models/**/*.py" = [
    "PLR0913", # Many hyperparameters allowed
    "C901",    # Complex ML algorithms
    "S101",    # Asserts for tensor validation
]
```

### ML-Friendly Settings

- **Higher complexity thresholds** for ML algorithms
- **Flexible documentation** for research code
- **Security exceptions** for ML-specific patterns
- **Performance optimizations** for data processing

## Integration with Development Workflow

### Pre-commit Hook (Recommended)

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
conda activate samo-dl
./scripts/lint.sh full
```

### Editor Integration

#### VS Code

1. Install the **Ruff extension**
2. Add to `settings.json`:

```json
{
    "[python]": {
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": true,
            "source.organizeImports.ruff": true
        }
    },
    "ruff.lint.args": ["--config=pyproject.toml"]
}
```

#### PyCharm

1. Go to **Settings → Tools → External Tools**
2. Add new tool:
   - Name: `Ruff Check`
   - Program: `conda`
   - Arguments: `run -n samo-dl ruff check --fix $FilePath$`
   - Working directory: `$ProjectFileDir$`

### CI/CD Integration

Add to your CI pipeline:

```yaml
- name: Lint with Ruff
  run: |
    conda activate samo-dl
    ./scripts/lint.sh full
```

## Common Issues and Solutions

### 1. Line Length (E501)

**Problem**: Lines exceed 88 characters

```python
# Bad
model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=27)

# Good
model = transformers.AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=27
)
```

### 2. Import Organization (I001)

**Problem**: Imports not properly sorted

```python
# Bad
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import sklearn

# Good (automatically fixed)
import numpy as np
import pandas as pd
import sklearn
import torch
from transformers import AutoTokenizer
```

### 3. Unused Imports (F401)

**Problem**: Imported modules not used

```python
# Bad
import numpy as np  # F401: not used
import pandas as pd

df = pd.DataFrame()

# Good
import pandas as pd

df = pd.DataFrame()
```

### 4. Documentation Style (D415)

**Problem**: Docstring doesn't end with punctuation

```python
# Bad
def preprocess_text(text: str) -> str:
    """Clean and normalize input text"""

# Good
def preprocess_text(text: str) -> str:
    """Clean and normalize input text."""
```

### 5. Security Issues (S101)

**Problem**: Using assert in production code

```python
# Bad (in production code)
assert len(input_ids) > 0

# Good
if len(input_ids) == 0:
    raise ValueError("Input IDs cannot be empty")
```

### 6. Type Annotations (UP035)

**Problem**: Using deprecated typing imports

```python
# Bad
from typing import List, Dict
def process_embeddings(embeddings: List[float]) -> Dict[str, float]:

# Good (automatically fixed)
def process_embeddings(embeddings: list[float]) -> dict[str, float]:
```

## Customizing Rules

### Temporary Rule Disabling

For specific lines:

```python
# ruff: noqa: E501
very_long_line_that_is_needed_for_some_specific_reason_and_cannot_be_shortened = True

# ruff: noqa: F401
import rarely_used_module  # Imported for side effects
```

For entire files, add to the top:

```python
# ruff: noqa
```

### Project-Level Rule Changes

Edit `pyproject.toml`:

```toml
[tool.ruff.lint]
ignore = [
    "E501",  # Ignore line length globally (not recommended)
    "D100",  # Ignore missing module docstrings
]
```

## Performance and Statistics

### Current Project Stats

- **Python files**: 17 analyzed
- **Configuration**: `pyproject.toml`
- **Line length**: 88 characters
- **Target Python**: 3.10

### Speed Benefits

- **10-100x faster** than traditional linters
- **Built in Rust** for maximum performance
- **Parallel processing** for large codebases
- **Incremental checking** for changed files only

## ML-Specific Best Practices

### 1. Model Training Code

```python
# Good: Clear parameter separation
def train_emotion_classifier(
    model_name: str = "bert-base-uncased",
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    num_epochs: int = 3,
    warmup_steps: int = 500,
) -> tuple[AutoModel, float]:
    """Train BERT model for emotion classification."""
```

### 2. Data Processing

```python
# Good: Explicit error handling
def load_goemotions_dataset(data_path: str) -> pd.DataFrame:
    """Load and validate GoEmotions dataset."""
    try:
        df = pd.read_csv(data_path)
        if df.empty:
            raise ValueError("Dataset is empty")
        return df
    except FileNotFoundError:
        logger.error(f"Dataset not found at {data_path}")
        raise
```

### 3. Notebook Development

- Ruff ignores exploration-specific issues in notebooks
- Focus on production code quality in `src/`
- Use `./scripts/lint.sh` regularly during development

## Integration with Black Formatter

Ruff is configured to work harmoniously with Black:

- **Same line length** (88 characters)
- **Compatible quote styles**
- **Consistent import formatting**
- **No formatting conflicts**

Run both tools:

```bash
# Format with Black first
conda activate samo-dl
black .

# Then lint with Ruff
./scripts/lint.sh
```

## Troubleshooting

### Configuration Issues

If you see TOML parsing errors:

```bash
# Test configuration
conda activate samo-dl
ruff check --config pyproject.toml src/
```

### Environment Issues

```bash
# Verify installation
conda activate samo-dl
ruff --version  # Should show 0.12.0+

# Reinstall if needed
conda install -y ruff
```

### Performance Issues

```bash
# Use cache for faster subsequent runs
export RUFF_CACHE_DIR=.ruff_cache

# Check only changed files
git diff --name-only | grep '\.py$' | xargs ruff check
```

## Next Steps

1. **Address Current Issues**: Run `./scripts/lint.sh fix` to auto-fix 578+ issues
2. **Manual Review**: Address remaining 550 issues requiring human judgment
3. **Set Up Editor**: Configure VS Code or PyCharm integration
4. **Establish Workflow**: Add pre-commit hooks
5. **Team Training**: Share this guide with team members

## Resources

- **Ruff Documentation**: <https://docs.astral.sh/ruff/>
- **Rule Reference**: <https://docs.astral.sh/ruff/rules/>
- **Configuration Guide**: <https://docs.astral.sh/ruff/configuration/>
- **Editor Integration**: <https://docs.astral.sh/ruff/editors/>

---

**Next Actions for SAMO-DL Team:**

1. Run `./scripts/lint.sh fix` to clean up existing codebase
2. Set up editor integration for real-time feedback
3. Add to CI/CD pipeline for quality gates
4. Focus on ML model development with clean, maintainable code! 🚀
