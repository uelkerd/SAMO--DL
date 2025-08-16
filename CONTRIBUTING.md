# Contributing to SAMO-DL

## 🎯 Welcome Contributors!

Thank you for your interest in contributing to the SAMO-DL project! This guide will help you get started and ensure your contributions align with our project standards.

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Code Review Guidelines](#code-review-guidelines)
- [Security Guidelines](#security-guidelines)
- [Documentation](#documentation)
- [Support](#support)

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.10+
- **Git**: Latest version
- **Docker**: 20.10+ (for containerized development)
- **Make**: For automation scripts

### Quick Start

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/SAMO--DL.git
   cd SAMO--DL
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Set up Git hooks**
   ```bash
   # Configure Git to use hooks from scripts/ directory
   ./scripts/setup-hooks.sh
   
   # Or manually configure:
   git config core.hooksPath scripts
   
   # Verify configuration:
   git config --get core.hooksPath
   # Should return: scripts
   ```

4. **Run tests**
   ```bash
   # Run all tests
   pytest
   
   # Run with coverage
   pytest --cov=.
   ```

## 🛠️ Development Setup

### Git Hooks Setup

This repository uses Git hooks located in the `scripts/` directory for code quality and security checks. After cloning:

1. **Automatic setup**:
   ```bash
   ./scripts/setup-hooks.sh
   ```

2. **Manual setup**:
   ```bash
   git config core.hooksPath scripts
   ```

3. **Verify setup**:
   ```bash
   git config --get core.hooksPath
   # Expected output: scripts
   ```

**Available hooks:**
- `pre-push`: Git LFS validation before pushing
- `post-merge`: Git LFS validation after merging
- `post-commit`: Git LFS validation after commits
- `post-checkout`: Git LFS validation after checkout

**Note**: If hooks aren't working, ensure you've run the setup script or configured `core.hooksPath` correctly.

### Environment Configuration

Create a `.env` file for local development:

```bash
# .env
ENVIRONMENT=development
DATABASE_URL=postgresql://user:pass@localhost:5432/samo_dl_dev
SECRET_KEY=dev-secret-key-change-in-production
API_KEY=dev-api-key
LOG_LEVEL=DEBUG
```

### Docker Development

```bash
# Build development container
docker build -f deployment/cloud-run/Dockerfile -t samo-dl-dev .

# Run with development settings
docker run -p 8080:8080 \
  -e ENVIRONMENT=development \
  -e DATABASE_URL=postgresql://user:pass@host:5432/db \
  samo-dl-dev
```

### Database Setup

```bash
# Install PostgreSQL (Ubuntu)
sudo apt install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb samo_dl_dev

# Run migrations
alembic upgrade head
```

## 📝 Code Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

```python
# ✅ Good
def predict_emotion(text: str) -> Dict[str, Any]:
    """Predict emotion from text input.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary containing emotion prediction and confidence
        
    Raises:
        ValueError: If text is empty or invalid
    """
    if not text or not isinstance(text, str):
        raise ValueError("Text must be a non-empty string")
    
    # Implementation here
    return {"emotion": "happy", "confidence": 0.95}

# ❌ Bad
def predict_emotion(text):
    if not text:
        return None
    # Implementation without type hints or docstrings
```

### Code Formatting

We use **Black** for code formatting and **Ruff** for linting:

```bash
# Format code
black .

# Lint code
ruff check .

# Auto-fix linting issues
ruff check --fix .
```

### Type Hints

All functions should include type hints:

```python
from typing import Dict, List, Optional, Any
import torch
from transformers import AutoTokenizer

def load_model(model_path: str) -> Optional[torch.nn.Module]:
    """Load PyTorch model from path."""
    pass

def predict_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """Predict emotions for multiple texts."""
    pass
```

### Documentation Standards

#### Docstrings

Use Google-style docstrings:

```python
def process_text(text: str, max_length: int = 512) -> str:
    """Process and clean input text.
    
    Args:
        text: Raw input text
        max_length: Maximum allowed text length
        
    Returns:
        Processed and cleaned text
        
    Raises:
        ValueError: If text exceeds maximum length
        TypeError: If text is not a string
        
    Example:
        >>> process_text("Hello, world!", max_length=10)
        "Hello, wor"
    """
    if not isinstance(text, str):
        raise TypeError("Text must be a string")
    
    if len(text) > max_length:
        text = text[:max_length]
    
    return text.strip()
```

#### Comments

- Use comments to explain **why**, not **what**
- Keep comments up-to-date with code changes
- Use TODO comments for future improvements

```python
# ✅ Good - explains why
# Use CPU for inference to avoid GPU memory issues in production
device = torch.device('cpu')

# ❌ Bad - explains what (obvious from code)
# Set device to CPU
device = torch.device('cpu')
```

## 🧪 Testing

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── e2e/           # End-to-end tests
├── fixtures/      # Test data and fixtures
└── conftest.py    # Pytest configuration
```

### Writing Tests

```python
# tests/unit/test_emotion_detector.py
import pytest
from src.emotion_detector import EmotionDetector

class TestEmotionDetector:
    """Test cases for EmotionDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create EmotionDetector instance for testing."""
        return EmotionDetector()
    
    def test_predict_happy_text(self, detector):
        """Test emotion prediction for happy text."""
        text = "I'm feeling really happy today!"
        result = detector.predict(text)
        
        assert result["emotion"] == "happy"
        assert result["confidence"] > 0.8
        assert "text" in result
    
    def test_predict_empty_text(self, detector):
        """Test emotion prediction with empty text."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            detector.predict("")
    
    def test_predict_invalid_input(self, detector):
        """Test emotion prediction with invalid input."""
        with pytest.raises(TypeError, match="Text must be a string"):
            detector.predict(123)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_emotion_detector.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run integration tests only
pytest tests/integration/

# Run tests in parallel
pytest -n auto
```

### Test Coverage

We aim for **90%+ test coverage**:

```bash
# Generate coverage report
pytest --cov=src --cov-report=term-missing

# View HTML coverage report
open htmlcov/index.html
```

## 🔄 Pull Request Process

### 1. Create Feature Branch

```bash
# Create and switch to feature branch
git checkout -b feature/your-feature-name

# Or use conventional commit format
git checkout -b feat/add-new-emotion-model
git checkout -b fix/security-vulnerability
git checkout -b docs/update-api-documentation
```

### 2. Make Changes

- Write code following our standards
- Add tests for new functionality
- Update documentation
- Ensure all tests pass

### 3. Commit Changes

Use conventional commit format:

```bash
# Format: type(scope): description
git commit -m "feat(api): add batch prediction endpoint"
git commit -m "fix(security): update dependencies to fix vulnerabilities"
git commit -m "docs(readme): update installation instructions"
git commit -m "test(emotion): add comprehensive test coverage"
```

**Commit Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### 4. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create Pull Request on GitHub
```

### 5. PR Template

Use our PR template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No security vulnerabilities introduced
```

## 👀 Code Review Guidelines

### For Contributors

**Before submitting PR:**
- [ ] Self-review your code
- [ ] Ensure all tests pass
- [ ] Update documentation
- [ ] Check for security issues
- [ ] Follow naming conventions

**During review:**
- Respond to feedback promptly
- Be open to suggestions
- Explain your reasoning when needed
- Make requested changes

### For Reviewers

**Review checklist:**
- [ ] Code follows project standards
- [ ] Tests are comprehensive
- [ ] Documentation is updated
- [ ] No security issues introduced
- [ ] Performance considerations addressed
- [ ] Error handling is appropriate

**Review comments:**
- Be constructive and specific
- Suggest alternatives when possible
- Focus on code quality and maintainability
- Consider security implications

## 🔒 Security Guidelines

### Security Best Practices

1. **Input Validation**
   ```python
   # ✅ Good
   def validate_text(text: str) -> str:
       if not isinstance(text, str):
           raise TypeError("Text must be a string")
       if len(text) > 1000:
           raise ValueError("Text too long")
       return text.strip()
   ```

2. **Secrets Management**
   ```python
   # ✅ Good - Use environment variables
   import os
   api_key = os.getenv('API_KEY')
   
   # ❌ Bad - Hardcoded secrets
   api_key = "your-api-key-here"  # Never commit real API keys
   ```

3. **SQL Injection Prevention**
   ```python
   # ✅ Good - Use parameterized queries
   cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
   
   # ❌ Bad - String concatenation
   cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
   ```

### Security Checklist

- [ ] No hardcoded secrets
- [ ] Input validation implemented
- [ ] SQL injection prevention
- [ ] XSS protection
- [ ] CSRF protection
- [ ] Rate limiting implemented
- [ ] Error messages don't leak information
- [ ] Dependencies are up-to-date

### Reporting Security Issues

**For security vulnerabilities:**
1. **DO NOT** create a public issue
2. Email: security@samo-project.com
3. Include detailed description and reproduction steps
4. We'll respond within 24 hours

## 📚 Documentation

### Documentation Standards

1. **README Updates**
   - Update README.md for user-facing changes
   - Include examples and usage instructions
   - Update installation steps if needed

2. **API Documentation**
   - Update OpenAPI specification
   - Add examples for new endpoints
   - Document error responses

3. **Code Documentation**
   - Add docstrings to all functions
   - Include type hints
   - Add inline comments for complex logic

### Documentation Checklist

- [ ] README updated
- [ ] API docs updated
- [ ] Code docstrings added
- [ ] Examples provided
- [ ] Installation instructions current
- [ ] Troubleshooting section updated

## 🆘 Support

### Getting Help

1. **Check existing issues** on GitHub
2. **Search documentation** for answers
3. **Ask in discussions** for general questions
4. **Create issue** for bugs or feature requests

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Email**: security@samo-project.com (security issues only)

### Issue Templates

Use our issue templates:
- **Bug Report**: For reporting bugs
- **Feature Request**: For requesting new features
- **Documentation**: For documentation issues

## 🎉 Recognition

### Contributors

We recognize contributors in several ways:
- **Contributors list** in README
- **Release notes** for significant contributions
- **Special thanks** for major features

### Contribution Levels

- **Bronze**: 1-5 contributions
- **Silver**: 6-20 contributions
- **Gold**: 21+ contributions
- **Platinum**: Core team member

## 📄 License

By contributing to SAMO-DL, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to SAMO-DL!** 🚀

Your contributions help make this project better for everyone in the community. 