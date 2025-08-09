# 🛠️ Development Setup Guide

Welcome, Developers! This guide will help you set up your development environment for SAMO Brain and get started with contributing to the project.

## 🚀 **Quick Setup (10 minutes)**

### **Prerequisites**
- Python 3.12+
- Git
- Docker (optional, for containerized development)
- Google Cloud SDK (for GCP deployment)

### **1. Clone the Repository**
```bash
git clone https://github.com/your-org/SAMO--DL.git
cd SAMO--DL
```

### **2. Set Up Virtual Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **3. Environment Configuration**
```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

**Required Environment Variables:**
```bash
# API Configuration
API_HOST=localhost
API_PORT=8000
DEBUG=True

# Model Configuration
MODEL_PATH=models/emotion_detection/
MODEL_VERSION=1.0.0

# Database Configuration (if using)
DATABASE_URL=sqlite:///./data/samo_brain.db

# Monitoring Configuration
ENABLE_METRICS=True
METRICS_PORT=9090

# Security Configuration
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here
```

### **4. Verify Installation**
```bash
# Run basic tests
python -m pytest tests/unit/ -v

# Start development server
python src/unified_ai_api.py

# Test API endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling happy today!"}'
```

---

## 🔧 **Development Environment Setup**

### **IDE Configuration**

**VS Code Setup:**
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

**PyCharm Setup:**
1. Open project in PyCharm
2. Configure Python interpreter: `File > Settings > Project > Python Interpreter`
3. Select virtual environment: `./venv/bin/python`
4. Install project dependencies
5. Configure pytest: `File > Settings > Tools > Python Integrated Tools`

### **Pre-commit Hooks**
```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

**Pre-commit Configuration:**
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.12

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]
```

---

## 🧪 **Testing Setup**

### **Test Environment Configuration**
```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock pytest-asyncio

# Create test configuration
mkdir -p tests/config
```

**Test Configuration:**
```python
# tests/conftest.py
import pytest
import tempfile
import os
from pathlib import Path

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "I am feeling happy today!",
        "I feel sad about the news",
        "I am excited for the party"
    ]

@pytest.fixture
def mock_model_response():
    """Mock model response for testing."""
    return {
        "predicted_emotion": "happy",
        "confidence": 0.95,
        "probabilities": {
            "happy": 0.95,
            "sad": 0.02,
            "excited": 0.03
        },
        "prediction_time_ms": 150,
        "model_version": "1.0.0"
    }
```

### **Running Tests**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v

# Run tests in parallel
pytest -n auto

# Run tests with verbose output
pytest -v --tb=short
```

### **Test Structure**
```
tests/
├── unit/                    # Unit tests
│   ├── test_api_models.py
│   ├── test_api_rate_limiter.py
│   └── test_emotion_detection.py
├── integration/             # Integration tests
│   ├── test_api_endpoints.py
│   └── test_database.py
├── e2e/                     # End-to-end tests
│   └── test_complete_workflows.py
├── conftest.py             # Test configuration
└── fixtures/               # Test fixtures
    ├── sample_data.json
    └── test_models/
```

---

## 🔒 **Security Setup**

### **Security Dependencies**
```bash
# Install security tools
pip install bandit safety pip-audit

# Run security checks
bandit -r src/
safety check
pip-audit
```

**Security Configuration:**
```ini
# .bandit
[bandit]
exclude: tests/,venv/
skips: B101,B601
```

### **Environment Security**
```bash
# Generate secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Set up API key rotation
# Add to .env
API_KEY_ROTATION_INTERVAL=86400  # 24 hours
API_KEY_BACKUP_COUNT=3
```

### **Docker Security**
```dockerfile
# Dockerfile.dev
FROM python:3.12-slim

# Create non-root user
RUN groupadd -r samo && useradd -r -g samo samo

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/

# Change ownership
RUN chown -R samo:samo /app

# Switch to non-root user
USER samo

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "src/unified_ai_api.py"]
```

---

## 📊 **Monitoring Setup**

### **Development Monitoring**
```python
# src/monitoring/dev_monitor.py
import logging
import time
from functools import wraps
from typing import Callable, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dev.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def monitor_performance(func: Callable) -> Callable:
    """Decorator to monitor function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    return wrapper

def log_api_request(request_data: dict, response_data: dict, execution_time: float):
    """Log API request details."""
    logger.info(f"API Request: {request_data} -> {response_data} ({execution_time:.3f}s)")
```

### **Health Check Endpoint**
```python
# src/health_check.py
from datetime import datetime
import psutil
import os

def get_system_health():
    """Get system health metrics."""
    return {
        "timestamp": datetime.now().isoformat(),
        "status": "healthy",
        "uptime": psutil.boot_time(),
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "process_id": os.getpid()
    }
```

---

## 🔄 **Development Workflow**

### **Git Workflow**
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: add new emotion detection feature"

# Push to remote
git push origin feature/your-feature-name

# Create pull request
# (Use GitHub web interface)
```

### **Code Quality Checks**
```bash
# Run linting
flake8 src/ tests/

# Run formatting
black src/ tests/
isort src/ tests/

# Run type checking
mypy src/

# Run security checks
bandit -r src/
safety check
```

### **Pre-commit Workflow**
```bash
# Install pre-commit hooks
pre-commit install

# Run on staged files
pre-commit run

# Run on all files
pre-commit run --all-files
```

---

## 🐳 **Docker Development**

### **Docker Compose for Development**
```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  samo-brain-api:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
    volumes:
      - ./src:/app/src
      - ./tests:/app/tests
      - ./logs:/app/logs
    environment:
      - DEBUG=True
      - API_HOST=0.0.0.0
      - API_PORT=8000
    command: python -m pytest tests/ && python src/unified_ai_api.py

  samo-brain-tests:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - ./src:/app/src
      - ./tests:/app/tests
    command: python -m pytest tests/ -v --cov=src

  monitoring:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
```

### **Development Commands**
```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Run tests in container
docker-compose -f docker-compose.dev.yml run samo-brain-tests

# View logs
docker-compose -f docker-compose.dev.yml logs -f samo-brain-api

# Stop development environment
docker-compose -f docker-compose.dev.yml down
```

---

## 📚 **Documentation Development**

### **Documentation Structure**
```
docs/
├── api/                     # API documentation
│   ├── endpoints.md
│   └── models.md
├── guides/                  # User guides
│   ├── getting-started.md
│   └── troubleshooting.md
├── development/             # Development docs
│   ├── contributing.md
│   └── architecture.md
└── wiki/                    # GitHub wiki
    ├── Home.md
    └── Integration-Guides/
```

### **Documentation Tools**
```bash
# Install documentation tools
pip install mkdocs mkdocs-material

# Build documentation
mkdocs build

# Serve documentation locally
mkdocs serve
```

---

## 🚨 **Troubleshooting**

### **Common Issues**

**Import Errors:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Port Conflicts:**
```bash
# Check if port is in use
lsof -i :8000

# Kill process using port
kill -9 <PID>
```

**Model Loading Issues:**
```bash
# Check model path
ls -la models/emotion_detection/

# Verify model files
python -c "import torch; print(torch.__version__)"
```

**Test Failures:**
```bash
# Run tests with verbose output
pytest -v --tb=long

# Run specific failing test
pytest tests/unit/test_specific.py::test_function -v -s
```

---

## 📞 **Support & Resources**

- **GitHub Issues**: [Report Issues](https://github.com/your-org/SAMO--DL/issues)
- **Discord Channel**: [Join Development Community](https://discord.gg/samo-brain)
- **Documentation**: [Complete API Reference](API-Reference)
- **Code Review**: [Pull Request Guidelines](CONTRIBUTING.md)

---

**Ready to start developing?** Follow the [Quick Setup](#-quick-setup-10-minutes) above and join our development community! 🚀👨‍💻 