# SAMO Deep Learning Repository

**SAMO** is an AI-powered, voice-first journaling companion designed to provide real emotional reflection. This repository contains the core AI/ML components including emotion detection, text summarization, and voice processing capabilities.

## 🎯 **Current Status: Week 1-2 Complete, Ahead of Schedule**

**🚀 Foundation Phase SUCCESS**: Infrastructure transformed from compromised state to production-ready ML pipeline
- **Security**: ✅ Resolved critical vulnerabilities, secured database credentials  
- **Code Quality**: ✅ Implemented Ruff linter with 578 automatic fixes
- **Architecture**: ✅ Clean repository (311→43 files, 86% reduction)
- **Emotion Detection**: ✅ Complete pipeline with excellent training progress (loss: 0.7016 → 0.1922)
- **Performance Tools**: ✅ GPU optimization, ONNX conversion, and benchmarking scripts ready

## 🚀 Quick Start

### Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd SAMO--DL

# Activate the ML environment
conda activate samo-dl

# Create environment file from template (see docs/environment-setup.md)
cp .env.template .env
# Edit .env with your database credentials

# Test database connection
python scripts/database/check_pgvector.py

# Run code quality check
./scripts/lint.sh
```

## 🧠 **Emotion Detection Pipeline - ACTIVE TRAINING**

### Current Training Status
- **Model**: BERT emotion classifier (43.7M parameters)
- **Dataset**: GoEmotions (54,263 examples, 27 emotions + neutral)
- **Progress**: Loss decreasing excellently (0.7016 → 0.1922)
- **Architecture**: Progressive unfreezing, class-weighted loss, early stopping

### Performance Optimization Ready
```bash
# Check GPU setup and optimization recommendations
python scripts/setup_gpu_training.py --check

# Test domain adaptation (Reddit → Journal entries)
python scripts/test_domain_adaptation.py --test-adaptation

# Performance benchmarking and ONNX conversion
python scripts/optimize_performance.py --check-gpu --benchmark
```

## 🛠 Development Tools

### Code Quality with Ruff
This project uses **Ruff** for fast, comprehensive linting optimized for ML/Data Science:

```bash
# Full quality check
./scripts/lint.sh

# Auto-fix issues  
./scripts/lint.sh fix

# Quick lint only
./scripts/lint.sh check

# Show statistics
./scripts/lint.sh stats
```

**Key Features:**
- 🚀 **10-100x faster** than traditional linters
- 🎯 **ML/Data Science optimized** rules
- 🔧 **Auto-fixes** 500+ types of issues (578 applied to this project)
- 📝 **Jupyter notebook** support
- ⚙️ **VS Code integration** ready

📖 **Full Documentation**: [docs/ruff-linter-guide.md](docs/ruff-linter-guide.md)

## 🧠 AI/ML Components

### Core Capabilities
- **✅ Emotion Detection**: BERT-based classification using GoEmotions dataset (28 emotions)
  - Progressive unfreezing training strategy
  - Class-weighted loss for imbalanced data (0.10-6.53 range)
  - Multi-label classification support (16.2% of examples)
  - Domain adaptation testing for journal entries
- **🔄 Smart Summarization**: T5/BART models for extracting emotional core (Next Phase)
- **🔄 Voice Processing**: OpenAI Whisper integration for voice-to-text (Next Phase)
- **🔄 Semantic Search**: Vector embeddings for Memory Lane features (Next Phase)

### Performance Targets & Current Status
- **Emotion Detection**: >80% F1 score target (Training in progress, excellent convergence)
- **Response Latency**: <500ms P95 target (ONNX optimization scripts ready)
- **Voice Transcription**: <10% Word Error Rate (Planned Week 3-4)
- **Model Uptime**: >99.5% availability (Infrastructure ready)

### Training & Optimization Scripts
```bash
# Emotion detection training (currently running)
python -m src.models.emotion_detection.training_pipeline

# GPU setup and optimization
python scripts/setup_gpu_training.py --check --create-config

# Performance benchmarking
python scripts/optimize_performance.py --benchmark --target-latency 500

# Domain adaptation analysis
python scripts/test_domain_adaptation.py --test-adaptation
```

## 📁 Project Structure

```
SAMO--DL/
├── src/                    # Core ML source code
│   ├── data/              # Data processing & database
│   ├── models/            # ML model implementations  
│   │   └── emotion_detection/  # ✅ Complete BERT pipeline
│   │       ├── __init__.py
│   │       ├── dataset_loader.py    # GoEmotions data processing
│   │       ├── bert_classifier.py   # BERT emotion model
│   │       ├── training_pipeline.py # End-to-end training
│   │       └── api_demo.py         # FastAPI endpoints
│   ├── training/          # Model training pipelines
│   ├── inference/         # Model inference APIs
│   └── evaluation/        # Model evaluation & metrics
├── scripts/               # Maintenance & utility scripts
│   ├── lint.sh           # ✅ Code quality automation
│   ├── optimize_performance.py    # ✅ GPU/ONNX optimization
│   ├── setup_gpu_training.py     # ✅ GPU transition helper
│   ├── test_domain_adaptation.py # ✅ Journal entry testing
│   └── database/         # Database management scripts
├── models/                # Trained model artifacts
│   ├── emotion_detection/ # 🔄 Training checkpoints
│   ├── summarization/    # 📋 Planned Week 3-4
│   └── voice_processing/ # 📋 Planned Week 3-4
├── data/                  # Dataset storage
│   ├── cache/            # ✅ GoEmotions cached datasets
│   ├── processed/        # ✅ Preprocessed training data
│   └── raw/              # Original datasets
├── test_checkpoints/     # ✅ Current training checkpoints
├── notebooks/            # Jupyter research notebooks
├── tests/                # Test suites
├── docs/                 # Comprehensive documentation
└── configs/              # Environment configurations
```

## 🔧 Technical Stack

- **ML Frameworks**: PyTorch, Transformers (Hugging Face), ONNX Runtime
- **Models**: BERT (GoEmotions fine-tuned) ✅, T5/BART 📋, OpenAI Whisper 📋
- **Data**: PostgreSQL with pgvector ✅, Pandas, NumPy
- **API**: FastAPI ✅, SQLAlchemy, Pydantic
- **Development**: Ruff (linting) ✅, Black (formatting), Pytest (testing)
- **Optimization**: ONNX Runtime ✅, GPU acceleration scripts ✅
- **Deployment**: Docker, Kubernetes (ready for GCP migration)

## 📊 Development Guidelines

### Code Quality Standards
- **Line Length**: 88 characters (Black/Ruff compatible)
- **Documentation**: Google-style docstrings required for public APIs
- **Testing**: Unit tests for core functions, integration tests for pipelines
- **Type Hints**: Encouraged for production code, required for APIs

### ML-Specific Best Practices  
- **Model Validation**: Assert tensor shapes and data types
- **Error Handling**: Graceful handling of inference failures
- **Logging**: Structured logging for debugging ML pipelines ✅
- **Performance**: Profile critical paths, optimize for <500ms response time ✅

### Git Workflow
- **Small Commits**: Focus on single functionality ✅
- **Clean History**: Squash commits before merging
- **Branch Naming**: `feature/`, `bugfix/`, `model/` prefixes
- **Code Review**: Required for all production code

## 🚀 Getting Started with Development

### 1. Environment Setup
```bash
conda activate samo-dl
./scripts/lint.sh check  # Verify linting works
python scripts/database/check_pgvector.py  # Test database
```

### 2. Emotion Detection Training (Currently Active)
```bash
# Monitor current training progress
python -m src.models.emotion_detection.training_pipeline

# Test trained model on journal entries
python scripts/test_domain_adaptation.py --test-adaptation

# Prepare for GPU acceleration
python scripts/setup_gpu_training.py --check
```

### 3. Performance Optimization
```bash
# Convert model to ONNX for production
python scripts/optimize_performance.py --convert-onnx

# Benchmark inference speed
python scripts/optimize_performance.py --benchmark --target-latency 500
```

### 4. Code Quality Check
```bash
./scripts/lint.sh  # Full quality analysis
./scripts/lint.sh fix  # Auto-fix issues
```

## 📖 Documentation

### Essential Reading
- [📋 Project Scope & Requirements](docs/samo-dl-prd.md)
- [🔧 Environment Setup Guide](docs/environment-setup.md) 
- [🛡️ Security Setup](docs/security-setup.md)
- [🏗️ Technical Architecture](docs/tech-architecture.md)
- [📏 Data Schema Registry](docs/data-documentation-schema-registry.md)
- [🎯 Ruff Linter Guide](docs/ruff-linter-guide.md)

### Development Guides
- [🚀 Model Training Playbook](docs/model-training-playbook.md)
- [📊 Track Scope](docs/track-scope.md)

## ✅ **Quality Metrics - EXCELLENT PROGRESS**

### Infrastructure Transformation (100% Complete)
- **Security**: ✅ Resolved leaked database credentials, implemented secure patterns
- **Code Quality**: ✅ Ruff linter with 578 automatic fixes applied
- **Repository**: ✅ Cleaned from 311 to 43 files (86% reduction)
- **Database**: ✅ PostgreSQL + pgvector setup and tested

### Model Development Status
- **Emotion Detection**: ✅ Complete pipeline, training in progress
  - Dataset: 54,263 GoEmotions examples processed
  - Model: BERT with 43.7M parameters
  - Training: Loss decreasing excellently (0.7016 → 0.1922)
  - Class balance: Weighted loss handling imbalanced data
- **Performance Tools**: ✅ GPU optimization, ONNX conversion, benchmarking ready
- **Domain Adaptation**: ✅ Journal entry testing framework implemented

### Development Progress (Ahead of Schedule)
- ✅ **Week 1-2 Foundation**: COMPLETE (Infrastructure + Emotion Detection)
- 🚀 **Week 3-4 Ready**: T5 summarization, Whisper integration, GPU acceleration
- 🎯 **Performance Target**: <500ms P95 latency optimization scripts ready
- 🔄 **Next Phase**: GCP migration for GPU training acceleration

### Training Metrics (Live)
- **Current Epoch**: 1/3 (in progress)
- **Loss Trajectory**: 0.7016 → 0.1922 (excellent convergence)
- **Learning Rate**: Warmup schedule working correctly
- **Memory Usage**: CPU training stable, GPU optimization ready
- **Dataset Stats**: 39,069 train, 4,341 val, 10,853 test samples

## 🤝 Contributing

1. **Activate Environment**: `conda activate samo-dl`
2. **Create Feature Branch**: `git checkout -b feature/summarization-pipeline`
3. **Write Clean Code**: Follow linting guidelines (`./scripts/lint.sh`)
4. **Add Tests**: Unit tests for new functionality
5. **Update Documentation**: Keep docs current
6. **Quality Check**: `./scripts/lint.sh full` before commit
7. **Submit PR**: Clear description with testing evidence

## 📞 Support

For questions about the SAMO Deep Learning components:
- **Code Quality**: See [Ruff Linter Guide](docs/ruff-linter-guide.md)  
- **Environment Issues**: See [Environment Setup](docs/environment-setup.md)
- **Security Concerns**: See [Security Setup](docs/security-setup.md)
- **Architecture Questions**: See [Technical Architecture](docs/tech-architecture.md)
- **Training Pipeline**: See model training logs and scripts/

## 🏆 **Achievement Summary**

**Foundation Phase (Weeks 1-2): COMPLETE & AHEAD OF SCHEDULE**
- ✅ Security vulnerabilities resolved
- ✅ Code quality infrastructure implemented (578 auto-fixes)
- ✅ Clean, maintainable codebase established  
- ✅ Emotion detection pipeline complete and training successfully
- ✅ Performance optimization tools ready for GCP deployment
- ✅ Domain adaptation testing framework for journal entries

**Ready for Phase 2**: T5/BART summarization, OpenAI Whisper integration, and GPU-accelerated training on GCP! 🚀

---

**Building emotionally intelligent AI with production-ready ML engineering! 🧠✨** 