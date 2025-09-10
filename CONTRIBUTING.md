# Contributing to SAMO-DL

Thank you for your interest in contributing to SAMO-DL! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/uelkerd/SAMO--DL.git
   cd SAMO--DL
   ```

2. **Set up development environment**
   ```bash
   make setup
   # or manually:
   # python scripts/setup_dev_environment.py
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Code Quality Standards

### Formatting
- **Black**: Code formatting (line length: 88)
- **isort**: Import sorting
- **Flake8**: Linting
- **Pylint**: Code analysis
- **MyPy**: Type checking

### Running Quality Checks
```bash
# Run all quality checks
make quality-check

# Format code
make format

# Run tests
make test

# Run specific test types
make test-unit
make test-integration
```

### Pre-commit Hooks
All commits are automatically checked with pre-commit hooks. To run manually:
```bash
pre-commit run --all-files
```

## Pull Request Guidelines

### PR Size Limits
- **Maximum 25 files** changed per PR
- **Maximum 500 lines** changed per PR
- **Maximum 5 commits** per PR
- **48-hour maximum** branch lifetime

### PR Structure
1. **One clear purpose** per PR
2. **Descriptive title** (e.g., "feat: add emotion detection endpoint")
3. **Detailed description** with:
   - What was changed
   - Why it was changed
   - How to test
   - Any breaking changes

### Branch Naming
- `feat/dl-<description>`: New features
- `fix/dl-<description>`: Bug fixes
- `refactor/dl-<description>`: Code refactoring
- `test/dl-<description>`: Test additions
- `docs/dl-<description>`: Documentation updates

## Testing

### Test Structure
- **Unit tests**: `tests/test_*.py`
- **Integration tests**: `tests/test_*_integration.py`
- **System tests**: `tests/test_system_*.py`

### Running Tests
```bash
# All tests
pytest

# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# With coverage
pytest --cov=src --cov-report=html
```

### Test Requirements
- **80% minimum** code coverage
- **All tests must pass** before merging
- **New features require tests**

## Security

### Security Checks
- **Bandit**: Security linting
- **Safety**: Dependency vulnerability scanning
- **Pre-commit hooks**: Automatic security checks

### Reporting Security Issues
Please report security issues privately to the maintainers.

## Documentation

### Code Documentation
- **Docstrings**: All public functions and classes
- **Type hints**: All function parameters and return values
- **Comments**: Complex logic explanations

### API Documentation
- **OpenAPI/Swagger**: Auto-generated from code
- **Examples**: Comprehensive usage examples
- **README**: Setup and usage instructions

## Commit Guidelines

### Commit Message Format
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Maintenance tasks

### Examples
```
feat(api): add emotion detection endpoint
fix(model): resolve CUDA memory leak
docs(readme): update installation instructions
```

## Review Process

### Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass and coverage is adequate
- [ ] Documentation is updated
- [ ] No security vulnerabilities
- [ ] Performance impact considered
- [ ] Breaking changes documented

### Review Timeline
- **Initial review**: Within 24 hours
- **Follow-up reviews**: Within 12 hours
- **Merge decision**: Within 48 hours

## Getting Help

### Resources
- **Issues**: GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions
- **Documentation**: README and code comments

### Contact
- **Maintainers**: @uelkerd
- **Project**: SAMO-DL

## License

By contributing to SAMO-DL, you agree that your contributions will be licensed under the MIT License.