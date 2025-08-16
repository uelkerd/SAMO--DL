# 🔒 SAMO Security Pipeline Guide

## Overview

The SAMO Security Pipeline is a **completely independent** CI/CD pipeline that runs alongside your existing working CI pipeline. It focuses exclusively on security scanning, code quality, and vulnerability prevention without interfering with your main development workflow.

## 🏗️ Architecture

### Independent Pipeline Design

```
┌─────────────────────────────────────────────────────────────┐
│                    EXISTING CI PIPELINE                    │
│  (config.yml) - Working fine, DO NOT TOUCH!              │
│  - Basic setup validation                                  │
│  - Code quality (non-blocking)                            │
│  - Unit tests                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                NEW SECURITY PIPELINE                       │
│  (security-quality.yml) - Completely separate!            │
│  - Security vulnerability scanning                         │
│  - Docker security analysis                               │
│  - Dependency auditing                                    │
│  - Code quality enforcement                               │
│  - Security verification tests                            │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Setup Pre-commit Hooks (Local Development)

```bash
# Run the setup script from project root
./scripts/setup-security-pipeline.sh
```

This will:
- Install pre-commit hooks
- Create all necessary configuration files
- Install security tools
- Set up linting configurations

### 2. Enable CircleCI Security Pipeline

The security pipeline will automatically run on:
- **Pull Requests** to `main` and `develop` branches
- **Daily at 2 AM UTC** (scheduled security scans)
- **Manual triggers** for security audits

## 📋 Pipeline Components

### Security Vulnerability Scanning

| Job | Purpose | Tools | Output |
|-----|---------|-------|---------|
| `security-scan` | Python security analysis | Safety, Bandit, Semgrep | JSON reports |
| `docker-security` | Container vulnerability scan | Trivy | SARIF format |
| `dependency-audit` | Python package vulnerabilities | pip-audit | JSON report |

### Code Quality Enforcement

| Job | Purpose | Tools | Output |
|-----|---------|-------|---------|
| `code-quality` | Code style and quality | Ruff, Black, isort, MyPy | JSON reports |
| `quality-gates` | Summary and reporting | Custom scripts | Markdown report |

### Security Verification

| Job | Purpose | Scripts | Verification |
|-----|---------|---------|--------------|
| `security-verification` | Custom security checks | Flask debug, Host binding, Monitoring | Pass/Fail status |

## 🔧 Configuration Files

### Pre-commit Configuration (`.pre-commit-config.yaml`)

```yaml
# Python code formatting and quality
- repo: https://github.com/psf/black
  rev: 23.12.1
  hooks:
    - id: black
      args: [--line-length=88, --target-version=py312]

# Security checks
- repo: https://github.com/PyCQA/bandit
  rev: 1.7.5
  hooks:
    - id: bandit
      args: [-r, ., -f, json, -o, bandit-report.json]

# Custom security verification hooks
- repo: local
  hooks:
    - id: flask-debug-security-check
      entry: python scripts/security/verify_flask_debug_security.py
```

### Ruff Configuration (`.ruff.toml`)

```toml
# Ruff configuration for SAMO project
target-version = "py312"
line-length = 88
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    # ... more rules
]

# Exclude patterns
exclude = [
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "notebooks",
    "scripts/legacy",
]
```

### MyPy Configuration (`.mypy.ini`)

```ini
[mypy]
python_version = 3.12
warn_return_any = True
disallow_untyped_defs = True
check_untyped_defs = True
ignore_missing_imports = True

# Per-module options
[mypy-src.*]
disallow_untyped_defs = False
```

## 🎯 Workflow Triggers

### Automatic Triggers

1. **Pull Request Workflow** (`security-and-quality-standalone`)
   - Runs on every PR to `main` and `develop`
   - All security and quality jobs execute
   - Results reported in PR comments

2. **Daily Security Scan** (`daily-security-scan`)
   - Runs at 2 AM UTC daily
   - Focuses on critical security checks
   - Monitors for new vulnerabilities

### Manual Triggers

3. **Manual Security Audit** (`manual-security-audit`)
   - Can be triggered manually from CircleCI
   - Useful for security reviews and compliance checks
   - Runs full security suite

## 📊 Artifacts and Reports

### Security Reports

- **Safety Report**: Python dependency vulnerabilities
- **Bandit Report**: Python code security issues
- **Semgrep Report**: Static analysis findings
- **Trivy Report**: Docker image vulnerabilities
- **Pip Audit Report**: Package vulnerability details

### Quality Reports

- **Ruff Report**: Linting and style issues
- **MyPy Report**: Type checking results
- **Quality Summary**: Overall pipeline status

### Accessing Reports

1. **CircleCI Dashboard**: Navigate to your project → Pipelines → Security Pipeline
2. **Artifacts Tab**: Download JSON reports for detailed analysis
3. **Job Logs**: View real-time execution and any failures

## 🚨 Security Alerts and False Positives

### Known False Positives

| Alert Type | Description | Why It's False | Resolution |
|------------|-------------|----------------|------------|
| `generic-api-key` | Python import paths | Import statements, not API keys | Documented in code comments |
| `dangerous-subprocess-use-audit` | Safe Popen calls | Path objects, not user input | Already secure implementation |
| `curl-auth-header` | Documentation examples | Environment variables, not hardcoded | Use `${VARIABLE}` syntax |

### Handling Security Alerts

1. **Review the Alert**: Check if it's a real vulnerability
2. **Document False Positives**: Add comments explaining why it's safe
3. **Fix Real Issues**: Address actual security problems immediately
4. **Update Documentation**: Keep security guides current

## 🛠️ Local Development Workflow

### Pre-commit Hooks

```bash
# Install hooks (one-time setup)
pre-commit install --install-hooks

# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run

# Run specific hook
pre-commit run black
pre-commit run ruff
pre-commit run mypy
```

### Manual Quality Checks

```bash
# Code formatting
black .
isort .

# Linting
ruff check .
ruff format .

# Type checking
mypy .

# Security scanning
safety check
bandit -r .
```

## 🔍 Troubleshooting

### Common Issues

1. **Pre-commit Hooks Fail**
   ```bash
   # Reinstall hooks
   pre-commit uninstall
   pre-commit install --install-hooks
   ```

2. **CircleCI Pipeline Fails**
   - Check job logs for specific error messages
   - Verify all required files exist
   - Ensure dependencies are properly specified

3. **Security Tools Not Found**
   ```bash
   # Reinstall security tools
   pip install safety bandit semgrep pip-audit
   ```

### Performance Optimization

- **Parallel Execution**: Jobs run in parallel where possible
- **Caching**: Pip cache enabled for faster dependency installation
- **Resource Classes**: Medium resource class for security jobs
- **Artifact Storage**: Reports stored for easy access

## 📈 Monitoring and Metrics

### Success Metrics

- **Security Scan Pass Rate**: Target: 100%
- **Code Quality Score**: Target: 95%+
- **Vulnerability Count**: Target: 0 critical/high
- **Pipeline Execution Time**: Target: < 15 minutes

### Alert Thresholds

- **Critical Vulnerabilities**: Immediate notification
- **High Vulnerabilities**: 24-hour response required
- **Medium Vulnerabilities**: 72-hour response required
- **Low Vulnerabilities**: Weekly review

## 🔄 Maintenance and Updates

### Regular Tasks

1. **Weekly**: Review security scan results
2. **Monthly**: Update security tool versions
3. **Quarterly**: Review and update security policies
4. **Annually**: Full security pipeline audit

### Tool Updates

```bash
# Update pre-commit hooks
pre-commit autoupdate

# Update security tools
pip install --upgrade safety bandit semgrep pip-audit

# Update CircleCI orbs (if used)
# Check CircleCI marketplace for updates
```

## 📚 Additional Resources

### Documentation

- [CircleCI Security Best Practices](https://circleci.com/docs/security/)
- [Pre-commit Framework](https://pre-commit.com/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Bandit Security Scanner](https://bandit.readthedocs.io/)

### SAMO Project Specific

- [Security Configuration Guide](../security/)
- [Docker Security Guide](../deployment/)
- [API Security Best Practices](../api/)

## 🎉 Success Formula

```
Secure-by-default patterns + 
Automated verification + 
Comprehensive documentation = 
Production-ready security architecture
```

---

**Remember**: This security pipeline is completely independent of your main CI pipeline. It won't break your existing workflow, but it will significantly improve your security posture and code quality!
