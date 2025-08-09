# 🔐 Security Disclaimer

## Documentation Contains Example Values Only

This documentation contains **placeholder values and example code** for demonstration purposes. These are **NOT real secrets or credentials**.

### ⚠️ Important Security Notes

1. **All placeholder values** starting with `REPLACE_WITH_` are examples only
2. **Never use placeholder values** in production environments
3. **Generate secure credentials** using appropriate tools for each service
4. **Never commit real secrets** to version control

### 🛡️ Secure Credential Generation

#### JWT Secrets
```bash
# Generate secure JWT secret (minimum 32 characters)
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

#### API Keys
```bash
# Generate secure API key
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

#### Encryption Keys
```bash
# Generate Fernet encryption key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### 📋 Common Placeholder Patterns

| Pattern | Description | Action Required |
|---------|-------------|-----------------|
| `REPLACE_WITH_*` | Placeholder values | Replace with actual secure values |
| `your-*-here` | Legacy placeholders | Replace with actual secure values |
| `example-*` | Example configurations | Replace with actual secure values |

### 🔍 Security Scanner Notes

If you're seeing security warnings about "hardcoded secrets" in this documentation:

- These are **false positives** - documentation examples only
- No real credentials are exposed
- All values are clearly marked as placeholders
- Use `skipcq: SCT-A000` pragma if needed to silence warnings

### 🚀 Production Deployment

For production deployment:

1. **Generate all credentials** using secure methods
2. **Store in environment variables** or secret management systems
3. **Never commit** real credentials to git
4. **Rotate credentials** regularly
5. **Use least-privilege access** principles

---

**Remember**: Documentation security is important, but these are intentionally obvious placeholder values designed to prevent accidental use in production.