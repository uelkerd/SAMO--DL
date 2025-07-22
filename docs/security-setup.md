# SAMO Deep Learning - Security Setup Guide

## ✅ Database Credentials Security Remediation (COMPLETED)

### What Was Done:
1. **Changed leaked password** for `samouser` account
2. **Created new secure user**: `samo_secure_1753200376`
3. **Verified PostgreSQL connection** using correct superuser (`minervae`)

### ⚠️ Important Security Notes:

The original credentials were **publicly exposed** in git history:
- ❌ Username: `samouser`
- ❌ Password: `samopassword` (now changed)
- ❌ Database: `samodb`

## 🔐 Current Secure Database Configuration

### PostgreSQL Connection Info:
- **Host**: `localhost:5432`
- **Database**: `samodb`
- **Secure User**: `samo_secure_1753200376` (with random password)
- **Admin User**: `minervae` (your macOS username)

### How to Connect:
```bash
# As admin (for maintenance):
psql -U minervae -d postgres

# As secure user (for application):
psql -U samo_secure_1753200376 -d samodb
```

## 🔧 Environment Configuration Template

Create a `.env` file (NEVER commit this!) with:

```bash
# Secure database connection
DATABASE_URL="postgresql://samo_secure_1753200376:SECURE_PASSWORD_HERE@localhost:5432/samodb?schema=public"

# AI/ML Configuration
OPENAI_API_KEY=your_openai_key_here
HF_TOKEN=your_huggingface_token_here
MODEL_SERVER_PORT=8000

# Security
JWT_SECRET=your_jwt_secret_here
LOG_LEVEL=info
```

## 🚨 Next Steps Required:

### 1. Update Your Application Configuration
- [ ] Update your Prisma database connection
- [ ] Test database connectivity with new credentials
- [ ] Update any deployment configurations

### 2. Rotate Any Other Potentially Exposed Secrets
- [ ] Generate new JWT secrets
- [ ] Rotate any API keys that were in the leaked .env
- [ ] Update production database credentials if applicable

### 3. Git History Considerations
⚠️ **The leaked credentials still exist in git history!**
- Consider using `git filter-branch` or BFG Repo-Cleaner if this is critical
- Monitor for unauthorized access using the old credentials
- Consider this when deploying to production

## 📋 Security Checklist Going Forward:

- [x] `.gitignore` configured to ignore `.env` files
- [x] Git LFS configured for large files
- [x] Database credentials changed and secured
- [ ] Environment variable validation in application code
- [ ] Regular security audits scheduled
- [ ] Monitoring setup for unusual database access

## 🔍 PostgreSQL Management Commands:

```bash
# List all users
psql -U minervae -d postgres -c "\du"

# List all databases
psql -U minervae -d postgres -c "\l"

# Create new user (if needed)
psql -U minervae -d postgres -c "CREATE USER newuser WITH PASSWORD 'securepass' CREATEDB;"

# Change user password
psql -U minervae -d postgres -c "ALTER USER username WITH PASSWORD 'newpassword';"

# Drop user (if needed)
psql -U minervae -d postgres -c "DROP USER username;"
```

Remember: Always use environment variables for sensitive configuration!
