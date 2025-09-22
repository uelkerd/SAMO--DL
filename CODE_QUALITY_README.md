# SAMO-DL Code Quality Standards

## 🎯 Mission: Prevent Monster PRs Forever

This document outlines the **strict rules** and **automated enforcement** designed to prevent the creation of massive, unfocused pull requests that slow down development and make code reviews impossible.

## 🚫 THE PROBLEM WE SOLVE

**Monster PRs** are pull requests that:
- Change 100+ files
- Add 1000+ lines of code
- Mix multiple concerns (API + tests + docs + infrastructure)
- Take weeks to review
- Cause merge conflicts
- Block progress

## ✅ THE SOLUTION: Micro-PRs Only

### **Hard Limits (Automated Enforcement)**
```yaml
# GitHub Actions - PR Size Guard
max_files_changed: 50     # HARD STOP at 50 files
max_lines_changed: 1500   # HARD STOP at 1500 lines
max_commits_per_pr: 5     # HARD STOP at 5 commits
branch_lifetime: 72h      # FORCE merge or close after 72h (extensions available for complex changes)
```

### **Single Purpose Rule**
**EVERY PR MUST HAVE EXACTLY ONE SENTENCE DESCRIBING ITS PURPOSE**
- ✅ "Add user authentication system"
- ✅ "Fix memory leak in model loading"
- ❌ "Improve model architecture and fix bugs" *(TWO THINGS!)*
- ❌ "Refactor training pipeline" *(TOO VAGUE!)*

## 🛠️ AUTOMATED ENFORCEMENT

### **1. Pre-Commit Hook**
The `pre-commit` framework, configured in `.pre-commit-config.yaml`, automatically runs scripts that:
- Validates branch naming (`feat/add-auth`, `fix/memory-leak`)
- Checks commit message format (`feat: add user auth`)
- Prevents commits with mixed concerns
- Runs before every commit

### **2. PR Scope Checker**
```bash
python scripts/check_pr_scope.py --strict
```
> **Note:** The PR scope checker script is available in the repository and actively maintained.
Validates:
- File count ≤ 50
- Line changes ≤ 1500
- Single purpose (no mixing concerns)
- Branch naming compliance

### **3. CI Pipeline Checks**
GitHub Actions automatically:
- Runs scope validation on PR creation
- Fails builds that exceed limits
- Prevents merging of out-of-scope PRs

## 📋 DEVELOPMENT WORKFLOW

### **Before Creating a Branch**
```bash
# Answer these questions:
1. Can I describe this in ONE sentence?
2. Will this affect < 50 files?
3. Can I complete this in < 4 hours?
4. Is this EXACTLY ONE concern?
5. Am I mixing API + tests + docs?

# If ANY answer is NO: Split into separate PRs
```

### **Branch Naming Convention**
```
feat/short-description      # New features
fix/short-description       # Bug fixes
chore/short-description     # Build/tooling changes
refactor/short-description  # Code restructuring
docs/short-description      # Documentation
test/short-description      # Test additions
```

**Examples:**
- ✅ `feat/add-user-auth`
- ✅ `fix/validate-input`
- ✅ `chore/update-deps`
- ✅ `refactor/simplify-logic`
- ❌ `feat/add-auth-and-fix-bugs` *(multiple concerns)*
- ❌ `fix-stuff` *(too vague)*

### **Commit Message Format**
```
<type>(<scope>): <subject>

<body - optional>
```
**Examples:**
```
feat: add JWT token authentication
fix: resolve memory leak in model loading
chore: update Python dependencies
refactor: simplify rate limiter logic
```

## 🏗️ CODE QUALITY TOOLS

### **Automated Tools**
- **Ruff**: Ultra-fast code formatting, linting, and import sorting (replaces Black, isort, flake8)
- **pylint**: Advanced code analysis
- **mypy**: Type checking
- **bandit**: Security scanning
- **safety**: Dependency vulnerability checks

### **Pre-commit Hooks**
Run automatically on commit:
```bash
pre-commit install  # Install hooks
pre-commit run --all-files  # Run on all files
```

### **Development Commands**
```bash
make format      # Format code
make lint        # Run linters
make test        # Run tests
make quality-check  # Run all quality checks
```

## 🚨 EMERGENCY OVERRIDES

**Only for critical production issues:**
```yaml
override_label: "EMERGENCY-OVERRIDE"
required_approvers: 2
max_override_per_week: 1
auto_close_after: 8h
```

## 📊 SUCCESS METRICS

**Weekly Tracking:**
- Average PR size: < 15 files
- PR lifetime: < 24 hours
- Number of scope violations: 0
- Merge conflicts: < 1 per week

**Red Flags (Auto-alert):**
- Any PR > 50 files
- Any branch > 48 hours old
- Any PR title with "and", "also", "plus"
- Any PR summary > 2 sentences (detailed descriptions in body are encouraged)

## 🎯 WHY THIS WORKS

### **Psychological Benefits**
- **Small wins**: Frequent merges build momentum
- **Fast feedback**: Quick reviews = faster iteration
- **Reduced risk**: Smaller changes = easier rollback
- **Team satisfaction**: Actually shipping features

### **Technical Benefits**
- **Fewer merge conflicts**: Smaller, focused changes
- **Easier reviews**: 50 files vs 100+ files
- **Better testing**: Isolated changes = targeted tests
- **Faster CI/CD**: Smaller PRs = faster pipelines

## 🚀 IMPLEMENTATION

### **Immediate Actions**
1. **Install pre-commit hooks**: `pre-commit install`
2. **Set commit template**: `git config commit.template .gitmessage.txt`
   > **Note:** The `.gitmessage.txt` file is included in the repository with a comprehensive commit message template following Conventional Commits standards.
3. **Run scope checker**: `python scripts/check_pr_scope.py`
4. **Review existing PRs**: Close any that violate rules

### **Team Adoption**
1. **Training session**: Walk through the rules
2. **Documentation**: Share this guide
3. **Examples**: Show good vs bad PR examples
4. **Celebrate**: Recognize teams following the rules

## 📞 SUPPORT

**Questions?** Ask in the development channel.

**Found a violation?** Use the PR comment template:
```
🚨 **SCOPE VIOLATION DETECTED**
This PR exceeds our size limits:
- Files: [count]/50 max
- Lines: [count]/1500 max
- Multiple concerns: [list them]

Please split into focused micro-PRs.
```

---

## 🎉 CONCLUSION

**Small PRs = Fast reviews = Quick merges = Happy developers = Successful project**

**NO EXCEPTIONS. NO EXCUSES. NO "JUST THIS ONCE".**

Welcome to the era of **productive, focused development!** 🚀
