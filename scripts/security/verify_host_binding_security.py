#!/usr/bin/env python3
"""
🔐 Host Binding Security Verification Script
===========================================
Verifies that all host binding security fixes are working correctly.
"""
import sys
import re
from pathlib import Path


def check_hardcoded_bindings(project_root: Path) -> tuple[bool, list]:
    """Check for any remaining hardcoded 0.0.0.0 bindings."""
    issues = []
    python_files = list(project_root.glob("**/*.py"))

    # Patterns that indicate hardcoded binding issues
    problematic_patterns = [
        r"host\s*=\s*['\"]0\.0\.0\.0['\"]",
        r"bind.*['\"]0\.0\.0\.0:",
        r"\.run\(\s*host\s*=\s*['\"]0\.0\.0\.0['\"]"
    ]

    for py_file in python_files:
        try:
            content = py_file.read_text(encoding='utf-8')
            for line_num, line in enumerate(content.splitlines(), 1):
                for pattern in problematic_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Check if this is in our allowlist of acceptable patterns
                        if not is_acceptable_binding(line, py_file):
                            issues.append({
                                'file': str(py_file.relative_to(project_root)),
                                'line': line_num,
                                'content': line.strip(),
                                'issue': 'Hardcoded 0.0.0.0 binding'
                            })
        except Exception as e:
            print(f"Warning: Could not read {py_file}: {e}")

    return len(issues) == 0, issues


def is_acceptable_binding(line: str, _file_path: Path) -> bool:
    """Check if a binding is acceptable (e.g., in comments or documentation)."""
    # Allow in comments
    if line.strip().startswith('#'):
        return True

    # Allow in documentation strings
    if '"""' in line or "'''" in line:
        return True

    # Allow in print statements or logging (informational only)
    if 'print(' in line or 'logger.' in line or '.info(' in line or '.debug(' in line:
        return True

    return False


def check_secure_patterns(project_root: Path) -> tuple[bool, list]:
    """Check that files use secure patterns with environment variables."""
    issues = []

    # Files that should have secure host binding patterns
    key_files = [
        'src/unified_ai_api.py',
        'health_app.py',
        'deployment/cloud-run/secure_api_server.py',
        'deployment/cloud-run/robust_predict.py',
        'deployment/gcp/predict.py'
    ]

    secure_patterns = [
        r"os\.getenv\(['\"]HOST['\"]",
        r"os\.environ\.get\(['\"]HOST['\"]",
        r"host\s*=\s*os\."
    ]

    for key_file in key_files:
        file_path = project_root / key_file
        if file_path.exists():
            try:
                content = file_path.read_text(encoding='utf-8')
                has_secure_pattern = any(
                    re.search(pattern, content, re.IGNORECASE)
                    for pattern in secure_patterns
                )

                if not has_secure_pattern:
                    issues.append({
                        'file': key_file,
                        'issue': 'Missing secure host configuration pattern',
                        'expected': 'host = os.getenv("HOST", "127.0.0.1")'
                    })
            except Exception as e:
                issues.append({
                    'file': key_file,
                    'issue': f'Could not verify: {e}'
                })
        else:
            print(f"Warning: Key file {key_file} not found")

    return len(issues) == 0, issues


def verify_deployment_configs(project_root: Path) -> tuple[bool, list]:
    """Verify deployment configurations are properly set."""
    # Check for Cloud Run deployment files
    cloud_run_configs = [
        'deployment/cloud-run/',
        '.github/workflows/',
        'cloudbuild.yaml',
        'app.yaml'
    ]

    recommendations = []

    for config_path in cloud_run_configs:
        full_path = project_root / config_path
        if full_path.exists():
            recommendations.append({
                'location': config_path,
                'action': 'Ensure HOST=0.0.0.0 environment variable is set '
                          'for Cloud Run deployment',
                'priority': 'High'
            })

    return True, recommendations  # These are recommendations, not issues


def run_security_verification():
    """Run all security verification checks."""
    print("🔐 Host Binding Security Verification")
    print("=" * 50)

    project_root = Path(__file__).parent.parent.parent
    print(f"Project root: {project_root}")
    print()

    all_passed = True

    # Check 1: Hardcoded bindings
    print("1️⃣ Checking for hardcoded 0.0.0.0 bindings...")
    passed, issues = check_hardcoded_bindings(project_root)

    if passed:
        print("   ✅ PASS: No hardcoded 0.0.0.0 bindings found")
    else:
        print("   ❌ FAIL: Found hardcoded bindings:")
        for issue in issues:
            print(f"      {issue['file']}:{issue['line']} - {issue['content']}")
        all_passed = False

    print()

    # Check 2: Secure patterns
    print("2️⃣ Checking for secure host configuration patterns...")
    passed, issues = check_secure_patterns(project_root)

    if passed:
        print("   ✅ PASS: All key files use secure host configuration")
    else:
        print("   ❌ FAIL: Missing secure patterns:")
        for issue in issues:
            print(f"      {issue['file']} - {issue['issue']}")
            if 'expected' in issue:
                print(f"      Expected: {issue['expected']}")
        all_passed = False

    print()

    # Check 3: Deployment recommendations
    print("3️⃣ Deployment configuration recommendations...")
    passed, recommendations = verify_deployment_configs(project_root)

    if recommendations:
        print("   📋 Cloud deployment checklist:")
        for rec in recommendations:
            print(f"      {rec['location']}: {rec['action']}")
    else:
        print("   ℹ️  No specific deployment configs found")

    print()

    # Summary
    print("🎯 Security Verification Summary")
    print("-" * 30)

    if all_passed:
        print("✅ ALL CHECKS PASSED")
        print("🔒 Host binding security vulnerabilities have been resolved")
        print("🚀 Safe to deploy with proper environment configuration")
        return True
    print("❌ SOME CHECKS FAILED")
    print("⚠️  Please address the issues above before deployment")
    return False


if __name__ == "__main__":
    success = run_security_verification()
    sys.exit(0 if success else 1)
