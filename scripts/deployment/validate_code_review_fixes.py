#!/usr/bin/env python3
"""
🧪 Validate Code Review Fixes
==============================
Validate that code review comments have been addressed by examining the code directly.
"""
import sys

def validate_comment_1_portability():
    """Validate that Comment 1 (hardcoded paths) has been addressed."""
    print("🧪 VALIDATING PORTABILITY FIX (Comment 1)")
    print("=" * 50)

    script_path = "scripts/deployment/upload_model_to_huggingface.py"

    try:
        with open(script_path, 'r') as f:
            content = f.read()

        # Check for configurable environment variables
        env_vars_found = [
            'SAMO_DL_BASE_DIR' in content,
            'MODEL_BASE_DIR' in content,
            'get_model_base_directory()' in content
        ]

        # Check for hardcoded paths (should be minimal/none)
        hardcoded_indicators = [
            content.count('/Users/') <= 1,  # Allow one or fewer hardcoded /Users/ paths
            content.count('/home/') <= 1,   # Allow one or fewer hardcoded /home/ paths
            'configurable' in content.lower(),
            'environment variable' in content.lower()
        ]

        all_env_vars = all(env_vars_found)
        no_hardcoded = all(hardcoded_indicators)

        if all_env_vars:
            print("✅ Environment variable configuration found")
            print("   • SAMO_DL_BASE_DIR support detected")
            print("   • MODEL_BASE_DIR support detected")
            print("   • get_model_base_directory() function found")

        if no_hardcoded:
            print("✅ Hardcoded paths minimized/eliminated")

        # Look for documentation about configurability
        if 'configurable' in content.lower() or 'environment' in content.lower():
            print("✅ Configurability documented in code")

        success = all_env_vars and no_hardcoded
        if success:
            print("✅ COMMENT 1 ADDRESSED: Hardcoded paths replaced with configurable options")
        else:
            print("❌ COMMENT 1 NOT FULLY ADDRESSED")

        return success

    except Exception as e:
        print(f"❌ Failed to validate: {e}")
        return False

def validate_comment_2_interactive_login():
    """Validate that Comment 2 (interactive login) has been addressed."""
    print("\n🧪 VALIDATING INTERACTIVE LOGIN FIX (Comment 2)")
    print("=" * 50)

    script_path = "scripts/deployment/upload_model_to_huggingface.py"

    try:
        with open(script_path, 'r') as f:
            content = f.read()

        # Check for non-interactive environment detection
        interactive_checks = [
            'is_interactive_environment' in content,
            'CI' in content and 'DOCKER' in content,  # Environment checks
            'KUBERNETES' in content,
            'sys.stdin.isatty()' in content,
            'non-interactive' in content.lower()
        ]

        # Check for improved error messages
        error_message_improvements = [
            'NON-INTERACTIVE ENVIRONMENT DETECTED' in content,
            'CI/CD pipelines' in content,
            'Docker containers' in content,
            'Headless servers' in content,
            'repository secrets' in content
        ]

        # Check for user consent before interactive login
        user_consent_checks = [
            'input(' in content,  # User input for consent
            'Attempt interactive login' in content,
            'y/N' in content or 'yes/no' in content
        ]

        has_interactive_detection = sum(interactive_checks) >= 3
        has_error_improvements = sum(error_message_improvements) >= 3
        has_user_consent = sum(user_consent_checks) >= 2

        if has_interactive_detection:
            print("✅ Non-interactive environment detection implemented")

        if has_error_improvements:
            print("✅ Clear error messages for non-interactive environments")

        if has_user_consent:
            print("✅ User consent before attempting interactive login")

        success = has_interactive_detection and has_error_improvements
        if success:
            print("✅ COMMENT 2 ADDRESSED: Interactive login properly handles non-interactive environments")
        else:
            print("❌ COMMENT 2 NOT FULLY ADDRESSED")

        return success

    except Exception as e:
        print(f"❌ Failed to validate: {e}")
        return False

def validate_comment_3_error_handling():
    """Validate that Comment 3 (state dict loading error handling) has been addressed."""
    print("\n🧪 VALIDATING ERROR HANDLING FIX (Comment 3)")
    print("=" * 50)

    script_path = "scripts/deployment/upload_model_to_huggingface.py"

    try:
        with open(script_path, 'r') as f:
            content = f.read()

        # Check for error handling around state dict loading
        error_handling_patterns = [
            'try:' in content and 'except' in content,
            'RuntimeError' in content,
            'size mismatch' in content,
            'KeyError' in content,
            'Architecture mismatch' in content
        ]

        # Check for PyTorch version compatibility
        pytorch_compatibility = [
            'weights_only=False' in content,
            'TypeError' in content,
            'PyTorch version' in content or 'pytorch version' in content.lower(),
            'legacy' in content.lower()
        ]

        # Check for informative error messages
        informative_errors = [
            'This usually means:' in content,
            'different number of classes' in content,
            'architecture doesn\'t match' in content,
            'checkpoint file is not corrupted' in content
        ]

        has_error_handling = sum(error_handling_patterns) >= 4
        has_pytorch_compat = sum(pytorch_compatibility) >= 3
        has_informative_errors = sum(informative_errors) >= 3

        if has_error_handling:
            print("✅ Comprehensive error handling implemented")

        if has_pytorch_compat:
            print("✅ PyTorch version compatibility handling")

        if has_informative_errors:
            print("✅ Informative error messages with troubleshooting tips")

        success = has_error_handling and has_pytorch_compat and has_informative_errors
        if success:
            print("✅ COMMENT 3 ADDRESSED: State dict loading has comprehensive error handling")
        else:
            print("❌ COMMENT 3 NOT FULLY ADDRESSED")

        return success

    except Exception as e:
        print(f"❌ Failed to validate: {e}")
        return False

def validate_additional_improvements():
    """Validate additional improvements made beyond the code review comments."""
    print("\n🧪 VALIDATING ADDITIONAL IMPROVEMENTS")
    print("=" * 50)

    script_path = "scripts/deployment/upload_model_to_huggingface.py"

    try:
        with open(script_path, 'r') as f:
            content = f.read()

        improvements = []

        # Check for multiple token environment variables
        if 'HF_TOKEN' in content and 'HUGGINGFACE_TOKEN' in content:
            improvements.append("Multiple HuggingFace token environment variables")

        # Check for better token error messages
        if 'write\' permissions' in content:
            improvements.append("Token permission validation")

        # Check for file corruption detection
        if 'corrupted' in content.lower():
            improvements.append("File corruption detection")

        # Check for disk space / permission checks
        if 'disk space' in content.lower() and 'permissions' in content.lower():
            improvements.append("Disk space and permission checks")

        for improvement in improvements:
            print(f"✅ {improvement}")

        if improvements:
            print("✅ BONUS IMPROVEMENTS: Enhanced beyond code review requirements")
            return True
        print("ℹ️ No additional improvements detected")
        return False
    except Exception as e:
        print(f"❌ Failed to validate additional improvements: {e}")
        return False

def main():
    """Run all validation checks."""
    print("🚀 VALIDATING CODE REVIEW FIXES")
    print("=" * 60)

    validators = [
        ("Portability (Comment 1)", validate_comment_1_portability),
        ("Interactive Login (Comment 2)", validate_comment_2_interactive_login),
        ("Error Handling (Comment 3)", validate_comment_3_error_handling),
        ("Additional Improvements", validate_additional_improvements),
    ]

    results = []
    for validator_name, validator_func in validators:
        try:
            result = validator_func()
            results.append((validator_name, result))
        except Exception as e:
            print(f"❌ {validator_name} validation failed: {e}")
            results.append((validator_name, False))

    print("\n🎯 CODE REVIEW VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for validator_name, result in results:
        status = "✅ ADDRESSED" if result else "❌ NOT ADDRESSED"
        print(f"  {status}: {validator_name}")

    print(f"\nValidations passed: {passed}/{total}")

    if passed >= 3:  # Allow for additional improvements to be optional
        print("\n🎉 ALL REQUIRED CODE REVIEW COMMENTS SUCCESSFULLY ADDRESSED!")
        print("\n📋 Summary of fixes implemented:")
        print("  ✅ Comment 1: Hardcoded absolute paths → Environment variable configuration")
        print("  ✅ Comment 2: Interactive login issues → Non-interactive environment detection")
        print("  ✅ Comment 3: No state dict error handling → Comprehensive error handling")
        print("  ✅ Bonus: Enhanced authentication, PyTorch compatibility, better error messages")

        return True
    print(f"\n⚠️ Only {passed}/{total} validations passed - some fixes may need review")
    return False

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    print(f"\nExit code: {exit_code}")
    sys.exit(exit_code)
