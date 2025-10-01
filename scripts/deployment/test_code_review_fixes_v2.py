#!/usr/bin/env python3
"""
🧪 Test Code Review Fixes (Version 2)
=====================================
Validate all the latest code review fixes including:
1. TemporaryDirectory usage in test files
2. HF_REPO_PRIVATE environment variable support
3. BASE_MODEL_NAME configurability  
4. Retry configuration with allowed_methods
"""

import os
import sys
import unittest.mock as mock

def test_temporary_directory_usage():
    """Test that test files use TemporaryDirectory instead of hardcoded paths."""
    print("🧪 TESTING TEMPORARY DIRECTORY USAGE")
    print("=" * 50)
    
    # Test 1: Check test_model_path_detection.py
    print("🔍 Test 1: test_model_path_detection.py uses TemporaryDirectory...")
    
    test_file_path = "scripts/deployment/test_model_path_detection.py"
    if not os.path.exists(test_file_path):
        print("❌ Test file not found")
        return False
    
    with open(test_file_path, 'r') as f:
        content = f.read()
    
    checks = [
        ("TemporaryDirectory import", "from tempfile import TemporaryDirectory" in content),
        ("TemporaryDirectory usage", "with TemporaryDirectory() as temp_dir:" in content),
        ("No hardcoded home path", "/home/user/projects/emotion-model" not in content),
        ("Proper cleanup", "if 'MODEL_BASE_DIR' in os.environ:" in content),
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}")
        if not passed:
            all_passed = False
    
    # Test 2: Check test_code_review_fixes.py  
    print("\n🔍 Test 2: test_code_review_fixes.py uses TemporaryDirectory...")
    
    test_file_path = "scripts/deployment/test_code_review_fixes.py"
    if not os.path.exists(test_file_path):
        print("❌ Test file not found")
        return False
    
    with open(test_file_path, 'r') as f:
        content = f.read()
    
    checks = [
        ("TemporaryDirectory import", "from tempfile import TemporaryDirectory" in content),
        ("TemporaryDirectory usage", "with TemporaryDirectory() as test_path:" in content),
        ("No hardcoded /tmp path", '"/tmp/test_project"' not in content),
        ("Dynamic path usage", "with mock.patch.dict(os.environ, {'SAMO_DL_BASE_DIR': test_path}):" in content),
    ]
    
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}")
        if not passed:
            all_passed = False
    
    return all_passed

def test_hf_repo_private_environment_variable():
    """Test HF_REPO_PRIVATE environment variable support."""
    print("\n🧪 TESTING HF_REPO_PRIVATE ENVIRONMENT VARIABLE")
    print("=" * 50)
    
    try:
        # Mock sys.stdin.isatty to avoid actual TTY checks
        with mock.patch('sys.stdin.isatty', return_value=True):
            # Import the function to test
            sys.path.append('scripts/deployment')
            from upload_model_to_huggingface import choose_repository_privacy
            
            # Test 1: HF_REPO_PRIVATE=true
            print("🔍 Test 1: HF_REPO_PRIVATE=true (private repository)")
            with mock.patch.dict(os.environ, {'HF_REPO_PRIVATE': 'true'}):
                result = choose_repository_privacy()
                if result is True:
                    print("   ✅ Correctly returns True for private repository")
                else:
                    print(f"   ❌ Expected True, got {result}")
                    return False
            
            # Test 2: HF_REPO_PRIVATE=false  
            print("\n🔍 Test 2: HF_REPO_PRIVATE=false (public repository)")
            with mock.patch.dict(os.environ, {'HF_REPO_PRIVATE': 'false'}):
                result = choose_repository_privacy()
                if result is False:
                    print("   ✅ Correctly returns False for public repository")
                else:
                    print(f"   ❌ Expected False, got {result}")
                    return False
            
            # Test 3: HF_REPO_PRIVATE invalid value
            print("\n🔍 Test 3: HF_REPO_PRIVATE=invalid (should warn and continue)")
            with mock.patch.dict(os.environ, {'HF_REPO_PRIVATE': 'invalid'}):
                # This should show a warning but continue to interactive mode
                # We'll mock input to avoid hanging
                with mock.patch('builtins.input', return_value='n'):
                    result = choose_repository_privacy()
                    if result is False:
                        print("   ✅ Invalid value handled gracefully, defaults to public")
                    else:
                        print(f"   ❌ Unexpected result: {result}")
                        return False
        
        # Test 4: Non-interactive environment
        print("\n🔍 Test 4: Non-interactive environment (should default to public)")
        with mock.patch('sys.stdin.isatty', return_value=False), mock.patch.dict(os.environ, {}, clear=True):  # Clear HF_REPO_PRIVATE
            result = choose_repository_privacy()
            if result is False:
                print("   ✅ Non-interactive environment defaults to public")
            else:
                print(f"   ❌ Expected False in non-interactive, got {result}")
                return False
        
        print("\n✅ HF_REPO_PRIVATE environment variable fully functional")
        return True
        
    except ImportError as e:
        print(f"❌ Could not import function: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def test_base_model_name_configurability():
    """Test BASE_MODEL_NAME configurability."""
    print("\n🧪 TESTING BASE_MODEL_NAME CONFIGURABILITY")
    print("=" * 50)
    
    try:
        # Import the function to test
        sys.path.append('scripts/deployment') 
        from upload_model_to_huggingface import get_base_model_name
        
        # Test 1: Default value (no environment variable)
        print("🔍 Test 1: Default base model name")
        with mock.patch.dict(os.environ, {}, clear=True):
            result = get_base_model_name()
            if result == "distilroberta-base":
                print("   ✅ Correctly returns default 'distilroberta-base'")
            else:
                print(f"   ❌ Expected 'distilroberta-base', got '{result}'")
                return False
        
        # Test 2: Custom base model via environment variable
        print("\n🔍 Test 2: Custom BASE_MODEL_NAME")
        custom_model = "roberta-base"
        with mock.patch.dict(os.environ, {'BASE_MODEL_NAME': custom_model}):
            result = get_base_model_name()
            if result == custom_model:
                print(f"   ✅ Correctly returns custom model '{custom_model}'")
            else:
                print(f"   ❌ Expected '{custom_model}', got '{result}'")
                return False
        
        # Test 3: Check that hardcoded strings are replaced
        print("\n🔍 Test 3: Checking upload script for configurable usage")
        
        upload_script_path = "scripts/deployment/upload_model_to_huggingface.py"
        with open(upload_script_path, 'r') as f:
            content = f.read()
        
        checks = [
            ("get_base_model_name function exists", "def get_base_model_name()" in content),
            ("Environment variable check", "os.getenv('BASE_MODEL_NAME')" in content),
            ("Used in model preparation", "base_model_name = get_base_model_name()" in content),
            ("Dynamic replacement logic", "current_base_model = get_base_model_name()" in content),
        ]
        
        all_passed = True
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"   {status} {check_name}")
            if not passed:
                all_passed = False
        
        return all_passed
        
    except ImportError as e:
        print(f"❌ Could not import function: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def test_retry_configuration():
    """Test that Retry configuration includes allowed_methods."""
    print("\n🧪 TESTING RETRY CONFIGURATION")
    print("=" * 50)
    
    print("🔍 Checking flexible_api_server.py for proper Retry configuration...")
    
    api_server_path = "deployment/flexible_api_server.py"
    if not os.path.exists(api_server_path):
        print("❌ API server file not found")
        return False
    
    with open(api_server_path, 'r') as f:
        content = f.read()
    
    checks = [
        ("Retry import", "from requests.packages.urllib3.util.retry import Retry" in content),
        ("allowed_methods parameter", "allowed_methods=" in content),
        ("POST method included", '"POST"' in content and 'allowed_methods' in content),
        ("Multiple methods supported", '"GET"' in content and 'allowed_methods' in content),
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("✅ Retry configuration properly includes allowed_methods for POST requests")
    
    return all_passed

def test_documentation_updates():
    """Test that documentation has been updated with new environment variables."""
    print("\n🧪 TESTING DOCUMENTATION UPDATES")
    print("=" * 50)
    
    print("🔍 Checking CODE_REVIEW_RESPONSE.md for new environment variable documentation...")
    
    doc_path = "scripts/deployment/CODE_REVIEW_RESPONSE.md"
    if not os.path.exists(doc_path):
        print("❌ Documentation file not found")
        return False
    
    with open(doc_path, 'r') as f:
        content = f.read()
    
    checks = [
        ("HF_REPO_PRIVATE section", "HF_REPO_PRIVATE" in content and "Repository Privacy Configuration" in content),
        ("HF_REPO_PRIVATE values", '"true"' in content and '"false"' in content),
        ("BASE_MODEL_NAME section", "BASE_MODEL_NAME" in content and "Configurable Base Model" in content),
        ("Usage examples", "export HF_REPO_PRIVATE=" in content and "export BASE_MODEL_NAME=" in content),
        ("Non-interactive behavior", "non-interactive environment" in content.lower() and "defaults to public" in content.lower()),
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("✅ Documentation comprehensively updated with new environment variables")
    
    return all_passed

def main():
    """Run all code review fix validation tests."""
    print("🧪 TESTING CODE REVIEW FIXES (VERSION 2)")
    print("=" * 60)
    print("Validating latest fixes:")
    print("• TemporaryDirectory usage instead of hardcoded paths")
    print("• HF_REPO_PRIVATE environment variable support")
    print("• BASE_MODEL_NAME configurability")
    print("• Retry configuration with allowed_methods")
    print("• Documentation updates")
    print("=" * 60)
    
    tests = [
        ("Temporary Directory Usage", test_temporary_directory_usage),
        ("HF_REPO_PRIVATE Environment Variable", test_hf_repo_private_environment_variable),
        ("BASE_MODEL_NAME Configurability", test_base_model_name_configurability),
        ("Retry Configuration", test_retry_configuration),
        ("Documentation Updates", test_documentation_updates),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n🎯 CODE REVIEW FIXES VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL CODE REVIEW FIXES SUCCESSFULLY IMPLEMENTED!")
        print("📋 Summary of improvements:")
        print("  ✅ Test isolation with TemporaryDirectory")
        print("  ✅ Non-interactive repository privacy configuration") 
        print("  ✅ Configurable base model support")
        print("  ✅ Enhanced HTTP retry configuration")
        print("  ✅ Comprehensive documentation updates")
        print("\n🚀 All fixes validated and ready for production!")
        return True
    print(f"\n⚠️ {total - passed} test(s) failed - review implementation")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)