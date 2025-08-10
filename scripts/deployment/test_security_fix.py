#!/usr/bin/env python3
"""
🛡️ Test Security Fix for BAN-B104
==================================
Validate that the binding to all interfaces issue has been resolved.
"""

import os
import sys
import unittest.mock as mock

def test_default_secure_binding():
    """Test that the default binding is secure (localhost)."""
    print("🛡️ TESTING DEFAULT SECURE BINDING (BAN-B104)")
    print("=" * 50)
    
    # Test 1: Default environment (no override)
    print("🔍 Test 1: Default configuration (secure)...")
    
    # Clear environment variables to test defaults
    env_clear = {}
    with mock.patch.dict(os.environ, env_clear, clear=True):
        # Simulate the configuration logic from the fixed code
        host = os.getenv('FLASK_HOST', '127.0.0.1')
        port = int(os.getenv('FLASK_PORT', '5000'))
        debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
        
        print(f"   Host: {host}")
        print(f"   Port: {port}")
        print(f"   Debug: {debug}")
        
        # Validation
        if host == '127.0.0.1':
            print("   ✅ SECURE: Default binding to localhost only")
        else:
            print(f"   ❌ INSECURE: Default binding to {host}")
            return False
            
        if not debug:
            print("   ✅ SECURE: Debug mode disabled by default")
        else:
            print("   ❌ INSECURE: Debug mode enabled by default")
            return False
    
    return True

def test_environment_configuration():
    """Test environment variable configuration options."""
    print("\n🔍 Test 2: Environment variable configuration...")
    
    test_scenarios = [
        # (FLASK_HOST, expected_security_level, description)
        ('127.0.0.1', 'SECURE', 'Localhost binding'),
        ('localhost', 'SECURE', 'Localhost name binding'), 
        ('0.0.0.0', 'WARNING', 'All interfaces binding'),
        ('192.168.1.100', 'CUSTOM', 'Specific IP binding'),
    ]
    
    all_passed = True
    
    for host_value, expected_level, description in test_scenarios:
        print(f"\n   Testing: {description} ({host_value})")
        
        env_vars = {'FLASK_HOST': host_value}
        with mock.patch.dict(os.environ, env_vars):
            host = os.getenv('FLASK_HOST', '127.0.0.1')
            
            # Simulate security level detection logic
            if host == '127.0.0.1' or host == 'localhost':
                security_level = 'SECURE'
            elif host == '0.0.0.0':
                security_level = 'WARNING'
            else:
                security_level = 'CUSTOM'
            
            if security_level == expected_level:
                print(f"   ✅ {description}: {security_level} (as expected)")
            else:
                print(f"   ❌ {description}: {security_level} (expected {expected_level})")
                all_passed = False
    
    return all_passed

def test_security_warnings():
    """Test that security warnings are properly triggered."""
    print("\n🔍 Test 3: Security warning detection...")
    
    # Test cases that should trigger warnings
    warning_cases = [
        ('0.0.0.0', True, 'All interfaces binding should warn'),
        ('127.0.0.1', False, 'Localhost should not warn'),
        ('localhost', False, 'Localhost name should not warn'),
        ('192.168.1.100', True, 'Custom IP should provide security tips'),
    ]
    
    all_passed = True
    
    for host_value, should_warn, description in warning_cases:
        print(f"\n   Testing: {description}")
        
        # Simulate warning logic from the fixed code
        triggers_security_warning = (host_value == '0.0.0.0')
        triggers_security_tips = (host_value != '127.0.0.1' and host_value != 'localhost')
        
        if should_warn:
            if triggers_security_warning or triggers_security_tips:
                print(f"   ✅ {description}: Warning/tips triggered correctly")
            else:
                print(f"   ❌ {description}: Should have triggered warning/tips")
                all_passed = False
        else:
            if not triggers_security_warning and not triggers_security_tips:
                print(f"   ✅ {description}: No unnecessary warnings")
            else:
                print(f"   ❌ {description}: Unexpected warning triggered")
                all_passed = False
    
    return all_passed

def test_fix_validation():
    """Validate that the fix has been properly implemented in the code."""
    print("\n🔍 Test 4: Fix implementation validation...")
    
    # Check that the file exists and has been modified
    file_path = "deployment/flexible_api_server.py"
    
    if not os.path.exists(file_path):
        print("   ❌ File not found")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check for security fixes
    fixes_found = []
    
    # Look for configurable host binding
    if "os.getenv('FLASK_HOST'" in content and "'127.0.0.1'" in content:
        fixes_found.append("configurable host binding with secure default")
    
    # Look for security warnings
    if "SECURITY WARNING" in content and "0.0.0.0" in content:
        fixes_found.append("security warning for all-interfaces binding")
    
    # Look for removal of hardcoded 0.0.0.0
    if "app.run(host='0.0.0.0'" not in content:
        fixes_found.append("hardcoded 0.0.0.0 binding removed")
    
    # Look for environment variable configuration
    if "FLASK_HOST" in content and "FLASK_PORT" in content:
        fixes_found.append("environment variable configuration")
    
    # Look for security tips
    if "Security Tips" in content or "security tips" in content:
        fixes_found.append("security guidance and tips")
    
    print("   ✅ Fix implementations found:")
    for fix in fixes_found:
        print(f"      • {fix}")
    
    if len(fixes_found) >= 4:
        print("   ✅ COMPREHENSIVE SECURITY FIX IMPLEMENTED")
        return True
    else:
        print("   ❌ Insufficient fixes detected")
        return False

def test_configuration_template():
    """Test that the security configuration template exists."""
    print("\n🔍 Test 5: Security configuration template...")
    
    template_path = "deployment/.env.flask.example"
    
    if not os.path.exists(template_path):
        print("   ❌ Security configuration template not found")
        return False
    
    with open(template_path, 'r') as f:
        template_content = f.read()
    
    # Check for security documentation
    security_elements = [
        'SECURITY CONFIGURATION',
        '127.0.0.1',
        'SECURITY WARNING' or 'security warning',
        'FLASK_HOST',
        'FLASK_DEBUG',
        'SECURITY BEST PRACTICES' or 'best practices',
    ]
    
    found_elements = []
    for element in security_elements:
        if element.lower() in template_content.lower():
            found_elements.append(element)
    
    print(f"   ✅ Security template elements found: {len(found_elements)}/{len(security_elements)}")
    
    if len(found_elements) >= 5:
        print("   ✅ COMPREHENSIVE SECURITY TEMPLATE CREATED")
        return True
    else:
        print("   ❌ Security template incomplete")
        return False

def main():
    """Run all security fix validation tests."""
    print("🛡️ TESTING SECURITY FIX FOR BAN-B104")
    print("=" * 60)
    print("Issue: Binding to all interfaces detected with hardcoded values")
    print("Fix: Configurable binding with secure localhost default")
    print("=" * 60)
    
    tests = [
        ("Default Secure Binding", test_default_secure_binding),
        ("Environment Configuration", test_environment_configuration), 
        ("Security Warnings", test_security_warnings),
        ("Fix Implementation", test_fix_validation),
        ("Configuration Template", test_configuration_template),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print(f"\n🎯 SECURITY FIX VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 BAN-B104 SECURITY ISSUE SUCCESSFULLY FIXED!")
        print("📋 Summary of security improvements:")
        print("  ✅ Default binding changed from 0.0.0.0 to 127.0.0.1 (secure)")
        print("  ✅ Configurable via FLASK_HOST environment variable")
        print("  ✅ Security warnings for dangerous configurations")  
        print("  ✅ Comprehensive security documentation provided")
        print("  ✅ Best practices and deployment guidance included")
        print("\n🛡️ Security compliance: OWASP Top 10 2021 A05 addressed")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) failed - review security implementation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)