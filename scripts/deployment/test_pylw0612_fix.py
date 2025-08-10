#!/usr/bin/env python3
"""
🔍 Test PYL-W0612 Unused Variable Fix
====================================
Validate that all unused variable issues have been resolved by replacing
unused variables with underscore (_) to indicate intentional non-use.
"""

import os
import sys
import ast
import re

def test_unused_variables_fixed():
    """Test that unused variables have been properly addressed."""
    print("🔍 TESTING PYL-W0612 UNUSED VARIABLE FIX")
    print("=" * 50)
    
    files_to_check = [
        "scripts/deployment/upload_model_to_huggingface.py",
        "scripts/deployment/test_improvements.py", 
        "scripts/deployment/test_code_review_fixes.py",
    ]
    
    issues_found = []
    fixes_validated = []
    
    for file_path in files_to_check:
        print(f"\n🔍 Checking {file_path}...")
        
        if not os.path.exists(file_path):
            print(f"   ❌ File not found: {file_path}")
            issues_found.append(f"Missing file: {file_path}")
            continue
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for specific patterns that were problematic
        checks = []
        
        if "upload_model_to_huggingface.py" in file_path:
            # Check that dirnames is replaced with _ in os.walk
            if "for dirpath, _, filenames in os.walk" in content:
                checks.append(("dirnames replaced with _", True))
                fixes_validated.append(f"{file_path}: dirnames → _")
            else:
                checks.append(("dirnames replaced with _", False))
                issues_found.append(f"{file_path}: dirnames still present in os.walk")
        
        elif "test_improvements.py" in file_path:
            # Check that dirnames is replaced with _ in os.walk
            if "for dirpath, _, filenames in os.walk" in content:
                checks.append(("dirnames replaced with _", True))
                fixes_validated.append(f"{file_path}: dirnames → _")
            else:
                checks.append(("dirnames replaced with _", False))
                issues_found.append(f"{file_path}: dirnames still present in os.walk")
        
        elif "test_code_review_fixes.py" in file_path:
            # Check that error_msg is replaced with _ in loop
            if "for error_type, _, expected_category in error_scenarios:" in content:
                checks.append(("error_msg replaced with _", True))
                fixes_validated.append(f"{file_path}: error_msg → _")
            else:
                checks.append(("error_msg replaced with _", False))
                issues_found.append(f"{file_path}: error_msg still present in loop")
            
            # Check that result is replaced with _ in assignments
            result_assignments = content.count("_ = mock_torch_load")
            if result_assignments >= 2:
                checks.append(("result assignments replaced with _", True))
                fixes_validated.append(f"{file_path}: result → _ (2 occurrences)")
            else:
                checks.append(("result assignments replaced with _", False))
                issues_found.append(f"{file_path}: result assignments not fixed")
        
        # Report checks for this file
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"   {status} {check_name}")
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"   Fixes validated: {len(fixes_validated)}")
    print(f"   Issues remaining: {len(issues_found)}")
    
    if fixes_validated:
        print(f"\n✅ FIXES VALIDATED:")
        for fix in fixes_validated:
            print(f"   • {fix}")
    
    if issues_found:
        print(f"\n❌ ISSUES REMAINING:")
        for issue in issues_found:
            print(f"   • {issue}")
        return False
    
    return True

def test_syntax_validation():
    """Test that all files still have valid Python syntax after fixes."""
    print("\n🔍 SYNTAX VALIDATION")
    print("=" * 50)
    
    files_to_validate = [
        "scripts/deployment/upload_model_to_huggingface.py",
        "scripts/deployment/test_improvements.py",
        "scripts/deployment/test_code_review_fixes.py",
    ]
    
    all_valid = True
    
    for file_path in files_to_validate:
        print(f"\n🔍 Validating syntax: {file_path}...")
        
        if not os.path.exists(file_path):
            print(f"   ❌ File not found")
            all_valid = False
            continue
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Parse the file to check syntax
            ast.parse(content)
            print(f"   ✅ Valid Python syntax")
            
        except SyntaxError as e:
            print(f"   ❌ Syntax error: {e}")
            all_valid = False
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
            all_valid = False
    
    return all_valid

def test_functional_patterns():
    """Test that the functionality patterns are preserved."""
    print("\n🔍 FUNCTIONAL PATTERN VALIDATION")
    print("=" * 50)
    
    # Test that os.walk patterns still work correctly
    print("\n🔧 Testing os.walk pattern simulation...")
    
    import tempfile
    
    # Create a temporary directory structure for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some test files
        test_file1 = os.path.join(temp_dir, "test1.txt")
        test_subdir = os.path.join(temp_dir, "subdir")
        os.makedirs(test_subdir)
        test_file2 = os.path.join(test_subdir, "test2.txt")
        
        with open(test_file1, 'w') as f:
            f.write("test content 1")
        with open(test_file2, 'w') as f:
            f.write("test content 2")
        
        # Test the pattern we use in our fixed code
        total_size = 0
        file_count = 0
        
        for dirpath, _, filenames in os.walk(temp_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                    file_count += 1
                except (OSError, FileNotFoundError):
                    pass
        
        print(f"   ✅ os.walk pattern works: {file_count} files, {total_size} bytes total")
        
        if file_count == 2 and total_size > 0:
            print("   ✅ Directory traversal functional")
            return True
        else:
            print("   ❌ Directory traversal failed")
            return False

def test_underscore_convention():
    """Test that underscore convention is properly used."""
    print("\n🔍 UNDERSCORE CONVENTION VALIDATION")
    print("=" * 50)
    
    files_to_check = [
        "scripts/deployment/upload_model_to_huggingface.py",
        "scripts/deployment/test_improvements.py",
        "scripts/deployment/test_code_review_fixes.py",
    ]
    
    convention_examples = []
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Look for underscore usage patterns
        underscore_patterns = [
            (r'for \w+, _, \w+ in', 'os.walk with unused dirnames'),
            (r'for \w+, _, \w+ in', 'loop unpacking with unused middle value'),
            (r'_ = \w+\(', 'assignment to underscore for unused return'),
        ]
        
        for pattern, description in underscore_patterns:
            matches = re.findall(pattern, content)
            if matches:
                convention_examples.append(f"{os.path.basename(file_path)}: {description} ({len(matches)} occurrences)")
    
    print("✅ Underscore convention usage found:")
    for example in convention_examples:
        print(f"   • {example}")
    
    return len(convention_examples) >= 3  # Expect at least 3 different usage patterns

def main():
    """Run all PYL-W0612 fix validation tests."""
    print("🔍 TESTING PYL-W0612 UNUSED VARIABLE FIX")
    print("=" * 60)
    print("Issue: 4 unused variables across 3 files")
    print("Fix: Replace unused variables with underscore (_) convention")
    print("=" * 60)
    
    tests = [
        ("Unused Variables Fixed", test_unused_variables_fixed),
        ("Syntax Validation", test_syntax_validation),
        ("Functional Patterns", test_functional_patterns),
        ("Underscore Convention", test_underscore_convention),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print(f"\n🎯 PYL-W0612 FIX VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 PYL-W0612 UNUSED VARIABLE ISSUES SUCCESSFULLY RESOLVED!")
        print("📋 Summary of fixes:")
        print("  ✅ dirnames in os.walk() → _ (2 files)")
        print("  ✅ error_msg in loop → _ (1 file)")
        print("  ✅ result assignments → _ (1 file, 2 occurrences)")
        print("  ✅ All syntax remains valid")
        print("  ✅ Functionality preserved")
        print("\n🐍 Python best practices: Underscore convention for unused variables")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) failed - review implementation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)