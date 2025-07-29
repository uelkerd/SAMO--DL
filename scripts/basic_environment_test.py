#!/usr/bin/env python3
"""
Basic Environment Test Script
Tests imports one by one to identify issues
"""

import sys
import os

def test_import(module_name, description):
    """Test importing a module and report status."""
    try:
        print(f"🔍 Testing {description}...")
        __import__(module_name)
        print(f"✅ {description} OK")
        return True
    except KeyboardInterrupt:
        print(f"❌ {description} - KeyboardInterrupt")
        return False
    except Exception as e:
        print(f"❌ {description} - Error: {e}")
        return False

def main():
    """Test all critical imports."""
    print("🧪 Basic Environment Test")
    print("=" * 50)
    
    # Test basic Python
    print("🔍 Testing basic Python...")
    print("✅ Basic Python OK")
    
    # Test core modules one by one
    tests = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
    ]
    
    results = []
    for module, description in tests:
        result = test_import(module, description)
        results.append(result)
        
        # Stop if we hit a KeyboardInterrupt
        if not result and "KeyboardInterrupt" in str(sys.exc_info()[1]):
            print(f"\n🚨 STOPPED: {description} caused KeyboardInterrupt")
            break
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY:")
    working = sum(results)
    total = len(results)
    print(f"✅ Working: {working}/{total}")
    print(f"❌ Failed: {total - working}/{total}")
    
    if working == total:
        print("🎉 All tests passed! Environment is working.")
        return True
    else:
        print("⚠️  Some tests failed. Environment has issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 