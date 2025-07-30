import sys

#!/usr/bin/env python3
"""
Basic Environment Test Script
Tests imports one by one to identify issues
"""



def test_import(module_name, description):
    """Test importing a module and report status."""
    try:
        print("🔍 Testing {description}...")
        __import__(module_name)
        print("✅ {description} OK")
        return True
    except KeyboardInterrupt:
        print("❌ {description} - KeyboardInterrupt")
        return False
    except Exception as e:
        print("❌ {description} - Error: {e}")
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
            print("\n🚨 STOPPED: {description} caused KeyboardInterrupt")
            break

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY:")
    working = sum(results)
    total = len(results)
    print("✅ Working: {working}/{total}")
    print("❌ Failed: {total - working}/{total}")

    if working == total:
        print("🎉 All tests passed! Environment is working.")
        return True
    else:
        print("⚠️  Some tests failed. Environment has issues.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
