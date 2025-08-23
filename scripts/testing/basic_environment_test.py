        # Stop if we hit a KeyboardInterrupt
    # Summary
    # Test basic Python
    # Test core modules one by one
#!/usr/bin/env python3
import logging
import sys

"""
Basic Environment Test Script
Tests imports one by one to identify issues
"""



def test_import(module_name, description):
    """Test importing a module and report status."""
    try:
        logging.info("🔍 Testing {description}...")
        __import__(module_name)
        logging.info("✅ {description} OK")
        return True
    except KeyboardInterrupt:
        logging.info("❌ {description} - KeyboardInterrupt")
        return False
    except Exception as e:
        logging.info(f"❌ {description} - Error: {e}")
        return False


def main():
    """Test all critical imports."""
    logging.info("🧪 Basic Environment Test")
    logging.info("=" * 50)

    logging.info("🔍 Testing basic Python...")
    logging.info("✅ Basic Python OK")

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

        if not result and "KeyboardInterrupt" in str(sys.exc_info()[1]):
            logging.info("\n🚨 STOPPED: {description} caused KeyboardInterrupt")
            break

    logging.info("\n{"=' * 50}")
    logging.info("📊 TEST SUMMARY:")
    working = sum(results)
    total = len(results)
    logging.info("✅ Working: {working}/{total}")
    logging.info("❌ Failed: {total - working}/{total}")

    if working == total:
        logging.info("🎉 All tests passed! Environment is working.")
        return True
    else:
        logging.info("⚠️  Some tests failed. Environment has issues.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
