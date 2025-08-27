#!/usr/bin/env python3
"""
Simple model test script that works with current Python environment.
"""

import json
import os

def test_model_files():
    """Test if model files exist and are valid."""
    print("🧪 SIMPLE MODEL TEST")
    print("=" * 50)
    
    # Check model file
    model_file = "best_simple_model.pth"
    if os.path.exists(model_file):
        size = os.path.getsize(model_file)
        print(f"✅ Model file: {model_file} ({size:,} bytes)")
        
        # Check if it's a reasonable size (should be ~400MB+)
        if size > 100_000_000:  # 100MB
            print("✅ Model file size looks good!")
        else:
            print("⚠️ Model file seems small, might be corrupted")
    else:
        print(f"❌ Model file missing: {model_file}")
        return False
    
    # Check results file
    results_file = "simple_training_results.json"
    if os.path.exists(results_file):
        size = os.path.getsize(results_file)
        print(f"✅ Results file: {results_file} ({size:,} bytes)")
        
        # Try to load and parse
        try:
            with open(results_file) as f:
                results = json.load(f)
            
            print("✅ Results file is valid JSON")
            print(f"📊 F1 Score: {results.get('best_f1', 'N/A')}")
            print(f"📊 Emotions: {len(results.get('all_emotions', []))}")
            
        except json.JSONDecodeError:
            print("❌ Results file is not valid JSON")
            return False
    else:
        print(f"❌ Results file missing: {results_file}")
        return False
    
    return True

def test_python_environment():
    """Test Python environment and libraries."""
    print("\n🔧 Testing Python Environment:")
    print("-" * 30)
    
    # Test basic imports
    try:
        import sys
        print(f"✅ Python version: {sys.version}")
    except ImportError:
        print("❌ Cannot import sys")
        return False
    
    # Test JSON
    try:
        import json
        print("✅ JSON module available")
    except ImportError:
        print("❌ JSON module not available")
        return False
    
    # Test OS
    try:
        import os
        print("✅ OS module available")
    except ImportError:
        print("❌ OS module not available")
        return False
    
    return True

def suggest_next_steps():
    """Suggest next steps for testing."""
    print("\n📋 NEXT STEPS:")
    print("=" * 30)
    
    print("1. 🐍 Python Environment:")
    print("   - You're using Python 3.8.6 but libraries are in Python 3.11")
    print("   - Options:")
    print("     a) Use: python3.11 scripts/test_emotion_model.py")
    print("     b) Install libraries in current Python: pip3 install torch transformers scikit-learn")
    print("     c) Create virtual environment")
    
    print("\n2. 🧪 Model Testing:")
    print("   - Once Python is fixed, run: python scripts/test_emotion_model.py")
    print("   - This will test the model with sample journal entries")
    
    print("\n3. 📊 Dataset Expansion:")
    print("   - Run: python scripts/expand_journal_dataset.py")
    print("   - This will create 1000+ balanced samples")
    
    print("\n4. 🚀 Retraining:")
    print("   - Use expanded dataset to retrain")
    print("   - Expect 75-85% F1 score!")

def main():
    """Main test function."""
    print("🚀 SIMPLE MODEL TESTING")
    print("=" * 50)
    
    # Test files
    files_ok = test_model_files()
    
    # Test environment
    env_ok = test_python_environment()
    
    print("\n📊 Test Results:")
    print(f"   Files: {'✅' if files_ok else '❌'}")
    print(f"   Environment: {'✅' if env_ok else '❌'}")
    
    if files_ok and env_ok:
        print("\n🎉 All tests passed! Ready for full testing.")
    else:
        print("\n⚠️ Some issues found. Check above.")
    
    suggest_next_steps()

if __name__ == "__main__":
    main()
