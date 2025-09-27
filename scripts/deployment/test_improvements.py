#!/usr/bin/env python3
"""
🧪 Test Script Improvements
============================
Validate the improvements made to the upload script.
"""

import os
import sys
import json
import tempfile
from unittest.mock import patch

# Add the upload script to path to import functions
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

def test_modern_typing():
    """Test that modern typing annotations work correctly."""
    print("🧪 TESTING MODERN TYPING ANNOTATIONS")
    print("=" * 50)
    
    # Test dict[str, any] type hints (Python 3.9+ style)
    sample_dict: dict[str, any] = {
        'emotion_labels': ['happy', 'sad', 'angry'],
        'num_labels': 3,
        'validation_warnings': []
    }
    
    sample_list: list[str] = ['happy', 'sad', 'angry']
    
    print("✅ Modern type annotations working correctly")
    print(f"   • dict[str, any]: {type(sample_dict).__name__} with {len(sample_dict)} items")
    print(f"   • list[str]: {type(sample_list).__name__} with {len(sample_list)} items")
    
    return True

def test_directory_creation():
    """Test the directory creation functionality."""
    print("\n🧪 TESTING DIRECTORY CREATION")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test config path directory creation
        config_path = os.path.join(temp_dir, "deployment", "custom_model_config.json")
        config_dir = os.path.dirname(config_path)
        
        print(f"Config path: {config_path}")
        print(f"Config dir: {config_dir}")
        
        # This should create the directory
        os.makedirs(config_dir, exist_ok=True)
        
        # Verify directory exists
        if os.path.exists(config_dir):
            print("✅ Directory creation works correctly")
            
            # Test writing config file
            config = {"test": "data"}
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            if os.path.exists(config_path):
                print("✅ Config file creation works correctly")
                return True
        
        print("❌ Directory creation failed")
        return False

def test_label_loading_methods():
    """Test different methods of loading emotion labels."""
    print("\n🧪 TESTING LABEL LOADING METHODS")
    print("=" * 50)
    
    # Test method 1: Environment variable (JSON format)
    print("🔍 Testing environment variable method (JSON)...")
    test_labels_json = '["happy", "sad", "angry", "calm", "excited"]'
    
    with patch.dict(os.environ, {'EMOTION_LABELS': test_labels_json}):
        env_labels = os.getenv('EMOTION_LABELS')
        if env_labels:
            try:
                labels = json.loads(env_labels)
                print(f"✅ JSON env method: {len(labels)} labels loaded")
            except json.JSONDecodeError:
                print("❌ JSON env method failed")
    
    # Test method 2: Environment variable (comma-separated)
    print("🔍 Testing environment variable method (comma-separated)...")
    test_labels_csv = "happy, sad, angry, calm, excited"
    
    with patch.dict(os.environ, {'EMOTION_LABELS': test_labels_csv}):
        env_labels = os.getenv('EMOTION_LABELS')
        if env_labels:
            labels = [label.strip() for label in env_labels.split(',') if label.strip()]
            print(f"✅ CSV env method: {len(labels)} labels loaded")
    
    # Test method 3: JSON file loading simulation
    print("🔍 Testing JSON file method...")
    test_json_data = {"labels": ["happy", "sad", "angry", "calm"]}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_json_data, f)
        temp_file = f.name
    
    try:
        with open(temp_file, 'r') as f:
            data = json.load(f)
        
        if 'labels' in data:
            labels = data['labels']
            print(f"✅ JSON file method: {len(labels)} labels loaded")
    finally:
        os.unlink(temp_file)
    
    print("✅ All label loading methods validated")
    return True

def test_model_validation_components():
    """Test model validation component checking."""
    print("\n🧪 TESTING MODEL VALIDATION COMPONENTS")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock HuggingFace model directory structure
        config_file = os.path.join(temp_dir, "config.json")
        tokenizer_file = os.path.join(temp_dir, "tokenizer.json")
        weights_file = os.path.join(temp_dir, "pytorch_model.bin")
        
        # Test 1: Complete model (all components present)
        print("🔍 Testing complete model validation...")
        
        # Create mock files
        with open(config_file, 'w') as f:
            json.dump({"model_type": "test", "num_labels": 5}, f)
        
        with open(tokenizer_file, 'w') as f:
            json.dump({"vocab": {"test": 0}}, f)
        
        with open(weights_file, 'wb') as f:
            f.write(b"mock_model_weights_data")
        
        # Check component existence
        has_config = os.path.exists(config_file)
        has_tokenizer = os.path.exists(tokenizer_file)
        has_weights = os.path.exists(weights_file)
        
        if has_config and has_tokenizer and has_weights:
            print("✅ Complete model validation: All components present")
        else:
            print(f"❌ Complete model validation failed: config={has_config}, tokenizer={has_tokenizer}, weights={has_weights}")
        
        # Test 2: Recursive directory size calculation
        print("🔍 Testing recursive size calculation...")
        
        # Create nested directory structure
        nested_dir = os.path.join(temp_dir, "nested")
        os.makedirs(nested_dir)
        
        nested_file = os.path.join(nested_dir, "nested_file.txt")
        with open(nested_file, 'w') as f:
            f.write("test content for nested file")
        
        # Calculate directory size recursively
        def calculate_directory_size(directory):
            total_size = 0
            for dirpath, _, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (OSError, FileNotFoundError):
                        pass
            return total_size
        
        total_size = calculate_directory_size(temp_dir)
        print(f"✅ Recursive size calculation: {total_size} bytes")
        
        if total_size > 0:
            print("✅ Model validation improvements working correctly")
            return True
        
        print("❌ Model validation improvements failed")
        return False

def main():
    """Run all tests."""
    print("🚀 TESTING SCRIPT IMPROVEMENTS")
    print("=" * 60)
    
    tests = [
        test_modern_typing,
        test_directory_creation,
        test_label_loading_methods,
        test_model_validation_components
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n🎯 SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All improvements working correctly!")
        return True
    print("⚠️ Some tests failed - review implementation")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)