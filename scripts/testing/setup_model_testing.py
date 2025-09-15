#!/usr/bin/env python3
"""
Setup script for testing the emotion detection model.
"""

import os
import json
import shutil

def check_model_files():
    """Check if required model files exist."""
    print("🔍 Checking for model files...")
    
    required_files = {
        'model': 'best_simple_model.pth',
        'results': 'simple_training_results.json'
    }
    
    missing_files = []
    existing_files = {}
    
    for file_type, filename in required_files.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            existing_files[file_type] = (filename, size)
            print(f"✅ {file_type.capitalize()}: {filename} ({size:,} bytes)")
        else:
            missing_files.append(file_type)
            print(f"❌ {file_type.capitalize()}: {filename} - MISSING")
    
    return existing_files, missing_files

def create_mock_results():
    """Create mock results file for testing if missing."""
    print("\n🔧 Creating mock results file for testing...")
    
    # Mock results based on our training
    mock_results = {
        "best_f1": 0.6692,
        "target_achieved": False,
        "num_labels": 12,
        "go_samples": 43410,
        "journal_samples": 150,
        "all_emotions": [
            "anxious", "calm", "content", "excited", "frustrated", 
            "grateful", "happy", "hopeful", "overwhelmed", "proud", "sad", "tired"
        ],
        "emotion_mapping": {
            "joy": "happy",
            "gratitude": "grateful",
            "pride": "proud",
            "excitement": "excited",
            "optimism": "hopeful",
            "sadness": "sad",
            "fear": "anxious",
            "anger": "frustrated",
            "disgust": "frustrated",
            "surprise": "excited",
            "love": "content",
            "caring": "content",
            "approval": "proud",
            "admiration": "proud",
            "amusement": "happy",
            "confusion": "anxious",
            "curiosity": "excited",
            "desire": "excited",
            "disappointment": "sad",
            "disapproval": "frustrated",
            "embarrassment": "anxious",
            "grief": "sad",
            "nervousness": "anxious",
            "realization": "content",
            "relief": "calm",
            "remorse": "sad",
            "neutral": "calm"
        }
    }
    
    with open('simple_training_results.json', 'w') as f:
        json.dump(mock_results, f, indent=2)
    
    print("✅ Created mock results file: simple_training_results.json")

def find_model_file():
    """Find the model file in common locations."""
    print("\n🔍 Searching for model file...")
    
    search_locations = [
        "best_simple_model.pth",
        "best_focal_model.pth",  # Fallback
        os.path.expanduser("~/Downloads/best_simple_model.pth"),
        os.path.expanduser("~/Desktop/best_simple_model.pth"),
        os.path.expanduser("~/best_simple_model.pth")
    ]
    
    for location in search_locations:
        if os.path.exists(location):
            size = os.path.getsize(location)
            print(f"✅ Found model: {location} ({size:,} bytes)")
            
            # Copy to current directory if not already here
            if location != "best_simple_model.pth":
                shutil.copy2(location, "best_simple_model.pth")
                print(f"✅ Copied to: best_simple_model.pth")
            
            return True
    
    print("❌ Model file not found in common locations")
    return False

def setup_testing():
    """Main setup function."""
    print("🚀 SETTING UP MODEL TESTING")
    print("=" * 50)
    
    # Check existing files
    existing_files, missing_files = check_model_files()
    
    # Find model file if missing
    if 'model' in missing_files and not find_model_file():
        print("\n❌ Cannot proceed without model file!")
        print("📋 Please download best_simple_model.pth from Colab and place it in this directory")
        return False
    
    # Create mock results if missing
    if 'results' in missing_files:
        create_mock_results()
    
    print("\n✅ Setup complete! Ready for testing.")
    return True

def run_quick_test():
    """Run a quick test to verify everything works."""
    print("\n🧪 Running quick test...")
    
    try:
        import torch
        import transformers
        from sklearn.preprocessing import LabelEncoder
        
        print("✅ All required libraries available")
        
        # Test model loading
        if os.path.exists('best_simple_model.pth'):
            print("✅ Model file exists")
            
            # Try to load a small part to verify it's valid
            checkpoint = torch.load('best_simple_model.pth', map_location='cpu')
            print(f"✅ Model checkpoint loaded with {len(checkpoint)} layers")
            
        return True
        
    except ImportError as e:
        print(f"❌ Missing library: {e}")
        print("📋 Install with: pip install torch transformers scikit-learn")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    if setup_testing():
        run_quick_test()
        print("\n🎉 Ready to test the model!")
        print("📋 Run: python scripts/test_emotion_model.py")
    else:
        print("\n❌ Setup failed. Please check the issues above.") 