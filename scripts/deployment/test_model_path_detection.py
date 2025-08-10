#!/usr/bin/env python3
"""
🧪 Test Model Path Detection
============================
Test the portable model directory detection logic.
"""

import os
import sys

# Add the upload script to path to import the function
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from upload_model_to_huggingface import get_model_base_directory

def test_path_detection():
    """Test the model path detection under different scenarios."""
    
    print("🧪 TESTING MODEL PATH DETECTION")
    print("=" * 50)
    
    # Test 1: No environment variable set (auto-detection)
    print("\n🔍 Test 1: Auto-detection (no env vars)")
    original_base_dir = os.getenv('SAMO_DL_BASE_DIR')
    original_model_dir = os.getenv('MODEL_BASE_DIR')
    
    # Temporarily clear environment variables
    if 'SAMO_DL_BASE_DIR' in os.environ:
        del os.environ['SAMO_DL_BASE_DIR']
    if 'MODEL_BASE_DIR' in os.environ:
        del os.environ['MODEL_BASE_DIR']
    
    detected_path = get_model_base_directory()
    print(f"  Detected path: {detected_path}")
    print(f"  Path exists: {os.path.exists(os.path.dirname(detected_path))}")
    
    # Test 2: With SAMO_DL_BASE_DIR set
    print("\n🔧 Test 2: With SAMO_DL_BASE_DIR environment variable")
    os.environ['SAMO_DL_BASE_DIR'] = "/tmp/test_project"
    
    detected_path = get_model_base_directory()
    print(f"  Environment var: {os.getenv('SAMO_DL_BASE_DIR')}")
    print(f"  Detected path: {detected_path}")
    print(f"  Expected: /tmp/test_project/deployment/models")
    print(f"  Match: {detected_path == '/tmp/test_project/deployment/models'}")
    
    # Test 3: With MODEL_BASE_DIR set  
    print("\n🔧 Test 3: With MODEL_BASE_DIR environment variable")
    if 'SAMO_DL_BASE_DIR' in os.environ:
        del os.environ['SAMO_DL_BASE_DIR']
    os.environ['MODEL_BASE_DIR'] = "/home/user/projects/emotion-model"
    
    detected_path = get_model_base_directory()
    print(f"  Environment var: {os.getenv('MODEL_BASE_DIR')}")
    print(f"  Detected path: {detected_path}")
    print(f"  Expected: /home/user/projects/emotion-model/deployment/models")
    print(f"  Match: {detected_path == '/home/user/projects/emotion-model/deployment/models'}")
    
    # Test 4: With expanduser (~) path
    print("\n🏠 Test 4: With home directory path expansion")
    os.environ['SAMO_DL_BASE_DIR'] = "~/Projects/SAMO-DL"
    
    detected_path = get_model_base_directory()
    expected_path = os.path.expanduser("~/Projects/SAMO-DL/deployment/models")
    print(f"  Environment var: {os.getenv('SAMO_DL_BASE_DIR')}")
    print(f"  Detected path: {detected_path}")  
    print(f"  Expected: {expected_path}")
    print(f"  Match: {detected_path == expected_path}")
    
    # Restore original environment
    if original_base_dir:
        os.environ['SAMO_DL_BASE_DIR'] = original_base_dir
    elif 'SAMO_DL_BASE_DIR' in os.environ:
        del os.environ['SAMO_DL_BASE_DIR']
        
    if original_model_dir:
        os.environ['MODEL_BASE_DIR'] = original_model_dir
    elif 'MODEL_BASE_DIR' in os.environ:
        del os.environ['MODEL_BASE_DIR']
    
    print("\n✅ Path detection tests completed!")
    print("\n📋 Usage Examples:")
    print("  export SAMO_DL_BASE_DIR='/path/to/your/project'")
    print("  export MODEL_BASE_DIR='~/Projects/SAMO-DL'")
    print("  # Or let script auto-detect project root")

if __name__ == "__main__":
    test_path_detection()