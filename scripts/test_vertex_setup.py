#!/usr/bin/env python3
"""
Simple test script to verify Vertex AI setup
"""

import json
import sys
import os


def main():
    print("🧪 Testing Vertex AI Setup...")

    # Test 1: Check arguments
    if len(sys.argv) != 4:
        print("Usage: python test_vertex_setup.py <project_id> <bucket_name> <data_path>")
        sys.exit(1)

    project_id = sys.argv[1]
    bucket_name = sys.argv[2]
    data_path = sys.argv[3]

    print("✅ Arguments parsed:")
    print(f"  - Project: {project_id}")
    print(f"  - Bucket: {bucket_name}")
    print(f"  - Data: {data_path}")

    # Test 2: Check data file
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        sys.exit(1)

    print(f"✅ Data file exists: {data_path}")

    # Test 3: Load and analyze data
    try:
        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)

        print("✅ Data loaded successfully:")
        print(f"  - Total entries: {len(data)}")

        if len(data) > 0:
            sample = data[0]
            print(f"  - Sample keys: {list(sample.keys())}")

            if "emotions" in sample:
                emotions = sample["emotions"]
                print(f"  - Sample emotions: {emotions}")
                print(f"  - Emotion type: {type(emotions)}")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)

    # Test 4: Try to import required packages
    try:
        import pandas as pd

        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")

    try:
        from google.cloud import storage

        print("✅ google-cloud-storage imported successfully")
    except ImportError as e:
        print(f"❌ google-cloud-storage import failed: {e}")

    print("\n🎉 All tests passed! Ready for data preparation.")


if __name__ == "__main__":
    main()
