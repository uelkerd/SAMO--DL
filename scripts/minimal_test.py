#!/usr/bin/env python3
"""
Minimal test script
"""

import sys
import json
import os


def main():
    print("🚀 Minimal test script running!")
    print(f"Arguments: {sys.argv}")

    if len(sys.argv) != 4:
        print("Usage: python minimal_test.py <project_id> <bucket_name> <data_path>")
        return

    project_id = sys.argv[1]
    bucket_name = sys.argv[2]
    data_path = sys.argv[3]

    print(f"✅ Project: {project_id}")
    print(f"✅ Bucket: {bucket_name}")
    print(f"✅ Data: {data_path}")

    # Test data file
    if os.path.exists(data_path):
        print("✅ Data file exists!")
        with open(data_path) as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} entries")
    else:
        print("❌ Data file not found")

    print("🎉 Minimal test complete!")


if __name__ == "__main__":
    main()
