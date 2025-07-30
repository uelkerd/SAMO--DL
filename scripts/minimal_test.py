import json
import os
import sys

#!/usr/bin/env python3
"""
Minimal test script
"""



def main():
    print("🚀 Minimal test script running!")
    print("Arguments: {sys.argv}")

    if len(sys.argv) != 4:
        print("Usage: python minimal_test.py <project_id> <bucket_name> <data_path>")
        return

    project_id = sys.argv[1]
    bucket_name = sys.argv[2]
    data_path = sys.argv[3]

    print("✅ Project: {project_id}")
    print("✅ Bucket: {bucket_name}")
    print("✅ Data: {data_path}")

    # Test data file
    if os.path.exists(data_path):
        print("✅ Data file exists!")
        with open(data_path) as f:
            data = json.load(f)
        print("✅ Loaded {len(data)} entries")
    else:
        print("❌ Data file not found")

    print("🎉 Minimal test complete!")


if __name__ == "__main__":
    main()
