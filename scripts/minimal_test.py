import logging

import json
import os
import sys

#!/usr/bin/env python3

"""
Minimal test script
"""



def main():
    logging.info("🚀 Minimal test script running!")
    logging.info("Arguments: {sys.argv}")

    if len(sys.argv) != 4:
        logging.info("Usage: python minimal_test.py <project_id> <bucket_name> <data_path>")
        return

    project_id = sys.argv[1]
    bucket_name = sys.argv[2]
    data_path = sys.argv[3]

    logging.info("✅ Project: {project_id}")
    logging.info("✅ Bucket: {bucket_name}")
    logging.info("✅ Data: {data_path}")

    # Test data file
    if os.path.exists(data_path):
        logging.info("✅ Data file exists!")
        with open(data_path) as f:
            data = json.load(f)
        logging.info("✅ Loaded {len(data)} entries")
    else:
        logging.info("❌ Data file not found")

    logging.info("🎉 Minimal test complete!")


if __name__ == "__main__":
    main()
