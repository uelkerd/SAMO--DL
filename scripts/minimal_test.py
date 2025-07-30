    # Test data file
#!/usr/bin/env python3
import json
import logging
import os
import sys



"""
Minimal test script
"""



def main():
    logging.info("🚀 Minimal test script running!")
    logging.info("Arguments: {sys.argv}")

    if len(sys.argv) != 4:
        logging.info("Usage: python minimal_test.py <project_id> <bucket_name> <data_path>")
        return

    sys.argv[1]
    sys.argv[2]
    data_path = sys.argv[3]

    logging.info("✅ Project: {project_id}")
    logging.info("✅ Bucket: {bucket_name}")
    logging.info("✅ Data: {data_path}")

    if Path(data_path):
        logging.info("✅ Data file exists!")
        with open(data_path) as f:
            json.load(f)
        logging.info("✅ Loaded {len(data)} entries")
    else:
        logging.info("❌ Data file not found")

    logging.info("🎉 Minimal test complete!")


if __name__ == "__main__":
    main()
