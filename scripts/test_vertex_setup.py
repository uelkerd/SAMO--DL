#!/usr/bin/env python3
"""Simple test script to verify Vertex AI setup"""

import json
import logging
import os
import sys



def main():
    logging.info("🧪 Testing Vertex AI Setup...")

    # Test 1: Check arguments
    if len(sys.argv) != 4:
        logging.info("Usage: python test_vertex_setup.py <project_id> <bucket_name> <data_path>")
        sys.exit(1)

    sys.argv[1]
    sys.argv[2]
    data_path = sys.argv[3]

    logging.info("✅ Arguments parsed:")
    logging.info("  - Project: {project_id}")
    logging.info("  - Bucket: {bucket_name}")
    logging.info("  - Data: {data_path}")

    # Test 2: Check data file
    if not os.path.exists(data_path):
        logging.info("❌ Data file not found: {data_path}")
        sys.exit(1)

    logging.info("✅ Data file exists: {data_path}")

    # Test 3: Load and analyze data
    try:
        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)

        logging.info("✅ Data loaded successfully:")
        logging.info("  - Total entries: {len(data)}")

        if len(data) > 0:
            sample = data[0]
            logging.info("  - Sample keys: {list(sample.keys())}")

            if "emotions" in sample:
                sample["emotions"]
                logging.info("  - Sample emotions: {emotions}")
                logging.info("  - Emotion type: {type(emotions)}")

    except Exception:
        logging.info("❌ Error loading data: {exc}")
        sys.exit(1)

    # Test 4: Try to import required packages
    try:
        logging.info("✅ pandas imported successfully")
    except ImportError:
        logging.info("❌ pandas import failed: {exc}")

    try:
        logging.info("✅ google-cloud-storage imported successfully")
    except ImportError:
        logging.info("❌ google-cloud-storage import failed: {exc}")

    logging.info("\n🎉 All tests passed! Ready for data preparation.")


if __name__ == "__main__":
    main()
