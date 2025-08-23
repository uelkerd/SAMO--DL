#!/usr/bin/env python3
"""CLI wrapper for uploading a custom trained model to HuggingFace Hub.

This refactors the previous monolithic script into modular components under
scripts/deployment/hf_upload/ and replaces prints with logging + argparse.
"""

import sys

from hf_upload.cli import main

if __name__ == "__main__":
    sys.exit(main())
