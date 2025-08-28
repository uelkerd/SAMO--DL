#!/usr/bin/env python3
"""Hello World script for Gradient workflows.

This file is referenced by the sample workflow.yaml file.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    """Main function that prints hello world and some system info."""
    print("=" * 50)
    print("Hello World from Paperspace Gradient Workflow!")
    print("=" * 50)

    # Print current timestamp
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    # Print Python version
    print(f"Python version: {sys.version}")

    # Print working directory
    print(f"Working directory: {Path.cwd()}")

    # Print environment variables
    print(f"PAPERSPACE_PROJECT_ID: {os.getenv('PAPERSPACE_PROJECT_ID', 'Not set')}")
    print(f"PAPERSPACE_WORKSPACE_ID: {os.getenv('PAPERSPACE_WORKSPACE_ID', 'Not set')}")

    # List files in current directory
    print("\nFiles in current directory:")
    enc = (sys.stdout.encoding or "").lower()
    use_emoji = "utf" in enc
    file_icon = "📄" if use_emoji else "-"
    dir_icon = "📁" if use_emoji else "[dir]"

    for p in Path(".").iterdir():
        if p.is_file():
            print(f"  {file_icon} {p.name}")
        elif p.is_dir():
            print(f"  {dir_icon} {p.name}/")

    print("\n" + "=" * 50)
    print("Workflow execution completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()
