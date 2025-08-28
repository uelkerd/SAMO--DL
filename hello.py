#!/usr/bin/env python3
"""
Simple hello world script for testing the Paperspace Gradient workflow.
This file is referenced by the sample workflow.yaml file.
"""

import os
import sys
from datetime import datetime

def main():
    """Main function that prints hello world and some system info."""
    print("=" * 50)
    print("Hello World from Paperspace Gradient Workflow!")
    print("=" * 50)
    
    # Print current timestamp
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Print Python version
    print(f"Python version: {sys.version}")
    
    # Print working directory
    print(f"Working directory: {os.getcwd()}")
    
    # Print environment variables
    print(f"PAPERSPACE_PROJECT_ID: {os.getenv('PAPERSPACE_PROJECT_ID', 'Not set')}")
    print(f"PAPERSPACE_WORKSPACE_ID: {os.getenv('PAPERSPACE_WORKSPACE_ID', 'Not set')}")
    
    # List files in current directory
    print("\nFiles in current directory:")
    for item in os.listdir('.'):
        if os.path.isfile(item):
            print(f"  📄 {item}")
        elif os.path.isdir(item):
            print(f"  📁 {item}/")
    
    print("\n" + "=" * 50)
    print("Workflow execution completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()
