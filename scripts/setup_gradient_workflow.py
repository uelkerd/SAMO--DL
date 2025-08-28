#!/usr/bin/env python3
"""
Paperspace Gradient Workflow Setup Script

This script helps set up and run the training pipeline workflow on Paperspace Gradient.
It handles authentication, workflow validation, and execution.
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class GradientWorkflowSetup:
    """Manages Paperspace Gradient workflow setup and execution."""

    def __init__(self, workflow_file: str = ".gradient/workflows/training-pipeline.yaml"):
        self.workflow_file = Path(workflow_file)
        self.api_key = os.getenv("PAPERSPACE_API_KEY")
        self.project_id = os.getenv("PAPERSPACE_PROJECT_ID")

    @staticmethod
    def check_gradient_cli() -> bool:
        """Check if gradient CLI is available."""
        try:
            result = subprocess.run(["gradient", "--version"], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ Gradient CLI found: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Gradient CLI not found. Please install it first:")
            print("   pip install gradient")
            return False

    @staticmethod
    def check_authentication() -> bool:
        """Check if user is authenticated with Paperspace."""
        try:
            result = subprocess.run(["gradient", "projects", "list"], 
                                  capture_output=True, text=True, check=True)
            if "No projects found" in result.stdout or "projects" in result.stdout:
                print("✅ Authentication successful")
                return True
            print("❌ Authentication failed")
            return False
        except subprocess.CalledProcessError:
            print("❌ Authentication failed. Please run:")
            print("   gradient apiKey YOUR_API_KEY")
            return False

    def authenticate(self, api_key: str) -> bool:
        """Authenticate with Paperspace using API key."""
        try:
            result = subprocess.run(["gradient", "apiKey", api_key], 
                                  capture_output=True, text=True, check=True)
            print("✅ Authentication successful")
            self.api_key = api_key
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Authentication failed: {e}")
            return False

    def validate_workflow_file(self) -> bool:
        """Validate the workflow YAML file."""
        if not self.workflow_file.exists():
            print(f"❌ Workflow file not found: {self.workflow_file}")
            return False

        try:
            with open(self.workflow_file, 'r') as f:
                workflow_config = yaml.safe_load(f)

            # Basic validation
            if 'workflows' not in workflow_config:
                print("❌ Invalid workflow file: missing 'workflows' section")
                return False

            workflow_name = list(workflow_config['workflows'].keys())[0]
            workflow = workflow_config['workflows'][workflow_name]

            if 'jobs' not in workflow:
                print("❌ Invalid workflow file: missing 'jobs' section")
                return False

            print(f"✅ Workflow file validated: {workflow_name}")
            print(f"   Jobs: {len(workflow['jobs'])}")

            # Print job names
            for job in workflow['jobs']:
                print(f"   - {job['name']}")

            return True

        except yaml.YAMLError as e:
            print(f"❌ YAML parsing error: {e}")
            return False
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return False

    @staticmethod
    def list_projects() -> None:
        """List available Paperspace projects."""
        try:
            result = subprocess.run(["gradient", "projects", "list"], 
                                  capture_output=True, text=True, check=True)
            print("📋 Available Projects:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to list projects: {e}")

    def run_workflow(self, project_id: Optional[str] = None) -> bool:
        """Run the workflow."""
        if not self.validate_workflow_file():
            return False

        cmd = ["gradient", "workflows", "run", str(self.workflow_file)]

        if project_id:
            cmd.extend(["--projectId", project_id])

        try:
            print(f"🚀 Running workflow: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)
            print("✅ Workflow started successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to run workflow: {e}")
            return False

    @staticmethod
    def create_datasets() -> None:
        """Create the required datasets for the workflow."""
        datasets = [
            "training-environment-setup",
            "prepared-training-data", 
            "trained-model-output",
            "training-logs",
            "evaluation-results",
            "training-pipeline-artifacts"
        ]

        print("📊 Creating required datasets...")
        for dataset_name in datasets:
            try:
                cmd = ["gradient", "datasets", "create", "--name", dataset_name]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                if result.returncode == 0:
                    print(f"   ✅ Created dataset: {dataset_name}")
                else:
                    print(f"   ⚠️  Dataset {dataset_name} may already exist")
            except Exception as e:
                print(f"   ❌ Failed to create dataset {dataset_name}: {e}")

    def setup_environment(self) -> bool:
        """Complete setup of the workflow environment."""
        print("🔧 Setting up Paperspace Gradient Workflow Environment")
        print("=" * 60)

        # Check CLI
        if not self.check_gradient_cli():
            return False

        # Check authentication
        if not self.check_authentication():
            print("\n🔑 Please provide your Paperspace API key:")
            api_key = input("API Key: ").strip()
            if not self.authenticate(api_key):
                return False

        # Validate workflow file
        if not self.validate_workflow_file():
            return False

        # List projects
        self.list_projects()

        # Create datasets
        self.create_datasets()

        print("\n✅ Environment setup completed!")
        return True

    def interactive_run(self) -> None:
        """Interactive workflow execution."""
        if not self.setup_environment():
            return

        print("\n🚀 Ready to run workflow!")

        # Ask for project ID
        project_id = input("Enter project ID (or press Enter to use default): ").strip()
        if not project_id:
            project_id = None

        # Confirm execution
        confirm = input("Run workflow now? (y/N): ").strip().lower()
        if confirm in ['y', 'yes']:
            self.run_workflow(project_id)
        else:
            print("Workflow execution cancelled.")


def main():
    """Main entry point."""
    setup = GradientWorkflowSetup()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "setup":
            setup.setup_environment()
        elif command == "run":
            project_id = sys.argv[2] if len(sys.argv) > 2 else None
            setup.run_workflow(project_id)
        elif command == "validate":
            setup.validate_workflow_file()
        elif command == "datasets":
            setup.create_datasets()
        elif command == "projects":
            setup.list_projects()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: setup, run, validate, datasets, projects")
    else:
        # Interactive mode
        setup.interactive_run()


if __name__ == "__main__":
    main()
