#!/usr/bin/env python3
"""
Paperspace Gradient Workflow Setup Script

This script helps set up and run the training pipeline workflow on Paperspace Gradient.
It handles authentication, workflow validation, and execution.
"""

import os
import sys
import subprocess
import json
import yaml
import argparse
from pathlib import Path
from typing import Optional


class GradientWorkflowSetup:
    """Manages Paperspace Gradient workflow setup and execution."""

    def __init__(self, workflow_file: str = 
                 ".gradient/workflows/training-pipeline.yaml"):
        self.workflow_file = Path(workflow_file)
        self.api_key = os.getenv("PAPERSPACE_API_KEY")
        self.project_id = os.getenv("PAPERSPACE_PROJECT_ID")

    @staticmethod
    def check_gradient_cli() -> bool:
        """Check if gradient CLI is available."""
        try:
            result = subprocess.run(["gradient", "--version"],
                                   capture_output=True, text=True, check=True,
                                   timeout=10)
            print(f"✅ Gradient CLI found: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Gradient CLI not found. Please install it first:")
            print("   pip install gradient")
            return False
        except subprocess.TimeoutExpired:
            print("❌ Gradient CLI check timed out")
            return False

    @staticmethod
    def check_authentication() -> bool:
        """Check if user is authenticated with Paperspace."""
        try:
            result = subprocess.run(["gradient", "projects", "list"],
                                   capture_output=True, text=True, check=True,
                                   timeout=30)
            if "No projects found" in result.stdout or "projects" in result.stdout:
                print("✅ Authentication successful")
                return True
            print("❌ Authentication failed")
            return False
        except subprocess.CalledProcessError:
            print("❌ Authentication failed. Please run:")
            print("   gradient apiKey YOUR_API_KEY")
            return False
        except subprocess.TimeoutExpired:
            print("❌ Authentication check timed out")
            return False

    def authenticate(self, api_key: str) -> bool:
        """Authenticate with Paperspace using API key."""
        try:
            subprocess.run(["gradient", "apiKey", api_key],
                           capture_output=True, text=True, check=True, timeout=30)
            print("✅ Authentication successful")
            self.api_key = api_key
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Authentication failed: {e}")
            return False
        except subprocess.TimeoutExpired:
            print("❌ Authentication timed out")
            return False

    @staticmethod
    def _validate_workflow_structure(workflow_config: dict) -> tuple[bool, Optional[str], Optional[dict]]:
        """Validate the basic structure of the workflow configuration."""
        if 'workflows' not in workflow_config:
            print("❌ Invalid workflow file: missing 'workflows' section")
            return False, None, None

        workflow_name = next(iter(workflow_config['workflows'].keys()))
        workflow = workflow_config['workflows'][workflow_name]

        if 'jobs' not in workflow:
            print("❌ Invalid workflow file: missing 'jobs' section")
            return False, None, None

        return True, workflow_name, workflow

    @staticmethod
    def _print_workflow_info(workflow_name: str, workflow: dict) -> None:
        """Print workflow information and job details."""
        print(f"✅ Workflow file validated: {workflow_name}")
        print(f"   Jobs: {len(workflow['jobs'])}")

        # Print job names
        for job in workflow['jobs']:
            print(f"   - {job.get('name', '<unnamed>')}")

    def validate_workflow_file(self) -> bool:
        """Validate the workflow YAML file."""
        if not self.workflow_file.exists():
            print(f"❌ Workflow file not found: {self.workflow_file}")
            return False

        try:
            workflow_config = yaml.safe_load(self.workflow_file.read_text())

            # Validate that YAML loading returned a valid dict
            if not isinstance(workflow_config, dict):
                print("❌ Invalid workflow file: YAML content is not a dictionary")
                return False

            # Basic validation
            is_valid, workflow_name, workflow = self._validate_workflow_structure(workflow_config)
            if not is_valid:
                return False

            # Print workflow information
            self._print_workflow_info(workflow_name, workflow)

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
                                   capture_output=True, text=True, check=True,
                                   timeout=30)
            print("📋 Available Projects:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("❌ Failed to list projects.")
            print(f"   STDOUT: {e.stdout or ''}".rstrip())
            print(f"   STDERR: {e.stderr or ''}".rstrip())
        except subprocess.TimeoutExpired:
            print("❌ Project listing timed out")

    def run_workflow(self, project_id: Optional[str] = None) -> bool:
        """Run the workflow."""
        if not self.validate_workflow_file():
            return False

        cmd = ["gradient", "workflows", "run", str(self.workflow_file)]

        # Use provided project_id or fall back to self.project_id
        project_id = project_id or self.project_id
        if project_id:
            cmd.extend(["--projectId", project_id])

        try:
            print(f"🚀 Running workflow: {' '.join(cmd)}")
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            print("✅ Workflow started successfully")
            return True
        except subprocess.CalledProcessError as e:
            print("❌ Failed to run workflow.")
            print(f"   STDOUT: {e.stdout or ''}".rstrip())
            print(f"   STDERR: {e.stderr or ''}".rstrip())
            return False
        except subprocess.TimeoutExpired:
            print("❌ Workflow execution timed out")
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

        print("📊 Checking and creating required datasets...")

        # List existing datasets
        try:
            list_cmd = ["gradient", "datasets", "list", "--json"]
            list_result = subprocess.run(list_cmd, capture_output=True, text=True,
                                       timeout=30, check=True)
            datasets_json = json.loads(list_result.stdout)
            existing_datasets = {
                ds.get("name") for ds in datasets_json if "name" in ds
            }
        except subprocess.CalledProcessError as e:
            print("   ❌ Failed to list existing datasets.")
            print(f"   STDOUT: {e.stdout or ''}".rstrip())
            print(f"   STDERR: {e.stderr or ''}".rstrip())
            existing_datasets = set()
        except json.JSONDecodeError as e:
            print(f"   ❌ Failed to parse datasets JSON: {e}")
            existing_datasets = set()
        except Exception as e:
            print(f"   ❌ Error listing datasets: {e}")
            existing_datasets = set()

        for dataset_name in datasets:
            if dataset_name in existing_datasets:
                print(f"   ⏩ Skipping existing dataset: {dataset_name}")
                continue
            try:
                cmd = ["gradient", "datasets", "create", "--name", dataset_name]
                subprocess.run(cmd, capture_output=True, text=True,
                             timeout=30, check=True)
                print(f"   ✅ Created dataset: {dataset_name}")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed to create dataset {dataset_name}: "
                      f"{e.stderr.strip()}")
            except Exception as e:
                print(f"   ❌ Exception creating dataset {dataset_name}: {e}")

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

        # Support project ID from command-line argument or environment variable
        parser = argparse.ArgumentParser(
            description="Run Gradient workflow interactively."
        )
        parser.add_argument("--project-id", type=str,
                           help="Project ID to use for the workflow.")
        args, _ = parser.parse_known_args()

        project_id = args.project_id or os.environ.get(
            "PAPERSPACE_PROJECT_ID"
        )
        if not project_id:
            # Ask for project ID (env/default respected)
            default_pid = self.project_id
            prompt = "Enter project ID (or press Enter to use default"
            prompt += f" [{default_pid}]" if default_pid else ""
            prompt += "): "
            project_id = input(prompt).strip() or default_pid

        # Check if we're in non-interactive mode
        if os.environ.get("GRADIENT_NON_INTERACTIVE"):
            print("Non-interactive mode detected. Running workflow automatically...")
            self.run_workflow(project_id)
            return

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
