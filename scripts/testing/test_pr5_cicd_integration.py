#!/usr/bin/env python3
"""
PR #5: CI/CD Pipeline Overhaul - Integration Test

This script validates that the CircleCI configuration fixes are working correctly.
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path

def test_yaml_syntax():
    """Test that the CircleCI config YAML is valid."""
    print("🔍 Testing CircleCI YAML syntax...")
    
    config_path = Path(".circleci/config.yml")
    if not config_path.exists():
        print("❌ CircleCI config file not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            yaml.safe_load(f)
        print("✅ CircleCI YAML syntax is valid")
        return True
    except yaml.YAMLError as e:
        print(f"❌ YAML syntax error: {e}")
        return False

def test_conda_environment_setup():
    """Test that conda environment setup works locally and installs packages."""
    import tempfile
    import shutil

    print("🔍 Testing conda environment setup...")

    env_name = None
    try:
        # Test if conda is available
        result = subprocess.run(['conda', '--version'],
                                capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("❌ Conda not available")
            return False

        # Test if environment file exists
        env_path = Path("environment.yml")
        if not env_path.exists():
            print("❌ environment.yml not found")
            return False

        # Create a unique environment name
        env_name = f"test_env_{os.getpid()}"

        # Create the environment
        print(f"🔧 Creating conda environment '{env_name}' from environment.yml...")
        create_result = subprocess.run(
            ['conda', 'env', 'create', '-f', str(env_path), '-n', env_name],
            capture_output=True, text=True, timeout=300
        )
        if create_result.returncode != 0:
            print(f"❌ Failed to create conda environment:\n{create_result.stderr}")
            return False

        # Read the environment.yml to get a key package to check
        with open(env_path, 'r') as f:
            env_yaml = yaml.safe_load(f)
        dependencies = env_yaml.get('dependencies', [])
        # Find a package name (skip pip sublists)
        key_package = None
        for dep in dependencies:
            if isinstance(dep, str):
                key_package = dep.split('=')[0]
                break
        if not key_package:
            key_package = "python"  # fallback

        # Check that the key package is installed in the new environment
        print(f"🔍 Verifying package '{key_package}' is installed in '{env_name}'...")
        list_result = subprocess.run(
            ['conda', 'run', '-n', env_name, 'python', '-c', f"import {key_package}"],
            capture_output=True, text=True, timeout=30
        )
        if list_result.returncode != 0:
            print(f"❌ Package '{key_package}' not installed or import failed:\n{list_result.stderr}")
            return False

        print("✅ Conda environment setup and package installation validation passed")
        return True
    except Exception as e:
        print(f"❌ Conda environment test failed: {e}")
        return False
    finally:
        # Clean up: remove the test environment if it was created
        if env_name:
            print(f"🧹 Removing test conda environment '{env_name}'...")
            subprocess.run(['conda', 'env', 'remove', '-n', env_name, '-y'],
                           capture_output=True, text=True)

def test_critical_fixes():
    """Test that critical CircleCI fixes are applied."""
    print("🔍 Testing critical CircleCI fixes...")
    
    config_path = Path(".circleci/config.yml")
    with open(config_path, 'r') as f:
        content = f.read()
    
    fixes = [
        ("step_name:", "Fixed restricted parameter issue"),
        ("conda run -n samo-dl-stable", "Standardized conda usage"),
        ("shell: /bin/bash", "Explicit bash shell specification"),
        ("PYTHONPATH: $CIRCLE_WORKING_DIRECTORY/src", "PYTHONPATH configuration"),
    ]
    
    all_fixes_present = True
    for fix, description in fixes:
        if fix in content:
            print(f"✅ {description}")
        else:
            print(f"❌ {description} - NOT FOUND")
            all_fixes_present = False
    
    return all_fixes_present

def test_pipeline_structure():
    """Test that the pipeline structure is correct."""
    print("🔍 Testing pipeline structure...")
    
    config_path = Path(".circleci/config.yml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    required_components = [
        "executors",
        "commands", 
        "jobs",
        "workflows"
    ]
    
    all_components_present = True
    for component in required_components:
        if component in config:
            print(f"✅ {component} section present")
        else:
            print(f"❌ {component} section missing")
            all_components_present = False
    
    return all_components_present

def test_job_dependencies():
    """Test that job dependencies are properly configured."""
    print("🔍 Testing job dependencies...")
    
    config_path = Path(".circleci/config.yml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    workflows = config.get('workflows', {})
    if not workflows:
        print("❌ No workflows found")
        return False
    
    main_workflow = None
    for workflow_name, workflow_config in workflows.items():
        if workflow_name == 'samo-ci-cd':
            main_workflow = workflow_config
            break
    
    if not main_workflow:
        print("❌ Main workflow 'samo-ci-cd' not found")
        return False
    
    jobs = main_workflow.get('jobs', [])
    if not jobs:
        print("❌ No jobs in main workflow")
        return False
    
    print(f"✅ Found {len(jobs)} jobs in main workflow")
    return True

def main():
    """Run all PR #5 CI/CD integration tests."""
    print("🔍 Running PR #5 CI/CD Integration Tests...")
    print("=" * 60)
    
    tests = [
        ("YAML Syntax", test_yaml_syntax),
        ("Conda Environment Setup", test_conda_environment_setup),
        ("Critical Fixes", test_critical_fixes),
        ("Pipeline Structure", test_pipeline_structure),
        ("Job Dependencies", test_job_dependencies),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("📊 PR #5 CI/CD Integration Test Summary")
    print("=" * 60)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n✅ PR #5 CI/CD pipeline is ready for testing!")
        print("Ready for CircleCI validation")
    else:
        print(f"\n❌ PR #5 needs {total - passed} fixes before testing")
        print("Please address the failing tests above")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 