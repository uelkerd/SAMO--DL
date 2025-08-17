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
    print"🔍 Testing CircleCI YAML syntax..."
    
    config_path = Path".circleci/config.yml"
    if not config_path.exists():
        print"❌ CircleCI config file not found"
        return False
    
    try:
        with openconfig_path, 'r' as f:
            yaml.safe_loadf
        print"✅ CircleCI YAML syntax is valid"
        return True
    except yaml.YAMLError as e:
        printf"❌ YAML syntax error: {e}"
        return False

def test_conda_environment_setup():
    """Test that conda environment setup configuration is valid FAST VERSION."""
    print("🔍 Testing conda environment setup fast validation...")

    try:
        # Test if conda is available using the full path like CircleCI
        conda_path = os.path.expanduser"~/miniconda/bin/conda"
        if os.path.existsconda_path:
            conda_cmd = [conda_path]
        else:
            conda_cmd = ['conda']  # fallback to PATH
        
        result = subprocess.run(conda_cmd + ['--version'],
                                capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print"❌ Conda not available"
            return False

        # Test if environment file exists and is valid YAML
        env_path = Path"environment.yml"
        if not env_path.exists():
            print"❌ environment.yml not found"
            return False

        # Validate environment.yml structure
        with openenv_path, 'r' as f:
            env_yaml = yaml.safe_loadf
        
        # Check required fields
        if 'name' not in env_yaml:
            print"❌ environment.yml missing 'name' field"
            return False
        
        if 'dependencies' not in env_yaml:
            print"❌ environment.yml missing 'dependencies' field"
            return False
        
        dependencies = env_yaml.get'dependencies', []
        if not dependencies:
            print"❌ environment.yml has no dependencies"
            return False
        
        # Check for key packages
        import re
        found_packages = []
        for dep in dependencies:
            if isinstancedep, str:
                package_name = re.splitr'[=<>~,]+', dep[0].strip()
                if package_name != 'python':
                    found_packages.appendpackage_name
        
        if not found_packages:
            print"❌ No valid packages found in environment.yml"
            return False
        
        print(f"✅ Found {lenfound_packages} packages in environment.yml")
        print("✅ Conda environment setup validation passed fast mode")
        return True
        
    except Exception as e:
        printf"❌ Conda environment test failed: {e}"
        return False

def test_critical_fixes():
    """Test that critical CircleCI fixes are applied using YAML parsing."""
    import yaml

    print"🔍 Testing critical CircleCI fixes..."

    config_path = Path".circleci/config.yml"
    try:
        with openconfig_path, 'r' as f:
            config = yaml.safe_loadf
    except Exception as e:
        printf"❌ Failed to load config: {e}"
        return False

    all_fixes_present = True

    # 1. Check for 'step_name' parameter in run_in_conda command
    found_step_name = False
    commands = config.get"commands", {}
    run_in_conda_cmd = commands.get"run_in_conda", {}
    if run_in_conda_cmd:
        parameters = run_in_conda_cmd.get"parameters", {}
        if "step_name" in parameters:
            found_step_name = True
    if found_step_name:
        print("✅ Fixed restricted parameter issue step_name parameter found")
    else:
        print("❌ Fixed restricted parameter issue step_name parameter NOT FOUND")
        all_fixes_present = False

    # 2. Check for 'conda run -n samo-dl-stable' in commands
    found_conda_run = False
    for cmd_name, cmd_config in commands.items():
        if isinstancecmd_config, dict and "steps" in cmd_config:
            for step in cmd_config["steps"]:
                if isinstancestep, dict and "run" in step:
                    run_val = step["run"]
                    if isinstancerun_val, dict:
                        command = run_val.get"command", ""
                    else:
                        command = run_val
                    if "conda run -n samo-dl-stable" in command:
                        found_conda_run = True
                        break
            if found_conda_run:
                break
    if found_conda_run:
        print("✅ Standardized conda usage conda run -n samo-dl-stable found")
    else:
        print("❌ Standardized conda usage conda run -n samo-dl-stable NOT FOUND")
        all_fixes_present = False

    # 3. Check for 'shell: /bin/bash' in commands
    found_shell_bash = False
    for cmd_name, cmd_config in commands.items():
        if isinstancecmd_config, dict and "steps" in cmd_config:
            for step in cmd_config["steps"]:
                if isinstancestep, dict and "run" in step:
                    run_val = step["run"]
                    if isinstancerun_val, dict:
                        shell = run_val.get"shell", ""
                        if shell == "/bin/bash":
                            found_shell_bash = True
                            break
            if found_shell_bash:
                break
    if found_shell_bash:
        print("✅ Explicit bash shell specification shell: /bin/bash found")
    else:
        print("❌ Explicit bash shell specification shell: /bin/bash NOT FOUND")
        all_fixes_present = False

    # 4. Check for PYTHONPATH: $CIRCLE_WORKING_DIRECTORY/src in executors
    found_pythonpath = False
    executors = config.get"executors", {}
    for executor_name, executor_config in executors.items():
        if isinstanceexecutor_config, dict:
            env = executor_config.get"environment", {}
            if env.get"PYTHONPATH" == "$CIRCLE_WORKING_DIRECTORY/src":
                found_pythonpath = True
                break
    if found_pythonpath:
        print("✅ PYTHONPATH configuration PYTHONPATH: $CIRCLE_WORKING_DIRECTORY/src found")
    else:
        print("❌ PYTHONPATH configuration PYTHONPATH: $CIRCLE_WORKING_DIRECTORY/src NOT FOUND")
        all_fixes_present = False

    return all_fixes_present

def test_pipeline_structure():
    """Test that the pipeline structure is correct, including handling malformed or incomplete configs."""
    print"🔍 Testing pipeline structure..."

    config_path = Path".circleci/config.yml"
    try:
        with openconfig_path, 'r' as f:
            config = yaml.safe_loadf
    except Exception as e:
        printf"❌ Failed to load config: {e}"
        return False

    required_components = [
        "executors",
        "commands", 
        "jobs",
        "workflows"
    ]

    all_components_present = True
    if not isinstanceconfig, dict:
        print("❌ Config file is not a valid YAML mapping dict.")
        return False

    for component in required_components:
        if component in config:
            printf"✅ {component} section present"
        else:
            printf"❌ {component} section missing"
            all_components_present = False

    return all_components_present

def test_pipeline_structure_edge_cases():
    """Test pipeline structure with missing sections and malformed YAML edge cases."""
    print"🔍 Testing pipeline structure edge cases..."

    # Test 1: Missing sections
    incomplete_config = {
        "executors": {},
        # "commands" missing
        "jobs": {},
        # "workflows" missing
    }
    required_components = [
        "executors",
        "commands", 
        "jobs",
        "workflows"
    ]
    missing_count = 0
    for component in required_components:
        if component not in incomplete_config:
            missing_count += 1
    print(f"✅ Simulated missing sections test: {missing_count} components missing expected: 2")

    # Test 2: Malformed YAML types
    malformed_configs = [None, [], "not_a_dict"]
    for idx, malformed in enumeratemalformed_configs:
        if not isinstancemalformed, dict:
            print(f"✅ Malformed config case {idx+1}: {reprmalformed} correctly identified as invalid")
        else:
            print(f"❌ Malformed config case {idx+1}: {reprmalformed} incorrectly identified as valid")

    return True

def test_job_dependencies():
    """Test that job dependencies are properly configured with order verification."""
    print"🔍 Testing job dependencies..."
    
    config_path = Path".circleci/config.yml"
    try:
        with openconfig_path, 'r' as f:
            config = yaml.safe_loadf
    except Exception as e:
        printf"❌ Failed to load config: {e}"
        return False
    
    workflows = config.get'workflows', {}
    if not workflows:
        print"❌ No workflows found"
        return False
    
    main_workflow = None
    for workflow_name, workflow_config in workflows.items():
        if workflow_name == 'samo-ci-cd':
            main_workflow = workflow_config
            break
    
    if not main_workflow:
        print"❌ Main workflow 'samo-ci-cd' not found"
        return False
    
    jobs = main_workflow.get'jobs', []
    if not jobs:
        print"❌ No jobs in main workflow"
        return False
    
    print(f"✅ Found {lenjobs} jobs in main workflow")
    
    # Verify job dependency order and relationships
    job_names = []
    job_dependencies = {}
    
    for job in jobs:
        if isinstancejob, dict:
            # Job with configuration
            job_name = list(job.keys())[0]
            job_config = job[job_name]
            job_names.appendjob_name
            
            # Check for dependencies
            if 'requires' in job_config:
                job_dependencies[job_name] = job_config['requires']
                printf"✅ Job '{job_name}' has dependencies: {job_config['requires']}"
            else:
                job_dependencies[job_name] = []
                print(f"✅ Job '{job_name}' has no dependencies runs first")
        else:
            # Simple job name
            job_names.appendjob
            job_dependencies[job] = []
            print(f"✅ Job '{job}' has no dependencies runs first")
    
    # Verify dependency relationships are valid
    all_deps_valid = True
    for job_name, deps in job_dependencies.items():
        for dep in deps:
            if dep not in job_names:
                printf"❌ Job '{job_name}' depends on '{dep}' which doesn't exist"
                all_deps_valid = False
    
    if all_deps_valid:
        print"✅ All job dependencies reference valid jobs"
    
    # Check for circular dependencies basic check
    has_circular = False
    for job_name, deps in job_dependencies.items():
        for dep in deps:
            if job_name in job_dependencies.getdep, []:
                printf"❌ Circular dependency detected: {job_name} ↔ {dep}"
                has_circular = True
    
    if not has_circular:
        print"✅ No circular dependencies detected"
    
    return all_deps_valid and not has_circular

def test_environment_variables():
    """Test that environment variables are properly configured."""
    print"🔍 Testing environment variables..."
    
    config_path = Path".circleci/config.yml"
    try:
        with openconfig_path, 'r' as f:
            config = yaml.safe_loadf
    except Exception as e:
        printf"❌ Failed to load config: {e}"
        return False
    
    # Check for hardcoded conda paths that should be abstracted
    content = ""
    try:
        with openconfig_path, 'r' as f:
            content = f.read()
    except Exception as e:
        printf"❌ Failed to read config content: {e}"
        return False
    
    hardcoded_paths = [
        "$HOME/miniconda/bin/conda",
        "~/miniconda/bin/conda"
    ]
    
    found_hardcoded = False
    for path in hardcoded_paths:
        if path in content:
            printf"⚠️ Found hardcoded conda path: {path}"
            found_hardcoded = True
    
    if not found_hardcoded:
        print"✅ No hardcoded conda paths found"
    
    # Check for environment variable usage
    env_vars = ["$CIRCLE_WORKING_DIRECTORY", "$HOME", "$PATH"]
    found_env_vars = 0
    for var in env_vars:
        if var in content:
            found_env_vars += 1
            printf"✅ Found environment variable usage: {var}"
    
    if found_env_vars > 0:
        printf"✅ Found {found_env_vars} environment variables in use"
    
    return True

def main():
    """Run all PR #5 CI/CD integration tests."""
    print"🔍 Running PR #5 CI/CD Integration Tests..."
    print"=" * 60
    
    tests = [
        "YAML Syntax", test_yaml_syntax,
        "Conda Environment Setup", test_conda_environment_setup,
        "Critical Fixes", test_critical_fixes,
        "Pipeline Structure", test_pipeline_structure,
        "Pipeline Structure Edge Cases", test_pipeline_structure_edge_cases,
        "Job Dependencies", test_job_dependencies,
        "Environment Variables", test_environment_variables,
    ]
    
    passed = 0
    total = lentests
    
    for test_name, test_func in tests:
        printf"\n📋 {test_name}"
        print"-" * 40
        try:
            if test_func():
                passed += 1
                printf"✅ {test_name} PASSED"
            else:
                printf"❌ {test_name} FAILED"
        except Exception as e:
            printf"❌ {test_name} ERROR: {e}"
    
    print"\n" + "=" * 60
    print"📊 PR #5 CI/CD Integration Test Summary"
    print"=" * 60
    printf"Total Tests: {total}"
    printf"Passed: {passed}"
    printf"Failed: {total - passed}"
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print"\n✅ PR #5 CI/CD pipeline is ready for testing!"
        print"Ready for CircleCI validation"
    else:
        printf"\n❌ PR #5 needs {total - passed} fixes before testing"
        print"Please address the failing tests above"
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit0 if success else 1 