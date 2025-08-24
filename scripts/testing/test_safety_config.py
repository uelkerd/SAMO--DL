#!/usr/bin/env python3
"""
Test script to validate Safety CLI v3 configuration
Verifies that our .safety-project.ini and .safety-policy.yml eliminate interactive prompts
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_safety_config():
    """Test Safety CLI v3 configuration for CI compatibility"""
    
    # Check if required files exist
    project_ini = Path(".safety-project.ini")
    policy_yml = Path(".safety-policy.yml")
    
    print("🔍 Checking Safety configuration files...")
    
    if not project_ini.exists():
        print("❌ .safety-project.ini not found")
        return False
    else:
        print("✅ .safety-project.ini found")
    
    if not policy_yml.exists():
        print("❌ .safety-policy.yml not found")
        return False
    else:
        print("✅ .safety-policy.yml found")
    
    # Test basic safety command availability
    try:
        result = subprocess.run(
            ["safety", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"✅ Safety CLI available: {result.stdout.strip()}")
        else:
            print(f"⚠️ Safety CLI available but returned non-zero: {result.stderr}")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("⚠️ Safety CLI not installed or not in PATH")
        return False
    
    # Test safety scan with policy (dry-run style)
    print("\n🧪 Testing Safety scan with pre-configured settings...")
    
    try:
        # Create artifacts directory
        os.makedirs("artifacts/security", exist_ok=True)
        
        # Test safety scan command (with continue on error to capture output)
        result = subprocess.run([
            "safety", "scan",
            "--policy-file", ".safety-policy.yml",
            "--output", "json",
            "--output-file", "artifacts/security/safety-test.json",
            "--continue-on-vulnerability-error"
        ], capture_output=True, text=True, timeout=60)
        
        print(f"Safety scan exit code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT: {result.stdout[:500]}...")
        if result.stderr:
            print(f"STDERR: {result.stderr[:500]}...")
            
        # Check if interactive prompts appeared
        if "Enter a name for this codebase" in result.stdout or "Enter a name for this codebase" in result.stderr:
            print("❌ Interactive prompts detected - configuration failed")
            return False
        else:
            print("✅ No interactive prompts detected")
            
        # Check if output file was created
        output_file = Path("artifacts/security/safety-test.json")
        if output_file.exists():
            print("✅ Safety report generated successfully")
            
            # Try to parse JSON output
            try:
                with open(output_file) as f:
                    report_data = json.load(f)
                print(f"✅ Valid JSON report with {len(report_data.get('vulnerabilities', []))} vulnerabilities")
            except json.JSONDecodeError:
                print("⚠️ Report file exists but contains invalid JSON")
        else:
            print("⚠️ No safety report generated")
            
    except subprocess.TimeoutExpired:
        print("❌ Safety scan timed out (possible interactive prompt)")
        return False
    except subprocess.SubprocessError as e:
        print(f"❌ Safety scan failed: {e}")
        return False
    
    print("\n✅ Safety configuration test completed")
    return True


def main():
    """Main test function"""
    print("🛡️ SAMO-DL Safety CLI v3 Configuration Test")
    print("=" * 50)
    
    if test_safety_config():
        print("\n🎉 All tests passed - Safety configuration is ready for CI")
        sys.exit(0)
    else:
        print("\n❌ Configuration issues detected")
        sys.exit(1)


if __name__ == "__main__":
    main()