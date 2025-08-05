#!/usr/bin/env python3
"""
Integration Tests for PR #4: Documentation & Security Enhancements

This script validates that the security configurations and documentation
implemented in PR #4 are properly integrated and functional.
"""

import os
import sys
import yaml
import json
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Any

class PR4IntegrationTester:
    """Integration tester for PR #4 security and documentation enhancements."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.security_config_path = self.project_root / "configs" / "security.yaml"
        self.openapi_spec_path = self.project_root / "docs" / "api" / "openapi.yaml"
        self.requirements_path = self.project_root / "requirements.txt"
        self.test_results = []
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests for PR #4."""
        print("🔍 Running PR #4 Integration Tests...")
        
        tests = [
            self.test_security_configuration,
            self.test_openapi_specification,
            self.test_dependencies_security,
            self.test_documentation_completeness,
            self.test_security_scanning_tools
        ]
        
        for test in tests:
            try:
                result = test()
                self.test_results.append(result)
                status = "✅ PASS" if result["passed"] else "❌ FAIL"
                print(f"{status} {result['name']}: {result['message']}")
            except Exception as e:
                error_result = {
                    "name": test.__name__,
                    "passed": False,
                    "message": f"Test failed with exception: {str(e)}",
                    "details": str(e)
                }
                self.test_results.append(error_result)
                print(f"❌ FAIL {test.__name__}: {str(e)}")
        
        return self.generate_summary()
    
    def test_security_configuration(self) -> Dict[str, Any]:
        """Test that security configuration is valid and complete."""
        if not self.security_config_path.exists():
            return {
                "name": "Security Configuration",
                "passed": False,
                "message": "Security configuration file not found",
                "details": f"Expected: {self.security_config_path}"
            }
        
        try:
            with open(self.security_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ['api', 'security_headers', 'logging', 'environment']
            missing_sections = [section for section in required_sections if section not in config]
            
            if missing_sections:
                return {
                    "name": "Security Configuration",
                    "passed": False,
                    "message": f"Missing required sections: {missing_sections}",
                    "details": f"Found sections: {list(config.keys())}"
                }
            
            # Check API security settings
            api_config = config.get('api', {})
            if not api_config.get('rate_limiting', {}).get('enabled'):
                return {
                    "name": "Security Configuration",
                    "passed": False,
                    "message": "Rate limiting not enabled in API configuration",
                    "details": "Rate limiting is required for production security"
                }
            
            return {
                "name": "Security Configuration",
                "passed": True,
                "message": "Security configuration is valid and complete",
                "details": f"All {len(required_sections)} required sections present"
            }
            
        except yaml.YAMLError as e:
            return {
                "name": "Security Configuration",
                "passed": False,
                "message": f"Invalid YAML in security configuration: {str(e)}",
                "details": str(e)
            }
    
    def test_openapi_specification(self) -> Dict[str, Any]:
        """Test that OpenAPI specification is valid and complete."""
        if not self.openapi_spec_path.exists():
            return {
                "name": "OpenAPI Specification",
                "passed": False,
                "message": "OpenAPI specification file not found",
                "details": f"Expected: {self.openapi_spec_path}"
            }
        
        try:
            with open(self.openapi_spec_path, 'r') as f:
                spec = yaml.safe_load(f)
            
            # Check OpenAPI version
            if spec.get('openapi') != '3.1.0':
                return {
                    "name": "OpenAPI Specification",
                    "passed": False,
                    "message": "OpenAPI version should be 3.1.0",
                    "details": f"Found version: {spec.get('openapi')}"
                }
            
            # Check required sections
            required_sections = ['info', 'paths', 'components']
            missing_sections = [section for section in required_sections if section not in spec]
            
            if missing_sections:
                return {
                    "name": "OpenAPI Specification",
                    "passed": False,
                    "message": f"Missing required sections: {missing_sections}",
                    "details": f"Found sections: {list(spec.keys())}"
                }
            
            # Check security definitions
            if 'security' not in spec:
                return {
                    "name": "OpenAPI Specification",
                    "passed": False,
                    "message": "Security definitions missing",
                    "details": "API security should be documented"
                }
            
            return {
                "name": "OpenAPI Specification",
                "passed": True,
                "message": "OpenAPI specification is valid and complete",
                "details": f"Version {spec.get('openapi')} with all required sections"
            }
            
        except yaml.YAMLError as e:
            return {
                "name": "OpenAPI Specification",
                "passed": False,
                "message": f"Invalid YAML in OpenAPI specification: {str(e)}",
                "details": str(e)
            }
    
    def test_dependencies_security(self) -> Dict[str, Any]:
        """Test that dependencies are secure and up-to-date."""
        if not self.requirements_path.exists():
            return {
                "name": "Dependencies Security",
                "passed": False,
                "message": "Requirements file not found",
                "details": f"Expected: {self.requirements_path}"
            }
        
        try:
            with open(self.requirements_path, 'r') as f:
                requirements = f.read()
            
            # Check for security scanning tools
            security_tools = ['bandit', 'safety']
            missing_tools = [tool for tool in security_tools if tool not in requirements]
            
            if missing_tools:
                return {
                    "name": "Dependencies Security",
                    "passed": False,
                    "message": f"Missing security scanning tools: {missing_tools}",
                    "details": "Security tools are required for vulnerability scanning"
                }
            
            # Check for critical security packages
            # The list of critical security packages is loaded from security.yaml under the 'critical_packages' key.
            # These packages are considered critical because:
            #   - cryptography: Provides secure cryptographic primitives for encryption, hashing, etc.
            #   - certifi: Ensures up-to-date CA certificates for secure HTTPS connections.
            #   - urllib3: Secure HTTP client with robust TLS/SSL support.
            try:
                with open(self.security_config_path, 'r') as secf:
                    security_config = yaml.safe_load(secf)
                critical_packages = security_config.get('critical_packages', ['cryptography', 'certifi', 'urllib3'])
                if 'critical_packages' not in security_config:
                    print("⚠️ Warning: 'critical_packages' not found in security.yaml, using default list.")
            except Exception as e:
                print(f"⚠️ Warning: Could not read security.yaml for critical_packages: {str(e)}. Using default list.")
                critical_packages = ['cryptography', 'certifi', 'urllib3']
            missing_critical = [pkg for pkg in critical_packages if pkg not in requirements]
            
            if missing_critical:
                return {
                    "name": "Dependencies Security",
                    "passed": False,
                    "message": f"Missing critical security packages: {missing_critical}",
                    "details": "Critical security packages are required"
                }
            
            return {
                "name": "Dependencies Security",
                "passed": True,
                "message": "Dependencies include required security packages",
                "details": f"All {len(security_tools)} security tools and {len(critical_packages)} critical packages present"
            }
            
        except Exception as e:
            return {
                "name": "Dependencies Security",
                "passed": False,
                "message": f"Error reading requirements file: {str(e)}",
                "details": str(e)
            }
    
    def test_documentation_completeness(self) -> Dict[str, Any]:
        """Test that documentation is complete and accessible."""
        required_docs = [
            "docs/deployment/PRODUCTION_DEPLOYMENT_GUIDE.md",
            "CONTRIBUTING.md",
            "docs/monster-pr-8-breakdown-strategy.md"
        ]
        
        missing_docs = []
        for doc_path in required_docs:
            if not (self.project_root / doc_path).exists():
                missing_docs.append(doc_path)
        
        if missing_docs:
            return {
                "name": "Documentation Completeness",
                "passed": False,
                "message": f"Missing required documentation: {missing_docs}",
                "details": "All required documentation should be present"
            }
        
        return {
            "name": "Documentation Completeness",
            "passed": True,
            "message": "All required documentation is present",
            "details": f"Found {len(required_docs)} required documentation files"
        }
    
    def test_security_scanning_tools(self) -> Dict[str, Any]:
        """Test that security scanning tools are available and functional."""
        try:
            # Test bandit availability
            bandit_path = shutil.which('bandit')
            if not bandit_path:
                return {
                    "name": "Security Scanning Tools",
                    "passed": False,
                    "message": "Bandit security scanner not found in PATH",
                    "details": "Install bandit: pip install bandit"
                }
            result = subprocess.run([bandit_path, '--version'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return {
                    "name": "Security Scanning Tools",
                    "passed": False,
                    "message": "Bandit security scanner not available",
                    "details": f"Bandit error: {result.stderr}"
                }
            
            # Test safety availability
            safety_path = shutil.which('safety')
            if safety_path is None:
                return {
                    "name": "Security Scanning Tools",
                    "passed": False,
                    "message": "Safety vulnerability scanner not found in PATH",
                    "details": "Install safety and ensure it is in a secure location"
                }
            result = subprocess.run([safety_path, '--version'],
                                  capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return {
                    "name": "Security Scanning Tools",
                    "passed": False,
                    "message": "Safety vulnerability scanner not available",
                    "details": f"Safety error: {result.stderr}"
                }
            
            return {
                "name": "Security Scanning Tools",
                "passed": True,
                "message": "Security scanning tools are available and functional",
                "details": "Bandit and Safety scanners are working"
            }
            
        except subprocess.TimeoutExpired:
            return {
                "name": "Security Scanning Tools",
                "passed": False,
                "message": "Security scanning tools timed out",
                "details": "Tools may not be properly installed"
            }
        except FileNotFoundError:
            return {
                "name": "Security Scanning Tools",
                "passed": False,
                "message": "Security scanning tools not found",
                "details": "Install bandit and safety: pip install bandit safety"
            }
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate test summary and recommendations."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["passed"])
        failed_tests = total_tests - passed_tests
        
        summary = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            "results": self.test_results,
            "recommendations": []
        }
        
        # Generate recommendations based on failures
        if failed_tests > 0:
            summary["recommendations"].append(
                f"Fix {failed_tests} failing tests before proceeding"
            )
        
        if summary["success_rate"] < 100:
            summary["recommendations"].append(
                "Complete integration testing before claiming PR #4 is ready"
            )
        
        return summary

def main():
    """Main function to run PR #4 integration tests."""
    tester = PR4IntegrationTester()
    summary = tester.run_all_tests()
    
    print("\n" + "="*60)
    print("📊 PR #4 Integration Test Summary")
    print("="*60)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    
    if summary['recommendations']:
        print("\n🔧 Recommendations:")
        for rec in summary['recommendations']:
            print(f"  - {rec}")
    
    if summary['failed'] > 0:
        print("\n❌ PR #4 is NOT ready for submission")
        sys.exit(1)
    else:
        print("\n✅ PR #4 integration tests passed!")
        print("Ready for final review and submission")

if __name__ == "__main__":
    main() 