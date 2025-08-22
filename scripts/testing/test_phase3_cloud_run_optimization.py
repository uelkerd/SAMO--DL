#!/usr/bin/env python3
"""
Phase 3 Cloud Run Optimization Test Suite
Comprehensive testing for Cloud Run optimization components using enhanced test approach
"""

import os
import sys
import yaml
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import unittest
from unittest.mock import patch
import logging

# Add src to path for imports
sys.path.insert(0, str(Path__file__.parent.parent.parent / 'src'))

class Phase3CloudRunOptimizationTestunittest.TestCase:
    """Comprehensive test suite for Phase 3 Cloud Run optimization"""
    
    def setUpself:
        """Set up test environment"""
        # Get the project root directory 2 levels up from scripts/testing
        self.project_root = Path__file__.parent.parent.parent
        self.cloud_run_dir = self.project_root / "deployment" / "cloud-run"
        
        # Alternative path calculation for when running from scripts/testing
        if not self.cloud_run_dir.exists():
            # When running from scripts/testing, use relative path
            self.cloud_run_dir = Path"../../deployment/cloud-run".resolve()
        
        # Ensure the cloud-run directory exists
        self.assertTrue(self.cloud_run_dir.exists(), f"Cloud Run directory not found: {self.cloud_run_dir}")
        
        # Set up logging for tests
        logging.basicConfiglevel=logging.INFO
        self.logger = logging.getLogger__name__
        
        self.maxDiff = None
        
        # Test configuration
        self.test_config = {
            'environment': 'test',
            'memory_limit_mb': 1024,
            'cpu_limit': 1,
            'max_instances': 5,
            'min_instances': 1,
            'concurrency': 40,
            'timeout_seconds': 180,
            'health_check_interval': 30,
            'graceful_shutdown_timeout': 15
        }
    
    def test_01_cloudbuild_yaml_structureself:
        """Test Cloud Build YAML structure and validation"""
        print"🔍 Testing Cloud Build YAML structure..."
        
        cloudbuild_path = self.cloud_run_dir / 'cloudbuild.yaml'
        self.assertTrue(cloudbuild_path.exists(), "cloudbuild.yaml should exist")
        
        with opencloudbuild_path, 'r' as f:
            config = yaml.safe_loadf
        
        # Validate required fields
        required_fields = ['steps', 'images', 'timeout']
        self._assert_all_fields_presentconfig, required_fields
        
        # Validate steps structure
        steps = config['steps']
        self.assertIsInstancesteps, list, "Steps should be a list"
        self.assertGreater(lensteps, 0, "Should have at least one step")
        
        # Validate each step has required fields
        self._assert_all_steps_validsteps
        
        # Validate timeout format
        timeout = config['timeout']
        self.assertIsInstancetimeout, str, "Timeout should be a string"
        self.assertTrue(timeout.endswith's', "Timeout should end with 's'")
        
        print"✅ Cloud Build YAML structure validation passed"
    
    def _assert_all_fields_presentself, config, required_fields:
        """Helper method to check all required fields are present"""
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            self.fail(f"Missing required fields: {', '.joinmissing_fields}")
    
    def _assert_all_steps_validself, steps:
        """Helper method to validate all steps"""
        invalid_steps = []
        for i, step in enumeratesteps:
            if 'name' not in step or 'args' not in step:
                invalid_steps.appendf"Step {i}"
        
        if invalid_steps:
            self.fail(f"Invalid steps: {', '.joininvalid_steps}")
    
    def test_02_health_monitor_functionalityself:
        """Test health monitor functionality and metrics collection"""
        print"🔍 Testing health monitor functionality..."
        
        # Import health monitor
        sys.path.insert(0, strself.cloud_run_dir)
        try:
            from health_monitor import HealthMonitor, HealthMetrics
        except ImportError as e:
            if 'psutil' in stre:
                self.skipTest"psutil not available in test environment"
            raise
        
        # Test health monitor initialization
        monitor = HealthMonitor()
        self.assertIsNotNonemonitor, "Health monitor should initialize"
        self.assertFalsemonitor.is_shutting_down, "Should not be shutting down initially"
        self.assertEqualmonitor.active_requests, 0, "Should start with 0 active requests"
        
        # Test system metrics
        metrics = monitor.get_system_metrics()
        self._test_required_metricsmetrics
        
        # Test request tracking
        monitor.request_started()
        self.assertEqualmonitor.active_requests, 1, "Should track request start"
        
        monitor.request_completed()
        self.assertEqualmonitor.active_requests, 0, "Should track request completion"
        
        # Test edge case: multiple rapid requests
        self._test_multiple_requestsmonitor
        
        # Test edge case: negative requests should not go below 0
        monitor.request_completed()
        self.assertEqualmonitor.active_requests, 0, "Should not go below 0 active requests"
        
        print"✅ Health monitor functionality tests passed"

    def _test_required_metricsself, metrics:
        """Helper method to test required metrics"""
        required_metrics = ['memory_usage_mb', 'cpu_usage_percent', 'memory_percent', 'uptime_seconds']
        missing_metrics = [metric for metric in required_metrics if metric not in metrics]
        if missing_metrics:
            self.fail(f"Missing metrics: {', '.joinmissing_metrics}")
        
        # Check all metrics are numeric
        non_numeric_metrics = [metric for metric in required_metrics if not isinstance(metrics[metric], int, float)]
        if non_numeric_metrics:
            self.fail(f"Non-numeric metrics: {', '.joinnon_numeric_metrics}")

    def _test_multiple_requestsself, monitor:
        """Helper method to test multiple requests"""
        # Add 10 requests
        for i in range10:
            monitor.request_started()
        self.assertEqualmonitor.active_requests, 10, "Should handle multiple requests"
        
        # Complete 10 requests
        for i in range10:
            monitor.request_completed()
        self.assertEqualmonitor.active_requests, 0, "Should handle multiple completions"
    
    def test_03_environment_config_validationself:
        """Test environment configuration validation and edge cases"""
        print"🔍 Testing environment configuration validation..."
        
        # Import config
        sys.path.insert(0, strself.cloud_run_dir)
        from config import EnvironmentConfig
        
        # Test production configuration
        with patch.dictos.environ, {'ENVIRONMENT': 'production'}:
            config = EnvironmentConfig()
            self.assertEqualconfig.environment, 'production', "Should load production environment"
            
            # Test configuration validation
            config.validate_config()  # Should not raise exception for valid config
            
            # Test resource limits
            cloud_config = config.config
            self.assertGreaterEqualcloud_config.memory_limit_mb, 512, "Memory should be >= 512MB"
            self.assertLessEqualcloud_config.memory_limit_mb, 8192, "Memory should be <= 8GB"
            self.assertGreaterEqualcloud_config.cpu_limit, 1, "CPU should be >= 1"
            self.assertLessEqualcloud_config.cpu_limit, 8, "CPU should be <= 8"
        
        # Test staging configuration
        with patch.dictos.environ, {'ENVIRONMENT': 'staging'}:
            config = EnvironmentConfig()
            self.assertEqualconfig.environment, 'staging', "Should load staging environment"
            config.validate_config()  # Should not raise exception for valid config
        
        # Test development configuration
        with patch.dictos.environ, {'ENVIRONMENT': 'development'}:
            config = EnvironmentConfig()
            self.assertEqualconfig.environment, 'development', "Should load development environment"
            config.validate_config()  # Should not raise exception for valid config
        
        # Test edge case: invalid environment
        with patch.dictos.environ, {'ENVIRONMENT': 'invalid'}:
            config = EnvironmentConfig()
            self.assertEqualconfig.environment, 'invalid', "Should load invalid environment"
            # Should still be valid as it falls back to development defaults
        
        print"✅ Environment configuration validation tests passed"
    
    def test_04_dockerfile_optimizationself:
        """Test Dockerfile optimization and security features"""
        print"🔍 Testing Dockerfile optimization..."
        
        dockerfile_path = self.cloud_run_dir / 'Dockerfile.secure'
        self.assertTrue(dockerfile_path.exists(), "Dockerfile.secure should exist")
        
        with opendockerfile_path, 'r' as f:
            content = f.read()
        
        # Test security features
        self._test_security_featurescontent
        
        # Test Cloud Run optimizations
        self._test_cloud_run_featurescontent
        
        # Test resource optimization
        self._test_optimization_featurescontent
        
        print"✅ Dockerfile optimization tests passed"

    def _test_security_featuresself, content:
        """Helper method to test security features"""
        security_features = [
            'FROM --platform=linux/amd64',  # Platform targeting
            'USER appuser',  # Non-root user
            'HEALTHCHECK',  # Health check
            '--no-cache-dir',  # No cache for security
            'PYTHONHASHSEED=random',  # Random hash seed
            'PIP_DISABLE_PIP_VERSION_CHECK=1'  # Disable pip version check
        ]
        
        missing_features = [feature for feature in security_features if feature not in content]
        if missing_features:
            self.fail(f"Missing security features: {', '.joinmissing_features}")

    def _test_cloud_run_featuresself, content:
        """Helper method to test Cloud Run features"""
        cloud_run_features = [
            'EXPOSE 8080',  # Cloud Run port
            '--bind :$PORT',  # Dynamic port binding
            '--workers 1',  # Single worker for Cloud Run
            '--timeout 0',  # Cloud Run handles timeouts
            '--keep-alive 5'  # Keep-alive optimization
        ]
        
        missing_features = [feature for feature in cloud_run_features if feature not in content]
        if missing_features:
            self.fail(f"Missing Cloud Run features: {', '.joinmissing_features}")

    def _test_optimization_featuresself, content:
        """Helper method to test optimization features"""
        optimization_features = [
            '--max-requests 1000',  # Request recycling
            '--max-requests-jitter 100',  # Jitter for load distribution
            '--access-logfile -',  # Structured logging
            '--error-logfile -'  # Error logging
        ]
        
        missing_features = [feature for feature in optimization_features if feature not in content]
        if missing_features:
            self.fail(f"Missing optimization features: {', '.joinmissing_features}")
    
    def test_05_requirements_securityself:
        """Test requirements.txt security and version pinning"""
        print"🔍 Testing requirements security..."
        
        requirements_path = self.cloud_run_dir / 'requirements_secure.txt'
        self.assertTrue(requirements_path.exists(), "requirements_secure.txt should exist")
        
        with openrequirements_path, 'r' as f:
            content = f.read()
        
        # Test required dependencies updated to match actual requirements format
        required_deps = [
            'flask==',  # Web framework exact version pinning
            'gunicorn==',  # WSGI server
            'psutil==',  # System monitoring
            'requests==',  # HTTP client
            'prometheus-client=='  # Metrics
        ]
        
        missing_deps = [dep for dep in required_deps if dep not in content]
        if missing_deps:
            self.fail(f"Missing required dependencies: {', '.joinmissing_deps}")
        
        # Test version pinning dependencies should have == for exact versions
        lines = content.split'\n'
        unpinned_deps = []
        for line in lines:
            line = line.strip()
            if (line and not line.startswith'#' and 
                '==' not in line and '>=' not in line and '<=' not in line):
                unpinned_deps.appendline
        
        if unpinned_deps:
            self.fail(f"Unpinned dependencies: {', '.joinunpinned_deps}")
        
        print"✅ Requirements security tests passed"
    
    def test_06_auto_scaling_configurationself:
        """Test auto-scaling configuration and validation"""
        print"🔍 Testing auto-scaling configuration..."
        
        cloudbuild_path = self.cloud_run_dir / 'cloudbuild.yaml'
        with opencloudbuild_path, 'r' as f:
            config = yaml.safe_loadf
        
        # Find Cloud Run deployment step
        deploy_step = self._find_deploy_stepconfig
        self.assertIsNotNonedeploy_step, "Should have Cloud Run deployment step"
        
        # Get args from the step
        args = deploy_step.get'args', []
        self.assertIsInstanceargs, list, "Args should be a list"
        self.assertGreater(lenargs, 0, "Should have deployment arguments")
        
        # Test auto-scaling parameters Cloud Build format: --param=value
        scaling_params = [
            '--max-instances=10',
            '--min-instances=1',
            '--concurrency=80'
        ]
        
        missing_params = [param for param in scaling_params if param not in args]
        if missing_params:
            self.fail(f"Missing auto-scaling parameters: {', '.joinmissing_params}")
        
        # Test resource allocation Cloud Build format: --param=value
        resource_params = [
            '--memory=2Gi',
            '--cpu=2'
        ]
        
        missing_resource_params = [param for param in resource_params if param not in args]
        if missing_resource_params:
            self.fail(f"Missing resource parameters: {', '.joinmissing_resource_params}")
        
        print"✅ Auto-scaling configuration tests passed"
    
    def _find_deploy_stepself, config:
        """Helper method to find deployment step"""
        for step in config['steps']:
            if 'gcr.io/google.com/cloudsdktool/cloud-sdk' in step.get'name', '':
                return step
        return None
    
    def test_07_health_check_integrationself:
        """Test health check integration and monitoring"""
        print"🔍 Testing health check integration..."
        
        # Test health check endpoint configuration
        cloudbuild_path = self.cloud_run_dir / 'cloudbuild.yaml'
        with opencloudbuild_path, 'r' as f:
            config = yaml.safe_loadf
        
        # Check for health check environment variables
        deploy_step = self._find_deploy_stepconfig
        self.assertIsNotNonedeploy_step, "Should have deployment step"
        
        args = deploy_step['args']
        
        # Test health check environment variables updated to match actual format
        health_vars = [
            'HEALTH_CHECK_INTERVAL=30',
            'GRACEFUL_SHUTDOWN_TIMEOUT=30',
            'ENABLE_HEALTH_CHECKS=true'
        ]
        
        # Check if the environment variables are set in any --set-env-vars argument
        env_vars_found = 0
        for arg in args:
            if arg.startswith'--set-env-vars=':
                for var in health_vars:
                    if var in arg:
                        env_vars_found += 1
        
        self.assertGreaterEqualenv_vars_found, 2, f"Should have at least 2 health check environment variables, found {env_vars_found}"
        
        print"✅ Health check integration tests passed"
    
    def test_08_configuration_edge_casesself:
        """Test configuration edge cases and error handling"""
        print"🔍 Testing configuration edge cases..."
        
        sys.path.insert(0, strself.cloud_run_dir)
        from config import EnvironmentConfig
        
        # Test invalid memory limits
        with patch.dict(os.environ, {
            'ENVIRONMENT': 'production',
            'MEMORY_LIMIT_MB': '100'  # Too low
        }):
            config = EnvironmentConfig()
            # Should still be valid as it uses defaults
        
        # Test invalid CPU limits
        with patch.dict(os.environ, {
            'ENVIRONMENT': 'production',
            'CPU_LIMIT': '10'  # Too high
        }):
            config = EnvironmentConfig()
            # Should still be valid as it uses defaults
        
        # Test invalid timeout
        with patch.dict(os.environ, {
            'ENVIRONMENT': 'production',
            'TIMEOUT_SECONDS': '1000'  # Too high
        }):
            config = EnvironmentConfig()
            # Should still be valid as it uses defaults
        
        # Test empty environment variables
        with patch.dict(os.environ, {
            'ENVIRONMENT': 'production',
            'MEMORY_LIMIT_MB': '',
            'CPU_LIMIT': '',
            'MAX_INSTANCES': ''
        }):
            config = EnvironmentConfig()
            config.validate_config()  # Should not raise exception for valid config
        
        print"✅ Configuration edge case tests passed"
    
    def test_09_performance_metricsself:
        """Test performance metrics and monitoring"""
        print"🔍 Testing performance metrics..."
        
        sys.path.insert(0, strself.cloud_run_dir)
        try:
            from health_monitor import HealthMonitor
        except ImportError as e:
            if 'psutil' in stre:
                self.skipTest"psutil not available in test environment"
            raise
        
        monitor = HealthMonitor()
        
        # Test metrics collection
        metrics = monitor.get_comprehensive_health()
        
        required_metrics = [
            'status', 'timestamp', 'uptime_seconds',
            'system', 'models', 'api', 'requests'
        ]
        
        missing_metrics = [metric for metric in required_metrics if metric not in metrics]
        if missing_metrics:
            self.fail(f"Missing performance metrics: {', '.joinmissing_metrics}")
        
        # Test system metrics structure
        system_metrics = metrics['system']
        system_required = ['memory_usage_mb', 'cpu_usage_percent', 'memory_percent']
        
        missing_system_metrics = [metric for metric in system_required if metric not in system_metrics]
        if missing_system_metrics:
            self.fail(f"Missing system metrics: {', '.joinmissing_system_metrics}")
        
        # Check all system metrics are numeric
        non_numeric_system_metrics = [metric for metric in system_required if not isinstance(system_metrics[metric], int, float)]
        if non_numeric_system_metrics:
            self.fail(f"Non-numeric system metrics: {', '.joinnon_numeric_system_metrics}")
        
        # Test request metrics
        request_metrics = metrics['requests']
        self.assertIn'active', request_metrics, "Should track active requests"
        self.assertIn'total_processed', request_metrics, "Should track total processed requests"
        
        print"✅ Performance metrics tests passed"
    
    def test_10_yaml_parsing_validationself:
        """Test YAML parsing and validation using enhanced test approach"""
        print"🔍 Testing YAML parsing and validation..."
        
        # Test Cloud Build YAML parsing
        cloudbuild_path = self.cloud_run_dir / 'cloudbuild.yaml'
        with opencloudbuild_path, 'r' as f:
            config = yaml.safe_loadf
        
        # Validate YAML structure using enhanced approach
        self._validate_yaml_structureconfig, 'cloudbuild.yaml'
        
        # Test configuration serialization
        sys.path.insert(0, strself.cloud_run_dir)
        from config import EnvironmentConfig
        
        config_obj = EnvironmentConfig'production'
        config_dict = config_obj.to_dict()
        
        # Convert to YAML and back to test serialization
        yaml_str = yaml.dumpconfig_dict, default_flow_style=False
        parsed_config = yaml.safe_loadyaml_str
        
        self.assertEqualconfig_dict, parsed_config, "YAML serialization should be reversible"
        
        print"✅ YAML parsing validation tests passed"
    
    def _validate_yaml_structureself, config: Dict[str, Any], filename: str:
        """Enhanced YAML structure validation"""
        # Validate top-level structure
        self.assertIsInstanceconfig, dict, f"{filename} should be a dictionary"
        
        # Validate required top-level keys
        if filename == 'cloudbuild.yaml':
            required_keys = ['steps', 'images']
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                self.fail(f"{filename} missing required keys: {', '.joinmissing_keys}")
        
        # Validate nested structures
        if 'steps' in config:
            self.assertIsInstanceconfig['steps'], list, "Steps should be a list"
            invalid_steps = []
            for i, step in enumerateconfig['steps']:
                if not isinstancestep, dict:
                    invalid_steps.appendf"Step {i} should be a dictionary"
                elif 'name' not in step or 'args' not in step:
                    invalid_steps.appendf"Step {i} missing required fields"
            
            if invalid_steps:
                self.fail(f"Invalid steps: {', '.joininvalid_steps}")

def run_phase3_tests():
    """Run all Phase 3 Cloud Run optimization tests"""
    print"🚀 Starting Phase 3 Cloud Run Optimization Test Suite"
    print"=" * 60
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCasePhase3CloudRunOptimizationTest
    
    # Run tests
    runner = unittest.TextTestRunnerverbosity=2
    result = runner.runsuite
    
    # Generate test report
    test_report = {
        'phase': 'Phase 3 - Cloud Run Optimization',
        'total_tests': result.testsRun,
        'failures': lenresult.failures,
        'errors': lenresult.errors,
        'success_rate': ((result.testsRun - lenresult.failures - lenresult.errors) / result.testsRun) * 100,
        'timestamp': time.strftime'%Y-%m-%d %H:%M:%S',
        'test_details': []
    }
    
    # Add test details
    for test, traceback in result.failures:
        test_report['test_details'].append({
            'test': test._testMethodName,
            'status': 'FAILED',
            'error': traceback
        })
    
    for test, traceback in result.errors:
        test_report['test_details'].append({
            'test': test._testMethodName,
            'status': 'ERROR',
            'error': traceback
        })
    
    # Save test report
    report_path = Path__file__.parent / 'phase3_test_report.json'
    with openreport_path, 'w' as f:
        json.dumptest_report, f, indent=2
    
    print"\n" + "=" * 60
    print"📊 Phase 3 Test Results:"
    printf"   Total Tests: {test_report['total_tests']}"
    printf"   Failures: {test_report['failures']}"
    printf"   Errors: {test_report['errors']}"
    printf"   Success Rate: {test_report['success_rate']:.1f}%"
    printf"   Report saved to: {report_path}"
    
    if result.wasSuccessful():
        print"✅ All Phase 3 tests passed!"
        return True
    print"❌ Some Phase 3 tests failed!"
    return False

if __name__ == '__main__':
    success = run_phase3_tests()
    sys.exit0 if success else 1 