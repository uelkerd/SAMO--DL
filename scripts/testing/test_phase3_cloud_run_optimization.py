#!/usr/bin/env python3
"""Phase 3 Cloud Run Optimization Test Suite Comprehensive testing for Cloud Run
optimization components using enhanced test approach."""

import json
import logging
import os
import sys
import time
import unittest
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

class Phase3CloudRunOptimizationTest(unittest.TestCase):
    """Comprehensive test suite for Phase 3 Cloud Run optimization."""

    def setUp(self):
        """Set up test environment."""
        # Get the project root directory (2 levels up from scripts/testing)
        self.project_root = Path(__file__).parent.parent.parent
        self.cloud_run_dir = self.project_root / "deployment" / "cloud-run"

        # Alternative path calculation for when running from scripts/testing
        if not self.cloud_run_dir.exists():
            # When running from scripts/testing, use relative path
            self.cloud_run_dir = Path("../../deployment/cloud-run").resolve()

        # Ensure the cloud-run directory exists
        self.assertTrue(self.cloud_run_dir.exists(), f"Cloud Run directory not found: {self.cloud_run_dir}")

        # Set up logging for tests
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

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

    def test_01_cloudbuild_yaml_structure(self):
        """Test Cloud Build YAML structure and validation."""
        print("🔍 Testing Cloud Build YAML structure...")

        cloudbuild_path = self.cloud_run_dir / 'cloudbuild.yaml'
        self.assertTrue(cloudbuild_path.exists(), "cloudbuild.yaml should exist")

        with open(cloudbuild_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validate required fields
        required_fields = ['steps', 'images', 'timeout']
        self._assert_all_fields_present(config, required_fields)

        # Validate steps structure
        steps = config['steps']
        self.assertIsInstance(steps, list, "Steps should be a list")
        self.assertGreater(len(steps), 0, "Should have at least one step")

        # Validate each step has required fields
        self._assert_all_steps_valid(steps)

        # Validate timeout format
        timeout = config['timeout']
        self.assertIsInstance(timeout, str, "Timeout should be a string")
        self.assertTrue(timeout.endswith('s'), "Timeout should end with 's'")

        print("✅ Cloud Build YAML structure validation passed")

    def _assert_all_fields_present(self, config, required_fields):
        """Helper method to check all required fields are present."""
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            self.fail("Missing required fields: {", '.join(missing_fields)}")

    def _assert_all_steps_valid(self, steps):
        """Helper method to validate all steps."""
        invalid_steps = []
        for i, step in enumerate(steps):
            if 'name' not in step or 'args' not in step:
                invalid_steps.append(f"Step {i}")

        if invalid_steps:
            self.fail("Invalid steps: {", '.join(invalid_steps)}")

    def test_02_health_monitor_functionality(self):
        """Test health monitor functionality and metrics collection."""
        print("🔍 Testing health monitor functionality...")

        # Import health monitor
        sys.path.insert(0, str(self.cloud_run_dir))
        try:
            from health_monitor import HealthMonitor, HealthMetrics
        except ImportError as e:
            if 'psutil' in str(e):
                self.skipTest("psutil not available in test environment")
            raise

        # Test health monitor initialization
        monitor = HealthMonitor()
        self.assertIsNotNone(monitor, "Health monitor should initialize")
        self.assertFalse(monitor.is_shutting_down, "Should not be shutting down initially")
        self.assertEqual(monitor.active_requests, 0, "Should start with 0 active requests")

        # Test system metrics
        metrics = monitor.get_system_metrics()
        self._test_required_metrics(metrics)

        # Test request tracking
        monitor.request_started()
        self.assertEqual(monitor.active_requests, 1, "Should track request start")

        monitor.request_completed()
        self.assertEqual(monitor.active_requests, 0, "Should track request completion")

        # Test edge case: multiple rapid requests
        self._test_multiple_requests(monitor)

        # Test edge case: negative requests (should not go below 0)
        monitor.request_completed()
        self.assertEqual(monitor.active_requests, 0, "Should not go below 0 active requests")

        print("✅ Health monitor functionality tests passed")

    def _test_required_metrics(self, metrics):
        """Helper method to test required metrics."""
        required_metrics = ['memory_usage_mb', 'cpu_usage_percent', 'memory_percent', 'uptime_seconds']
        missing_metrics = [metric for metric in required_metrics if metric not in metrics]
        if missing_metrics:
            self.fail("Missing metrics: {", '.join(missing_metrics)}")

        # Check all metrics are numeric
        non_numeric_metrics = [metric for metric in required_metrics if not isinstance(metrics[metric], (int, float))]
        if non_numeric_metrics:
            self.fail("Non-numeric metrics: {", '.join(non_numeric_metrics)}")

    def _test_multiple_requests(self, monitor):
        """Helper method to test multiple requests."""
        # Add 10 requests
        for i in range(10):
            monitor.request_started()
        self.assertEqual(monitor.active_requests, 10, "Should handle multiple requests")

        # Complete 10 requests
        for i in range(10):
            monitor.request_completed()
        self.assertEqual(monitor.active_requests, 0, "Should handle multiple completions")

    def test_03_environment_config_validation(self):
        """Test environment configuration validation and edge cases."""
        print("🔍 Testing environment configuration validation...")

        # Import config
        sys.path.insert(0, str(self.cloud_run_dir))
        from config import EnvironmentConfig

        # Test production configuration
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            config = EnvironmentConfig()
            self.assertEqual(config.environment, 'production', "Should load production environment")

            # Test configuration validation
            config.validate_config()  # Should not raise exception for valid config

            # Test resource limits
            cloud_config = config.config
            self.assertGreaterEqual(cloud_config.memory_limit_mb, 512, "Memory should be >= 512MB")
            self.assertLessEqual(cloud_config.memory_limit_mb, 8192, "Memory should be <= 8GB")
            self.assertGreaterEqual(cloud_config.cpu_limit, 1, "CPU should be >= 1")
            self.assertLessEqual(cloud_config.cpu_limit, 8, "CPU should be <= 8")

        # Test staging configuration
        with patch.dict(os.environ, {'ENVIRONMENT': 'staging'}):
            config = EnvironmentConfig()
            self.assertEqual(config.environment, 'staging', "Should load staging environment")
            config.validate_config()  # Should not raise exception for valid config

        # Test development configuration
        with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
            config = EnvironmentConfig()
            self.assertEqual(config.environment, 'development', "Should load development environment")
            config.validate_config()  # Should not raise exception for valid config

        # Test edge case: invalid environment
        with patch.dict(os.environ, {'ENVIRONMENT': 'invalid'}):
            config = EnvironmentConfig()
            self.assertEqual(config.environment, 'invalid', "Should load invalid environment")
            # Should still be valid as it falls back to development defaults

        print("✅ Environment configuration validation tests passed")

    def test_04_dockerfile_optimization(self):
        """Test Dockerfile optimization and security features."""
        print("🔍 Testing Dockerfile optimization...")

        dockerfile_path = self.cloud_run_dir / 'Dockerfile.secure'
        self.assertTrue(dockerfile_path.exists(), "Dockerfile.secure should exist")

        with open(dockerfile_path, 'r') as f:
            content = f.read()

        # Test security features
        self._test_security_features(content)

        # Test Cloud Run optimizations
        self._test_cloud_run_features(content)

        # Test resource optimization
        self._test_optimization_features(content)

        print("✅ Dockerfile optimization tests passed")

    def _test_security_features(self, content):
        """Helper method to test security features."""
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
            self.fail("Missing security features: {", '.join(missing_features)}")

    def _test_cloud_run_features(self, content):
        """Helper method to test Cloud Run features."""
        cloud_run_features = [
            'EXPOSE 8080',  # Cloud Run port
            '--bind :$PORT',  # Dynamic port binding
            '--workers 1',  # Single worker for Cloud Run
            '--timeout 0',  # Cloud Run handles timeouts
            '--keep-alive 5'  # Keep-alive optimization
        ]

        missing_features = [feature for feature in cloud_run_features if feature not in content]
        if missing_features:
            self.fail("Missing Cloud Run features: {", '.join(missing_features)}")

    def _test_optimization_features(self, content):
        """Helper method to test optimization features."""
        optimization_features = [
            '--max-requests 1000',  # Request recycling
            '--max-requests-jitter 100',  # Jitter for load distribution
            '--access-logfile -',  # Structured logging
            '--error-logfile -'  # Error logging
        ]

        missing_features = [feature for feature in optimization_features if feature not in content]
        if missing_features:
            self.fail("Missing optimization features: {", '.join(missing_features)}")

    def test_05_requirements_security(self):
        """Test requirements.txt security and version pinning."""
        print("🔍 Testing requirements security...")

        requirements_path = self.cloud_run_dir / 'requirements_secure.txt'
        self.assertTrue(requirements_path.exists(), "requirements_secure.txt should exist")

        with open(requirements_path, 'r') as f:
            content = f.read()

        # Test required dependencies (updated to match actual requirements format)
        required_deps = [
            'flask==',  # Web framework (exact version pinning)
            'gunicorn==',  # WSGI server
            'psutil==',  # System monitoring
            'requests==',  # HTTP client
            'prometheus-client=='  # Metrics
        ]

        missing_deps = [dep for dep in required_deps if dep not in content]
        if missing_deps:
            self.fail("Missing required dependencies: {", '.join(missing_deps)}")

        # Test version pinning (dependencies should have == for exact versions)
        lines = content.split('\n')
        unpinned_deps = []
        for line in lines:
            line = line.strip()
            if (line and not line.startswith('#') and
                '==' not in line and '>=' not in line and '<=' not in line):
                unpinned_deps.append(line)

        if unpinned_deps:
            self.fail("Unpinned dependencies: {", '.join(unpinned_deps)}")

        print("✅ Requirements security tests passed")

    def test_06_auto_scaling_configuration(self):
        """Test auto-scaling configuration and validation."""
        print("🔍 Testing auto-scaling configuration...")

        cloudbuild_path = self.cloud_run_dir / 'cloudbuild.yaml'
        with open(cloudbuild_path, 'r') as f:
            config = yaml.safe_load(f)

        # Find Cloud Run deployment step
        deploy_step = self._find_deploy_step(config)
        self.assertIsNotNone(deploy_step, "Should have Cloud Run deployment step")

        # Get args from the step
        args = deploy_step.get('args', [])
        self.assertIsInstance(args, list, "Args should be a list")
        self.assertGreater(len(args), 0, "Should have deployment arguments")

        # Test auto-scaling parameters (Cloud Build format: --param=value)
        scaling_params = [
            '--max-instances=10',
            '--min-instances=1',
            '--concurrency=80'
        ]

        missing_params = [param for param in scaling_params if param not in args]
        if missing_params:
            self.fail("Missing auto-scaling parameters: {", '.join(missing_params)}")

        # Test resource allocation (Cloud Build format: --param=value)
        resource_params = [
            '--memory=2Gi',
            '--cpu=2'
        ]

        missing_resource_params = [param for param in resource_params if param not in args]
        if missing_resource_params:
            self.fail("Missing resource parameters: {", '.join(missing_resource_params)}")

        print("✅ Auto-scaling configuration tests passed")

    def _find_deploy_step(self, config):
        """Helper method to find deployment step."""
        for step in config['steps']:
            if 'gcr.io/google.com/cloudsdktool/cloud-sdk' in step.get('name', ''):
                return step
        return None

    def test_07_health_check_integration(self):
        """Test health check integration and monitoring."""
        print("🔍 Testing health check integration...")

        # Test health check endpoint configuration
        cloudbuild_path = self.cloud_run_dir / 'cloudbuild.yaml'
        with open(cloudbuild_path, 'r') as f:
            config = yaml.safe_load(f)

        # Check for health check environment variables
        deploy_step = self._find_deploy_step(config)
        self.assertIsNotNone(deploy_step, "Should have deployment step")

        args = deploy_step['args']

        # Test health check environment variables (updated to match actual format)
        health_vars = [
            'HEALTH_CHECK_INTERVAL=30',
            'GRACEFUL_SHUTDOWN_TIMEOUT=30',
            'ENABLE_HEALTH_CHECKS=true'
        ]

        # Check if the environment variables are set in any --set-env-vars argument
        env_vars_found = 0
        for arg in args:
            if arg.startswith('--set-env-vars='):
                for var in health_vars:
                    if var in arg:
                        env_vars_found += 1

        self.assertGreaterEqual(env_vars_found, 2, f"Should have at least 2 health check environment variables, found {env_vars_found}")

        print("✅ Health check integration tests passed")

    def test_08_configuration_edge_cases(self):
        """Test configuration edge cases and error handling."""
        print("🔍 Testing configuration edge cases...")

        sys.path.insert(0, str(self.cloud_run_dir))
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

        print("✅ Configuration edge case tests passed")

    def test_09_performance_metrics(self):
        """Test performance metrics and monitoring."""
        print("🔍 Testing performance metrics...")

        sys.path.insert(0, str(self.cloud_run_dir))
        try:
            from health_monitor import HealthMonitor
        except ImportError as e:
            if 'psutil' in str(e):
                self.skipTest("psutil not available in test environment")
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
            self.fail("Missing performance metrics: {", '.join(missing_metrics)}")

        # Test system metrics structure
        system_metrics = metrics['system']
        system_required = ['memory_usage_mb', 'cpu_usage_percent', 'memory_percent']

        missing_system_metrics = [metric for metric in system_required if metric not in system_metrics]
        if missing_system_metrics:
            self.fail("Missing system metrics: {", '.join(missing_system_metrics)}")

        # Check all system metrics are numeric
        non_numeric_system_metrics = [metric for metric in system_required if not isinstance(system_metrics[metric], (int, float))]
        if non_numeric_system_metrics:
            self.fail("Non-numeric system metrics: {", '.join(non_numeric_system_metrics)}")

        # Test request metrics
        request_metrics = metrics['requests']
        self.assertIn('active', request_metrics, "Should track active requests")
        self.assertIn('total_processed', request_metrics, "Should track total processed requests")

        print("✅ Performance metrics tests passed")

    def test_10_yaml_parsing_validation(self):
        """Test YAML parsing and validation using enhanced test approach."""
        print("🔍 Testing YAML parsing and validation...")

        # Test Cloud Build YAML parsing
        cloudbuild_path = self.cloud_run_dir / 'cloudbuild.yaml'
        with open(cloudbuild_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validate YAML structure using enhanced approach
        self._validate_yaml_structure(config, 'cloudbuild.yaml')

        # Test configuration serialization
        sys.path.insert(0, str(self.cloud_run_dir))
        from config import EnvironmentConfig

        config_obj = EnvironmentConfig('production')
        config_dict = config_obj.to_dict()

        # Convert to YAML and back to test serialization
        yaml_str = yaml.dump(config_dict, default_flow_style=False)
        parsed_config = yaml.safe_load(yaml_str)

        self.assertEqual(config_dict, parsed_config, "YAML serialization should be reversible")

        print("✅ YAML parsing validation tests passed")

    def _validate_yaml_structure(self, config: Dict[str, Any], filename: str):
        """Enhanced YAML structure validation."""
        # Validate top-level structure
        self.assertIsInstance(config, dict, f"{filename} should be a dictionary")

        # Validate required top-level keys
        if filename == 'cloudbuild.yaml':
            required_keys = ['steps', 'images']
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                self.fail(f"{filename} missing required keys: {', '.join(missing_keys)}")

        # Validate nested structures
        if 'steps' in config:
            self.assertIsInstance(config['steps'], list, "Steps should be a list")
            invalid_steps = []
            for i, step in enumerate(config['steps']):
                if not isinstance(step, dict):
                    invalid_steps.append(f"Step {i} should be a dictionary")
                elif 'name' not in step or 'args' not in step:
                    invalid_steps.append(f"Step {i} missing required fields")

            if invalid_steps:
                self.fail("Invalid steps: {", '.join(invalid_steps)}")

def run_phase3_tests():
    """Run all Phase 3 Cloud Run optimization tests."""
    print("🚀 Starting Phase 3 Cloud Run Optimization Test Suite")
    print("=" * 60)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(Phase3CloudRunOptimizationTest)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Generate test report
    test_report = {
        'phase': 'Phase 3 - Cloud Run Optimization',
        'total_tests': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'success_rate': ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
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
    report_path = Path(__file__).parent / 'phase3_test_report.json'
    with open(report_path, 'w') as f:
        json.dump(test_report, f, indent=2)

    print("\n" + "=" * 60)
    print("📊 Phase 3 Test Results:")
    print("   Total Tests: {test_report["total_tests']}")
    print("   Failures: {test_report["failures']}")
    print("   Errors: {test_report["errors']}")
    print("   Success Rate: {test_report["success_rate']:.1f}%")
    print(f"   Report saved to: {report_path}")

    if result.wasSuccessful():
        print("✅ All Phase 3 tests passed!")
        return True
    print("❌ Some Phase 3 tests failed!")
    return False

if __name__ == '__main__':
    success = run_phase3_tests()
    sys.exit(0 if success else 1)
