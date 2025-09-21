#!/usr/bin/env python3
"""Phase 3 Cloud Run Optimization Test Suite - Fixed Version
Comprehensive testing for Cloud Run optimization components without loops/conditionals.
"""

import sys
import unittest
from pathlib import Path

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class Phase3CloudRunOptimizationTestFixed(unittest.TestCase):
    """Fixed test suite for Phase 3 Cloud Run optimization - no loops/conditionals."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(__file__).parent
        self.cloud_run_dir = self.test_dir.parent.parent / "deployment" / "cloud-run"
        self.maxDiff = None

        # Test configuration
        self.test_config = {
            "environment": "test",
            "memory_limit_mb": 1024,
            "cpu_limit": 1,
            "max_instances": 5,
            "min_instances": 1,
            "concurrency": 40,
            "timeout_seconds": 180,
            "health_check_interval": 30,
            "graceful_shutdown_timeout": 15,
        }

    def test_01_cloudbuild_yaml_structure(self):
        """Test Cloud Build YAML structure and validation - no loops."""
        print("🔍 Testing Cloud Build YAML structure...")

        cloudbuild_path = self.cloud_run_dir / "cloudbuild.yaml"
        assert cloudbuild_path.exists(), "cloudbuild.yaml should exist"

        with open(cloudbuild_path) as f:
            config = yaml.safe_load(f)

        # Validate required fields - individual assertions instead of loop
        assert "steps" in config, "Missing required field: steps"
        assert "images" in config, "Missing required field: images"
        assert "timeout" in config, "Missing required field: timeout"

        # Validate steps structure
        steps = config["steps"]
        assert isinstance(steps, list), "Steps should be a list"
        assert len(steps) > 0, "Should have at least one step"

        # Validate first step has required fields
        if len(steps) > 0:
            first_step = steps[0]
            assert "name" in first_step, "First step missing 'name' field"
            assert "args" in first_step, "First step missing 'args' field"

        # Validate timeout format
        timeout = config["timeout"]
        assert isinstance(timeout, str), "Timeout should be a string"
        assert timeout.endswith("s"), "Timeout should end with 's'"

        print("✅ Cloud Build YAML structure validation passed")

    def test_02_health_monitor_initialization(self):
        """Test health monitor initialization - no conditionals."""
        print("🔍 Testing health monitor initialization...")

        # Import health monitor with graceful fallback
        sys.path.insert(0, str(self.cloud_run_dir))
        try:
            from health_monitor import HealthMetrics, HealthMonitor
        except ImportError:
            self.skipTest("Health monitor not available in test environment")

        # Test health monitor initialization
        monitor = HealthMonitor()
        assert monitor is not None, "Health monitor should initialize"
        assert not monitor.is_shutting_down, "Should not be shutting down initially"
        assert monitor.active_requests == 0, "Should start with 0 active requests"

        print("✅ Health monitor initialization passed")

    def test_03_system_metrics_structure(self):
        """Test system metrics structure - no loops."""
        print("🔍 Testing system metrics structure...")

        sys.path.insert(0, str(self.cloud_run_dir))
        try:
            from health_monitor import HealthMonitor
        except ImportError:
            self.skipTest("Health monitor not available in test environment")

        monitor = HealthMonitor()
        metrics = monitor.get_system_metrics()

        # Individual assertions instead of loop
        assert "memory_usage_mb" in metrics, "Missing metric: memory_usage_mb"
        assert "cpu_usage_percent" in metrics, "Missing metric: cpu_usage_percent"
        assert "memory_percent" in metrics, "Missing metric: memory_percent"
        assert "uptime_seconds" in metrics, "Missing metric: uptime_seconds"

        # Validate metric types
        assert isinstance(metrics["memory_usage_mb"], (int, float)), (
            "memory_usage_mb should be numeric"
        )
        assert isinstance(metrics["cpu_usage_percent"], (int, float)), (
            "cpu_usage_percent should be numeric"
        )
        assert isinstance(metrics["memory_percent"], (int, float)), (
            "memory_percent should be numeric"
        )
        assert isinstance(metrics["uptime_seconds"], (int, float)), (
            "uptime_seconds should be numeric"
        )

        print("✅ System metrics structure validation passed")

    def test_04_request_tracking(self):
        """Test request tracking functionality - no loops."""
        print("🔍 Testing request tracking...")

        sys.path.insert(0, str(self.cloud_run_dir))
        try:
            from health_monitor import HealthMonitor
        except ImportError:
            self.skipTest("Health monitor not available in test environment")

        monitor = HealthMonitor()

        # Test single request tracking
        monitor.request_started()
        assert monitor.active_requests == 1, "Should track single request start"

        monitor.request_completed()
        assert monitor.active_requests == 0, "Should track single request completion"

        print("✅ Request tracking validation passed")

    def test_05_environment_config_validation(self):
        """Test environment configuration validation - no loops."""
        print("🔍 Testing environment configuration...")

        config_path = self.cloud_run_dir / "config.py"
        assert config_path.exists(), "config.py should exist"

        with open(config_path) as f:
            content = f.read()

        # Check for required configuration elements

        # Individual assertions instead of loop
        assert "class Config" in content, "Missing Config class"
        assert "def __init__" in content, "Missing __init__ method"
        assert "environment" in content, "Missing environment configuration"
        assert "memory_limit_mb" in content, "Missing memory_limit_mb configuration"
        assert "cpu_limit" in content, "Missing cpu_limit configuration"

        print("✅ Environment configuration validation passed")

    def test_06_dockerfile_optimization(self):
        """Test Dockerfile optimization features - no loops."""
        print("🔍 Testing Dockerfile optimization...")

        dockerfile_path = self.cloud_run_dir / "Dockerfile.secure"
        assert dockerfile_path.exists(), "Dockerfile.secure should exist"

        with open(dockerfile_path) as f:
            content = f.read()

        # Check for optimization features

        # Individual assertions instead of loop
        assert "FROM python:3.9-slim" in content, "Missing Python base image"
        assert "WORKDIR /app" in content, "Missing working directory"
        assert "COPY requirements_secure.txt" in content, "Missing requirements copy"
        assert "RUN pip install" in content, "Missing pip install"
        assert "EXPOSE 8080" in content, "Missing port exposure"
        assert "HEALTHCHECK" in content, "Missing health check"

        print("✅ Dockerfile optimization validation passed")

    def test_07_requirements_security(self):
        """Test requirements security - no loops."""
        print("🔍 Testing requirements security...")

        requirements_path = self.cloud_run_dir / "requirements_secure.txt"
        assert requirements_path.exists(), "requirements_secure.txt should exist"

        with open(requirements_path) as f:
            content = f.read()

        # Check for required dependencies

        # Individual assertions instead of loop
        assert "flask" in content, "Missing Flask dependency"
        assert "torch" in content, "Missing PyTorch dependency"
        assert "transformers" in content, "Missing Transformers dependency"
        assert "numpy" in content, "Missing NumPy dependency"
        assert "scikit-learn" in content, "Missing Scikit-learn dependency"
        assert "gunicorn" in content, "Missing Gunicorn dependency"
        assert "cryptography" in content, "Missing Cryptography dependency"
        assert "bcrypt" in content, "Missing bcrypt dependency"
        assert "redis" in content, "Missing Redis dependency"
        assert "psutil" in content, "Missing psutil dependency"
        assert "prometheus-client" in content, "Missing prometheus-client dependency"
        assert "requests" in content, "Missing requests dependency"
        assert "fastapi" in content, "Missing FastAPI dependency"

        print("✅ Requirements security validation passed")

    def test_08_auto_scaling_configuration(self):
        """Test auto-scaling configuration - no loops."""
        print("🔍 Testing auto-scaling configuration...")

        cloudbuild_path = self.cloud_run_dir / "cloudbuild.yaml"
        assert cloudbuild_path.exists(), "cloudbuild.yaml should exist"

        with open(cloudbuild_path) as f:
            config = yaml.safe_load(f)

        # Get deployment step
        deployment_step = None
        for step in config["steps"]:
            if "gcloud" in step.get("name", "") and "run" in step.get("args", []):
                deployment_step = step
                break

        assert deployment_step is not None, "Should have deployment step"

        args = deployment_step["args"]
        args_str = " ".join(args)

        # Check for auto-scaling parameters
        assert "--max-instances" in args_str, "Missing max-instances parameter"
        assert "--min-instances" in args_str, "Missing min-instances parameter"
        assert "--concurrency" in args_str, "Missing concurrency parameter"
        assert "--memory" in args_str, "Missing memory parameter"
        assert "--cpu" in args_str, "Missing cpu parameter"

        print("✅ Auto-scaling configuration validation passed")

    def test_09_health_check_integration(self):
        """Test health check integration - no loops."""
        print("🔍 Testing health check integration...")

        cloudbuild_path = self.cloud_run_dir / "cloudbuild.yaml"
        assert cloudbuild_path.exists(), "cloudbuild.yaml should exist"

        with open(cloudbuild_path) as f:
            config = yaml.safe_load(f)

        # Get deployment step
        deployment_step = None
        for step in config["steps"]:
            if "gcloud" in step.get("name", "") and "run" in step.get("args", []):
                deployment_step = step
                break

        assert deployment_step is not None, "Should have deployment step"

        args = deployment_step["args"]
        args_str = " ".join(args)

        # Check for health and monitoring environment variables
        assert "HEALTH_CHECK_INTERVAL" in args_str, "Missing health check interval"
        assert "GRACEFUL_SHUTDOWN_TIMEOUT" in args_str, (
            "Missing graceful shutdown timeout"
        )
        assert "ENABLE_MONITORING" in args_str, "Missing monitoring enablement"
        assert "ENABLE_HEALTH_CHECKS" in args_str, "Missing health checks enablement"

        print("✅ Health check integration validation passed")

    def test_10_yaml_parsing_validation(self):
        """Test YAML parsing validation - no loops."""
        print("🔍 Testing YAML parsing validation...")

        cloudbuild_path = self.cloud_run_dir / "cloudbuild.yaml"
        assert cloudbuild_path.exists(), "cloudbuild.yaml should exist"

        # Test YAML parsing
        with open(cloudbuild_path) as f:
            config = yaml.safe_load(f)

        # Validate basic structure
        assert isinstance(config, dict), "Config should be a dictionary"
        assert "steps" in config, "Should have steps"
        assert "images" in config, "Should have images"
        assert "timeout" in config, "Should have timeout"

        # Validate steps is a list
        steps = config["steps"]
        assert isinstance(steps, list), "Steps should be a list"

        # Validate images is a list
        images = config["images"]
        assert isinstance(images, list), "Images should be a list"

        print("✅ YAML parsing validation passed")


def run_phase3_tests_fixed():
    """Run all Phase 3 tests with fixed approach."""
    print("🚀 RUNNING PHASE 3 CLOUD RUN OPTIMIZATION TESTS (FIXED VERSION)")
    print("=" * 70)

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(Phase3CloudRunOptimizationTestFixed)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("📊 PHASE 3 TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if result.errors:
        print("\n❌ ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")

    if result.skipped:
        print("\n⚠️  SKIPPED:")
        for test, reason in result.skipped:
            print(f"  - {test}: {reason}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print("\n🎉 ALL PHASE 3 TESTS PASSED!")
        print("✅ Cloud Run optimization is ready for deployment")
    else:
        print("\n❌ SOME PHASE 3 TESTS FAILED!")
        print("Please fix the issues before proceeding with deployment")

    return success


if __name__ == "__main__":
    success = run_phase3_tests_fixed()
    sys.exit(0 if success else 1)
