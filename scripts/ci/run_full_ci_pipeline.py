#!/usr/bin/env python3
"""Comprehensive CI Pipeline Runner for SAMO Deep Learning.

This script runs the complete CI pipeline end-to-end, including:
- Environment validation
- Model loading tests
- API health checks
- Performance benchmarks
- GPU compatibility (when available)

Designed to work in both local and Colab environments.
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

# Use shared truthy parsing
try:
    from src.common.env import is_truthy
except ImportError:  # Fallback to local helper if import path not available

    def is_truthy(value: str | None) -> bool:
        return bool(value) and value.strip().lower() in {"1", "true", "yes"}

# Add src to path for local imports and import BERT classifier
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from models.emotion_detection.bert_classifier import BERTEmotionClassifier
except ImportError:
    from src.models.emotion_detection.bert_classifier import BERTEmotionClassifier


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ci_pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)


class CIPipelineRunner:
    """Comprehensive CI Pipeline Runner."""

    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        self.ci_scripts = [
            "scripts/ci/api_health_check.py",
            "scripts/ci/bert_model_test.py",
            "scripts/ci/t5_summarization_test.py",
            "scripts/ci/whisper_transcription_test.py",
            "scripts/ci/model_calibration_test.py",
            "scripts/ci/onnx_conversion_test.py",
        ]

    @staticmethod
    def _run_subprocess_command(
        command: list, timeout: int
    ) -> subprocess.CompletedProcess:
        """Run a subprocess command with standardized parameters.

        Parameters
        ----------
        command : list
            The command to run as a list of strings
        timeout : int
            Timeout in seconds

        Returns
        -------
        subprocess.CompletedProcess
            The completed process result
        """
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _get_test_stats(self) -> tuple[dict, int, int]:
        """Calculate statistics on test results.

        Returns
        -------
            tuple: (test_results dict, total_tests, passed_tests)
        """
        test_results = {
            name: result
            for name, result in self.results.items()
            if isinstance(result, bool)
        }
        total_tests = len(test_results)
        # Booleans can be summed directly (True=1, False=0)
        passed_tests = sum(test_results.values())
        return test_results, total_tests, passed_tests

    def detect_environment(self) -> Dict[str, str]:
        """Detect the current environment (local vs Colab)."""
        logger.info("🔍 Detecting environment...")

        env_info = {
            "platform": sys.platform,
            "python_version": sys.version,
            "is_colab": "COLAB_GPU" in os.environ,
            "gpu_available": False,
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        }

        # Check for GPU
        try:
            import torch

            env_info["gpu_available"] = torch.cuda.is_available()
            if env_info["gpu_available"]:
                env_info["gpu_count"] = torch.cuda.device_count()
                env_info["gpu_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            logger.warning("⚠️ PyTorch not available for GPU detection")

        # Check for Colab
        if env_info["is_colab"]:
            logger.info("🎯 Running in Google Colab environment")
            env_info["colab_gpu"] = os.environ.get("COLAB_GPU", "unknown")
        else:
            logger.info("💻 Running in local environment")

        logger.info("📊 Environment: %s", env_info)
        return env_info

    def validate_dependencies(self) -> bool:
        """Validate that all required dependencies are available."""
        logger.info("📦 Validating dependencies...")

        required_packages = [
            "torch",
            "transformers",
            "fastapi",
            "pydantic",
            "datasets",
            "tokenizers",
            "numpy",
            "pandas",
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                logger.info("✅ %s available", package)
            except ImportError:
                missing_packages.append(package)
                logger.error("❌ %s missing", package)

        if missing_packages:
            logger.error("❌ Missing packages: %s", missing_packages)
            return False

        logger.info("✅ All dependencies validated")
        return True

    def run_ci_script(self, script_path: str) -> Tuple[bool, str]:
        """Run a single CI script and return success status and output."""
        logger.info("🚀 Running %s...", script_path)

        try:
            # Use the correct Python interpreter
            python_executable = sys.executable

            # Run the script - script_path is controlled internally by self.ci_scripts
            # No command injection risk as paths are static and predefined
            result = self._run_subprocess_command(
                [python_executable, script_path],
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0:
                logger.info("✅ %s PASSED", script_path)
                return True, result.stdout
            else:
                logger.error("❌ %s FAILED", script_path)
                logger.error("Error output: %s", result.stderr)
                logger.error("Standard output: %s", result.stdout)
                return False, result.stderr

        except subprocess.TimeoutExpired:
            logger.error("⏰ %s TIMEOUT", script_path)
            return False, "Script timed out after 5 minutes"
        except Exception as e:
            logger.exception("💥 %s ERROR: %s", script_path, e)
            return False, str(e)

    def run_unit_tests(self) -> bool:
        """Run unit tests."""
        logger.info("🧪 Running unit tests...")

        try:
            # Static command - no injection risk, all arguments are literals
            result = self._run_subprocess_command(
                [sys.executable, "-m", "pytest", "tests/unit/", "-v"],
                timeout=1200,  # 20 minute timeout (increased from 10)
            )

            if result.returncode == 0:
                logger.info("✅ Unit tests PASSED")
                return True
            else:
                logger.error("❌ Unit tests FAILED")
                logger.error("Return code: %s", result.returncode)
                logger.error("Error output: %s", result.stderr)
                logger.error("Standard output: %s", result.stdout)
                return False

        except subprocess.TimeoutExpired:
            logger.error("⏰ Unit tests TIMEOUT")
            return False
        except Exception as e:
            logger.exception("💥 Unit tests ERROR: %s", e)
            return False

    def run_e2e_tests(self) -> bool:
        """Run end-to-end tests."""
        logger.info("🎯 Running E2E tests...")

        try:
            # Static command - no injection risk, all arguments are literals
            result = self._run_subprocess_command(
                [sys.executable, "-m", "pytest", "tests/e2e/", "-v"],
                timeout=900,  # 15 minute timeout
            )

            if result.returncode == 0:
                logger.info("✅ E2E tests PASSED")
                return True
            else:
                logger.error("❌ E2E tests FAILED")
                logger.error("Error output: %s", result.stderr)
                logger.error("Standard output: %s", result.stdout)
                return False

        except Exception as e:
            logger.exception("💥 E2E tests ERROR: %s", e)
            return False

    @staticmethod
    def _test_gpu_model_forward_pass() -> bool:
        """Test GPU model forward pass with BERT classifier.

        Returns
        -------
        bool
            True if GPU forward pass succeeds, False otherwise
        """
        try:
            import torch

            device = torch.device("cuda")

            # Use the module-level BERT classifier import
            model = BERTEmotionClassifier().to(device)

            # Test forward pass
            dummy_input = torch.randint(0, 1000, (2, 512)).to(device)
            with torch.no_grad():
                output = model(dummy_input, torch.ones_like(dummy_input))

            logger.info("✅ GPU forward pass successful, output shape: %s", output.shape)
            return True

        except Exception as e:
            logger.exception("❌ GPU model forward pass failed: %s", e)
            return False

    def test_gpu_compatibility(self) -> bool:
        """Test GPU compatibility if available."""
        logger.info("🖥️ Testing GPU compatibility...")

        try:
            import torch

            if not torch.cuda.is_available():
                logger.info("ℹ️ No GPU available, skipping GPU tests")
                return True

            logger.info("🎮 GPU detected: %s", torch.cuda.get_device_name(0))

            # Test GPU model loading and forward pass
            return self._test_gpu_model_forward_pass()

        except Exception as e:
            logger.exception("❌ GPU compatibility test failed: %s", e)
            return False

    @staticmethod
    def _measure_model_loading_time() -> float:
        """Measure BERT model loading time.

        Returns
        -------
        float
            Loading time in seconds
        """
        start_time = time.time()
        _ = BERTEmotionClassifier()  # Instantiate model to measure loading time
        loading_time = time.time() - start_time

        logger.info("✅ Model loading time: %.2fs", loading_time)
        return loading_time

    @staticmethod
    def _measure_inference_time(model) -> float:
        """Measure model inference time.

        Parameters
        ----------
        model : torch.nn.Module
            The model to test inference on

        Returns
        -------
        float
            Inference time in seconds
        """
        import torch

        start_time = time.time()
        dummy_input = torch.randint(0, 1000, (1, 512))
        with torch.no_grad():
            model(dummy_input, torch.ones_like(dummy_input))
        inference_time = time.time() - start_time

        logger.info("✅ Inference time: %.2fs", inference_time)
        return inference_time

    @staticmethod
    def _validate_performance_thresholds(
        loading_time: float, inference_time: float
    ) -> bool:
        """Validate that performance times are within acceptable thresholds.

        Parameters
        ----------
        loading_time : float
            Model loading time in seconds
        inference_time : float
            Inference time in seconds

        Returns
        -------
        bool
            True if performance is acceptable, False otherwise
        """
        # Increased threshold for CPU environments
        if loading_time < 10.0 and inference_time < 5.0:
            logger.info("✅ Performance benchmarks passed")
            return True

        logger.error(
            "❌ Performance too slow - loading: %.2fs, inference: %.2fs",
            loading_time,
            inference_time,
        )
        return False

    def run_performance_benchmarks(self) -> bool:
        """Run performance benchmarks."""
        logger.info("⚡ Running performance benchmarks...")

        try:
            # Measure model loading time
            loading_time = self._measure_model_loading_time()

            # Import model again for inference test (avoid reusing loaded model)
            model = BERTEmotionClassifier()

            # Measure inference time
            inference_time = self._measure_inference_time(model)

            # Validate performance thresholds
            return self._validate_performance_thresholds(loading_time, inference_time)

        except Exception as e:
            logger.exception("❌ Performance benchmark failed: %s", e)
            return False

    def run_pipeline_and_exit(self) -> None:
        """Run the CI pipeline and exit with appropriate code.

        This method handles the complete pipeline execution flow including:
        - Running all tests
        - Generating reports
        - Writing CI artifacts
        - Exiting with proper status codes
        """
        try:
            _ = self.run_full_pipeline()
            report = self.generate_report()

            print(report)
            # Only write report to file in CI so it can be uploaded as an artifact
            write_ci_report_if_needed(report)

            # Exit with appropriate code
            _, total_tests, passed_tests = self._get_test_stats()

            if total_tests == 0:
                logger.error("❌ CI Pipeline failed - no boolean tests were executed!")
                sys.exit(1)
            elif passed_tests == total_tests:
                logger.info("🎉 CI Pipeline completed successfully!")
                sys.exit(0)
            else:
                logger.error("❌ CI Pipeline failed!")
                sys.exit(1)

        except KeyboardInterrupt:
            logger.info("⏹️ CI Pipeline interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.exception("💥 CI Pipeline crashed: %s", e)
            sys.exit(1)

    def run_full_pipeline(self) -> Dict[str, bool]:
        """Run the complete CI pipeline."""
        logger.info("🚀 Starting Comprehensive CI Pipeline")
        logger.info("=" * 60)

        # Environment detection
        env_info = self.detect_environment()
        self.results["environment"] = env_info

        # Dependency validation
        self.results["dependencies"] = self.validate_dependencies()

        # Run individual CI scripts
        for script in self.ci_scripts:
            script_name = Path(script).stem
            success, output = self.run_ci_script(script)
            self.results[script_name] = success

            if not success:
                logger.error("❌ %s failed, but continuing...", script_name)

        # Run unit tests
        self.results["unit_tests"] = self.run_unit_tests()

        # Run E2E tests
        self.results["e2e_tests"] = self.run_e2e_tests()

        # Test GPU compatibility
        self.results["gpu_compatibility"] = self.test_gpu_compatibility()

        # Run performance benchmarks
        self.results["performance"] = self.run_performance_benchmarks()

        return self.results

    def generate_report(self) -> str:
        """Generate a comprehensive CI report."""
        logger.info("📊 Generating CI Report")
        logger.info("=" * 60)

        # Only count boolean results as actual tests
        test_results, total_tests, passed_tests = self._get_test_stats()

        # Handle case where no boolean tests were collected
        if total_tests == 0:
            success_rate = 0.0
        else:
            success_rate = (passed_tests / total_tests) * 100.0

        report = f"""
🎯 COMPREHENSIVE CI PIPELINE REPORT
{"=" * 60}

📊 SUMMARY:
- Total Tests: {total_tests}
- Passed: {passed_tests}
- Failed: {total_tests - passed_tests}
- Success Rate: {success_rate:.1f}%

🔍 DETAILED RESULTS:
"""

        for test_name, result in self.results.items():
            if isinstance(result, bool):
                status = "✅ PASSED" if result else "❌ FAILED"
                report += f"- {test_name}: {status}\n"
            elif isinstance(result, dict):
                report += f"- {test_name}: {result}\n"

        report += f"""
⏱️ EXECUTION TIME: {time.time() - self.start_time:.1f}s

🎯 RECOMMENDATIONS:
"""

        if total_tests == 0:
            report += (
                "⚠️ No boolean tests were executed. Treating pipeline as failed.\n"
            )
        elif passed_tests == total_tests:
            report += "🎉 All tests passed! Pipeline is ready for deployment.\n"
        else:
            failed_test_names = [
                name for name, result in test_results.items() if not result
            ]
            report += f"⚠️ Failed tests: {', '.join(failed_test_names)}\n"
            report += "🔧 Please fix the failed tests before deployment.\n"

        return report


def write_ci_report_if_needed(report: str) -> None:
    """Write the CI report artifact only when running in CI environment."""
    if is_truthy(os.environ.get("CI")):
        with open("ci_pipeline_report.txt", "w") as f:
            f.write(report)


def main():
    """Main function to run the CI pipeline."""
    runner = CIPipelineRunner()
    runner.run_pipeline_and_exit()


if __name__ == "__main__":
    main()
