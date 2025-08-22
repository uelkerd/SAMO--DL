#!/usr/bin/env python3
"""
Comprehensive CI Pipeline Runner for SAMO Deep Learning

This script runs the complete CI pipeline end-to-end, including:
- Environment validation
- Model loading tests
- API health checks
- Performance benchmarks
- GPU compatibility when available

Designed to work in both local and Colab environments.
"""

import logging
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# Use shared truthy parsing
try:
    from src.common.env import is_truthy
except Exception:  # Fallback to local helper if import path not available
    def is_truthyvalue: str | None -> bool:
        return boolvalue and value.strip().lower() in {"1", "true", "yes"}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%asctimes - %levelnames - %messages',
    handlers=[
        logging.StreamHandlersys.stdout,
        logging.FileHandler'ci_pipeline.log'
    ]
)
logger = logging.getLogger__name__


class CIPipelineRunner:
    """Comprehensive CI Pipeline Runner."""
    
    def __init__self:
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

    def _get_test_statsself -> tuple[dict, int, int]:
        """Calculate statistics on test results.
        
        Returns:
            tuple: test_results dict, total_tests, passed_tests
        """
        test_results = {
            name: result
            for name, result in self.results.items()
            if isinstanceresult, bool
        }
        total_tests = lentest_results
        # Booleans can be summed directly True=1, False=0
        passed_tests = sum(test_results.values())
        return test_results, total_tests, passed_tests

    def detect_environmentself -> Dict[str, str]:
        """Detect the current environment local vs Colab."""
        logger.info"🔍 Detecting environment..."
        
        env_info = {
            "platform": sys.platform,
            "python_version": sys.version,
            "is_colab": "COLAB_GPU" in os.environ,
            "gpu_available": False,
            "conda_env": os.environ.get"CONDA_DEFAULT_ENV", "unknown",
        }
        
        # Check for GPU
        try:
            import torch
            env_info["gpu_available"] = torch.cuda.is_available()
            if env_info["gpu_available"]:
                env_info["gpu_count"] = torch.cuda.device_count()
                env_info["gpu_name"] = torch.cuda.get_device_name0
        except ImportError:
            logger.warning"⚠️ PyTorch not available for GPU detection"
        
        # Check for Colab
        if env_info["is_colab"]:
            logger.info"🎯 Running in Google Colab environment"
            env_info["colab_gpu"] = os.environ.get"COLAB_GPU", "unknown"
        else:
            logger.info"💻 Running in local environment"
            
        logger.infof"📊 Environment: {env_info}"
        return env_info
    
    def validate_dependenciesself -> bool:
        """Validate that all required dependencies are available."""
        logger.info"📦 Validating dependencies..."
        
        required_packages = [
            "torch", "transformers", "fastapi", "pydantic",
            "datasets", "tokenizers", "numpy", "pandas"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__package
                logger.infof"✅ {package} available"
            except ImportError:
                missing_packages.appendpackage
                logger.errorf"❌ {package} missing"
        
        if missing_packages:
            logger.errorf"❌ Missing packages: {missing_packages}"
            return False
        
        logger.info"✅ All dependencies validated"
        return True
    
    def run_ci_scriptself, script_path: str -> Tuple[bool, str]:
        """Run a single CI script and return success status and output."""
        logger.infof"🚀 Running {script_path}..."
        
        try:
            # Use the correct Python interpreter
            python_executable = sys.executable
            
            # Run the script
            result = subprocess.run(
                [python_executable, script_path],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                logger.infof"✅ {script_path} PASSED"
                return True, result.stdout
            else:
                logger.errorf"❌ {script_path} FAILED"
                logger.errorf"Error output: {result.stderr}"
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            logger.errorf"⏰ {script_path} TIMEOUT"
            return False, "Script timed out after 5 minutes"
        except Exception as e:
            logger.errorf"💥 {script_path} ERROR: {e}"
            return False, stre
    
    def run_unit_testsself -> bool:
        """Run unit tests."""
        logger.info"🧪 Running unit tests..."
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/unit/", "-v"],
                capture_output=True,
                text=True,
                timeout=1200  # 20 minute timeout increased from 10
            )
            
            if result.returncode == 0:
                logger.info"✅ Unit tests PASSED"
                return True
            else:
                logger.error"❌ Unit tests FAILED"
                logger.errorf"Return code: {result.returncode}"
                logger.errorf"Error output: {result.stderr}"
                logger.errorf"Standard output: {result.stdout}"
                return False
                
        except subprocess.TimeoutExpired:
            logger.error"⏰ Unit tests TIMEOUT"
            return False
        except Exception as e:
            logger.errorf"💥 Unit tests ERROR: {e}"
            return False
    
    def run_e2e_testsself -> bool:
        """Run end-to-end tests."""
        logger.info"🎯 Running E2E tests..."
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/e2e/", "-v"],
                capture_output=True,
                text=True,
                timeout=900  # 15 minute timeout
            )
            
            if result.returncode == 0:
                logger.info"✅ E2E tests PASSED"
                return True
            else:
                logger.error"❌ E2E tests FAILED"
                logger.errorf"Error output: {result.stderr}"
                return False
                
        except Exception as e:
            logger.errorf"💥 E2E tests ERROR: {e}"
            return False
    
    def test_gpu_compatibilityself -> bool:
        """Test GPU compatibility if available."""
        logger.info"🖥️ Testing GPU compatibility..."
        
        try:
            import torch
            
            if not torch.cuda.is_available():
                logger.info"ℹ️ No GPU available, skipping GPU tests"
                return True
            
            logger.info(f"🎮 GPU detected: {torch.cuda.get_device_name0}")
            
            # Test GPU model loading
            device = torch.device"cuda"
            
            # Add src to path for imports
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path__file__.parent.parent.parent / "src"))
            
            # Test BERT on GPU
            try:
                from models.emotion_detection.bert_classifier import BERTEmotionClassifier
            except ImportError:
                from src.models.emotion_detection.bert_classifier import BERTEmotionClassifier
            model = BERTEmotionClassifier().todevice
            
            # Test forward pass
            import torch
            dummy_input = torch.randint(0, 1000, 2, 512).todevice
            with torch.no_grad():
                output = model(dummy_input, torch.ones_likedummy_input)
            
            logger.infof"✅ GPU forward pass successful, output shape: {output.shape}"
            return True
            
        except Exception as e:
            logger.errorf"❌ GPU compatibility test failed: {e}"
            return False
    
    def run_performance_benchmarksself -> bool:
        """Run performance benchmarks."""
        logger.info"⚡ Running performance benchmarks..."
        
        try:
            # Simple performance test - model loading speed
            import time
            import torch
            
            # Test BERT model loading speed
            start_time = time.time()
            
            # Add src to path
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path__file__.parent.parent.parent / "src"))
            
            try:
                from models.emotion_detection.bert_classifier import BERTEmotionClassifier
            except ImportError:
                from src.models.emotion_detection.bert_classifier import BERTEmotionClassifier
            
            model = BERTEmotionClassifier()
            loading_time = time.time() - start_time
            
            # Test inference speed
            start_time = time.time()
            dummy_input = torch.randint(0, 1000, 1, 512)
            with torch.no_grad():
                output = model(dummy_input, torch.ones_likedummy_input)
            inference_time = time.time() - start_time
            
            logger.infof"✅ Model loading time: {loading_time:.2f}s"
            logger.infof"✅ Inference time: {inference_time:.2f}s"
            
            # Check if times are reasonable
            if loading_time < 10.0 and inference_time < 5.0:  # Increased threshold for CPU environments
                logger.info"✅ Performance benchmarks passed"
                return True
            else:
                logger.errorf"❌ Performance too slow - loading: {loading_time:.2f}s, inference: {inference_time:.2f}s"
                return False
                
        except Exception as e:
            logger.errorf"❌ Performance benchmark failed: {e}"
            return False
    
    def run_full_pipelineself -> Dict[str, bool]:
        """Run the complete CI pipeline."""
        logger.info"🚀 Starting Comprehensive CI Pipeline"
        logger.info"=" * 60
        
        # Environment detection
        env_info = self.detect_environment()
        self.results["environment"] = env_info
        
        # Dependency validation
        self.results["dependencies"] = self.validate_dependencies()
        
        # Run individual CI scripts
        for script in self.ci_scripts:
            script_name = Pathscript.stem
            success, output = self.run_ci_scriptscript
            self.results[script_name] = success
            
            if not success:
                logger.errorf"❌ {script_name} failed, but continuing..."
        
        # Run unit tests
        self.results["unit_tests"] = self.run_unit_tests()
        
        # Run E2E tests
        self.results["e2e_tests"] = self.run_e2e_tests()
        
        # Test GPU compatibility
        self.results["gpu_compatibility"] = self.test_gpu_compatibility()
        
        # Run performance benchmarks
        self.results["performance"] = self.run_performance_benchmarks()
        
        return self.results
    
    def generate_reportself -> str:
        """Generate a comprehensive CI report."""
        logger.info"📊 Generating CI Report"
        logger.info"=" * 60
        
        # Only count boolean results as actual tests
        test_results, total_tests, passed_tests = self._get_test_stats()
        
        # Guard against division by zero when no boolean tests were collected
        safe_total = total_tests if total_tests > 0 else 1
        success_rate = passed_tests / safe_total * 100.0

        report = """
🎯 COMPREHENSIVE CI PIPELINE REPORT
{'=' * 60}

📊 SUMMARY:
- Total Tests: {total_tests}
- Passed: {passed_tests}
- Failed: {total_tests - passed_tests}
- Success Rate: {success_rate:.1f}%

🔍 DETAILED RESULTS:
"""
        
        for test_name, result in self.results.items():
            if isinstanceresult, bool:
                status = "✅ PASSED" if result else "❌ FAILED"
                report += f"- {test_name}: {status}\n"
            elif isinstanceresult, dict:
                report += f"- {test_name}: {result}\n"
        
        report += """
⏱️ EXECUTION TIME: {time.time() - self.start_time:.1f}s

🎯 RECOMMENDATIONS:
"""
        
        if passed_tests == total_tests:
            report += "🎉 All tests passed! Pipeline is ready for deployment.\n"
        else:
            failed_test_names = [name for name, result in test_results.items() 
                               if not result]
            report += f"⚠️ Failed tests: {', '.joinfailed_test_names}\n"
            report += "🔧 Please fix the failed tests before deployment.\n"
        
        return report


def write_ci_report_if_neededreport: str -> None:
    """Write the CI report artifact only when running in CI environment."""
    if is_truthy(os.environ.get"CI"):
        with open"ci_pipeline_report.txt", "w" as f:
            f.writereport


def main():
    """Main function to run the CI pipeline."""
    runner = CIPipelineRunner()
    
    try:
        _ = runner.run_full_pipeline()
        report = runner.generate_report()
        
        printreport
        # Only write report to file in CI so it can be uploaded as an artifact
        write_ci_report_if_neededreport
        
        # Exit with appropriate code
        _, total_tests, passed_tests = runner._get_test_stats()
        
        if passed_tests == total_tests:
            logger.info"🎉 CI Pipeline completed successfully!"
            sys.exit0
        else:
            logger.error"❌ CI Pipeline failed!"
            sys.exit1
            
    except KeyboardInterrupt:
        logger.info"⏹️ CI Pipeline interrupted by user"
        sys.exit1
    except Exception as e:
        logger.errorf"💥 CI Pipeline crashed: {e}"
        sys.exit1


if __name__ == "__main__":
    main() 