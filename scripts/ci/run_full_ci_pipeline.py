#!/usr/bin/env python3
"""
Comprehensive CI Pipeline Runner for SAMO Deep Learning

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
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ci_pipeline.log')
    ]
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
            
        logger.info(f"📊 Environment: {env_info}")
        return env_info
    
    def validate_dependencies(self) -> bool:
        """Validate that all required dependencies are available."""
        logger.info("📦 Validating dependencies...")
        
        required_packages = [
            "torch", "transformers", "fastapi", "pydantic",
            "datasets", "tokenizers", "numpy", "pandas"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package} available")
            except ImportError:
                missing_packages.append(package)
                logger.error(f"❌ {package} missing")
        
        if missing_packages:
            logger.error(f"❌ Missing packages: {missing_packages}")
            return False
        
        logger.info("✅ All dependencies validated")
        return True
    
    def run_ci_script(self, script_path: str) -> Tuple[bool, str]:
        """Run a single CI script and return success status and output."""
        logger.info(f"🚀 Running {script_path}...")
        
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
                logger.info(f"✅ {script_path} PASSED")
                return True, result.stdout
            else:
                logger.error(f"❌ {script_path} FAILED")
                logger.error(f"Error output: {result.stderr}")
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {script_path} TIMEOUT")
            return False, "Script timed out after 5 minutes"
        except Exception as e:
            logger.error(f"💥 {script_path} ERROR: {e}")
            return False, str(e)
    
    def run_unit_tests(self) -> bool:
        """Run unit tests."""
        logger.info("🧪 Running unit tests...")
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/unit/", "-v"],
                capture_output=True,
                text=True,
                timeout=1200  # 20 minute timeout (increased from 10)
            )
            
            if result.returncode == 0:
                logger.info("✅ Unit tests PASSED")
                return True
            else:
                logger.error("❌ Unit tests FAILED")
                logger.error(f"Return code: {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                logger.error(f"Standard output: {result.stdout}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("⏰ Unit tests TIMEOUT")
            return False
        except Exception as e:
            logger.error(f"💥 Unit tests ERROR: {e}")
            return False
    
    def run_e2e_tests(self) -> bool:
        """Run end-to-end tests."""
        logger.info("🎯 Running E2E tests...")
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/e2e/", "-v"],
                capture_output=True,
                text=True,
                timeout=900  # 15 minute timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ E2E tests PASSED")
                return True
            else:
                logger.error("❌ E2E tests FAILED")
                logger.error(f"Error output: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"💥 E2E tests ERROR: {e}")
            return False
    
    def test_gpu_compatibility(self) -> bool:
        """Test GPU compatibility if available."""
        logger.info("🖥️ Testing GPU compatibility...")
        
        try:
            import torch
            
            if not torch.cuda.is_available():
                logger.info("ℹ️ No GPU available, skipping GPU tests")
                return True
            
            logger.info(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
            
            # Test GPU model loading
            device = torch.device("cuda")
            
            # Add src to path for imports
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
            
            # Test BERT on GPU
            try:
                from models.emotion_detection.bert_classifier import BERTEmotionClassifier
            except ImportError:
                from src.models.emotion_detection.bert_classifier import BERTEmotionClassifier
            model = BERTEmotionClassifier().to(device)
            
            # Test forward pass
            import torch
            dummy_input = torch.randint(0, 1000, (2, 512)).to(device)
            with torch.no_grad():
                output = model(dummy_input, torch.ones_like(dummy_input))
            
            logger.info(f"✅ GPU forward pass successful, output shape: {output.shape}")
            return True
            
        except Exception as e:
            logger.error(f"❌ GPU compatibility test failed: {e}")
            return False
    
    def run_performance_benchmarks(self) -> bool:
        """Run performance benchmarks."""
        logger.info("⚡ Running performance benchmarks...")
        
        try:
            # Simple performance test - model loading speed
            import time
            import torch
            
            # Test BERT model loading speed
            start_time = time.time()
            
            # Add src to path
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
            
            try:
                from models.emotion_detection.bert_classifier import BERTEmotionClassifier
            except ImportError:
                from src.models.emotion_detection.bert_classifier import BERTEmotionClassifier
            
            model = BERTEmotionClassifier()
            loading_time = time.time() - start_time
            
            # Test inference speed
            start_time = time.time()
            dummy_input = torch.randint(0, 1000, (1, 512))
            with torch.no_grad():
                output = model(dummy_input, torch.ones_like(dummy_input))
            inference_time = time.time() - start_time
            
            logger.info(f"✅ Model loading time: {loading_time:.2f}s")
            logger.info(f"✅ Inference time: {inference_time:.2f}s")
            
            # Check if times are reasonable
            if loading_time < 10.0 and inference_time < 5.0:  # Increased threshold for CPU environments
                logger.info("✅ Performance benchmarks passed")
                return True
            else:
                logger.error(f"❌ Performance too slow - loading: {loading_time:.2f}s, inference: {inference_time:.2f}s")
                return False
                
        except Exception as e:
            logger.error(f"❌ Performance benchmark failed: {e}")
            return False
    
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
                logger.error(f"❌ {script_name} failed, but continuing...")
        
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
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if isinstance(result, bool) and result)
        
        report = f"""
🎯 COMPREHENSIVE CI PIPELINE REPORT
{'=' * 60}

📊 SUMMARY:
- Total Tests: {total_tests}
- Passed: {passed_tests}
- Failed: {total_tests - passed_tests}
- Success Rate: {(passed_tests/total_tests)*100:.1f}%

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
        
        if passed_tests == total_tests:
            report += "🎉 All tests passed! Pipeline is ready for deployment.\n"
        else:
            failed_tests = [name for name, result in self.results.items() 
                          if isinstance(result, bool) and not result]
            report += f"⚠️ Failed tests: {', '.join(failed_tests)}\n"
            report += "🔧 Please fix the failed tests before deployment.\n"
        
        return report


def main():
    """Main function to run the CI pipeline."""
    runner = CIPipelineRunner()
    
    try:
        results = runner.run_full_pipeline()
        report = runner.generate_report()
        
        print(report)
        
        # Write report to file
        with open("ci_pipeline_report.txt", "w") as f:
            f.write(report)
        
        # Exit with appropriate code
        total_tests = len([r for r in results.values() if isinstance(r, bool)])
        passed_tests = sum(1 for r in results.values() if isinstance(r, bool) and r)
        
        if passed_tests == total_tests:
            logger.info("🎉 CI Pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ CI Pipeline failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️ CI Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 CI Pipeline crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 