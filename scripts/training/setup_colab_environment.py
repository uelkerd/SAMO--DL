#!/usr/bin/env python3
"""
Colab Environment Setup Script for SAMO Deep Learning

This script sets up the environment for Google Colab with GPU support.
It installs all required dependencies and configures the environment
for optimal performance in the Colab environment.
"""

import logging
import os
import subprocess
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_colab_environment():
    """Detect if running in Google Colab."""
    is_colab = "COLAB_GPU" in os.environ
    if is_colab:
        logger.info("🎯 Detected Google Colab environment")
        logger.info("📊 Colab GPU: {os.environ.get("COLAB_GPU', 'unknown')}")
        return True
    else:
        logger.info("💻 Running in local environment")
        return False


def install_dependencies():
    """Install all required dependencies."""
    logger.info("📦 Installing dependencies...")

    # Core ML dependencies
    packages = [
        "torch>=2.1.0,<2.2.0",
        "torchvision>=0.16.0,<0.17.0",
        "torchaudio>=2.1.0,<2.2.0",
        "transformers>=4.30.0,<5.0.0",
        "datasets>=2.10.0,<3.0.0",
        "tokenizers>=0.13.0,<1.0.0",
        "pandas>=2.0.0,<3.0.0",
        "numpy>=1.24.0,<2.0.0",
        "scikit-learn>=1.3.0,<2.0.0",
        "fastapi>=0.100.0,<1.0.0",
        "uvicorn>=0.20.0,<1.0.0",
        "pydantic>=2.0.0,<3.0.0",
        "pytest>=7.0.0,<8.0.0",
        "pytest-cov>=4.0.0,<5.0.0",
        "pytest-asyncio>=0.21.0,<1.0.0",
        "black>=23.0.0,<24.0.0",
        "ruff>=0.1.0,<1.0.0",
        "sentencepiece>=0.1.99",
        "openai-whisper>=20231117",
        "pydub>=0.25.1",
        "jiwer>=3.0.3",
        "onnx>=1.14.0,<2.0.0",
        "onnxruntime>=1.15.0,<2.0.0",
        "python-dotenv>=1.0.0,<2.0.0",
        "accelerate>=0.20.0,<1.0.0",
    ]

    for package in packages:
        try:
            logger.info(f"📦 Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package],
                         check=True, capture_output=True, text=True)
            logger.info(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install {package}: {e}")
            return False

    return True


def setup_gpu_environment():
    """Set up GPU environment for optimal performance."""
    logger.info("🖥️ Setting up GPU environment...")

    try:
        import torch

        if torch.cuda.is_available():
            logger.info(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"🎮 GPU count: {torch.cuda.device_count()}")
            logger.info(f"🎮 CUDA version: {torch.version.cuda}")

            # Set environment variables for optimal GPU performance
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            # Test GPU functionality
            device = torch.device("cuda")
            test_tensor = torch.randn(100, 100).to(device)
            result = torch.matmul(test_tensor, test_tensor.T)
            logger.info(f"✅ GPU test successful, result shape: {result.shape}")

            return True
        else:
            logger.warning("⚠️ No GPU available, using CPU")
            return True

    except ImportError:
        logger.error("❌ PyTorch not available for GPU setup")
        return False
    except Exception as e:
        logger.error(f"❌ GPU setup failed: {e}")
        return False


def create_colab_notebook():
    """Create a Colab-ready notebook template."""
    logger.info("📓 Creating Colab notebook template...")

    notebook_content = '''{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "setup"
      },
      "source": [
        "# SAMO Deep Learning - Colab Environment Setup\\n",
        "\\n",
        "This notebook sets up the environment for SAMO Deep Learning with GPU support."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "install_deps"
      },
      "outputs": [],
      "source": [
        "# Install dependencies\\n",
        "!pip install torch>=2.1.0,<2.2.0 torchvision>=0.16.0,<0.17.0 torchaudio>=2.1.0,<2.2.0\\n",
        "!pip install transformers>=4.30.0,<5.0.0 datasets>=2.10.0,<3.0.0 tokenizers>=0.13.0,<1.0.0\\n",
        "!pip install fastapi>=0.100.0,<1.0.0 uvicorn>=0.20.0,<1.0.0 pydantic>=2.0.0,<3.0.0\\n",
        "!pip install sentencepiece>=0.1.99 openai-whisper>=20231117 pydub>=0.25.1 jiwer>=3.0.3\\n",
        "!pip install onnx>=1.14.0,<2.0.0 onnxruntime>=1.15.0,<2.0.0\\n",
        "!pip install pytest>=7.0.0,<8.0.0 black>=23.0.0,<24.0.0 ruff>=0.1.0,<1.0.0"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "clone_repo"
      },
      "outputs": [],
      "source": [
        "# Clone the repository\\n",
        "!git clone https://github.com/your-username/SAMO--DL.git\\n",
        "%cd SAMO--DL"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "test_gpu"
      },
      "outputs": [],
      "source": [
        "# Test GPU availability\\n",
        "import torch\\n",
        "print(f\"CUDA available: {torch.cuda.is_available()}\")\\n",
        "if torch.cuda.is_available():\\n",
        "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\\n",
        "    print(f\"GPU count: {torch.cuda.device_count()}\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "run_ci"
      },
      "outputs": [],
      "source": [
        "# Run CI pipeline\\n",
        "!python scripts/ci/run_full_ci_pipeline.py"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "name": "SAMO Deep Learning Setup",
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.10.0"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}'''

    with open("samo_dl_colab_setup.ipynb", "w") as f:
        f.write(notebook_content)

    logger.info("✅ Colab notebook template created: samo_dl_colab_setup.ipynb")
    return True


def run_ci_pipeline():
    """Run the CI pipeline to verify everything is working."""
    logger.info("🚀 Running CI pipeline verification...")

    try:
        result = subprocess.run(
            [sys.executable, "scripts/ci/run_full_ci_pipeline.py"],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode == 0:
            logger.info("✅ CI pipeline verification passed")
            logger.info("📊 CI Results:")
            logger.info(result.stdout)
            return True
        else:
            logger.error("❌ CI pipeline verification failed")
            logger.error(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        logger.error("⏰ CI pipeline verification timed out")
        return False
    except Exception as e:
        logger.error(f"💥 CI pipeline verification error: {e}")
        return False


def main():
    """Main setup function."""
    logger.info("🚀 Starting Colab Environment Setup")
    logger.info("=" * 50)

    # Detect environment
    is_colab = detect_colab_environment()

    # Install dependencies
    if not install_dependencies():
        logger.error("❌ Dependency installation failed")
        sys.exit(1)

    # Setup GPU environment
    if not setup_gpu_environment():
        logger.error("❌ GPU environment setup failed")
        sys.exit(1)

    # Create Colab notebook
    if is_colab:
        create_colab_notebook()

    # Run CI pipeline verification
    if not run_ci_pipeline():
        logger.error("❌ CI pipeline verification failed")
        sys.exit(1)

    logger.info("🎉 Colab environment setup completed successfully!")
    logger.info("=" * 50)
    logger.info("📋 Next steps:")
    logger.info("1. Upload the repository to Colab")
    logger.info("2. Run the CI pipeline: python scripts/ci/run_full_ci_pipeline.py")
    logger.info("3. Start developing with GPU acceleration!")

    if is_colab:
        logger.info("📓 Colab notebook template created: samo_dl_colab_setup.ipynb")


if __name__ == "__main__":
    main()
