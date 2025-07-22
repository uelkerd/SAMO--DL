#!/usr/bin/env python3
"""GPU Training Setup Script for SAMO Deep Learning.

This script helps transition the current CPU training to GPU training
with optimal settings for performance and memory efficiency.

Usage:
    python scripts/setup_gpu_training.py --check
    python scripts/setup_gpu_training.py --resume-training --checkpoint ./test_checkpoints/best_model.pt
"""

import argparse
import logging
import os
from pathlib import Path

import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_gpu_environment() -> None:
    """Set up environment variables for optimal GPU training."""
    # Disable tokenizers parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Enable CUDA optimizations
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Async CUDA kernels for speed
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6"  # Support modern GPUs

    logger.info("✅ GPU environment configured")


def check_gpu_availability() -> bool:
    """Check GPU setup and provide optimization recommendations."""
    logger.info("🔍 Checking GPU availability...")

    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available. Install PyTorch with CUDA support:")
        print(
            "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        )
        return False

    # GPU Info
    device_name = torch.cuda.get_device_name()
    memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9

    logger.info(f"✅ GPU Available: {device_name}")
    logger.info(f"   Memory: {memory_total:.1f} GB")

    # Optimization recommendations
    print("\n💡 GPU Training Optimizations:")

    if memory_total >= 12:  # 12GB+ GPU
        print(f"   • Use batch_size=32 (you have {memory_total:.1f}GB memory)")
        print("   • Enable mixed precision training (fp16)")
        print("   • Consider gradient accumulation for larger effective batch sizes")
    elif memory_total >= 8:  # 8-12GB GPU
        print(f"   • Use batch_size=16-24 (you have {memory_total:.1f}GB memory)")
        print("   • Enable mixed precision training (fp16)")
        print("   • Monitor memory usage")
    else:  # <8GB GPU
        print(f"   • Use batch_size=8-12 (you have {memory_total:.1f}GB memory)")
        print("   • Enable mixed precision training (fp16) - REQUIRED")
        print("   • Consider gradient checkpointing to save memory")

    # Speed estimates
    if "A100" in device_name or "V100" in device_name:
        print("   • Expected training speedup: 15-20x vs CPU")
    elif "RTX" in device_name or "T4" in device_name:
        print("   • Expected training speedup: 8-12x vs CPU")
    else:
        print("   • Expected training speedup: 5-8x vs CPU")

    return True


def create_gpu_training_config():
    """Create optimized training configuration for GPU."""
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

    # Determine optimal batch size
    if gpu_memory >= 12:
        batch_size = 32
    elif gpu_memory >= 8:
        batch_size = 24
    else:
        batch_size = 12

    config = f"""# GPU Training Configuration for SAMO Deep Learning
# Auto-generated based on your GPU: {torch.cuda.get_device_name()}
# Memory: {gpu_memory:.1f} GB

import os
from src.models.emotion_detection.training_pipeline import EmotionDetectionTrainer

# Environment setup
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# GPU-optimized training parameters
trainer = EmotionDetectionTrainer(
    model_name="bert-base-uncased",
    cache_dir="./data/cache",
    output_dir="./models/checkpoints",
    batch_size={batch_size},  # Optimized for your GPU
    learning_rate=3e-5,  # Slightly higher LR for larger batches
    num_epochs=5,
    warmup_steps=500,
    weight_decay=0.01,
    freeze_initial_layers=6,
    unfreeze_schedule=[2, 4],  # Progressive unfreezing
    device="cuda"  # Force GPU usage
)

# Train the model
results = trainer.train()

print(f"\\nTraining completed!")
print(f"Best validation score: {{results['best_validation_score']:.4f}}")
print(f"Final test Macro F1: {{results['final_test_metrics']['macro_f1']:.4f}}")
"""

    # Save configuration
    config_path = Path("train_gpu.py")
    config_path.write_text(config)

    logger.info(f"✅ GPU training script created: {config_path}")
    logger.info("Run with: python train_gpu.py")

    return config_path


def resume_training_on_gpu(checkpoint_path: str) -> None:
    """Resume training from CPU checkpoint on GPU."""
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    logger.info(f"📁 Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", 0.0)

    logger.info(f"✅ Checkpoint loaded - Epoch: {epoch}, Best F1: {best_score:.4f}")

    # Create resume script
    resume_script = f"""#!/usr/bin/env python3
# Auto-generated GPU resume script

import torch
from src.models.emotion_detection.training_pipeline import EmotionDetectionTrainer

# Resume training on GPU from epoch {epoch}
trainer = EmotionDetectionTrainer(
    model_name="bert-base-uncased",
    cache_dir="./data/cache",
    output_dir="./models/checkpoints",
    batch_size=24,  # Optimized for GPU
    learning_rate=2e-5,
    num_epochs=5,  # Continue for additional epochs
    device="cuda"
)

# Load checkpoint and continue training
print("Resuming training on GPU from checkpoint...")
print("Note: You may need to manually implement checkpoint loading in the trainer")
print("Checkpoint path: {checkpoint_path}")

# Train normally - the trainer will create a new model
# TODO: Implement checkpoint resume functionality in trainer class
results = trainer.train()

print(f"\\nGPU training completed!")
print(f"Best validation score: {{results['best_validation_score']:.4f}}")
"""

    script_path = Path("resume_gpu_training.py")
    script_path.write_text(resume_script)

    logger.info(f"✅ Resume script created: {script_path}")
    print("\n💡 To resume training on GPU:")
    print("   1. Let current CPU training complete")
    print(f"   2. Run: python {script_path}")
    print("   3. Monitor GPU usage with: watch -n 1 nvidia-smi")


def main() -> None:
    parser = argparse.ArgumentParser(description="SAMO GPU Training Setup")
    parser.add_argument("--check", action="store_true", help="Check GPU availability")
    parser.add_argument("--create-config", action="store_true", help="Create GPU training config")
    parser.add_argument("--resume-training", action="store_true", help="Resume training on GPU")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path for resuming")

    args = parser.parse_args()

    # Setup environment
    setup_gpu_environment()

    if args.check or not any([args.create_config, args.resume_training]):
        if check_gpu_availability():
            print("\n🚀 Ready for GPU training!")
        else:
            return

    if args.create_config:
        if torch.cuda.is_available():
            create_gpu_training_config()
        else:
            logger.error("GPU not available. Install CUDA-compatible PyTorch first.")

    if args.resume_training:
        checkpoint_path = args.checkpoint or "./test_checkpoints/best_model.pt"
        resume_training_on_gpu(checkpoint_path)


if __name__ == "__main__":
    main()
