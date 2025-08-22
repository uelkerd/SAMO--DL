    # Analyze convergence
    # Check model files
    # Extract metrics
    # F1 score curve
    # Generate plots
    # Load and analyze training history
    # Loss curve
    # Next Steps
    # Performance Metrics
    # Performance analysis
    # Recommendations
    # Save analysis report
    # Training Progress
    # Training time analysis
# Add src to path for imports
#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import sys



"""
Training Monitor for SAMO Emotion Detection Model

This script monitors the training progress of the emotion detection model
and provides insights on performance, convergence, and next steps.
"""

sys.path.append(str(Path(__file__).parent.parent / "src"))
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

def load_training_history(checkpoint_dir: str = "test_checkpoints_dev") -> list[dict]:
    """Load training history from checkpoint directory."""
    history_file = Path(checkpoint_dir) / "training_history.json"

    if not history_file.exists():
        logging.info(f"❌ Training history not found at {history_file}")
        return []

    with open(history_file) as f:
        history = json.load(f)

    return history

def analyze_training_progress(history: list[dict]) -> dict:
    """Analyze training progress and provide insights."""
    if not history:
        return {"error": "No training history found"}

    analysis = {
        "total_epochs": len(history),
        "latest_epoch": history[-1]["epoch"],
        "loss_progress": [],
        "f1_progress": [],
        "training_time": [],
        "learning_rate": [],
        "convergence_status": "unknown",
        "recommendations": []
    }

    for epoch_data in history:
        analysis["loss_progress"].append(epoch_data["train_loss"])
        analysis["f1_progress"].append(epoch_data["micro_f1"])
        analysis["training_time"].append(epoch_data["epoch_time"])
        analysis["learning_rate"].append(epoch_data["learning_rate"])

    if len(analysis["loss_progress"]) >= 2:
        latest_loss = analysis["loss_progress"][-1]
        previous_loss = analysis["loss_progress"][-2]
        loss_improvement = previous_loss - latest_loss

        if loss_improvement > 0.01:
            analysis["convergence_status"] = "excellent"
            analysis["recommendations"].append("✅ Loss decreasing significantly - continue training")
        elif loss_improvement > 0.001:
            analysis["convergence_status"] = "good"
            analysis["recommendations"].append("✅ Loss decreasing - continue training")
        elif loss_improvement > -0.001:
            analysis["convergence_status"] = "plateauing"
            analysis["recommendations"].append("⚠️ Loss plateauing - consider learning rate adjustment")
        else:
            analysis["convergence_status"] = "diverging"
            analysis["recommendations"].append("❌ Loss increasing - check learning rate and data")

    latest_f1 = analysis["f1_progress"][-1]
    if latest_f1 > 0.8:
        analysis["recommendations"].append("🎯 Excellent F1 score achieved!")
    elif latest_f1 > 0.6:
        analysis["recommendations"].append("📈 Good F1 score - continue training")
    else:
        analysis["recommendations"].append("📊 F1 score needs improvement - consider data augmentation")

    avg_epoch_time = np.mean(analysis["training_time"])
    analysis["avg_epoch_time_minutes"] = avg_epoch_time / 60

    if avg_epoch_time > 1200:  # 20 minutes
        analysis["recommendations"].append("⏱️ Training time is high - consider GPU acceleration")

    return analysis

def generate_training_report(analysis: dict) -> str:
    """Generate a comprehensive training report."""
    report = []
    report.append("=" * 60)
    report.append("🧠 SAMO Emotion Detection Training Report")
    report.append("=" * 60)
    report.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    report.append("📊 TRAINING PROGRESS")
    report.append("-" * 30)
    report.append(f"Total Epochs: {analysis['total_epochs']}")
    report.append(f"Latest Epoch: {analysis['latest_epoch']}")
    report.append(f"Convergence Status: {analysis['convergence_status'].upper()}")
    report.append("")

    if analysis["loss_progress"]:
        latest_loss = analysis["loss_progress"][-1]
        initial_loss = analysis["loss_progress"][0]
        loss_reduction = ((initial_loss - latest_loss) / initial_loss) * 100

        report.append("📈 PERFORMANCE METRICS")
        report.append("-" * 30)
        report.append(f"Initial Loss: {initial_loss:.4f}")
        report.append(f"Latest Loss: {latest_loss:.4f}")
        report.append(f"Loss Reduction: {loss_reduction:.1f}%")

        if analysis["f1_progress"]:
            latest_f1 = analysis["f1_progress"][-1]
            report.append(f"Latest F1 Score: {latest_f1:.4f}")

        report.append(f"Average Epoch Time: {analysis['avg_epoch_time_minutes']:.1f} minutes")
        report.append("")

    report.append("💡 RECOMMENDATIONS")
    report.append("-" * 30)
    for rec in analysis["recommendations"]:
        report.append(f"• {rec}")
    report.append("")

    report.append("🚀 NEXT STEPS")
    report.append("-" * 30)
    if analysis["convergence_status"] in ["excellent", "good"]:
        report.append("• Continue training for more epochs")
        report.append("• Monitor validation metrics")
        report.append("• Consider fine-tuning hyperparameters")
    elif analysis["convergence_status"] == "plateauing":
        report.append("• Reduce learning rate")
        report.append("• Add data augmentation")
        report.append("• Consider early stopping")
    else:
        report.append("• Check data quality")
        report.append("• Reduce learning rate significantly")
        report.append("• Verify model architecture")

    report.append("")
    report.append("=" * 60)

    return "\n".join(report)

def plot_training_curves(history: list[dict], save_path: Optional[str] = None):
    """Plot training curves for visualization."""
    if not history:
        logging.info("❌ No training history to plot")
        return

    epochs = [epoch["epoch"] for epoch in history]
    losses = [epoch["train_loss"] for epoch in history]
    f1_scores = [epoch["micro_f1"] for epoch in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(epochs, losses, 'b-o', linewidth=2, markersize=6)
    ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2.plot(epochs, f1_scores, 'g-o', linewidth=2, markersize=6)
    ax2.set_title('F1 Score Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Micro F1 Score')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"📊 Training curves saved to {save_path}")
    else:
        plt.show()

def check_model_files(checkpoint_dir: str = "test_checkpoints_dev") -> dict:
    """Check if model files exist and are valid."""
    checkpoint_path = Path(checkpoint_dir)

    files = {
        "training_history": checkpoint_path / "training_history.json",
        "best_model": checkpoint_path / "best_model.pt",
        "config": checkpoint_path / "config.json"
    }

    status = {}
    for name, file_path in files.items():
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            status[name] = {
                "exists": True,
                "size_mb": size_mb,
                "path": str(file_path)
            }
        else:
            status[name] = {
                "exists": False,
                "size_mb": 0,
                "path": str(file_path)
            }

    return status

def main():
    """Main monitoring function."""
    logging.info("🔍 SAMO Training Monitor")
    logging.info("=" * 40)

    logging.info("\n📁 Checking model files...")
    model_status = check_model_files()

    for name, info in model_status.items():
        if info["exists"]:
            logging.info(f"✅ {name}: {info['size_mb']:.1f}MB")
        else:
            logging.info(f"❌ {name}: Not found")

    logging.info("\n📊 Analyzing training progress...")
    history = load_training_history()

    if not history:
        logging.info("❌ No training history found. Run training first:")
        logging.info("   python -m src.models.emotion_detection.training_pipeline")
        return

    analysis = analyze_training_progress(history)
    report = generate_training_report(analysis)

    logging.info("\n" + report)

    logging.info("\n📈 Generating training curves...")
    plots_dir = Path("logs/plots")
    plots_dir.mkdir(exist_ok=True)
    plot_path = plots_dir / f"training_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_training_curves(history, str(plot_path))

    report_path = plots_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, 'w') as f:
        f.write(report)

    logging.info(f"\n📄 Analysis report saved to {report_path}")
    logging.info(f"📊 Training curves saved to {plot_path}")

if __name__ == "__main__":
    main()
