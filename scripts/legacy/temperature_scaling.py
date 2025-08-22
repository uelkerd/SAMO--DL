        # Calibrate temperature
        # Create tokenized dataset
        # Extract raw validation data
        # Load checkpoint
        # Load dataset
        # Load trained model
        # Save calibrated model
        from src.models.emotion_detection.bert_classifier import EmotionDataset
        from transformers import AutoTokenizer
        import traceback
    # Collect logits and labels
    # Concatenate all batches
    # Create temperature scaling layer
    # Optimize temperature parameter
    # Setup device
# Add project root to path
# Configure logging
#!/usr/bin/env python3
from pathlib import Path
from src.models.emotion_detection.dataset_loader import GoEmotionsDataLoader
from src.models.emotion_detection.training_pipeline import create_bert_emotion_classifier
from torch import nn
import logging
import os
import sys
import torch
import traceback






"""
Temperature Scaling for Model Calibration

This script applies temperature scaling to improve model calibration
and potentially boost F1 score by 5-10%.
"""

project_root = Path__file__.parent.parent.resolve()
sys.path.append(strproject_root)

logging.basicConfig(level=logging.INFO, format="%asctimes - %levelnames - %messages")
logger = logging.getLogger__name__


class TemperatureScalingnn.Module:
    """Temperature scaling for model calibration."""

    def __init__self:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones1 * 1.5)

    def forwardself, logits:
        """Apply temperature scaling to logits."""
        return logits / self.temperature


def calibrate_temperaturemodel, val_loader, device:
    """Calibrate temperature parameter on validation set."""
    logger.info"🔧 Calibrating temperature parameter..."

    temperature_scaling = TemperatureScaling().todevice

    all_logits = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].todevice
            attention_mask = batch["attention_mask"].todevice
            labels = batch["labels"].float().todevice

            outputs = modelinput_ids, attention_mask=attention_mask
            logits = outputs["logits"]

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.catall_logits, dim=0
    all_labels = torch.catall_labels, dim=0

    optimizer = torch.optim.LBFGS[temperature_scaling.temperature], lr=0.01, max_iter=50

    def eval():
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(
            temperature_scalingall_logits, all_labels
        )
        loss.backward()
        return loss

    optimizer.stepeval

    optimal_temperature = temperature_scaling.temperature.item()
    logger.info"✅ Optimal temperature: {optimal_temperature:.3f}"

    return temperature_scaling


def apply_temperature_scaling():
    """Apply temperature scaling to improve model calibration."""

    logger.info"🌡️ Starting Temperature Scaling"
    logger.info"   • Expected improvement: 5-10% F1 score"
    logger.info"   • Method: Model calibration"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info"Using device: {device}"

    try:
        logger.info"Loading validation dataset..."
        data_loader = GoEmotionsDataLoader()
        datasets = data_loader.prepare_datasets()

        val_raw = datasets["validation"]
        val_texts = [item["text"] for item in val_raw]
        val_labels = [item["labels"] for item in val_raw]

        tokenizer = AutoTokenizer.from_pretrained"bert-base-uncased"
        val_dataset = EmotionDatasetval_texts, val_labels, tokenizer, max_length=512
        val_loader = torch.utils.data.DataLoaderval_dataset, batch_size=16, shuffle=False

        model_path = "./models/checkpoints/focal_loss_best_model.pt"
        if not Pathmodel_path:
            logger.error"❌ Model not found: {model_path}"
            logger.info"   • Please run focal_loss_training.py first"
            return False

        logger.info"Loading model from {model_path}"
        model, _ = create_bert_emotion_classifier(
            model_name="bert-base-uncased",
            class_weights=None,
            freeze_bert_layers=4,
        )
        model.todevice

        checkpoint = torch.loadmodel_path, map_location=device
        model.load_state_dictcheckpoint["model_state_dict"]
        logger.info"✅ Model loaded successfully"

        temperature_scaling = calibrate_temperaturemodel, val_loader, device

        output_dir = "./models/checkpoints"
        os.makedirsoutput_dir, exist_ok=True
        calibrated_path = Pathoutput_dir, "temperature_scaled_model.pt"

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "temperature_scaling_state_dict": temperature_scaling.state_dict(),
                "temperature": temperature_scaling.temperature.item(),
                "original_checkpoint": checkpoint,
            },
            calibrated_path,
        )

        logger.info"✅ Calibrated model saved to: {calibrated_path}"
        logger.info("   • Temperature: {temperature_scaling.temperature.item():.3f}")

        return True

    except Exception as e:
        logger.error"❌ Temperature scaling failed: {e}"
        traceback.print_exc()
        return False


def main():
    """Main function."""
    logger.info"🌡️ Temperature Scaling Script"
    logger.info"This script calibrates the model for better F1 scores"

    success = apply_temperature_scaling()

    if success:
        logger.info"✅ Temperature scaling completed successfully!"
        sys.exit0
    else:
        logger.error"❌ Temperature scaling failed. Check the logs above."
        sys.exit1


if __name__ == "__main__":
    main()
