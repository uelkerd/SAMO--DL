    # --- Calibration Search ---
    # --- Load Data ---
    # --- Load Model ---
    # --- Report Results ---
#!/usr/bin/env python3
from src
    .models.emotion_detection
    .bert_classifier import create_bert_emotion_classifier, EmotionDataset
from src.models.emotion_detection.dataset_loader import GoEmotionsDataLoader
from pathlib import Path
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
import logging
import numpy as np
import sys
import torch







"""
Model Calibration Script

This script finds the optimal temperature and threshold for the emotion detection
model by evaluating its performance on the validation set across a range of values.
"""

sys.path.append(str(Path.cwd() / "src"))

def calibrate_model():
    """Find the best temperature and threshold for the model."""
    logging.info("🚀 Starting Model Calibration Script")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: {device}")

    logging.info("🤖 Loading trained model...")
    checkpoint_path = Path("test_checkpoints/best_model.pt")
    if not checkpoint_path.exists():
        logging.info("❌ Model checkpoint not found!")
        return

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )  # Set to False
    model, _ = create_bert_emotion_classifier()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logging.info("✅ Model loaded successfully.")

    logging.info("📊 Loading validation data...")
    data_loader = GoEmotionsDataLoader()
    datasets = data_loader.prepare_datasets()
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)

    val_dataset = EmotionDataset(
        texts=datasets["validation"]["text"],
        labels=datasets["validation"]["labels"],
        tokenizer=tokenizer,
        max_length=128,  # Use a reasonable max length
    )
    val_dataloader = DataLoader(val_dataset, batch_size=64)
    logging.info("✅ Loaded {len(val_dataset)} validation samples.")

    temperatures = np.linspace(1.0, 15.0, 15)
    thresholds = np.linspace(0.1, 0.9, 9)
    best_f1 = 0

    results = []

    logging.info("\n🌡️ Starting calibration search...")
    for temp in temperatures:
        model.set_temperature(temp)

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Temp: {temp:.1f}", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask)
                probabilities = torch.sigmoid(logits)

                all_probs.append(probabilities.cpu())
                all_labels.append(labels.cpu())

        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        for thresh in thresholds:
            predictions = (all_probs > thresh).astype(int)
            micro_f1 = f1_score(
                                all_labels,
                                predictions,
                                average="micro",
                                zero_division=0
                               )

            results.append((temp, thresh, micro_f1))

            best_f1 = max(best_f1, micro_f1)

    logging.info("\n🎉 Calibration Complete!")
    logging.info("=" * 50)
    logging.info("🏆 Best Micro F1 Score: {best_f1:.4f}")
    logging.info("🔥 Best Temperature:     {best_temp:.2f}")
    logging.info("🎯 Best Threshold:       {best_thresh:.2f}")
    logging.info("=" * 50)

    logging.info("\nTop 5 Results:")
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    for _i, (_temp, _thresh, _f1) in enumerate(sorted_results[:5]):
        logging.info(" {i+1}. Temp: {temp:.2f}, Thresh: {thresh:.2f}, F1: {f1:.4f}")


if __name__ == "__main__":
    calibrate_model()
