import logging

import sys

#!/usr/bin/env python3
from pathlib import Path

import torch
from models.emotion_detection.bert_classifier import (
from models.emotion_detection.dataset_loader import create_goemotions_loader


    from torch.utils.data import DataLoader
    from models.emotion_detection.bert_classifier import EmotionDataset

    # Create GoEmotions loader
    from transformers import AutoTokenizer


"""
Simple Temperature Scaling Test - Direct Model Loading.
"""

sys.path.append(str(Path.cwd() / "src"))

    create_bert_emotion_classifier,
    evaluate_emotion_classifier,
)
def simple_temperature_test():
    logging.info("🌡️ Simple Temperature Scaling Test")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: {device}")

    # Load checkpoint
    checkpoint_path = Path("test_checkpoints/best_model.pt")
    if not checkpoint_path.exists():
        logging.info("❌ Model not found")
        return

    logging.info("📦 Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create data loader for evaluation
    logging.info("📊 Preparing validation data...")
    goemotions_loader = create_goemotions_loader()
    datasets = goemotions_loader.prepare_datasets()

    # Get validation data
    val_texts = datasets["validation"]["text"]
    val_labels = datasets["validation"]["labels"]

    # Use smaller dataset for quick test (10% of validation)
    val_size = len(val_texts)
    dev_val_size = int(val_size * 0.1)
    val_indices = torch.randperm(val_size)[:dev_val_size].tolist()
    val_texts = [val_texts[i] for i in val_indices]
    val_labels = [val_labels[i] for i in val_indices]

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Create dataset and dataloader
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, max_length=512)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Create model
    logging.info("🤖 Creating model...")
    model, _ = create_bert_emotion_classifier(
        model_name="bert-base-uncased", class_weights=None, freeze_bert_layers=0
    )

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logging.info("✅ Model loaded successfully")

    # Test temperatures
    temperatures = [1.0, 2.0, 3.0, 4.0]
    threshold = 0.5

    logging.info("\n🎯 Testing temperatures with threshold {threshold}")
    logging.info("-" * 50)

    for temp in temperatures:
        logging.info("\n🌡️ Temperature: {temp}")

        # Update temperature
        model.set_temperature(temp)

        # Quick evaluation
        metrics = evaluate_emotion_classifier(model, val_loader, device, threshold=threshold)

        logging.info("  📊 Macro F1: {metrics['macro_f1']:.4f}")
        logging.info("  📊 Micro F1: {metrics['micro_f1']:.4f}")

    logging.info("\n🎉 Temperature scaling test complete!")


if __name__ == "__main__":
    simple_temperature_test()
