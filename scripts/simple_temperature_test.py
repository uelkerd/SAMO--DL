#!/usr/bin/env python3
"""
Simple Temperature Scaling Test - Direct Model Loading.
"""

import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / "src"))

import torch
from models.emotion_detection.bert_classifier import (
    create_bert_emotion_classifier,
    evaluate_emotion_classifier,
)
from models.emotion_detection.dataset_loader import create_goemotions_loader


def simple_temperature_test():
    print("🌡️ Simple Temperature Scaling Test")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint_path = Path("test_checkpoints/best_model.pt")
    if not checkpoint_path.exists():
        print("❌ Model not found")
        return

    print("📦 Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create data loader for evaluation
    print("📊 Preparing validation data...")
    from torch.utils.data import DataLoader
    from models.emotion_detection.bert_classifier import EmotionDataset

    # Create GoEmotions loader
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
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Create dataset and dataloader
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, max_length=512)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Create model
    print("🤖 Creating model...")
    model, _ = create_bert_emotion_classifier(
        model_name="bert-base-uncased", class_weights=None, freeze_bert_layers=0
    )

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully")

    # Test temperatures
    temperatures = [1.0, 2.0, 3.0, 4.0]
    threshold = 0.5

    print(f"\n🎯 Testing temperatures with threshold {threshold}")
    print("-" * 50)

    for temp in temperatures:
        print(f"\n🌡️ Temperature: {temp}")

        # Update temperature
        model.set_temperature(temp)

        # Quick evaluation
        metrics = evaluate_emotion_classifier(model, val_loader, device, threshold=threshold)

        print(f"  📊 Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  📊 Micro F1: {metrics['micro_f1']:.4f}")

    print("\n🎉 Temperature scaling test complete!")


if __name__ == "__main__":
    simple_temperature_test()
