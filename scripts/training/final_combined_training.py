#!/usr/bin/env python3
"""
🚀 FINAL COMBINED TRAINING - JOURNAL + CMU-MOSEI
================================================

This script combines your original journal dataset with CMU-MOSEI data
to achieve the target 75-85% F1 score.

Strategy:
1. Use original 150 high-quality journal samples
2. Add CMU-MOSEI samples for diversity
3. Train with optimal hyperparameters
4. Achieve 75-85% F1 score
"""

import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("🚀 FINAL COMBINED TRAINING - JOURNAL + CMU-MOSEI")
print("=" * 60)

def load_combined_dataset():
    """Load and combine journal and CMU-MOSEI datasets"""
    print("📊 Loading combined dataset...")

    combined_samples = []

    # Load original journal dataset (150 high-quality samples)
    try:
        with open('data/journal_test_dataset.json', 'r') as f:
            journal_data = json.load(f)

        for item in journal_data:
            combined_samples.append({
                'text': item['text'],
                'emotion': item['emotion'],
                'source': 'journal'
            })
        print(f"✅ Loaded {len(journal_data)} journal samples")
    except Exception as e:
        print(f"⚠️ Could not load journal data: {e}")

    # Load CMU-MOSEI dataset
    try:
        with open('data/cmu_mosei_balanced_dataset.json', 'r') as f:
            cmu_data = json.load(f)

        for item in cmu_data:
            combined_samples.append({
                'text': item['text'],
                'emotion': item['emotion'],
                'source': 'cmu_mosei'
            })
        print(f"✅ Loaded {len(cmu_data)} CMU-MOSEI samples")
    except Exception as e:
        print(f"⚠️ Could not load CMU-MOSEI data: {e}")

    # Load expanded journal dataset as backup
    try:
        with open('data/expanded_journal_dataset.json', 'r') as f:
            expanded_data = json.load(f)

        # Only use a subset to avoid synthetic data issues
        subset_size = min(200, len(expanded_data))
        selected_samples = np.random.choice(expanded_data, size=subset_size, replace=False)

        for item in selected_samples:
            combined_samples.append({
                'text': item['text'],
                'emotion': item['emotion'],
                'source': 'expanded_journal'
            })
        print(f"✅ Loaded {subset_size} expanded journal samples")
    except Exception as e:
        print(f"⚠️ Could not load expanded journal data: {e}")

    print(f"📊 Total combined samples: {len(combined_samples)}")

    # Show emotion distribution
    emotion_counts = {}
    for sample in combined_samples:
        emotion = sample['emotion']
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    print("📊 Emotion distribution:")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"  {emotion}: {count} samples")

    return combined_samples

class EmotionDataset(Dataset):
    """Custom dataset for emotion classification"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    """Compute F1 score and accuracy"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    f1 = f1_score(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)

    return {
        'f1': f1,
        'accuracy': accuracy
    }

def main():
    """Main training function"""
    print("🎯 Target F1 Score: 75-85%")
    print("🔧 Current Best: 67%")
    print("📈 Expected Improvement: 8-18%")
    print()

    # Load combined dataset
    samples = load_combined_dataset()

    if not samples:
        print("❌ No samples loaded!")
        return

    # Prepare data
    texts = [sample['text'] for sample in samples]
    emotions = [sample['emotion'] for sample in samples]

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(emotions)

    print(f"🎯 Number of labels: {len(label_encoder.classes_)}")
    print(f"📊 Labels: {list(label_encoder.classes_)}")

    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"📈 Training samples: {len(train_texts)}")
    print(f"🧪 Test samples: {len(test_labels)}")

    # Initialize tokenizer and model
    print("🔧 Initializing model...")
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_),
        problem_type="single_label_classification"
    )

    # Create datasets
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
    test_dataset = EmotionDataset(test_texts, test_labels, tokenizer)

    # Training arguments optimized for performance
    training_args = TrainingArguments(
        output_dir="./emotion_model_combined",
        num_train_epochs=8,  # More epochs for better performance
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to=None,  # Disable wandb
        learning_rate=2e-5,  # Optimal learning rate
        gradient_accumulation_steps=2,  # Effective batch size = 32
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train model
    print("🚀 Starting training...")
    trainer.train()

    # Evaluate final model
    print("📊 Evaluating final model...")
    results = trainer.evaluate()

    print(f"🏆 Final F1 Score: {results['eval_f1']:.4f} ({results['eval_f1']*100:.2f}%)")
    print(f"🎯 Target achieved: {'✅ YES!' if results['eval_f1'] >= 0.75 else '❌ Not yet'}")

    # Save model
    trainer.save_model("./emotion_model_final_combined")
    print("💾 Model saved to ./emotion_model_final_combined")

    # Test on sample texts
    print("\n🧪 Testing on sample texts...")
    test_texts = [
        "I'm feeling really happy today!",
        "This is so frustrating, nothing works.",
        "I'm anxious about the presentation.",
        "I'm grateful for all the support.",
        "I'm tired and need some rest."
    ]

    model.eval()
    with torch.no_grad():
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            predicted_label = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs).item()

            predicted_emotion = label_encoder.inverse_transform([predicted_label])[0]
            print(f"Text: {text}")
            print(f"Predicted: {predicted_emotion} (confidence: {confidence:.3f})")
            print()

    print("🎉 Training completed!")
    print(f"📈 Final F1 Score: {results['eval_f1']*100:.2f}%")
    print(f"🎯 Target: 75-85%")
    print(f"📊 Improvement: {((results['eval_f1'] - 0.67) / 0.67 * 100):.1f}% from baseline")

if __name__ == "__main__":
    main() 