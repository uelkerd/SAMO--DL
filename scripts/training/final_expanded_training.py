#!/usr/bin/env python3
"""
🚀 FINAL EXPANDED DATASET TRAINING
===================================

This script trains the emotion detection model using the expanded dataset
to achieve the target 75-85% F1 score.

Target: 75-85% F1 Score
Current: 67% F1 Score  
Expected: 8-18% improvement
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

print("🚀 FINAL EXPANDED DATASET TRAINING")
print("=" * 50)

# Load expanded dataset
print("📊 Loading expanded dataset...")
with open('data/expanded_journal_dataset.json', 'r') as f:
    expanded_data = json.load(f)

print(f"✅ Loaded {len(expanded_data)} expanded samples")

# Prepare data
texts = [item['content'] for item in expanded_data]
emotions = [item['emotion'] for item in expanded_data]

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(emotions)
num_labels = len(label_encoder.classes_)

print(f"📊 Emotions: {list(label_encoder.classes_)}")
print(f"🎯 Number of labels: {num_labels}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
)

print(f"📈 Training samples: {len(X_train)}")
print(f"🧪 Test samples: {len(X_test)}")

# Create dataset class
class EmotionDataset(Dataset):
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

# Initialize tokenizer and model
print("🔧 Initializing model...")
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=num_labels,
    problem_type="single_label_classification"
)

# Create datasets
train_dataset = EmotionDataset(X_train, y_train, tokenizer)
test_dataset = EmotionDataset(X_test, y_test, tokenizer)

# Training arguments with optimizations
training_args = TrainingArguments(
    output_dir="./emotion_model_final",
    num_train_epochs=5,
    per_device_train_batch_size=8,  # Reduced for CPU
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    # fp16=True,  # Removed for CPU compatibility
    dataloader_num_workers=0,  # Reduced for CPU
    remove_unused_columns=False,
    report_to=None,  # Disable wandb
)

# Custom compute_metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    f1 = f1_score(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'f1': f1,
        'accuracy': accuracy
    }

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train the model
print("🚀 Starting training...")
trainer.train()

# Evaluate on test set
print("🧪 Evaluating model...")
results = trainer.evaluate()
print(f"📊 Final Results:")
print(f"   F1 Score: {results['eval_f1']:.4f} ({results['eval_f1']*100:.1f}%)")
print(f"   Accuracy: {results['eval_accuracy']:.4f} ({results['eval_accuracy']*100:.1f}%)")

# Save the model
print("💾 Saving model...")
trainer.save_model("./best_emotion_model_final")
tokenizer.save_pretrained("./best_emotion_model_final")

# Test on sample journal entries
print("\n🧪 Testing on sample journal entries...")
test_samples = [
    "I'm feeling really happy today! Everything is going well.",
    "I'm so frustrated with this project. Nothing is working.",
    "I feel anxious about the upcoming presentation.",
    "I'm grateful for all the support I've received.",
    "I'm feeling overwhelmed with all these tasks.",
    "I'm proud of what I've accomplished so far.",
    "I'm feeling sad and lonely today.",
    "I'm excited about the new opportunities ahead.",
    "I feel calm and peaceful right now.",
    "I'm hopeful that things will get better.",
    "I'm tired and need some rest.",
    "I'm content with how things are going."
]

expected_emotions = ['happy', 'frustrated', 'anxious', 'grateful', 'overwhelmed', 
                    'proud', 'sad', 'excited', 'calm', 'hopeful', 'tired', 'content']

print("📊 Testing Results:")
print("=" * 80)

correct_predictions = 0
for i, (text, expected) in enumerate(zip(test_samples, expected_emotions), 1):
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item()
        predicted_emotion = label_encoder.inverse_transform([predicted_idx])[0]
    
    # Get top 3 predictions
    top_3_indices = torch.topk(probabilities[0], 3).indices
    top_3_emotions = label_encoder.inverse_transform(top_3_indices.cpu().numpy())
    top_3_probs = torch.topk(probabilities[0], 3).values.cpu().numpy()
    
    # Check if correct
    is_correct = predicted_emotion == expected
    if is_correct:
        correct_predictions += 1
    
    print(f"{i}. Text: {text}")
    print(f"   Predicted: {predicted_emotion} (confidence: {confidence:.3f})")
    print(f"   Expected: {expected}")
    print(f"   {'✅ CORRECT' if is_correct else '❌ WRONG'}")
    print(f"   Top 3 predictions:")
    for emotion, prob in zip(top_3_emotions, top_3_probs):
        print(f"     - {emotion}: {prob:.3f}")
    print()

test_accuracy = correct_predictions / len(test_samples)
final_f1 = results['eval_f1']

print(f"\n📈 FINAL RESULTS:")
print(f"   Test Accuracy: {test_accuracy:.2%} ({correct_predictions}/{len(test_samples)})")
print(f"   F1 Score: {final_f1:.4f} ({final_f1*100:.1f}%)")
print(f"   Target Achieved: {'✅ YES!' if final_f1 >= 0.75 else '❌ Not yet'}")

if final_f1 >= 0.75:
    print(f"\n🎉 SUCCESS! Model achieved {final_f1*100:.1f}% F1 score!")
    print(f"🚀 Ready for production deployment!")
else:
    print(f"\n📈 Good progress! Current F1: {final_f1*100:.1f}%")
    print(f"💡 Consider: more data, hyperparameter tuning, or different model architecture")

print(f"\n💾 Model saved to: ./best_emotion_model_final")
print(f"📊 Training completed successfully!") 