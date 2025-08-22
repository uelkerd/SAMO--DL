#!/usr/bin/env python3
"""
Retrain the emotion detection model with the expanded dataset.
"""

import json
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

def load_expanded_dataset():
    """Load the expanded journal dataset."""
    print("📊 Loading expanded dataset...")

    with open('data/expanded_journal_dataset.json', 'r') as f:
        data = json.load(f)

    print(f"✅ Loaded {len(data)} samples")

    # Analyze distribution
    emotion_counts = {}
    for entry in data:
        emotion = entry['emotion']
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    print("📈 Emotion distribution:")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"  {emotion}: {count} samples")

    return data

class ExpandedEmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
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

class ExpandedEmotionClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_labels=12):
        super().__init__()
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(self.dropout(pooled_output))
        return logits

def prepare_expanded_data(data, test_size=0.2, val_size=0.1):
    """Prepare data for training with expanded dataset."""
    print("🔧 Preparing expanded data...")

    # Extract texts and emotions
    texts = [entry['content'] for entry in data]
    emotions = [entry['emotion'] for entry in data]

    # Create label encoder
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(emotions)

    print(f"✅ Label encoder created with {len(label_encoder.classes_)} classes")
    print(f"📊 Classes: {list(label_encoder.classes_)}")

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
    )

    print("📊 Data split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder

def train_expanded_model(train_data, val_data, label_encoder, epochs=5, batch_size=16):
    """Train the model with expanded dataset."""
    print("🚀 Training with expanded dataset...")

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Create datasets
    X_train, y_train = train_data
    X_val, y_val = val_data

    train_dataset = ExpandedEmotionDataset(X_train, y_train, tokenizer)
    val_dataset = ExpandedEmotionDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = ExpandedEmotionClassifier(num_labels=len(label_encoder.classes_))
    model.to(device)

    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_f1 = 0
    training_history = []

    for epoch in range(epochs):
        print(f"\n🔄 Epoch {epoch + 1}/{epochs}")

        # Training
        model.train()
        total_loss = 0

        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 50 == 0:
                print(f"  Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        accuracy = accuracy_score(all_labels, all_preds)

        print(f"📊 Epoch {epoch + 1} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val F1 (Macro): {f1_macro:.4f}")
        print(f"  Val Accuracy: {accuracy:.4f}")

        # Save best model
        if f1_macro > best_f1:
            best_f1 = f1_macro
            torch.save(model.state_dict(), 'best_expanded_model.pth')
            print(f"💾 New best model saved! F1: {best_f1:.4f}")

        training_history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_f1_macro': f1_macro,
            'val_accuracy': accuracy
        })

    return model, training_history, best_f1

def save_expanded_results(training_history, best_f1, label_encoder, test_data):
    """Save training results."""
    print("💾 Saving results...")

    # Test final model
    X_test, y_test = test_data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load best model
    model = ExpandedEmotionClassifier(num_labels=len(label_encoder.classes_))
    model.load_state_dict(torch.load('best_expanded_model.pth'))
    model.to(device)
    model.eval()

    # Test predictions
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    test_dataset = ExpandedEmotionDataset(X_test, y_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate final metrics
    final_f1 = f1_score(all_labels, all_preds, average='macro')
    final_accuracy = accuracy_score(all_labels, all_preds)

    # Save results
    results = {
        'best_f1': best_f1,
        'final_f1': final_f1,
        'final_accuracy': final_accuracy,
        'target_achieved': final_f1 >= 0.70,
        'num_labels': len(label_encoder.classes_),
        'all_emotions': list(label_encoder.classes_),
        'training_history': training_history,
        'expanded_samples': len(X_test) + len([x for x in train_data[0]]) + len([x for x in val_data[0]]),
        'test_samples': len(X_test)
    }

    with open('expanded_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("✅ Results saved!")
    print(f"📊 Final F1 Score: {final_f1:.4f}")
    print(f"📊 Final Accuracy: {final_accuracy:.4f}")
    print(f"🎯 Target Achieved: {final_f1 >= 0.70}")

def main():
    """Main training function."""
    print("🚀 RETRAINING WITH EXPANDED DATASET")
    print("=" * 60)

    # Load expanded dataset
    data = load_expanded_dataset()

    # Prepare data
    train_data, val_data, test_data, label_encoder = prepare_expanded_data(data)

    # Train model
    model, training_history, best_f1 = train_expanded_model(train_data, val_data, label_encoder)

    # Save results
    save_expanded_results(training_history, best_f1, label_encoder, test_data)

    print("\n🎉 Retraining completed!")
    print("📋 Next steps:")
    print("  1. Test the new model")
    print("  2. Compare performance")
    print("  3. Deploy if target achieved!")

if __name__ == "__main__":
    main()
