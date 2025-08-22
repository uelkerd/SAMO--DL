#!/usr/bin/env python3
"""
Fix the label mapping issue between GoEmotions and Journal datasets.
"""

import subprocess
import sys
import json
from pathlib import Path


def install_dependencies() -> bool:
    """Install required dependencies."""
    print("🔧 Installing dependencies...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "datasets", "pandas", "transformers"]
        )
        print("✅ Dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    return True


# Install dependencies first
if not install_dependencies():
    print("❌ Cannot proceed without dependencies")
    sys.exit(1)


def analyze_label_mapping():
    """Analyze the label mapping issue."""
    print("🔍 Analyzing label mapping issue...")

    # Lazy imports after install
    from datasets import load_dataset  # type: ignore
    import pandas as pd  # type: ignore

    # Load datasets
    go_emotions = load_dataset("go_emotions", "simplified")
    with open('data/journal_test_dataset.json', 'r') as f:
        journal_entries = json.load(f)
    journal_df = pd.DataFrame(journal_entries)
    
    # Analyze GoEmotions labels
    print("\n📊 GoEmotions Analysis:")
    go_label_counts_map = {}
    for example in go_emotions['train']:
        if example['labels']:
            for label_val in example['labels']:
                go_label_counts_map[label_val] = go_label_counts_map.get(label_val, 0) + 1
    
    print(f"GoEmotions unique labels: {len(go_label_counts_map)}")
    print(f"GoEmotions labels: {sorted(list(go_label_counts_map.keys()))}")
    top10 = dict(
        sorted(go_label_counts_map.items(), key=lambda x: x[1], reverse=True)[:10]
    )
    print(f"Top 10 GoEmotions labels: {top10}")
    
    # Analyze Journal labels
    print("\n📊 Journal Analysis:")
    journal_label_counts_map = journal_df['emotion'].value_counts().to_dict()
    print(f"Journal unique labels: {len(journal_label_counts_map)}")
    print(f"Journal labels: {sorted(list(journal_label_counts_map.keys()))}")
    print(f"Journal label counts: {journal_label_counts_map}")
    
    # Check for any common labels
    go_labels_set = set(go_label_counts_map.keys())
    journal_labels_set = set(journal_label_counts_map.keys())
    common_labels = go_labels_set.intersection(journal_labels_set)
    
    print(f"\n🔍 Common labels: {len(common_labels)}")
    if common_labels:
        print(f"Common labels: {sorted(list(common_labels))}")
    else:
        print("❌ NO COMMON LABELS FOUND!")
        print("This is why we get 0 GoEmotions samples!")
    
    return go_label_counts_map, journal_label_counts_map


def create_emotion_mapping():
    """Create a mapping between GoEmotions and Journal emotions."""
    print("\n🔧 Creating emotion mapping...")
    
    # GoEmotions emotion labels (from their documentation)
    go_emotions_mapping = {
        'admiration': 'admiration',
        'amusement': 'happy',
        'anger': 'frustrated',
        'annoyance': 'frustrated',
        'approval': 'proud',
        'caring': 'content',
        'confusion': 'overwhelmed',
        'curiosity': 'excited',
        'desire': 'excited',
        'disappointment': 'sad',
        'disapproval': 'frustrated',
        'disgust': 'frustrated',
        'embarrassment': 'anxious',
        'excitement': 'excited',
        'fear': 'anxious',
        'gratitude': 'grateful',
        'grief': 'sad',
        'joy': 'happy',
        'love': 'content',
        'nervousness': 'anxious',
        'optimism': 'hopeful',
        'pride': 'proud',
        'realization': 'content',
        'relief': 'calm',
        'remorse': 'sad',
        'sadness': 'sad',
        'surprise': 'excited',
        'neutral': 'calm'
    }
    
    print(f"Created mapping with {len(go_emotions_mapping)} emotions")
    return go_emotions_mapping


def create_fixed_bulletproof_cell():
    """Create a fixed bulletproof cell with proper emotion mapping."""
    
    cell_code = '''# 🚀 BULLETPROOF TRAINING CELL - FIXED LABEL MAPPING
# Runtime → Change runtime type → GPU (T4 or V100)
# Kernel → Restart and run all

print("🚀 BULLETPROOF TRAINING FOR REQ-DL-012 - FIXED LABEL MAPPING")
print("=" * 60)

# Step 1: Clear everything and validate environment
import os
import sys
import json
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

print("✅ Imports successful")

# Clear GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"✅ GPU memory cleared: {torch.cuda.get_device_name()}")
else:
    print("⚠️ CUDA not available, using CPU")

# Test basic operations
try:
    test_tensor = torch.randn(2, 3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_tensor.to(device)
    print("✅ Basic tensor operations work")
except Exception as e:
    print(f"❌ Basic tensor operations failed: {e}")
    raise

# Step 2: Ensure we are at the repository root
start_path = Path(__file__).resolve() if "__file__" in globals() else Path.cwd()

def _find_repo_root(start: Path) -> Path:
    for d in [start] + list(start.parents):
        if (d / "src").exists() or (d / ".git").exists() or (d / "pyproject.toml").exists():
            return d
    return start

REPO_ROOT = _find_repo_root(start_path)
os.chdir(str(REPO_ROOT))

# Step 3: Create emotion mapping
print("\n🔧 Creating emotion mapping...")

# GoEmotions to Journal emotion mapping
emotion_mapping = {
    'admiration': 'proud',
    'amusement': 'happy',
    'anger': 'frustrated',
    'annoyance': 'frustrated',
    'approval': 'proud',
    'caring': 'content',
    'confusion': 'overwhelmed',
    'curiosity': 'excited',
    'desire': 'excited',
    'disappointment': 'sad',
    'disapproval': 'frustrated',
    'disgust': 'frustrated',
    'embarrassment': 'anxious',
    'excitement': 'excited',
    'fear': 'anxious',
    'gratitude': 'grateful',
    'grief': 'sad',
    'joy': 'happy',
    'love': 'content',
    'nervousness': 'anxious',
    'optimism': 'hopeful',
    'pride': 'proud',
    'realization': 'content',
    'relief': 'calm',
    'remorse': 'sad',
    'sadness': 'sad',
    'surprise': 'excited',
    'neutral': 'calm'
}

print(f"✅ Created mapping with {len(emotion_mapping)} emotions")

# Step 4: Load and prepare data with mapping
print("\n📊 Loading and preparing data with mapping...")

go_emotions = load_dataset("go_emotions", "simplified")
with open('data/journal_test_dataset.json', 'r') as f:
    journal_entries = json.load(f)
journal_df = pd.DataFrame(journal_entries)

# Get journal emotions
journal_emotions = set(journal_df['emotion'].unique())
print(f"📊 Journal emotions: {sorted(list(journal_emotions))}")

# Filter GoEmotions data using mapping
go_texts = []
go_labels = []
for example in go_emotions['train']:
    if example['labels']:
        for label in example['labels']:
            if label in emotion_mapping:
                mapped_emotion = emotion_mapping[label]
                if mapped_emotion in journal_emotions:
                    go_texts.append(example['text'])
                    go_labels.append(mapped_emotion)
                    break

# Prepare journal data
journal_texts = list(journal_df['content'])
journal_labels = list(journal_df['emotion'])

print(f"📊 Mapped GoEmotions: {len(go_texts)} samples")
print(f"📊 Journal: {len(journal_texts)} samples")

# Create unified label encoder
all_emotions = sorted(list(set(go_labels + journal_labels)))
print(f"📊 All emotions: {all_emotions}")

label_encoder = LabelEncoder()
label_encoder.fit(all_emotions)
label_to_id = {label: idx for idx, label in enumerate(label_encoder.classes_)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

print(f"✅ Label encoder created with {len(label_encoder.classes_)} classes")

# Convert labels to IDs
go_label_ids = [label_to_id[label] for label in go_labels]
journal_label_ids = [label_to_id[label] for label in journal_labels]

print(f"📊 GoEmotions label range: {min(go_label_ids)} to {max(go_label_ids)}")
print(f"📊 Journal label range: {min(journal_label_ids)} to {max(journal_label_ids)}")

# Step 5: Create simple dataset class
class SimpleEmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Validate data
        if len(texts) != len(labels):
            raise ValueError(f"Texts and labels have different lengths: {len(texts)} vs {len(labels)}")
        
        # Validate labels
        for i, label in enumerate(labels):
            if not isinstance(label, int) or label < 0:
                raise ValueError(f"Invalid label at index {i}: {label}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Validate inputs
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid text at index {idx}")
        
        if not isinstance(label, int) or label < 0:
            raise ValueError(f"Invalid label at index {idx}: {label}")
        
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

# Step 6: Create simple model
class SimpleEmotionClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_labels=None):
        super().__init__()
        
        if num_labels is None or num_labels <= 0:
            raise ValueError(f"Invalid num_labels: {num_labels}")
        
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        print(f"✅ Model initialized with {num_labels} labels")
    
    def forward(self, input_ids, attention_mask):
        # Validate inputs
        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids to be 2D, got {input_ids.dim()}D")
        
        if attention_mask.dim() != 2:
            raise ValueError(f"Expected attention_mask to be 2D, got {attention_mask.dim()}D")
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(self.dropout(pooled_output))
        
        # Validate outputs
        if logits.shape[-1] != self.num_labels:
            raise ValueError(f"Expected {self.num_labels} output classes, got {logits.shape[-1]}")
        
        return logits

# Step 7: Setup training
print("\n🚀 Setting up training...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
num_labels = len(label_encoder.classes_)
model = SimpleEmotionClassifier(model_name="bert-base-uncased", num_labels=num_labels)
model = model.to(device)

# Create datasets
go_dataset = SimpleEmotionDataset(go_texts, go_label_ids, tokenizer)
journal_dataset = SimpleEmotionDataset(journal_texts, journal_label_ids, tokenizer)

# Split journal data
journal_train_texts, journal_val_texts, journal_train_labels, journal_val_labels = train_test_split(
    journal_texts, journal_label_ids, test_size=0.3, random_state=42, stratify=journal_label_ids
)

journal_train_dataset = SimpleEmotionDataset(journal_train_texts, journal_train_labels, tokenizer)
journal_val_dataset = SimpleEmotionDataset(journal_val_texts, journal_val_labels, tokenizer)

# Create dataloaders
go_loader = DataLoader(go_dataset, batch_size=8, shuffle=True)
journal_train_loader = DataLoader(journal_train_dataset, batch_size=8, shuffle=True)
journal_val_loader = DataLoader(journal_val_dataset, batch_size=8, shuffle=False)

print(f"✅ Training samples: {len(go_dataset)} GoEmotions + {len(journal_train_dataset)} Journal")
print(f"✅ Validation samples: {len(journal_val_dataset)} Journal")

# Step 8: Training loop
print("\n🚀 Starting training...")

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

num_epochs = 3  # Reduced for testing
best_f1 = 0.0

for epoch in range(num_epochs):
    print(f"\n🔄 Epoch {epoch + 1}/{num_epochs}")
    
    # Training
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Train on GoEmotions
    print("  📚 Training on GoEmotions...")
    for i, batch in enumerate(go_loader):
        try:
            # Validate batch
            if 'input_ids' not in batch or 'attention_mask' not in batch or 'labels' not in batch:
                print(f"⚠️ Invalid batch structure at batch {i}")
                continue
            
            # Move to device with validation
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Validate labels
            if torch.any(labels >= num_labels) or torch.any(labels < 0):
                print(f"⚠️ Invalid labels in batch {i}: {labels}")
                continue
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if i % 50 == 0:
                print(f"    Batch {i}/{len(go_loader)}, Loss: {loss.item():.4f}")
                
        except Exception as e:
            print(f"❌ Error in batch {i}: {e}")
            continue
    
    # Train on journal data
    print("  📝 Training on journal data...")
    for i, batch in enumerate(journal_train_loader):
        try:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            if torch.any(labels >= num_labels) or torch.any(labels < 0):
                continue
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if i % 10 == 0:
                print(f"    Batch {i}/{len(journal_train_loader)}, Loss: {loss.item():.4f}")
                
        except Exception as e:
            print(f"❌ Error in journal batch {i}: {e}")
            continue
    
    # Validation
    print("  🎯 Validating...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in journal_val_loader:
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            except Exception as e:
                print(f"❌ Error in validation batch: {e}")
                continue
    
    # Calculate metrics
    if all_preds and all_labels:
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        accuracy = accuracy_score(all_labels, all_preds)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        print(f"  📊 Epoch {epoch + 1} Results:")
        print(f"    Average Loss: {avg_loss:.4f}")
        print(f"    Validation F1 (Macro): {f1_macro:.4f}")
        print(f"    Validation Accuracy: {accuracy:.4f}")
        
        # Save best model
        if f1_macro > best_f1:
            best_f1 = f1_macro
            torch.save(model.state_dict(), 'best_simple_model.pth')
            print(f"    💾 New best model saved! F1: {best_f1:.4f}")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\n🏆 Training completed! Best F1 Score: {best_f1:.4f}")

# Step 9: Save results
results = {
    'best_f1': best_f1,
    'num_labels': num_labels,
    'target_achieved': best_f1 >= 0.7,
    'go_samples': len(go_texts),
    'journal_samples': len(journal_texts),
    'emotion_mapping': emotion_mapping
}

with open('simple_training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ Training completed successfully!")
print(f"📊 Final F1 Score: {best_f1:.4f}")
print(f"🎯 Target Met: {'✅' if best_f1 >= 0.7 else '❌'}")

# Download results
from google.colab import files
files.download('best_simple_model.pth')
files.download('simple_training_results.json')

print("\n🎉 BULLETPROOF TRAINING COMPLETED!")
print("📁 Files downloaded: best_simple_model.pth, simple_training_results.json")'''
    
    # Write to file
    with open('bulletproof_training_cell_fixed.py', 'w') as f:
        f.write(cell_code)
    
    print("✅ Created fixed bulletproof training cell: bulletproof_training_cell_fixed.py")
    print("📋 This version has proper emotion mapping!")

if __name__ == "__main__":
    # Analyze the issue
    go_label_counts, journal_label_counts = analyze_label_mapping()
    
    # Create emotion mapping
    emotion_mapping = create_emotion_mapping()
    
    # Create fixed bulletproof cell
    create_fixed_bulletproof_cell()
    
    print("\n🎯 SUMMARY:")
    print("The issue was that GoEmotions uses emotion names (like 'admiration')")
    print("while Journal uses different emotion names (like 'proud').")
    print("The fixed version maps GoEmotions emotions to Journal emotions!") 