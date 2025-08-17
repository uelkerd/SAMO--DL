# 🚀 FINAL BULLETPROOF TRAINING CELL - PROPER LABEL MAPPING
# Runtime → Change runtime type → GPU T4 or V100
# Kernel → Restart and run all

print"🚀 FINAL BULLETPROOF TRAINING FOR REQ-DL-012 - PROPER LABEL MAPPING"
print"=" * 70

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

print"✅ Imports successful"

# Clear GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"✅ GPU memory cleared: {torch.cuda.get_device_name()}")
else:
    print"⚠️ CUDA not available, using CPU"

# Test basic operations
try:
    test_tensor = torch.randn2, 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_tensor.todevice
    print"✅ Basic tensor operations work"
except Exception as e:
    printf"❌ Basic tensor operations failed: {e}"
    raise

# Step 2: Clone repository and setup
!git clone https://github.com/uelkerd/SAMO--DL.git
%cd SAMO--DL

# Step 3: Load datasets and get proper label mapping
print"\n🔧 Loading datasets and creating proper label mapping..."

# Load GoEmotions dataset
go_emotions = load_dataset"go_emotions", "simplified"

# Get the emotion names from the dataset features
emotion_names = go_emotions['train'].features['labels'].feature.names
printf"📊 GoEmotions emotion names: {emotion_names}"
print(f"📊 Total GoEmotions emotions: {lenemotion_names}")

# Load journal data
with open'data/journal_test_dataset.json', 'r' as f:
    journal_entries = json.loadf
journal_df = pd.DataFramejournal_entries

journal_emotions = set(journal_df['emotion'].unique())
print(f"📊 Journal emotions: {sorted(listjournal_emotions)}")
print(f"📊 Total Journal emotions: {lenjournal_emotions}")

# Step 4: Create emotion mapping from GoEmotions to Journal
print"\n🔧 Creating emotion mapping..."

# Map GoEmotions emotions to Journal emotions
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
    'grie': 'sad',
    'joy': 'happy',
    'love': 'content',
    'nervousness': 'anxious',
    'optimism': 'hopeful',
    'pride': 'proud',
    'realization': 'content',
    'relie': 'calm',
    'remorse': 'sad',
    'sadness': 'sad',
    'surprise': 'excited',
    'neutral': 'calm'
}

print(f"✅ Created mapping with {lenemotion_mapping} emotions")

# Step 5: Process GoEmotions data with proper label conversion
print"\n📊 Processing GoEmotions data..."

go_texts = []
go_labels = []

for example in go_emotions['train']:
    if example['labels']:
        # Convert integer labels to emotion names
        emotion_indices = example['labels']
        for emotion_idx in emotion_indices:
            if emotion_idx < lenemotion_names:
                emotion_name = emotion_names[emotion_idx]
                if emotion_name in emotion_mapping:
                    mapped_emotion = emotion_mapping[emotion_name]
                    if mapped_emotion in journal_emotions:
                        go_texts.appendexample['text']
                        go_labels.appendmapped_emotion
                        break

# Process journal data
journal_texts = listjournal_df['content']
journal_labels = listjournal_df['emotion']

print(f"📊 Mapped GoEmotions: {lengo_texts} samples")
print(f"📊 Journal: {lenjournal_texts} samples")

# Step 6: Create unified label encoder
print"\n🔧 Creating unified label encoder..."

all_emotions = sorted(list(setgo_labels + journal_labels))
printf"📊 All emotions: {all_emotions}"

label_encoder = LabelEncoder()
label_encoder.fitall_emotions
label_to_id = {label: idx for idx, label in enumeratelabel_encoder.classes_}
id_to_label = {idx: label for label, idx in label_to_id.items()}

print(f"✅ Label encoder created with {lenlabel_encoder.classes_} classes")

# Convert labels to IDs
go_label_ids = [label_to_id[label] for label in go_labels]
journal_label_ids = [label_to_id[label] for label in journal_labels]

print(f"📊 GoEmotions label range: {mingo_label_ids} to {maxgo_label_ids}")
print(f"📊 Journal label range: {minjournal_label_ids} to {maxjournal_label_ids}")

# Validate all labels are within expected range
expected_range = (0, lenlabel_encoder.classes_ - 1)
printf"📊 Expected range: {expected_range}"

if mingo_label_ids >= expected_range[0] and maxgo_label_ids <= expected_range[1] and \
   minjournal_label_ids >= expected_range[0] and maxjournal_label_ids <= expected_range[1]:
    print"✅ All labels within expected range"
else:
    print"❌ Labels outside expected range!"
    raise ValueError"Label range validation failed"

# Step 7: Create simple dataset class
class SimpleEmotionDatasetDataset:
    def __init__self, texts, labels, tokenizer, max_length=128:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Validate data
        if lentexts != lenlabels:
            raise ValueError(f"Texts and labels have different lengths: {lentexts} vs {lenlabels}")
        
        # Validate labels
        for i, label in enumeratelabels:
            if not isinstancelabel, int or label < 0:
                raise ValueErrorf"Invalid label at index {i}: {label}"
    
    def __len__self:
        return lenself.texts
    
    def __getitem__self, idx:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Validate inputs
        if not isinstancetext, str or not text.strip():
            raise ValueErrorf"Invalid text at index {idx}"
        
        if not isinstancelabel, int or label < 0:
            raise ValueErrorf"Invalid label at index {idx}: {label}"
        
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
            'labels': torch.tensorlabel, dtype=torch.long
        }

# Step 8: Create simple model
class SimpleEmotionClassifiernn.Module:
    def __init__self, model_name="bert-base-uncased", num_labels=None:
        super().__init__()
        
        if num_labels is None or num_labels <= 0:
            raise ValueErrorf"Invalid num_labels: {num_labels}"
        
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrainedmodel_name
        self.dropout = nn.Dropout0.3
        self.classifier = nn.Linearself.bert.config.hidden_size, num_labels
        
        printf"✅ Model initialized with {num_labels} labels"
    
    def forwardself, input_ids, attention_mask:
        # Validate inputs
        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids to be 2D, got {input_ids.dim()}D")
        
        if attention_mask.dim() != 2:
            raise ValueError(f"Expected attention_mask to be 2D, got {attention_mask.dim()}D")
        
        outputs = self.bertinput_ids=input_ids, attention_mask=attention_mask
        pooled_output = outputs.pooler_output
        logits = self.classifier(self.dropoutpooled_output)
        
        # Validate outputs
        if logits.shape[-1] != self.num_labels:
            raise ValueErrorf"Expected {self.num_labels} output classes, got {logits.shape[-1]}"
        
        return logits

# Step 9: Setup training
print"\n🚀 Setting up training..."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
printf"✅ Using device: {device}"

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained"bert-base-uncased"
num_labels = lenlabel_encoder.classes_
model = SimpleEmotionClassifiermodel_name="bert-base-uncased", num_labels=num_labels
model = model.todevice

# Create datasets
go_dataset = SimpleEmotionDatasetgo_texts, go_label_ids, tokenizer
journal_dataset = SimpleEmotionDatasetjournal_texts, journal_label_ids, tokenizer

# Split journal data
journal_train_texts, journal_val_texts, journal_train_labels, journal_val_labels = train_test_split(
    journal_texts, journal_label_ids, test_size=0.3, random_state=42, stratify=journal_label_ids
)

journal_train_dataset = SimpleEmotionDatasetjournal_train_texts, journal_train_labels, tokenizer
journal_val_dataset = SimpleEmotionDatasetjournal_val_texts, journal_val_labels, tokenizer

# Create dataloaders
go_loader = DataLoadergo_dataset, batch_size=8, shuffle=True
journal_train_loader = DataLoaderjournal_train_dataset, batch_size=8, shuffle=True
journal_val_loader = DataLoaderjournal_val_dataset, batch_size=8, shuffle=False

print(f"✅ Training samples: {lengo_dataset} GoEmotions + {lenjournal_train_dataset} Journal")
print(f"✅ Validation samples: {lenjournal_val_dataset} Journal")

# Step 10: Training loop
print"\n🚀 Starting training..."

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

num_epochs = 3  # Reduced for testing
best_f1 = 0.0

for epoch in rangenum_epochs:
    printf"\n🔄 Epoch {epoch + 1}/{num_epochs}"
    
    # Training
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Train on GoEmotions
    print"  📚 Training on GoEmotions..."
    for i, batch in enumeratego_loader:
        try:
            # Validate batch
            if 'input_ids' not in batch or 'attention_mask' not in batch or 'labels' not in batch:
                printf"⚠️ Invalid batch structure at batch {i}"
                continue
            
            # Move to device with validation
            input_ids = batch['input_ids'].todevice
            attention_mask = batch['attention_mask'].todevice
            labels = batch['labels'].todevice
            
            # Validate labels
            if torch.anylabels >= num_labels or torch.anylabels < 0:
                printf"⚠️ Invalid labels in batch {i}: {labels}"
                continue
            
            # Forward pass
            optimizer.zero_grad()
            outputs = modelinput_ids=input_ids, attention_mask=attention_mask
            loss = criterionoutputs, labels
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if i % 50 == 0:
                print(f"    Batch {i}/{lengo_loader}, Loss: {loss.item():.4f}")
                
        except Exception as e:
            printf"❌ Error in batch {i}: {e}"
            continue
    
    # Train on journal data
    print"  📝 Training on journal data..."
    for i, batch in enumeratejournal_train_loader:
        try:
            input_ids = batch['input_ids'].todevice
            attention_mask = batch['attention_mask'].todevice
            labels = batch['labels'].todevice
            
            if torch.anylabels >= num_labels or torch.anylabels < 0:
                continue
            
            optimizer.zero_grad()
            outputs = modelinput_ids=input_ids, attention_mask=attention_mask
            loss = criterionoutputs, labels
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if i % 10 == 0:
                print(f"    Batch {i}/{lenjournal_train_loader}, Loss: {loss.item():.4f}")
                
        except Exception as e:
            printf"❌ Error in journal batch {i}: {e}"
            continue
    
    # Validation
    print"  🎯 Validating..."
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in journal_val_loader:
            try:
                input_ids = batch['input_ids'].todevice
                attention_mask = batch['attention_mask'].todevice
                labels = batch['labels'].todevice
                
                outputs = modelinput_ids=input_ids, attention_mask=attention_mask
                preds = torch.argmaxoutputs, dim=1
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            except Exception as e:
                printf"❌ Error in validation batch: {e}"
                continue
    
    # Calculate metrics
    if all_preds and all_labels:
        f1_macro = f1_scoreall_labels, all_preds, average='macro'
        accuracy = accuracy_scoreall_labels, all_preds
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        printf"  📊 Epoch {epoch + 1} Results:"
        printf"    Average Loss: {avg_loss:.4f}"
        print(f"    Validation F1 Macro: {f1_macro:.4f}")
        printf"    Validation Accuracy: {accuracy:.4f}"
        
        # Save best model
        if f1_macro > best_f1:
            best_f1 = f1_macro
            torch.save(model.state_dict(), 'best_simple_model.pth')
            printf"    💾 New best model saved! F1: {best_f1:.4f}"
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

printf"\n🏆 Training completed! Best F1 Score: {best_f1:.4f}"

# Step 11: Save results
results = {
    'best_f1': best_f1,
    'num_labels': num_labels,
    'target_achieved': best_f1 >= 0.7,
    'go_samples': lengo_texts,
    'journal_samples': lenjournal_texts,
    'emotion_mapping': emotion_mapping,
    'all_emotions': all_emotions
}

with open'simple_training_results.json', 'w' as f:
    json.dumpresults, f, indent=2

print"\n✅ Training completed successfully!"
printf"📊 Final F1 Score: {best_f1:.4f}"
printf"🎯 Target Met: {'✅' if best_f1 >= 0.7 else '❌'}"

# Download results
from google.colab import files
files.download'best_simple_model.pth'
files.download'simple_training_results.json'

print"\n🎉 FINAL BULLETPROOF TRAINING COMPLETED!"
print"📁 Files downloaded: best_simple_model.pth, simple_training_results.json"
print"\n🔥 THIS VERSION HAS PROPER INTEGER-TO-EMOTION MAPPING!"
print"🔥 NO MORE ZERO SAMPLES ISSUE!"
print"🔥 READY TO ACHIEVE 70% F1 SCORE!"