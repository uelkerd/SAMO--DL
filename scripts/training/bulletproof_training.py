#!/usr/bin/env python3
"""Bulletproof training script for REQ-DL-012 that handles notebook state corruption.

This script can be run in a fresh kernel and will validate everything step by step.
"""
import sys
import json
import pickle
import torch
import torch.nn as nn
import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_environment():
    """Validate the environment and clear any corrupted state."""
    logger.info("🔍 Validating environment...")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("✅ GPU memory cleared")
    
    # Check CUDA
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        logger.info(f"✅ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("⚠️ CUDA not available, using CPU")
    
    # Test basic operations
    try:
        test_tensor = torch.randn(2, 3)
        test_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("✅ Basic tensor operations work")
    except Exception as e:
        logger.error(f"❌ Basic tensor operations failed: {e}")
        return False
    
    return True

def create_unified_label_encoder():
    """Create a unified label encoder for both datasets."""
    logger.info("🔧 Creating unified label encoder...")
    
    # Load datasets
    go_emotions = load_dataset("go_emotions", "simplified")
    with open('data/journal_test_dataset.json', 'r') as f:
        journal_entries = json.load(f)
    journal_df = pd.DataFrame(journal_entries)
    
    # Extract labels
    go_labels = set()
    for example in go_emotions['train']:
        if example['labels']:
            go_labels.update(example['labels'])
    
    journal_labels = set(journal_df['emotion'].unique())
    
    # Find common labels
    common_labels = sorted(list(go_labels.intersection(journal_labels)))
    if not common_labels:
        logger.warning("⚠️ No common labels found! Using all labels...")
        common_labels = sorted(list(go_labels.union(journal_labels)))
    
    logger.info(f"📊 Using {len(common_labels)} labels: {common_labels}")
    
    # Create encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(common_labels)
    
    # Save encoder
    with open('unified_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save mappings
    label_to_id = {label: idx for idx, label in enumerate(label_encoder.classes_)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    
    with open('label_mappings.json', 'w') as f:
        json.dump({
            'label_to_id': label_to_id,
            'id_to_label': id_to_label,
            'num_labels': len(label_encoder.classes_),
            'classes': label_encoder.classes_.tolist()
        }, f, indent=2)
    
    logger.info(f"✅ Label encoder created with {len(label_encoder.classes_)} classes")
    return label_encoder, label_to_id, id_to_label

def prepare_filtered_data(label_encoder, label_to_id):
    """Prepare filtered data using only common labels."""
    logger.info("📊 Preparing filtered data...")
    
    # Load datasets
    go_emotions = load_dataset("go_emotions", "simplified")
    with open('data/journal_test_dataset.json', 'r') as f:
        journal_entries = json.load(f)
    journal_df = pd.DataFrame(journal_entries)
    
    valid_labels = set(label_encoder.classes_)
    
    # Filter GoEmotions data
    go_texts = []
    go_labels = []
    for example in go_emotions['train']:
        if example['labels']:
            for label in example['labels']:
                if label in valid_labels:
                    go_texts.append(example['text'])
                    go_labels.append(label_to_id[label])
                    break
    
    # Filter journal data
    journal_texts = []
    journal_labels = []
    for _, row in journal_df.iterrows():
        if row['emotion'] in valid_labels:
            journal_texts.append(row['content'])
            journal_labels.append(label_to_id[row['emotion']])
    
    logger.info(f"📊 Filtered GoEmotions: {len(go_texts)} samples")
    logger.info(f"📊 Filtered Journal: {len(journal_texts)} samples")
    
    # Validate label ranges - FIX: Convert to integers for comparison
    if go_labels:
        go_label_range = (min(go_labels), max(go_labels))
    else:
        go_label_range = (0, 0)
        
    if journal_labels:
        journal_label_range = (min(journal_labels), max(journal_labels))
    else:
        journal_label_range = (0, 0)
        
    expected_range = (0, len(label_encoder.classes_) - 1)
    
    logger.info(f"📊 GoEmotions label range: {go_label_range}")
    logger.info(f"📊 Journal label range: {journal_label_range}")
    logger.info(f"📊 Expected range: {expected_range}")
    
    if go_label_range[0] < expected_range[0] or go_label_range[1] > expected_range[1]:
        logger.error(f"❌ GoEmotions labels out of range!")
        return None, None, None, None
    
    if journal_label_range[0] < expected_range[0] or journal_label_range[1] > expected_range[1]:
        logger.error(f"❌ Journal labels out of range!")
        return None, None, None, None
    
    logger.info("✅ All labels within expected range")
    return go_texts, go_labels, journal_texts, journal_labels

class SimpleEmotionDataset(Dataset):
    """Simple dataset class with validation."""
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

class SimpleEmotionClassifier(nn.Module):
    """Simple emotion classifier with validation."""
    def __init__(self, model_name="bert-base-uncased", num_labels=None):
        super().__init__()
        
        if num_labels is None or num_labels <= 0:
            raise ValueError(f"Invalid num_labels: {num_labels}")
        
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        logger.info(f"✅ Model initialized with {num_labels} labels")
    
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

def train_model_simple(go_texts, go_labels, journal_texts, journal_labels, num_labels):
    """Simple training function with comprehensive validation."""
    logger.info("🚀 Starting simple training...")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"✅ Using device: {device}")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = SimpleEmotionClassifier(model_name="bert-base-uncased", num_labels=num_labels)
    model = model.to(device)
    
    # Create datasets
    go_dataset = SimpleEmotionDataset(go_texts, go_labels, tokenizer)
    journal_dataset = SimpleEmotionDataset(journal_texts, journal_labels, tokenizer)
    
    # Split journal data
    journal_train_texts, journal_val_texts, journal_train_labels, journal_val_labels = train_test_split(
        journal_texts, journal_labels, test_size=0.3, random_state=42, stratify=journal_labels
    )
    
    journal_train_dataset = SimpleEmotionDataset(journal_train_texts, journal_train_labels, tokenizer)
    journal_val_dataset = SimpleEmotionDataset(journal_val_texts, journal_val_labels, tokenizer)
    
    # Create dataloaders
    go_loader = DataLoader(go_dataset, batch_size=8, shuffle=True)
    journal_train_loader = DataLoader(journal_train_dataset, batch_size=8, shuffle=True)
    journal_val_loader = DataLoader(journal_val_dataset, batch_size=8, shuffle=False)
    
    logger.info(f"✅ Training samples: {len(go_dataset)} GoEmotions + {len(journal_train_dataset)} Journal")
    logger.info(f"✅ Validation samples: {len(journal_val_dataset)} Journal")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 3  # Reduced for testing
    best_f1 = 0.0
    
    for epoch in range(num_epochs):
        logger.info(f"🔄 Epoch {epoch + 1}/{num_epochs}")
        
        # Training
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Train on GoEmotions
        logger.info("  📚 Training on GoEmotions...")
        for i, batch in enumerate(go_loader):
            try:
                # Validate batch
                if 'input_ids' not in batch or 'attention_mask' not in batch or 'labels' not in batch:
                    logger.warning(f"⚠️ Invalid batch structure at batch {i}")
                    continue
                
                # Move to device with validation
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Validate labels
                if torch.any(labels >= num_labels) or torch.any(labels < 0):
                    logger.warning(f"⚠️ Invalid labels in batch {i}: {labels}")
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
                    logger.info(f"    Batch {i}/{len(go_loader)}, Loss: {loss.item():.4f}")
                    
            except Exception as e:
                logger.error(f"❌ Error in batch {i}: {e}")
                continue
        
        # Train on journal data
        logger.info("  📝 Training on journal data...")
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
                    logger.info(f"    Batch {i}/{len(journal_train_loader)}, Loss: {loss.item():.4f}")
                    
            except Exception as e:
                logger.error(f"❌ Error in journal batch {i}: {e}")
                continue
        
        # Validation
        logger.info("  🎯 Validating...")
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
                    logger.error(f"❌ Error in validation batch: {e}")
                    continue
        
        # Calculate metrics
        if all_preds and all_labels:
            f1_macro = f1_score(all_labels, all_preds, average='macro')
            accuracy = accuracy_score(all_labels, all_preds)
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            
            logger.info(f"  📊 Epoch {epoch + 1} Results:")
            logger.info(f"    Average Loss: {avg_loss:.4f}")
            logger.info(f"    Validation F1 (Macro): {f1_macro:.4f}")
            logger.info(f"    Validation Accuracy: {accuracy:.4f}")
            
            # Save best model
            if f1_macro > best_f1:
                best_f1 = f1_macro
                torch.save(model.state_dict(), 'best_simple_model.pth')
                logger.info(f"    💾 New best model saved! F1: {best_f1:.4f}")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.info(f"🏆 Training completed! Best F1 Score: {best_f1:.4f}")
    return best_f1

def main():
    """Main function with comprehensive error handling."""
    logger.info("🚀 Starting bulletproof training for REQ-DL-012...")
    
    try:
        # Step 1: Validate environment
        if not validate_environment():
            logger.error("❌ Environment validation failed")
            return False
        
        # Step 2: Create unified label encoder
        label_encoder, label_to_id, id_to_label = create_unified_label_encoder()
        
        # Step 3: Prepare filtered data
        go_texts, go_labels, journal_texts, journal_labels = prepare_filtered_data(label_encoder, label_to_id)
        
        if go_texts is None:
            logger.error("❌ Data preparation failed")
            return False
        
        # Step 4: Train model
        num_labels = len(label_encoder.classes_)
        best_f1 = train_model_simple(go_texts, go_labels, journal_texts, journal_labels, num_labels)
        
        # Step 5: Save results
        results = {
            'best_f1': best_f1,
            'num_labels': num_labels,
            'target_achieved': best_f1 >= 0.7,
            'go_samples': len(go_texts),
            'journal_samples': len(journal_texts)
        }
        
        with open('simple_training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"📊 Final F1 Score: {best_f1:.4f}")
        logger.info(f"🎯 Target Met: {'✅' if best_f1 >= 0.7 else '❌'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)