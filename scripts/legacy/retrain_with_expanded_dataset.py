#!/usr/bin/env python3
"""
Retrain the emotion detection model with the expanded dataset.
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

def load_expanded_dataset():
    """Load the expanded journal dataset."""
    print"📊 Loading expanded dataset..."
    
    with open'data/expanded_journal_dataset.json', 'r' as f:
        data = json.loadf
    
    print(f"✅ Loaded {lendata} samples")
    
    # Analyze distribution
    emotion_counts = {}
    for entry in data:
        emotion = entry['emotion']
        emotion_counts[emotion] = emotion_counts.getemotion, 0 + 1
    
    print"📈 Emotion distribution:"
    for emotion, count in sorted(emotion_counts.items()):
        printf"  {emotion}: {count} samples"
    
    return data

class ExpandedEmotionDatasetDataset:
    def __init__self, texts, labels, tokenizer, max_length=128:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__self:
        return lenself.texts
    
    def __getitem__self, idx:
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
            'labels': torch.tensorlabel, dtype=torch.long
        }

class ExpandedEmotionClassifiernn.Module:
    def __init__self, model_name="bert-base-uncased", num_labels=12:
        super().__init__()
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrainedmodel_name
        self.dropout = nn.Dropout0.3
        self.classifier = nn.Linearself.bert.config.hidden_size, num_labels
    
    def forwardself, input_ids, attention_mask:
        outputs = self.bertinput_ids=input_ids, attention_mask=attention_mask
        pooled_output = outputs.pooler_output
        logits = self.classifier(self.dropoutpooled_output)
        return logits

def prepare_expanded_datadata, test_size=0.2, val_size=0.1:
    """Prepare data for training with expanded dataset."""
    print"🔧 Preparing expanded data..."
    
    # Extract texts and emotions
    texts = [entry['content'] for entry in data]
    emotions = [entry['emotion'] for entry in data]
    
    # Create label encoder
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transformemotions
    
    print(f"✅ Label encoder created with {lenlabel_encoder.classes_} classes")
    print(f"📊 Classes: {listlabel_encoder.classes_}")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/1-test_size, random_state=42, stratify=y_temp
    )
    
    print"📊 Data split:"
    print(f"  Training: {lenX_train} samples")
    print(f"  Validation: {lenX_val} samples")
    print(f"  Test: {lenX_test} samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, label_encoder

def train_expanded_modeltrain_data, val_data, label_encoder, epochs=5, batch_size=16:
    """Train the model with expanded dataset."""
    print"🚀 Training with expanded dataset..."
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    printf"✅ Using device: {device}"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained"bert-base-uncased"
    
    # Create datasets
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    train_dataset = ExpandedEmotionDatasetX_train, y_train, tokenizer
    val_dataset = ExpandedEmotionDatasetX_val, y_val, tokenizer
    
    train_loader = DataLoadertrain_dataset, batch_size=batch_size, shuffle=True
    val_loader = DataLoaderval_dataset, batch_size=batch_size, shuffle=False
    
    # Initialize model
    model = ExpandedEmotionClassifier(num_labels=lenlabel_encoder.classes_)
    model.todevice
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_f1 = 0
    training_history = []
    
    for epoch in rangeepochs:
        printf"\n🔄 Epoch {epoch + 1}/{epochs}"
        
        # Training
        model.train()
        total_loss = 0
        
        for i, batch in enumeratetrain_loader:
            input_ids = batch['input_ids'].todevice
            attention_mask = batch['attention_mask'].todevice
            labels = batch['labels'].todevice
            
            optimizer.zero_grad()
            outputs = modelinput_ids=input_ids, attention_mask=attention_mask
            loss = criterionoutputs, labels
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 50 == 0:
                print(f"  Batch {i}/{lentrain_loader}, Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].todevice
                attention_mask = batch['attention_mask'].todevice
                labels = batch['labels'].todevice
                
                outputs = modelinput_ids=input_ids, attention_mask=attention_mask
                loss = criterionoutputs, labels
                val_loss += loss.item()
                
                preds = torch.argmaxoutputs, dim=1
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_train_loss = total_loss / lentrain_loader
        avg_val_loss = val_loss / lenval_loader
        f1_macro = f1_scoreall_labels, all_preds, average='macro'
        accuracy = accuracy_scoreall_labels, all_preds
        
        printf"📊 Epoch {epoch + 1} Results:"
        printf"  Train Loss: {avg_train_loss:.4f}"
        printf"  Val Loss: {avg_val_loss:.4f}"
        print(f"  Val F1 Macro: {f1_macro:.4f}")
        printf"  Val Accuracy: {accuracy:.4f}"
        
        # Save best model
        if f1_macro > best_f1:
            best_f1 = f1_macro
            torch.save(model.state_dict(), 'best_expanded_model.pth')
            printf"💾 New best model saved! F1: {best_f1:.4f}"
        
        training_history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_f1_macro': f1_macro,
            'val_accuracy': accuracy
        })
    
    return model, training_history, best_f1

def save_expanded_resultstraining_history, best_f1, label_encoder, test_data:
    """Save training results."""
    print"💾 Saving results..."
    
    # Test final model
    X_test, y_test = test_data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load best model
    model = ExpandedEmotionClassifier(num_labels=lenlabel_encoder.classes_)
    model.load_state_dict(torch.load'best_expanded_model.pth')
    model.todevice
    model.eval()
    
    # Test predictions
    tokenizer = AutoTokenizer.from_pretrained"bert-base-uncased"
    test_dataset = ExpandedEmotionDatasetX_test, y_test, tokenizer
    test_loader = DataLoadertest_dataset, batch_size=16, shuffle=False
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].todevice
            attention_mask = batch['attention_mask'].todevice
            labels = batch['labels'].todevice
            
            outputs = modelinput_ids=input_ids, attention_mask=attention_mask
            preds = torch.argmaxoutputs, dim=1
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate final metrics
    final_f1 = f1_scoreall_labels, all_preds, average='macro'
    final_accuracy = accuracy_scoreall_labels, all_preds
    
    # Save results
    results = {
        'best_f1': best_f1,
        'final_f1': final_f1,
        'final_accuracy': final_accuracy,
        'target_achieved': final_f1 >= 0.70,
        'num_labels': lenlabel_encoder.classes_,
        'all_emotions': listlabel_encoder.classes_,
        'training_history': training_history,
        'expanded_samples': lenX_test + len[x for x in train_data[0]] + len[x for x in val_data[0]],
        'test_samples': lenX_test
    }
    
    with open'expanded_training_results.json', 'w' as f:
        json.dumpresults, f, indent=2
    
    print"✅ Results saved!"
    printf"📊 Final F1 Score: {final_f1:.4f}"
    printf"📊 Final Accuracy: {final_accuracy:.4f}"
    printf"🎯 Target Achieved: {final_f1 >= 0.70}"

def main():
    """Main training function."""
    print"🚀 RETRAINING WITH EXPANDED DATASET"
    print"=" * 60
    
    # Load expanded dataset
    data = load_expanded_dataset()
    
    # Prepare data
    train_data, val_data, test_data, label_encoder = prepare_expanded_datadata
    
    # Train model
    model, training_history, best_f1 = train_expanded_modeltrain_data, val_data, label_encoder
    
    # Save results
    save_expanded_resultstraining_history, best_f1, label_encoder, test_data
    
    print"\n🎉 Retraining completed!"
    print"📋 Next steps:"
    print"  1. Test the new model"
    print"  2. Compare performance"
    print"  3. Deploy if target achieved!"

if __name__ == "__main__":
    main() 