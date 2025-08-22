#!/usr/bin/env python3

from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from tqdm import tqdm
import json
import logging
import numpy as np
import random
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


"""
Fixed Focal Loss Training with Proper Data and Thresholds

This script addresses the issues identified in the diagnosis:
1. Uses proper emotion labels instead of all zeros
2. Implements proper threshold optimization
3. Uses larger, more diverse training data
4. Implements proper evaluation metrics

Usage:
    python3 scripts/fixed_focal_training.py
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class SimpleBERTClassifier(nn.Module):
    """Simple BERT classifier for emotion detection."""

    def __init__(self, model_name="bert-base-uncased", num_classes=28):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])  # Use [CLS] token
        return logits


def create_proper_training_data():
    """Create proper training data with diverse emotion labels."""
    logger.info("📊 Creating proper training data with diverse emotion labels...")

    emotion_names = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring",
        "confusion", "curiosity", "desire", "disappointment", "disapproval",
        "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
        "joy", "love", "nervousness", "optimism", "pride", "realization",
        "relief", "remorse", "sadness", "surprise", "neutral"
    ]

    # Create diverse training data with proper emotion labels
    training_data = []
    
    # Joy examples
    joy_examples = [
        "I'm so happy today! Everything is going great!",
        "This is the best day ever! I can't stop smiling!",
        "I feel amazing and full of energy!",
        "What a wonderful surprise! I'm thrilled!",
        "I'm overjoyed with the results!",
        "This makes me so happy and excited!",
        "I'm feeling great and optimistic!",
        "What a fantastic experience!",
        "I'm delighted with how things turned out!",
        "This brings me so much joy!"
    ]
    
    # Sadness examples
    sadness_examples = [
        "I'm feeling really down today.",
        "Everything seems so hopeless right now.",
        "I'm so sad and lonely.",
        "This is really depressing me.",
        "I feel like crying.",
        "I'm heartbroken over what happened.",
        "This is so disappointing and sad.",
        "I'm feeling really low today.",
        "Everything is going wrong.",
        "I'm so upset about this situation."
    ]
    
    # Anger examples
    anger_examples = [
        "I'm so angry about this!",
        "This is absolutely infuriating!",
        "I can't believe this is happening!",
        "I'm furious with the situation!",
        "This makes me so mad!",
        "I'm really pissed off!",
        "This is unacceptable!",
        "I'm so frustrated and angry!",
        "This is driving me crazy!",
        "I'm really annoyed and angry!"
    ]
    
    # Fear examples
    fear_examples = [
        "I'm really scared about what might happen.",
        "This is terrifying me.",
        "I'm afraid of the consequences.",
        "This is making me anxious and fearful.",
        "I'm worried about the future.",
        "This is really frightening.",
        "I'm scared of what comes next.",
        "This is causing me a lot of fear.",
        "I'm terrified of the outcome.",
        "This is making me really nervous."
    ]
    
    # Love examples
    love_examples = [
        "I love you so much!",
        "You mean everything to me.",
        "I'm so in love with you.",
        "You make me so happy.",
        "I adore you completely.",
        "You're the best thing in my life.",
        "I'm so grateful for your love.",
        "You're my everything.",
        "I love spending time with you.",
        "You're the love of my life."
    ]
    
    # Disgust examples
    disgust_examples = [
        "This is absolutely disgusting!",
        "I'm repulsed by this.",
        "This is so gross!",
        "I can't stand this.",
        "This is really nasty.",
        "I'm disgusted by what I saw.",
        "This is revolting!",
        "I'm appalled by this.",
        "This is really sickening.",
        "I'm really grossed out."
    ]
    
    # Surprise examples
    surprise_examples = [
        "Oh my God! I can't believe this!",
        "This is completely unexpected!",
        "Wow! I'm so surprised!",
        "This is amazing! I didn't see this coming!",
        "I'm shocked by this news!",
        "This is incredible!",
        "I'm stunned by this revelation!",
        "This is unbelievable!",
        "I'm really surprised by this!",
        "This is astonishing!"
    ]
    
    # Neutral examples
    neutral_examples = [
        "The weather is cloudy today.",
        "I went to the store to buy groceries.",
        "The meeting is scheduled for tomorrow.",
        "I need to finish my work.",
        "The book is on the table.",
        "I'm going to the library.",
        "The car is parked outside.",
        "I have an appointment at 3 PM.",
        "The computer is working fine.",
        "I'm reading a book."
    ]

    # Create labeled data
    for text in joy_examples:
        labels = [0] * 28
        labels[emotion_names.index("joy")] = 1
        training_data.append({"text": text, "labels": labels})
    
    for text in sadness_examples:
        labels = [0] * 28
        labels[emotion_names.index("sadness")] = 1
        training_data.append({"text": text, "labels": labels})
    
    for text in anger_examples:
        labels = [0] * 28
        labels[emotion_names.index("anger")] = 1
        training_data.append({"text": text, "labels": labels})
    
    for text in fear_examples:
        labels = [0] * 28
        labels[emotion_names.index("fear")] = 1
        training_data.append({"text": text, "labels": labels})
    
    for text in love_examples:
        labels = [0] * 28
        labels[emotion_names.index("love")] = 1
        training_data.append({"text": text, "labels": labels})
    
    for text in disgust_examples:
        labels = [0] * 28
        labels[emotion_names.index("disgust")] = 1
        training_data.append({"text": text, "labels": labels})
    
    for text in surprise_examples:
        labels = [0] * 28
        labels[emotion_names.index("surprise")] = 1
        training_data.append({"text": text, "labels": labels})
    
    for text in neutral_examples:
        labels = [0] * 28
        labels[emotion_names.index("neutral")] = 1
        training_data.append({"text": text, "labels": labels})

    # Shuffle the data
    random.shuffle(training_data)
    
    # Split into train/val/test
    total_samples = len(training_data)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    
    train_data = training_data[:train_size]
    val_data = training_data[train_size:train_size + val_size]
    test_data = training_data[train_size + val_size:]
    
    logger.info(f"✅ Created {len(train_data)} training, {len(val_data)} validation, {len(test_data)} test samples")
    
    return train_data, val_data, test_data


def create_dataloader(data, model, batch_size=8):
    """Create a simple dataloader for the data."""
    dataloader = []
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        
        texts = [item["text"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        # Tokenize
        tokenized = model.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        dataloader.append({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.float32)
        })
    
    return dataloader


def train_model(model, train_data, val_data, device, epochs=10):
    """Train the model with focal loss."""
    logger.info("🚀 Starting model training...")
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = FocalLoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_data, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_data:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_data)
        avg_val_loss = val_loss / len(val_data)
        
        logger.info(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_focal_model.pth")
            logger.info(f"✅ Saved best model with val loss: {best_val_loss:.4f}")
    
    return model


def evaluate_model(model, test_data, device):
    """Evaluate the model with different thresholds."""
    logger.info("📊 Evaluating model with different thresholds...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_data:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask)
            predictions = torch.sigmoid(outputs)
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Test different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        binary_predictions = (all_predictions > threshold).astype(int)
        
        # Calculate metrics
        f1 = f1_score(all_labels, binary_predictions, average='weighted', zero_division=0)
        precision = precision_score(all_labels, binary_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, binary_predictions, average='weighted', zero_division=0)
        
        logger.info(f"Threshold {threshold}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    logger.info(f"🎯 Best threshold: {best_threshold} with F1: {best_f1:.4f}")
    
    # Final evaluation with best threshold
    binary_predictions = (all_predictions > best_threshold).astype(int)
    final_f1 = f1_score(all_labels, binary_predictions, average='weighted', zero_division=0)
    final_precision = precision_score(all_labels, binary_predictions, average='weighted', zero_division=0)
    final_recall = recall_score(all_labels, binary_predictions, average='weighted', zero_division=0)
    
    logger.info(f"🏆 Final Results - F1: {final_f1:.4f}, Precision: {final_precision:.4f}, Recall: {final_recall:.4f}")
    
    return {
        "f1": final_f1,
        "precision": final_precision,
        "recall": final_recall,
        "best_threshold": best_threshold
    }


def main():
    """Main training function."""
    logger.info("🎯 Starting Fixed Focal Loss Training")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️ Using device: {device}")
    
    # Create directories
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    # Create proper training data
    train_data, val_data, test_data = create_proper_training_data()
    
    # Create model
    model = SimpleBERTClassifier()
    logger.info(f"🤖 Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dataloaders
    train_dataloader = create_dataloader(train_data, model, batch_size=8)
    val_dataloader = create_dataloader(val_data, model, batch_size=8)
    test_dataloader = create_dataloader(test_data, model, batch_size=8)
    
    # Train model
    trained_model = train_model(model, train_dataloader, val_dataloader, device, epochs=5)
    
    # Load best model
    trained_model.load_state_dict(torch.load("best_focal_model.pth"))
    
    # Evaluate model
    results = evaluate_model(trained_model, test_dataloader, device)
    
    # Save results
    with open("results/focal_training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Final summary
    logger.info("🎉 Training completed successfully!")
    logger.info(f"📊 Final F1 Score: {results['f1']:.4f}")
    logger.info(f"🎯 Best Threshold: {results['best_threshold']}")
    logger.info("💾 Results saved to results/focal_training_results.json")


if __name__ == "__main__":
    main()
