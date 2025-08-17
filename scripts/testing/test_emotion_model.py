#!/usr/bin/env python3
"""
Test the trained emotion detection model with sample journal entries.
"""

import json
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_trained_model():
    """Load the trained emotion detection model."""
    print"🔧 Loading trained model..."
    
    # Load model weights
    model_path = 'best_simple_model.pth'
    model = SimpleEmotionClassifiermodel_name="bert-base-uncased", num_labels=12
    model.load_state_dict(torch.loadmodel_path, map_location='cpu')
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained"bert-base-uncased"
    
    # Load label encoder
    with open'simple_training_results.json', 'r' as f:
        results = json.loadf
    
    # Create label encoder from results
    all_emotions = results.get'all_emotions', []
    label_encoder = LabelEncoder()
    label_encoder.fitall_emotions
    
    print(f"✅ Model loaded with {lenlabel_encoder.classes_} emotions: {label_encoder.classes_}")
    return model, tokenizer, label_encoder

class SimpleEmotionClassifiernn.Module:
    def __init__self, model_name="bert-base-uncased", num_labels=None:
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

def predict_emotiontext, model, tokenizer, label_encoder, device='cpu':
    """Predict emotion for a given text."""
    model.todevice
    
    # Tokenize input
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].todevice
    attention_mask = encoding['attention_mask'].todevice
    
    # Predict
    with torch.no_grad():
        outputs = modelinput_ids=input_ids, attention_mask=attention_mask
        probabilities = torch.softmaxoutputs, dim=1
        predicted_class = torch.argmaxprobabilities, dim=1.item()
        confidence = probabilities[0][predicted_class].item()
    
    # Get emotion label
    emotion = label_encoder.inverse_transform[predicted_class][0]
    
    return emotion, confidence, probabilities[0].cpu().numpy()

def test_model():
    """Test the model with sample journal entries."""
    print"🧪 Testing emotion detection model..."
    
    # Load model
    model, tokenizer, label_encoder = load_trained_model()
    
    # Sample journal entries for testing
    test_entries = [
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
    
    print"\n📊 Testing Results:"
    print"=" * 80
    
    for i, text in enumeratetest_entries, 1:
        emotion, confidence, all_probs = predict_emotiontext, model, tokenizer, label_encoder
        
        printf"\n{i}. Text: {text}"
        print(f"   Predicted: {emotion} confidence: {confidence:.3f}")
        
        # Show top 3 predictions
        top_indices = np.argsortall_probs[-3:][::-1]
        print"   Top 3 predictions:"
        for idx in top_indices:
            prob = all_probs[idx]
            emotion_name = label_encoder.inverse_transform[idx][0]
            printf"     - {emotion_name}: {prob:.3f}"
    
    print"\n✅ Model testing completed!"

def analyze_performance():
    """Analyze model performance on validation data."""
    print"\n📈 Performance Analysis:"
    print"=" * 40
    
    # Load results
    with open'simple_training_results.json', 'r' as f:
        results = json.loadf
    
    printf"Final F1 Score: {results['best_f1']:.4f}"
    printf"Target Achieved: {results['target_achieved']}"
    printf"Number of Labels: {results['num_labels']}"
    printf"GoEmotions Samples: {results['go_samples']}"
    printf"Journal Samples: {results['journal_samples']}"
    
    # Show emotion mapping
    print"\nEmotion Mapping Used:"
    for go_emotion, journal_emotion in results['emotion_mapping'].items():
        printf"  {go_emotion} → {journal_emotion}"

if __name__ == "__main__":
    test_model()
    analyze_performance() 