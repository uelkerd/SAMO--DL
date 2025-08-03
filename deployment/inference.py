#!/usr/bin/env python3
"""
🚀 EMOTION DETECTION INFERENCE SCRIPT
=====================================
Standalone script to run emotion detection on text.
"""

import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder

class EmotionDetector:
    def __init__(self, model_path="./model"):
        """Initialize the emotion detector"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load label encoder
        with open(f"{model_path}/label_encoder.json", 'r') as f:
            label_data = json.load(f)
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.array(label_data['classes'])
        
        print(f"✅ Model loaded successfully!")
        print(f"🎯 Device: {self.device}")
        print(f"📊 Emotions: {list(self.label_encoder.classes_)}")
    
    def predict(self, text, return_confidence=True):
        """Predict emotion for given text"""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Decode prediction
        predicted_emotion = self.label_encoder.inverse_transform([predicted_class])[0]
        
        if return_confidence:
            return {
                'text': text,
                'emotion': predicted_emotion,
                'confidence': confidence,
                'probabilities': {
                    emotion: prob.item() 
                    for emotion, prob in zip(self.label_encoder.classes_, probabilities[0])
                }
            }
        else:
            return predicted_emotion
    
    def predict_batch(self, texts):
        """Predict emotions for multiple texts"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

def main():
    """Example usage"""
    # Initialize detector
    detector = EmotionDetector()
    
    # Test examples
    test_texts = [
        "I'm feeling really happy today!",
        "I'm so frustrated with this project.",
        "I feel anxious about the presentation.",
        "I'm grateful for all the support.",
        "I'm feeling overwhelmed with tasks."
    ]
    
    print("🧪 Testing Emotion Detection Model")
    print("=" * 50)
    
    for text in test_texts:
        result = detector.predict(text)
        print(f"Text: {text}")
        print(f"Emotion: {result['emotion']} (confidence: {result['confidence']:.3f})")
        print(f"Top 3 predictions:")
        sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
        for emotion, prob in sorted_probs[:3]:
            print(f"  - {emotion}: {prob:.3f}")
        print()

if __name__ == "__main__":
    main()
