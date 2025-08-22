#!/usr/bin/env python3
"""
EMOTION DETECTION INFERENCE SCRIPT
=====================================
Standalone script to run emotion detection on text.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

class EmotionDetector:
    def __init__self, model_path=None:
        """Initialize the emotion detector"""
        if model_path is None:
            # Use the model directory relative to this script
            model_path = Path__file__.parent / "model"
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        printf"🔧 Loading model from: {model_path}"
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained"roberta-base"
        self.model = AutoModelForSequenceClassification.from_pretrained(strmodel_path)
        self.model.toself.device
        self.model.eval()
        
        # Define emotion mapping based on training order
        self.emotion_mapping = ['anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful', 'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired']
        
        printf"✅ Model loaded successfully on {self.device}"
    
    def predictself, text:
        """Predict emotion for given text"""
        # Tokenize
        inputs = self.tokenizertext, return_tensors="pt", truncation=True, max_length=512, padding=True
        inputs = {k: v.toself.device for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model**inputs
            probabilities = torch.softmaxoutputs.logits, dim=1
            predicted_class = torch.argmaxprobabilities, dim=1.item()
            confidence = probabilities[0][predicted_class].item()
        
        # Map to emotion name
        emotion = self.emotion_mapping[predicted_class]
        
        return {
            "emotion": emotion,
            "confidence": confidence,
            "text": text
        }
    
    def predict_batchself, texts:
        """Predict emotions for multiple texts"""
        results = []
        for text in texts:
            result = self.predicttext
            results.appendresult
        return results

def main():
    """Main function for command line usage"""
    import sys
    
    if lensys.argv < 2:
        print"Usage: python inference.py 'Your text here'"
        print"Example: python inference.py 'I am feeling happy today!'"
        return
    
    text = sys.argv[1]
    
    # Initialize detector
    detector = EmotionDetector()
    
    # Make prediction
    result = detector.predicttext
    
    print"\n🎯 EMOTION DETECTION RESULT"
    print"=" * 40
    printf"Text: {result['text']}"
    printf"Emotion: {result['emotion']}"
    printf"Confidence: {result['confidence']:.3f}"

if __name__ == "__main__":
    main()
