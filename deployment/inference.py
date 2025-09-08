#!/usr/bin/env python3
"""EMOTION DETECTION INFERENCE SCRIPT.
=====================================
Standalone script to run emotion detection on text.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

class EmotionDetector:
    def __init__(self, model_path=None) -> None:
        """Initialize the emotion detector."""
        if model_path is None:
            # Use the model directory relative to this script
            model_path = Path(__file__).parent / "model"
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        self.model.to(self.device)
        self.model.eval()
        
        # Define emotion mapping based on training order
        self.emotion_mapping = ['anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful', 'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired']
        
    
    def predict(self, text):
        """Predict emotion for given text."""
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Map to emotion name
        emotion = self.emotion_mapping[predicted_class]
        
        return {
            "emotion": emotion,
            "confidence": confidence,
            "text": text
        }
    
    def predict_batch(self, texts):
        """Predict emotions for multiple texts."""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results

def main() -> None:
    """Main function for command line usage."""
    import sys
    
    if len(sys.argv) < 2:
        return
    
    text = sys.argv[1]
    
    # Initialize detector
    detector = EmotionDetector()
    
    # Make prediction
    detector.predict(text)
    

if __name__ == "__main__":
    main()
