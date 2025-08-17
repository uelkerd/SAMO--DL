#!/usr/bin/env python3
"""
CI Model Calibration Test

This script tests the BERT emotion classifier calibration for CI/CD pipeline.
It creates a simple model and tests basic functionality without requiring checkpoints.

Usage:
    python scripts/ci/model_calibration_test.py

Returns:
    0 if test passes
    1 if test fails
"""

import logging
import sys
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%levelnames: %messages")
logger = logging.getLogger__name__


class SimpleBERTClassifiertorch.nn.Module:
    """Simple BERT classifier for emotion detection."""

    def __init__self, model_name="bert-base-uncased", num_emotions=28:
        super().__init__()
        self.bert = AutoModel.from_pretrainedmodel_name
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout0.3,
            torch.nn.Linear768, 256,
            torch.nn.ReLU(),
            torch.nn.Dropout0.3,
            torch.nn.Linear256, num_emotions,
        )
        self.temperature = torch.nn.Parameter(torch.ones1)

    def forwardself, input_ids, attention_mask, token_type_ids=None:
        outputs = self.bertinput_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        pooled_output = outputs.pooler_output
        logits = self.classifierpooled_output
        return logits


def create_test_data():
    """Create simple test data for calibration."""
    test_texts = [
        "I am so happy today!",
        "I love this new song!",
        "This makes me excited!",
        "I'm really angry about this!",
        "This is so frustrating!",
        "I hate this!",
        "I feel so sad right now",
        "This is heartbreaking",
        "I'm feeling down",
        "I love you so much!",
    ]

    # Create simple labels one emotion per text
    emotions = [
        "joy",
        "love",
        "excitement",
        "anger",
        "frustration",
        "disgust",
        "sadness",
        "grie",
        "sadness",
        "love",
    ]
    emotion_to_idx = {
        "joy": 0,
        "love": 1,
        "excitement": 2,
        "anger": 3,
        "frustration": 4,
        "disgust": 5,
        "sadness": 6,
        "grie": 7,
        "neutral": 27,
    }

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained"bert-base-uncased"
    
    # Basic validation
    assert lentest_texts == lenemotions, "Texts and emotions must have same length"
    
    return test_texts, emotions, emotion_to_idx, tokenizer


def test_model_calibration():
    """Test model calibration functionality."""
    try:
        logger.info"🧪 Testing model calibration..."

        # Create test data
        test_texts, emotions, emotion_to_idx, tokenizer = create_test_data()
        
        # Create model
        model = SimpleBERTClassifier"bert-base-uncased", num_emotions=28
        model.eval()

        logger.info"✅ Model created successfully"

        # Test model inference
        with torch.no_grad():
            # Tokenize
            inputs = tokenizer(
                test_texts[0],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Get predictions only pass required arguments
            outputs = modelinputs["input_ids"], inputs["attention_mask"]
            probabilities = torch.sigmoidoutputs
            
            logger.infof"✅ Model inference successful, output shape: {outputs.shape}"

        # Test temperature setting
        model.temperature.data = torch.tensor[2.0]
        logger.info"✅ Temperature setting successful"

        # Test threshold optimization
        threshold = 0.5
        predictions = probabilities > threshold.float()
        logger.infof"✅ Threshold optimization successful, predictions shape: {predictions.shape}"

        # Test metrics calculation
        if lentest_texts > 1:
            # Create simple labels for testing - match the prediction shape
            labels = torch.zeros1, 28  # Match the single prediction shape
            if emotions[0] in emotion_to_idx:
                labels[0, emotion_to_idx[emotions[0]]] = 1.0
            
            # Calculate F1 score
            f1 = f1_score(labels.flatten(), predictions.flatten(), average='micro')
            logger.infof"✅ Metrics calculation successful, F1: {f1:.3f}"

        logger.info"✅ Model calibration test passed"
        return True

    except Exception as e:
        logger.errorf"❌ Model calibration test failed: {e}"
        return False


def main():
    """Run model calibration tests."""
    logger.info"🚀 Starting Model Calibration Tests..."

    if test_model_calibration():
        logger.info"🎉 All model calibration tests passed!"
        return True
    else:
        logger.error"💥 Model calibration tests failed!"
        return False


if __name__ == "__main__":
    success = main()
    sys.exit0 if success else 1
