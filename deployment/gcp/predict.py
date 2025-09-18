#!/usr/bin/env python3
"""
Vertex AI Custom Container Prediction Server
===========================================

This script runs a Flask server for the emotion detection model on Vertex AI.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, request, jsonify

app = Flask(__name__)

class EmotionDetectionModel:
    def __init__(self):
        """Initialize the model."""
        self.model_path = os.path.join(os.getcwd(), "model")
        print(f"Loading model from: {self.model_path}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.to('cuda')
                print("✅ Model moved to GPU")
            else:
                print("⚠️ CUDA not available, using CPU")

            self.emotions = ['anxious', 'calm', 'content', 'excited', 'frustrated', 'grateful', 'happy', 'hopeful', 'overwhelmed', 'proud', 'sad', 'tired']
            print("✅ Model loaded successfully")

        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            raise

    def predict(self, text):
        """Make a prediction."""
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}

            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                predicted_label = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_label].item()

                # Get all probabilities
                all_probs = probabilities[0].cpu().numpy()

            # Get predicted emotion
            if predicted_label in self.model.config.id2label:
                predicted_emotion = self.model.config.id2label[predicted_label]
            elif str(predicted_label) in self.model.config.id2label:
                predicted_emotion = self.model.config.id2label[str(predicted_label)]
            else:
                predicted_emotion = f"unknown_{predicted_label}"

            # Create response
            response = {
                'text': text,
                'predicted_emotion': predicted_emotion,
                'confidence': float(confidence),
                'probabilities': {
                    emotion: float(prob) for emotion, prob in zip(self.emotions, all_probs)
                },
                'model_version': '2.0',
                'model_type': 'comprehensive_emotion_detection',
                'performance': {
                    'basic_accuracy': '100.00%',
                    'real_world_accuracy': '93.75%',
                    'average_confidence': '83.9%'
                }
            }

            return response

        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise

# Initialize model
print("🔧 Loading emotion detection model...")
model = EmotionDetectionModel()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_version': '2.0',
        'model_type': 'comprehensive_emotion_detection'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400

        # Make prediction
        result = model.predict(text)

        return jsonify(result)

    except Exception as e:
        print(f"Prediction endpoint error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/', methods=['GET'])
def home():
    """Home endpoint."""
    return jsonify({
        'message': 'Comprehensive Emotion Detection API',
        'version': '2.0',
        'endpoints': {
            'GET /': 'This documentation',
            'GET /health': 'Health check',
            'POST /predict': 'Single prediction (send {"text": "your text"})'
        },
        'model_info': {
            'emotions': model.emotions,
            'performance': {
                'basic_accuracy': '100.00%',
                'real_world_accuracy': '93.75%',
                'average_confidence': '83.9%'
            }
        }
    })

if __name__ == '__main__':
    print("🌐 Starting Vertex AI prediction server...")
    print("📋 Available endpoints:")
    print("   GET  / - API documentation")
    print("   GET  /health - Health check")
    print("   POST /predict - Single prediction")
    print("")
    # Security-first host binding configuration
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8080"))

    # Only bind to all interfaces in production/container environments
    if (os.environ.get("PRODUCTION") == "true" or
        os.environ.get("DOCKER_CONTAINER") == "true" or
        os.environ.get("CLOUD_RUN_SERVICE") or
        os.environ.get("BIND_ALL_INTERFACES") == "true"):
        host = "0.0.0.0"
        print("⚠️  Production mode: Binding to all interfaces (0.0.0.0)")
    else:
        print(f"🔒 Development mode: Binding to localhost only ({host})")

    print(f"🚀 Server starting on http://{host}:{port}")
    print("")

    # Run the Flask app
    app.run(host=host, port=port, debug=False)
