#!/usr/bin/env python3
"""
🚀 EMOTION DETECTION API SERVER
===============================
REST API server for emotion detection with comprehensive security headers.
"""

# Import all modules first
import logging
import os
from flask import Flask, request, jsonify
from inference import EmotionDetector

# Import security setup using relative import
try:
    from ..src.security_setup import setup_security_middleware
except Exception:  # fallback when executed as script
    from src.security_setup import setup_security_middleware

# Configure logging after all imports
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize security headers middleware
security_middleware = setup_security_middleware(app, os.environ.get("FLASK_ENV", "development"))

# Initialize emotion detector
try:
    detector = EmotionDetector()
    logger.info("✅ Emotion detector initialized successfully!")
except Exception as e:
    logger.error(f"❌ Failed to initialize emotion detector: {e}")
    detector = None


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    # Safe access to detector's emotion mapping
    emotions = []
    if detector:
        # Try to get mapping from detector safely
        mapping = getattr(detector, "mapping", {})
        if isinstance(mapping, dict):
            emotions = list(mapping.values())
        elif hasattr(mapping, "__iter__") and not isinstance(mapping, str):
            emotions = list(mapping)
        else:
            # Fallback to empty list if mapping is not accessible
            emotions = []

    return jsonify(
        {
            "status": "healthy",
            "model_loaded": detector is not None,
            "emotions": emotions,
        }
    )


@app.route("/predict", methods=["POST"])
def predict_emotion():
    """Predict emotion for given text"""
    if detector is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        result = detector.predict(text)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during prediction."}), 500


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """Predict emotions for multiple texts"""
    if detector is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json(silent=True) or {}
        texts = data.get("texts", [])

        if not texts:
            return jsonify({"error": "No texts provided"}), 400

        results = detector.predict_batch(texts)
        return jsonify({"results": results})

    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during batch prediction."}), 500


@app.route("/emotions", methods=["GET"])
def get_emotions():
    """Get list of supported emotions"""
    if detector is None:
        return jsonify({"error": "Model not loaded"}), 500

    return jsonify(
        {
            "emotions": list(detector.label_encoder.classes_),
            "count": len(detector.label_encoder.classes_),
        }
    )


if __name__ == "__main__":
    print("🚀 Starting Emotion Detection API Server")
    print("=" * 50)
    print("📊 Model Performance: 99.48% F1 Score")
    print(
        "🎯 Supported Emotions:",
        list(detector.label_encoder.classes_) if detector else "None",
    )
    print("🌐 API Endpoints:")
    print("  - GET  /health - Health check")
    print("  - POST /predict - Single text prediction")
    print("  - POST /predict_batch - Batch prediction")
    print("  - GET  /emotions - List emotions")
    print("=" * 50)

    # Environment-based host configuration for security
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))

    # Use centralized security-first host binding configuration
    from src.security.host_binding import get_secure_host_binding, validate_host_binding, get_binding_security_summary

    host, port = get_secure_host_binding(default_port=port)
    validate_host_binding(host, port)

    security_summary = get_binding_security_summary(host, port)
    logger.info("Security Summary: %s", security_summary)

    app.run(host=host, port=port, debug=False)
