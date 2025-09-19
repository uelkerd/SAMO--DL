# (shebang removed; run via `python deployment/local/simple_server.py`)
"""
Simple Local API Server for Development
========================================

A lightweight Flask server for local development testing.
Serves static files and provides basic CORS support.
"""

import argparse
import logging
import os

import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

# Configure Flask for file uploads (16MB max)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Configure logging
logging.basicConfig(level=logging.INFO)

# Resolve once
WEBSITE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "website")
)

# Environment-configurable upstream settings
UPSTREAM_BASE = os.getenv(
    "SAMO_UNIFIED_API_BASE",
    "https://samo-unified-api-optimized-frrnetyhfa-uc.a.run.app",
)
API_KEY = os.getenv("SAMO_API_KEY")  # optional
COMMON_HEADERS = (
    {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
)


def create_mock_voice_response(filename):
    """Create a mock voice processing response for development when upstream API doesn't support voice."""
    import time
    import random

    # Sample transcription text based on filename or random
    sample_texts = [
        "Hello, this is a test recording. I'm speaking into the microphone to "
        "test the voice processing functionality.",
        "The weather is beautiful today. I think I'll go for a walk in the park "
        "after finishing this demo.",
        "Voice recognition technology has come a long way. It's amazing how "
        "accurately it can transcribe speech now.",
        "Testing the SAMO voice analysis system. This should analyze both the "
        "transcription and emotions.",
        "I'm feeling quite optimistic about this new feature. It will make the "
        "demo much more interactive."
    ]

    transcribed_text = random.choice(sample_texts)

    # Mock emotion analysis matching the expected format
    mock_emotions = {
        'admiration': random.uniform(0.05, 0.15),
        'amusement': random.uniform(0.05, 0.12),
        'anger': random.uniform(0.01, 0.05),
        'annoyance': random.uniform(0.01, 0.04),
        'approval': random.uniform(0.10, 0.20),
        'caring': random.uniform(0.05, 0.10),
        'confusion': random.uniform(0.02, 0.06),
        'curiosity': random.uniform(0.15, 0.25),
        'desire': random.uniform(0.03, 0.08),
        'disappointment': random.uniform(0.01, 0.04),
        'disapproval': random.uniform(0.01, 0.03),
        'disgust': random.uniform(0.01, 0.02),
        'embarrassment': random.uniform(0.01, 0.03),
        'excitement': random.uniform(0.60, 0.85),
        'fear': random.uniform(0.01, 0.04),
        'gratitude': random.uniform(0.08, 0.15),
        'grief': random.uniform(0.01, 0.02),
        'joy': random.uniform(0.55, 0.75),
        'love': random.uniform(0.05, 0.12),
        'nervousness': random.uniform(0.02, 0.05),
        'optimism': random.uniform(0.50, 0.70),
        'pride': random.uniform(0.04, 0.08),
        'realization': random.uniform(0.04, 0.08),
        'relief': random.uniform(0.03, 0.06),
        'remorse': random.uniform(0.01, 0.02),
        'sadness': random.uniform(0.01, 0.04),
        'surprise': random.uniform(0.10, 0.18),
        'neutral': random.uniform(0.05, 0.10)
    }

    # Create top emotions array
    top_emotions = sorted(
        mock_emotions.items(), key=lambda x: x[1], reverse=True
    )[:5]
    top_emotions_array = [
        {"emotion": emotion, "confidence": confidence} 
        for emotion, confidence in top_emotions
    ]

    # Mock summary
    if len(transcribed_text) > 100:
        summary_text = transcribed_text[:100] + "..."
    else:
        summary_text = transcribed_text

    return {
        "transcription": {
            "text": transcribed_text,
            "confidence": random.uniform(0.85, 0.95),
            "duration": random.uniform(3.0, 8.0)
        },
        "emotion_analysis": {
            "text": transcribed_text,
            "emotions": mock_emotions,
            "predicted_emotion": top_emotions[0][0],
            "top_emotions": top_emotions_array,
            "confidence": top_emotions[0][1]
        },
        "summary": {
            "summary": summary_text,
            "original_length": len(transcribed_text),
            "summary_length": len(summary_text),
            "compression_ratio": round(len(summary_text) / len(transcribed_text), 2)
        },
        "processing_info": {
            "filename": filename,
            "mock": True,
            "timestamp": time.time(),
            "request_id": f"mock-{int(time.time())}-{random.randint(1000, 9999)}",
            "models_used": ["Mock Whisper", "Mock DeBERTa", "Mock T5"]
        }
    }


# Serve static files from website directory
@app.route("/")
def index():
    """Serve the main demo page."""
    return send_from_directory(WEBSITE_DIR, "comprehensive-demo.html")


@app.route("/<path:filename>")
def static_files(filename):
    """Serve static files from the website directory."""
    return send_from_directory(WEBSITE_DIR, filename)


# CORS Proxy for Real API
@app.route("/api/emotion", methods=["POST"])
def proxy_emotion():
    """Proxy emotion analysis requests to the real API."""
    try:
        # Accept JSON body or query param
        data = request.get_json(silent=True) or {}
        text = (
            data.get("text") or request.args.get("text", "")
        ).strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Call real API with JSON body
        api_url = f"{UPSTREAM_BASE}/analyze/emotion"
        response = requests.post(
            api_url, 
            json={"text": text}, 
            headers=COMMON_HEADERS, 
            timeout=30
        )

        if response.ok:
            return jsonify(response.json())
        return (
            jsonify({"error": f"API error: {response.status_code}"}),
            response.status_code,
        )

    except Exception:
        logging.exception("Unhandled exception in /api/emotion")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/summarize", methods=["POST"])
def proxy_summarize():
    """Proxy text summarization requests to the real API."""
    try:
        # Accept JSON body or query param
        data = request.get_json(silent=True) or {}
        text = (
            data.get("text") or request.args.get("text", "")
        ).strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Call real API with JSON body
        api_url = f"{UPSTREAM_BASE}/analyze/summarize"
        response = requests.post(
            api_url, 
            json={"text": text}, 
            headers=COMMON_HEADERS, 
            timeout=30
        )

        if response.ok:
            return jsonify(response.json())
        return (
            jsonify({"error": f"API error: {response.status_code}"}),
            response.status_code,
        )

    except Exception:
        logging.exception("Unhandled exception in /api/summarize")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/voice-journal", methods=["POST"])
def proxy_voice_journal():
    """Proxy voice journal requests to the real API with ephemeral file handling."""
    try:
        # Check for audio file in the request
        if 'audio_file' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio_file']
        if audio_file.filename == '':
            return jsonify({"error": "No audio file selected"}), 400

        # Validate MIME type
        allowed_types = ['audio/webm', 'audio/wav', 'audio/mp4', 'audio/mpeg']
        if audio_file.content_type not in allowed_types:
            return jsonify({
                "error": (
                    f"Unsupported audio format: {audio_file.content_type}. "
                    f"Supported: {', '.join(allowed_types)}"
                )
            }), 400

        # Log the upload attempt
        logging.info(
            f"🎙️ Processing audio upload: {audio_file.filename} "
            f"({audio_file.content_type})"
        )

        # Create files dict for requests - keeps file in memory only
        files = {
            'audio_file': (
                audio_file.filename,
                audio_file.stream,
                audio_file.content_type
            )
        }

        # Call real API with extended timeout for audio processing
        api_url = f"{UPSTREAM_BASE}/analyze/voice-journal"
        try:
            response = requests.post(
                api_url,
                files=files,
                headers=COMMON_HEADERS,
                timeout=60  # Extended timeout for audio processing
            )

            if response.ok:
                logging.info("✅ Voice processing successful")
                return jsonify(response.json())
            if response.status_code == 404:
                # Upstream doesn't support voice processing, provide mock response
                logging.info(
                    "⚠️ Upstream API doesn't support voice processing, "
                    "returning mock response"
                )
                return jsonify(create_mock_voice_response(audio_file.filename))
            else:
                logging.warning(f"⚠️ Upstream API error: {response.status_code}")
                return (
                    jsonify({
                        "error": f"Voice processing failed: {response.status_code}",
                        "details": response.text,
                    }),
                    response.status_code,
                )
        except requests.exceptions.ConnectionError:
            # Network error, provide mock response for development
            logging.warning(
                "🌐 Network error, providing mock voice response for development"
            )
            return jsonify(create_mock_voice_response(audio_file.filename))

    except requests.exceptions.Timeout:
        logging.exception("⏰ Voice processing timeout")
        return jsonify({
            "error": "Voice processing timeout. Please try with a shorter recording."
        }), 504

    except requests.exceptions.RequestException as e:
        logging.exception(f"🌐 Network error during voice processing: {e}")
        return jsonify({
            "error": "Network error during voice processing. Please try again."
        }), 502

    except Exception:
        logging.exception("❌ Unhandled exception in /api/voice-journal")
        return jsonify({
            "error": "Internal server error during voice processing"
        }), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint for the local development server."""
    return jsonify({"status": "healthy", "server": "simple_local_dev"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Local Development Server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--host", 
        default="127.0.0.1", 
        help="Host to bind to (default: 127.0.0.1)"
    )
    args = parser.parse_args()

    print("🚀 SIMPLE LOCAL DEVELOPMENT SERVER")
    print("==================================")
    print(f"🌐 Server starting at: http://{args.host}:{args.port}")
    print("📁 Serving website files with CORS enabled")
    print("🔧 Proxy AI endpoints available for testing")
    print("Press Ctrl+C to stop the server")
    print("")

    app.run(host=args.host, port=args.port, debug=False)
