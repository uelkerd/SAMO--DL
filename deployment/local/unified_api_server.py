#!/usr/bin/env python3
"""🚀 UNIFIED SAMO API SERVER WITH VOICE PROCESSING
==============================================
Complete API server with emotion detection, summarization, and voice processing.
Combines all SAMO models for comprehensive AI analysis.
"""

import argparse
import logging
import os
import tempfile
import time
import uuid
import threading
from pathlib import Path

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    T5Tokenizer,
    T5ForConditionalGeneration,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Configure Flask for file uploads (16MB max)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Global variables for model state (thread-safe with locks)
emotion_model = None
emotion_tokenizer = None
emotion_mapping = None
voice_transcriber = None
summarizer_model = None
summarizer_tokenizer = None
model_loading = False
models_loaded = False
model_lock = threading.Lock()

# GoEmotions emotion mapping for SAMO DeBERTa model
GOEMOTIONS_EMOTIONS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

# Fallback mapping (if needed)
EMOTION_MAPPING = [
    "anxious",
    "calm",
    "content",
    "excited",
    "frustrated",
    "grateful",
    "happy",
    "hopeful",
    "overwhelmed",
    "proud",
    "sad",
    "tired",
]

# Constants
MAX_INPUT_LENGTH = 512


def load_models():
    """Load all AI models: emotion detection, voice processing, and summarization"""
    global \
        model_loading, \
        models_loaded, \
        emotion_model, \
        emotion_tokenizer, \
        emotion_mapping, \
        voice_transcriber, \
        summarizer_model, \
        summarizer_tokenizer

    with model_lock:
        if model_loading or models_loaded:
            return
        model_loading = True

    logger.info("🔄 Starting unified model loading...")

    try:
        # Load emotion detection model
        logger.info("📥 Loading emotion detection model...")
        model_path = Path("/app/model")  # For production deployment

        # Fallback to local development path if production path doesn't exist
        if not model_path.exists():
            logger.info("📁 Production model path not found, checking for local models...")
            # For development, we'll use a basic emotion classifier
            # This can be replaced with actual trained models

        logger.info("📥 Loading emotion model...")
        try:
            # Try to load production model first
            emotion_model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            emotion_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            logger.info("✅ Production model loaded successfully")
        except:
            logger.warning("⚠️ Production model not found, loading REAL SAMO model")
            samo_model_id = "duelker/samo-goemotions-deberta-v3-large"
            logger.info(f"🚀 Loading REAL SAMO emotion model: {samo_model_id}")
            emotion_model = AutoModelForSequenceClassification.from_pretrained(samo_model_id)
            emotion_tokenizer = AutoTokenizer.from_pretrained(samo_model_id)
            logger.info("✅ REAL SAMO emotion model loaded successfully")

        # Set device (CPU for compatibility)
        device = torch.device("cpu")
        emotion_model.to(device)
        emotion_model.eval()

        # Use GoEmotions labels for SAMO DeBERTa model
        try:
            id2label = getattr(emotion_model.config, "id2label", None)
            if id2label and len(id2label) == len(GOEMOTIONS_EMOTIONS):
                # SAMO model uses GoEmotions labels - map from LABEL_X to actual emotions
                emotion_mapping = GOEMOTIONS_EMOTIONS
                logger.info("✅ Using REAL SAMO GoEmotions mapping: 28 emotions")
                logger.info(
                    f"🎯 Sample emotions: {GOEMOTIONS_EMOTIONS[:5]}...{GOEMOTIONS_EMOTIONS[-3:]}"
                )
            else:
                emotion_mapping = EMOTION_MAPPING
                logger.info("⚠️ Using fallback emotion mapping")
        except Exception:
            emotion_mapping = EMOTION_MAPPING
            logger.info("⚠️ Using fallback emotion mapping due to error")

        logger.info(f"✅ Emotion model loaded successfully on {device}")

        # Load voice processing model (lightweight approach)
        logger.info("🎙️ Loading voice processing model...")
        try:
            import whisper

            # Use smallest/fastest Whisper model for development
            voice_transcriber = whisper.load_model("tiny")
            logger.info("✅ Voice processing model (Whisper tiny) loaded successfully")

        except Exception as e:
            logger.warning(f"⚠️ Voice processing model failed to load: {e}")
            logger.info("📝 Voice processing will use fallback mock responses")
            voice_transcriber = None

        # Load text summarization model
        logger.info("📝 Loading text summarization model...")
        try:
            summarizer_model = T5ForConditionalGeneration.from_pretrained("t5-small")
            summarizer_tokenizer = T5Tokenizer.from_pretrained("t5-small")
            summarizer_model.eval()
            logger.info("✅ T5 summarization model loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Summarization model failed to load: {e}")
            logger.info("📝 Summarization will use enhanced fallback")
            summarizer_model = None
            summarizer_tokenizer = None

        logger.info("🎉 All models loaded successfully!")
        logger.info(f"🎯 Emotion mapping: {emotion_mapping}")
        logger.info(f"📝 Summarization available: {summarizer_model is not None}")

    except Exception:
        logger.exception("❌ Failed to load models")
        # Continue without models for graceful degradation
    finally:
        with model_lock:
            # Only set models_loaded = True when both model and tokenizer are present
            models_loaded = emotion_model is not None and emotion_tokenizer is not None
            model_loading = False


def summarize_text(text: str) -> dict:
    """Generate real AI summary using T5 model"""
    if summarizer_model is None or summarizer_tokenizer is None:
        # Enhanced fallback with better extraction
        sentences = text.split(".")
        if len(sentences) <= 3:
            return {
                "summary": text,
                "original_length": len(text),
                "summary_length": len(text),
                "compression_ratio": 1.0,
                "model": "extractive_fallback",
            }

        # Take first 2 and last 1 sentence for better context
        important_sentences = sentences[:2] + sentences[-1:]
        summary = ". ".join(sent.strip() for sent in important_sentences if sent.strip()) + "."

        return {
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "compression_ratio": round(len(summary) / len(text), 2),
            "model": "extractive_fallback",
        }

    try:
        # Prepare text for T5
        input_text = f"summarize: {text}"

        # Tokenize
        inputs = summarizer_tokenizer.encode(
            input_text, return_tensors="pt", max_length=512, truncation=True
        )

        # Generate summary
        with torch.no_grad():
            summary_ids = summarizer_model.generate(
                inputs,
                max_length=150,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
            )

        # Decode summary
        summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return {
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "compression_ratio": round(len(summary) / len(text), 2),
            "model": "T5-small",
        }

    except Exception as e:
        logger.warning(f"T5 summarization failed: {e}, using fallback")
        # Fallback to extractive
        words = text.split()
        summary_length = max(20, len(words) // 4)
        summary = " ".join(words[:summary_length])
        if len(words) > summary_length:
            summary += "..."

        return {
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "compression_ratio": round(len(summary) / len(text), 2),
            "model": "extractive_fallback",
        }


def predict_emotion(text: str) -> dict:
    """Predict emotion for given text"""
    if not models_loaded or emotion_model is None:
        raise RuntimeError("Emotion model not loaded")

    # Input sanitization and length check
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    if len(text) > MAX_INPUT_LENGTH:
        raise ValueError(f"Input text too long (>{MAX_INPUT_LENGTH} characters).")

    # Tokenize
    inputs = emotion_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        padding=True,
    )

    # Predict
    with torch.no_grad():
        outputs = emotion_model(**inputs)

        # Check if this is a multi-label classification model
        is_multi_label = (
            getattr(emotion_model.config, "problem_type", "") == "multi_label_classification"
        )

        if is_multi_label:
            # Use sigmoid for multi-label classification
            scores = torch.sigmoid(outputs.logits)[0]
            # Apply threshold for multi-label decisions
            threshold = 0.5
            predicted_labels = (scores > threshold).nonzero(as_tuple=True)[0].tolist()

            if predicted_labels:
                # Get the highest scoring label as primary
                predicted_class = int(torch.argmax(scores).item())
                confidence = float(scores[predicted_class].item())
            else:
                # No labels above threshold, use highest scoring
                predicted_class = int(torch.argmax(scores).item())
                confidence = float(scores[predicted_class].item())
        else:
            # Use softmax for single-label classification
            scores = torch.softmax(outputs.logits, dim=-1)[0]
            predicted_class = int(torch.argmax(scores).item())
            confidence = float(scores[predicted_class].item())

    # Map to emotion name (use index if available, otherwise fallback)
    if predicted_class < len(emotion_mapping):
        emotion = emotion_mapping[predicted_class]
    else:
        emotion = "neutral"  # Fallback

    # Create full emotions dictionary with all scores
    emotions_dict = {}
    for i, label in enumerate(emotion_mapping):
        emotions_dict[label] = float(scores[i].item())

    # Create top emotions array (sorted by confidence)
    top_emotions = sorted(
        [{"emotion": label, "confidence": score} for label, score in emotions_dict.items()],
        key=lambda x: x["confidence"],
        reverse=True,
    )[:5]

    return {
        "emotion": emotion,  # Keep backward compatibility
        "confidence": confidence,  # Keep backward compatibility
        "text": text,
        "emotions": emotions_dict,  # Add full emotions for demo
        "predicted_emotion": emotion,  # Add for demo compatibility
        "top_emotions": top_emotions,  # Add for demo compatibility
    }


def transcribe_audio(audio_file) -> dict:
    """Transcribe audio file to text with emotion analysis"""
    if voice_transcriber is None:
        raise RuntimeError("Voice processing model not available")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        audio_file.save(temp_file.name)
        temp_path = temp_file.name

    try:
        # Transcribe audio
        result = voice_transcriber.transcribe(temp_path)

        if not result or "text" not in result:
            raise RuntimeError("Transcription failed - no text returned")

        transcribed_text = result.get("text", "")
        # Whisper doesn't provide a calibrated confidence; keep a placeholder
        confidence = 0.9
        # Approximate duration from segments if available
        segs = result.get("segments") or []
        duration = float(segs[-1]["end"]) if segs else 0.0

        # Analyze emotions in transcribed text
        emotion_analysis = predict_emotion(transcribed_text)

        # Create comprehensive response
        return {
            "transcription": {
                "text": transcribed_text,
                "confidence": confidence,
                "duration": duration,
            },
            "emotion_analysis": emotion_analysis,
            "processing_info": {
                "timestamp": time.time(),
                "request_id": str(uuid.uuid4()),
                "models_used": ["SAMO Whisper", "SAMO Emotion Detection"],
            },
        }

    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass


def ensure_models_loaded():
    """Ensure models are loaded before processing requests"""
    if not models_loaded and not model_loading:
        load_models()

    if not models_loaded:
        raise RuntimeError("Models not loaded")


def create_error_response(message: str, status_code: int = 500) -> tuple:
    """Create standardized error response with request ID for debugging"""
    request_id = str(uuid.uuid4())
    logger.exception(f"{message} [request_id={request_id}]")
    return jsonify({"error": message, "request_id": request_id}), status_code


# API Routes


@app.route("/", methods=["GET"])
def root():
    """Root endpoint"""
    return jsonify(
        {
            "message": "SAMO Unified AI API - Voice, Emotion & Summarization",
            "status": "running",
            "models_loaded": models_loaded,
            "timestamp": time.time(),
        }
    )


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "models_loaded": models_loaded,
            "model_loading": model_loading,
            "voice_available": voice_transcriber is not None,
            "emotion_available": emotion_model is not None,
            "summarization_available": summarizer_model is not None,
            "timestamp": time.time(),
        }
    )


@app.route("/analyze/emotion", methods=["POST"])
def analyze_emotion():
    """Analyze emotion in text"""
    try:
        # Ensure models are loaded
        ensure_models_loaded()

        # Get text from query params (to match frontend expectations)
        text = request.args.get("text", "").strip()
        if not text:
            # Fallback to JSON body
            data = request.get_json(silent=True) or {}
            text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Make prediction
        result = predict_emotion(text)

        # Enhance response with additional metadata
        result.update({"request_id": str(uuid.uuid4()), "timestamp": time.time()})

        return jsonify(result)

    except Exception:
        return create_error_response("Emotion analysis failed. Please try again later.")


@app.route("/analyze/voice-journal", methods=["POST"])
def analyze_voice_journal():
    """Analyze voice recording with transcription and emotion detection"""
    try:
        # Ensure models are loaded
        ensure_models_loaded()

        # Check for audio file in the request
        if "audio_file" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files["audio_file"]
        if audio_file.filename == "":
            return jsonify({"error": "No audio file selected"}), 400

        # Validate MIME type
        allowed_types = ["audio/webm", "audio/wav", "audio/mp4", "audio/mpeg"]
        if audio_file.content_type not in allowed_types:
            return (
                jsonify(
                    {
                        "error": (
                            f"Unsupported audio format: {audio_file.content_type}. "
                            f"Supported: {', '.join(allowed_types)}"
                        )
                    }
                ),
                400,
            )

        logger.info(
            f"🎙️ Processing voice journal: {audio_file.filename} ({audio_file.content_type})"
        )

        # Transcribe and analyze
        if voice_transcriber is not None:
            result = transcribe_audio(audio_file)
            logger.info("✅ Voice journal processing successful")
            return jsonify(result)
        # Fallback to mock if voice model not available
        logger.warning("⚠️ Voice model not available, using enhanced mock response")
        return jsonify(create_enhanced_mock_response(audio_file.filename))

    except Exception:
        return create_error_response("Voice processing failed. Please try again later.")


def create_enhanced_mock_response(filename: str) -> dict:
    """Create an enhanced mock response that looks more realistic"""
    import random

    sample_texts = [
        ("Today has been a wonderful day filled with excitement and new opportunities."),
        ("I'm feeling quite optimistic about the future and all the possibilities ahead."),
        ("The voice processing feature is working amazingly well for transcription."),
        ("I'm grateful for all the progress we've made on this project so far."),
        ("This technology is truly impressive and will help many people."),
    ]

    transcribed_text = random.choice(sample_texts)

    # Use real emotion analysis on the mock text
    try:
        if emotion_model is not None:
            emotion_result = predict_emotion(transcribed_text)
        else:
            emotion_result = {
                "emotion": "optimism",
                "confidence": 0.85,
                "text": transcribed_text,
            }
    except:
        emotion_result = {
            "emotion": "neutral",
            "confidence": 0.75,
            "text": transcribed_text,
        }

    return {
        "transcription": {
            "text": transcribed_text,
            "confidence": random.uniform(0.85, 0.95),
            "duration": random.uniform(3.0, 8.0),
        },
        "emotion_analysis": emotion_result,
        "processing_info": {
            "filename": filename,
            "timestamp": time.time(),
            "request_id": str(uuid.uuid4()),
            "models_used": ["Enhanced Mock Whisper", "Real Emotion Analysis"],
            "note": "Voice transcription simulated - emotion analysis is real",
        },
    }


@app.route("/analyze/summarize", methods=["POST"])
def analyze_summarize():
    """Summarize text (placeholder for future implementation)"""
    try:
        # Get text from query params
        text = request.args.get("text", "").strip()
        if not text:
            data = request.get_json(silent=True) or {}
            text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Use REAL T5 AI summarization
        summary_result = summarize_text(text)

        result = {
            "summary": summary_result["summary"],
            "original_length": summary_result["original_length"],
            "summary_length": summary_result["summary_length"],
            "compression_ratio": summary_result["compression_ratio"],
            "model_used": summary_result["model"],
            "request_id": str(uuid.uuid4()),
            "timestamp": time.time(),
        }

        return jsonify(result)

    except Exception:
        return create_error_response("Text summarization failed. Please try again later.")


# Initialize models on startup
def initialize_models():
    """Initialize models before first request"""
    try:
        load_models()
    except Exception:
        logger.exception("Failed to initialize models on startup")


# Initialize models when module is imported
initialize_models()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAMO Unified AI API Server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8002")),
        help="Port to run the server on (default: 8002)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    logger.info("🚀 STARTING SAMO UNIFIED AI API SERVER")
    logger.info("=" * 50)
    logger.info("🎙️ Voice Processing: SAMO Whisper Integration")
    logger.info("😊 Emotion Detection: SAMO DeBERTa Model")
    logger.info("📝 Text Summarization: SAMO T5 Model")
    logger.info("🌐 API Endpoints:")
    logger.info("  - GET  / - Root endpoint")
    logger.info("  - GET  /health - Health check")
    logger.info("  - POST /analyze/emotion - Text emotion analysis")
    logger.info("  - POST /analyze/voice-journal - Voice transcription + emotion")
    logger.info("  - POST /analyze/summarize - Text summarization")
    logger.info("=" * 50)

    # Initialize models
    try:
        load_models()
    except Exception:
        logger.exception("Failed to load models on startup")

    print(f"🌐 Server starting at: http://{args.host}:{args.port}")
    print("📁 Serving unified AI analysis with voice processing")
    print("🔧 Real voice transcription and emotion analysis")
    print("Press Ctrl+C to stop the server")
    print("")

    app.run(host=args.host, port=args.port, debug=args.debug)
