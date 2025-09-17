#!/usr/bin/env python3
"""Minimal Unified AI API for SAMO Deep Learning.

This is a simplified version that loads models on-demand to avoid startup timeout issues.
"""
import os
import logging
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SAMO Unified AI API",
    description="Unified AI API for emotion detection, text summarization, and voice processing",
    version="1.0.0"
)

# CORS configuration from environment variables
def get_cors_origins():
    """Get allowed CORS origins from environment variable or use safe defaults."""
    origins_env = os.environ.get("CORS_ORIGINS", "")

    if origins_env:
        # Split CSV and strip whitespace
        origins = [origin.strip() for origin in origins_env.split(",") if origin.strip()]
        logger.info(f"CORS origins from environment: {origins}")
        return origins
    # Safe development defaults when no config provided
    dev_origins = [
        "http://localhost:3000",
        "http://localhost:8080", 
        "http://localhost:8082",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8082"
    ]
    logger.warning("No CORS_ORIGINS configured, using development defaults")
    return dev_origins

def get_cors_origin_regex():
    """Get CORS origin regex patterns for dynamic hosts."""
    regex_env = os.environ.get("CORS_ORIGIN_REGEX", "")

    if regex_env:
        # Split CSV and strip whitespace for multiple regex patterns
        patterns = [pattern.strip() for pattern in regex_env.split(",") if pattern.strip()]
        # Combine patterns with alternation operator and wrap in non-capturing group
        combined_regex = f"^(?:{'|'.join(patterns)})$"
        logger.info(f"CORS origin regex patterns: {patterns}")
        logger.info(f"Combined CORS origin regex: {combined_regex}")
        return combined_regex
    # Default patterns for common development and staging environments
    default_patterns = [
        r"https://.*\.vercel\.app$",  # Vercel deployments
        r"https://.*\.netlify\.app$",  # Netlify deployments
        r"https://.*\.github\.io$",    # GitHub Pages
        r"http://localhost:\d+$",      # Local development with any port
        r"http://127\.0\.0\.1:\d+$",   # Local development with any port
    ]
    # Combine default patterns with alternation operator and wrap in non-capturing group
    combined_regex = f"^(?:{'|'.join(default_patterns)})$"
    logger.info(f"Default CORS origin regex patterns: {default_patterns}")
    logger.info(f"Combined CORS origin regex: {combined_regex}")
    return combined_regex

# Add CORS middleware with secure configuration
cors_origins = get_cors_origins()
cors_origin_regex = get_cors_origin_regex()

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=cors_origin_regex,
    allow_credentials=True,  # Safe because we're not using "*" for origins
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Pydantic models
class EmotionRequest(BaseModel):
    text: str

# Global variables for lazy loading
emotion_model = None

@lru_cache(maxsize=1)
def _load_emotion_model():
    """Load and cache emotion detection model."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_name = 'duelker/samo-goemotions-deberta-v3-large'
    logger.info(f"Loading emotion model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Set model to evaluation mode and move to appropriate device
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    logger.info("Emotion model loaded and cached successfully")
    return tokenizer, model, device
summarization_model = None
whisper_model = None

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "SAMO Unified AI API",
        "version": "1.0.0",
        "status": "running",
        "features": ["emotion_detection", "text_summarization", "voice_processing"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "API is running"}

@app.get("/api/health")
async def api_health():
    """API health check endpoint."""
    return {"status": "healthy", "message": "API is running"}

@app.post("/analyze/emotion")
async def analyze_emotion(request: EmotionRequest):
    """Analyze emotion in text using cached model with safe inference."""
    try:
        # Input validation
        if not request.text or not isinstance(request.text, str) or not request.text.strip():
            raise ValueError("Text input is required and cannot be empty")

        logger.info("Starting emotion analysis", extra={"text_length": len(request.text)})

        # Load cached model
        tokenizer, model, device = _load_emotion_model()

        # Perform emotion analysis with safe inference
        inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits.sigmoid()  # Use sigmoid for multi-label classification

        # Detach and move to CPU before converting to Python lists
        predictions = predictions.detach().cpu()

        # Get emotion labels (28 emotions from our DeBERTa-v3 model)
        emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]
        emotion_scores = predictions[0].tolist()
        predicted_emotion = emotion_labels[emotion_scores.index(max(emotion_scores))]

        logger.info("Emotion analysis completed successfully", extra={
            "predicted_emotion": predicted_emotion,
            "max_confidence": max(emotion_scores)
        })

        result = {
            "text": request.text,
            "emotions": dict(zip(emotion_labels, emotion_scores)),
            "predicted_emotion": predicted_emotion
        }

        return result

    except ValueError:
        logger.warning("Invalid input for emotion analysis")
        raise HTTPException(status_code=400, detail="Invalid input for emotion analysis")
    except Exception as e:
        logger.exception("Error in emotion analysis")
        raise HTTPException(status_code=500, detail="Emotion analysis failed") from e

@app.post("/analyze/summarize")
async def summarize_text(text: str):
    """Summarize text using T5 model."""
    try:
        # Input validation
        if not text or not isinstance(text, str) or not text.strip():
            raise HTTPException(status_code=400, detail="Text input is required and cannot be empty")

        # Enforce maximum length
        MAX_TEXT_LENGTH = 10000
        if len(text) > MAX_TEXT_LENGTH:
            raise HTTPException(status_code=400, detail=f"Text too long. Maximum length is {MAX_TEXT_LENGTH} characters")

        # Lazy load summarization model
        global summarization_model
        if summarization_model is None:
            logger.info("Loading summarization model...")
            try:
                import sentencepiece  # Required for T5 tokenizer
            except ImportError:
                logger.error("sentencepiece not installed. Please install it: pip install sentencepiece")
                raise HTTPException(status_code=500, detail="Summarization model requires sentencepiece. Please install it.")

            from transformers import T5Tokenizer, T5ForConditionalGeneration

            model_name = 't5-small'
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)

            # Set model to evaluation mode and move to appropriate device
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            summarization_model = {"tokenizer": tokenizer, "model": model, "device": device}
            logger.info("Summarization model loaded successfully")

        # Perform summarization with safe inference
        device = summarization_model["device"]
        inputs = summarization_model["tokenizer"](f"summarize: {text}", return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = summarization_model["model"].generate(
                inputs["input_ids"],
                max_length=150,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )

        # Detach and move to CPU before decoding
        outputs = outputs.detach().cpu()
        summary = summarization_model["tokenizer"].decode(outputs[0], skip_special_tokens=True)

        return {
            "original_text": text,
            "summary": summary
        }

    except ImportError as e:
        logger.exception("Missing dependency for summarization")
        raise HTTPException(status_code=500, detail="Summarization model requires sentencepiece. Please install it.") from e
    except Exception as e:
        logger.exception("Error in text summarization")
        raise HTTPException(status_code=500, detail="Text summarization failed") from e

@app.post("/analyze/transcribe")
async def transcribe_audio(audio_file: bytes):
    """Transcribe audio using Whisper model."""
    try:
        # Lazy load Whisper model
        global whisper_model
        if whisper_model is None:
            logger.info("Loading Whisper model...")
            import whisper
            whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded successfully")

        # Save audio file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_file)
            tmp_file_path = tmp_file.name

        try:
            # Perform transcription
            result = whisper_model.transcribe(tmp_file_path)
            transcription = result["text"]

            return {
                "transcription": transcription,
                "language": result.get("language", "unknown")
            }
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)

    except Exception:
        logger.exception("Error in audio transcription")
        raise HTTPException(status_code=500, detail="Audio transcription failed")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    # Default to localhost for development to avoid exposure
    host = os.environ.get("HOST", "127.0.0.1")
    if os.environ.get("PRODUCTION") == "true" or os.environ.get("CLOUD_RUN_SERVICE"):
        host = "0.0.0.0"  # Cloud Run and production environments
    uvicorn.run(app, host=host, port=port)
