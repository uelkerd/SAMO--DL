#!/usr/bin/env python3
"""Bulletproof startup API with pre-loaded models for Cloud Run."""
import os
import logging
import traceback
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="SAMO Unified AI API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for pre-loaded models
emotion_model = None
summarization_model = None
whisper_model = None
models_loaded = False
startup_error = None

def load_emotion_model():
    """Load emotion analysis model from cache."""
    global emotion_model
    try:
        logger.info("🚀 Loading DeBERTa-v3 emotion model from cache...")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_name = "duelker/samo-goemotions-deberta-v3-large"

        # Verify cache directory exists
        cache_dir = "/app/models"
        if not os.path.exists(cache_dir):
            raise FileNotFoundError(f"Cache directory {cache_dir} not found")

        # Load from cache only - no network downloads
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True  # Critical: prevent network downloads
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True  # Critical: prevent network downloads
        )

        emotion_model = {"tokenizer": tokenizer, "model": model}
        logger.info("✅ DeBERTa-v3 emotion model loaded successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to load emotion model: {e}")
        logger.error(traceback.format_exc())
        raise

def load_summarization_model():
    """Load T5 summarization model from cache."""
    global summarization_model
    try:
        logger.info("🚀 Loading T5 summarization model from cache...")
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        model_name = "t5-small"
        cache_dir = "/app/models"

        # Load from cache only - no network downloads
        tokenizer = T5Tokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True  # Critical: prevent network downloads
        )
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True  # Critical: prevent network downloads
        )

        summarization_model = {"tokenizer": tokenizer, "model": model}
        logger.info("✅ T5 summarization model loaded successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to load summarization model: {e}")
        logger.error(traceback.format_exc())
        raise

def load_whisper_model():
    """Load Whisper model from cache."""
    global whisper_model
    try:
        logger.info("🚀 Loading Whisper model from cache...")
        import whisper

        model_name = "base"
        download_root = "/app/models"

        # Verify Whisper model files exist
        expected_path = os.path.join(download_root, f"{model_name}.pt")
        if not os.path.exists(expected_path):
            raise FileNotFoundError(f"Whisper model not found at {expected_path}")

        # Load from cache only
        whisper_model = whisper.load_model(model_name, download_root=download_root)
        logger.info("✅ Whisper model loaded successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to load Whisper model: {e}")
        logger.error(traceback.format_exc())
        raise

@app.on_event("startup")
async def startup_load_models():
    """Load all models during FastAPI startup - CRITICAL for Cloud Run success."""
    global models_loaded, startup_error

    try:
        logger.info("🔥 STARTING MODEL LOADING SEQUENCE - CRITICAL FOR CLOUD RUN")

        # Log memory usage before loading
        try:
            import psutil
            memory_before = psutil.virtual_memory()
            logger.info(f"Memory before loading: {memory_before.used / (1024**3):.2f}GB used / {memory_before.total / (1024**3):.2f}GB total")
        except ImportError:
            logger.info("psutil not available - cannot monitor memory usage")

        # Sequential loading to prevent memory spikes
        logger.info("Step 1/3: Loading emotion model...")
        load_emotion_model()

        logger.info("Step 2/3: Loading summarization model...")
        load_summarization_model()

        logger.info("Step 3/3: Loading Whisper model...")
        try:
            load_whisper_model()
        except Exception as e:
            logger.warning(f"⚠️ Whisper model failed to load (non-critical): {e}")
            logger.info("Continuing without Whisper - core emotion/summarization models loaded successfully")

        # Log memory usage after loading
        try:
            memory_after = psutil.virtual_memory()
            logger.info(f"Memory after loading: {memory_after.used / (1024**3):.2f}GB used / {memory_after.total / (1024**3):.2f}GB total")
            logger.info(f"Memory increase: {(memory_after.used - memory_before.used) / (1024**3):.2f}GB")
        except:
            pass

        models_loaded = True
        logger.info("🎉 CORE MODELS LOADED SUCCESSFULLY - CLOUD RUN DEPLOYMENT READY!")

    except Exception as e:
        startup_error = str(e)
        models_loaded = False
        logger.error(f"💥 CRITICAL STARTUP FAILURE: {e}")
        logger.error(traceback.format_exc())
        # Don't raise here - let the app start but mark as not ready

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "SAMO Unified AI API",
        "status": "running",
        "models_loaded": models_loaded
    }

@app.get("/health")
async def health():
    """Liveness probe - always returns healthy if app is running."""
    return {"status": "healthy"}

@app.get("/ready")
async def ready():
    """Readiness probe - only returns ready after all models are loaded."""
    if not models_loaded:
        if startup_error:
            raise HTTPException(
                status_code=503,
                detail=f"Models not loaded due to startup error: {startup_error}"
            )
        else:
            raise HTTPException(
                status_code=503,
                detail="Models still loading, please wait..."
            )

    return {
        "status": "ready",
        "models_loaded": True,
        "available_endpoints": ["/analyze/emotion", "/analyze/summarize"]
    }

@app.post("/analyze/emotion")
async def analyze_emotion(text: str):
    """Analyze emotion in text using pre-loaded DeBERTa model."""
    global emotion_model

    # Verify model is loaded
    if not models_loaded or emotion_model is None:
        raise HTTPException(
            status_code=503,
            detail="Emotion model not loaded. Check /ready endpoint."
        )

    try:
        # Perform analysis with pre-loaded model
        inputs = emotion_model["tokenizer"](text, return_tensors="pt", truncation=True, max_length=512)
        outputs = emotion_model["model"](**inputs)
        predictions = outputs.logits.sigmoid()

        emotion_labels = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring",
            "confusion", "curiosity", "desire", "disappointment", "disapproval",
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization",
            "relief", "remorse", "sadness", "surprise", "neutral"
        ]

        emotion_scores = predictions[0].tolist()
        return {
            "text": text,
            "emotions": dict(zip(emotion_labels, emotion_scores)),
            "predicted_emotion": emotion_labels[emotion_scores.index(max(emotion_scores))]
        }

    except Exception as e:
        logger.error(f"Error in emotion analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/summarize")
async def summarize_text(text: str):
    """Summarize text using pre-loaded T5 model."""
    global summarization_model

    # Verify model is loaded
    if not models_loaded or summarization_model is None:
        raise HTTPException(
            status_code=503,
            detail="Summarization model not loaded. Check /ready endpoint."
        )

    try:
        # Perform summarization with pre-loaded model
        inputs = summarization_model["tokenizer"](f"summarize: {text}", return_tensors="pt", max_length=512, truncation=True)
        outputs = summarization_model["model"].generate(
            inputs["input_ids"],
            max_length=150,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = summarization_model["tokenizer"].decode(outputs[0], skip_special_tokens=True)

        return {
            "original_text": text,
            "summary": summary
        }

    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    # Default to localhost for development to avoid exposure
    host = os.environ.get("HOST", "127.0.0.1")
    if os.environ.get("PRODUCTION") == "true" or os.environ.get("CLOUD_RUN_SERVICE"):
        host = "0.0.0.0"  # Cloud Run and production environments
    logger.info(f"Starting bulletproof server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)