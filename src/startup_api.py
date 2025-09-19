#!/usr/bin/env python3
"""
Bulletproof startup API with pre-loaded models for Cloud Run.

SECURITY NOTE: This application binds to 0.0.0.0 only in production environments
(Cloud Run, Docker containers) where it's required for proper operation. In development,
it defaults to 127.0.0.1 to prevent external access. This is a deliberate design choice
for containerized deployments and is not a security vulnerability.
"""
import asyncio
import logging
import os
import traceback

import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import torch
import httpx
from pydantic import BaseModel

# Import security-first host binding
from src.security.host_binding import (
    get_secure_host_binding,
    validate_host_binding,
    get_binding_security_summary,
)

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="SAMO Unified AI API", version="1.0.0")


# Pydantic models for request/response
class OpenAIRequest(BaseModel):
    prompt: str
    max_tokens: int = 4000
    temperature: float = 0.7


class OpenAIResponse(BaseModel):
    text: str
    model: str
    usage: dict = None


# CORS configuration from environment variables
def get_cors_origins():
    """Get allowed CORS origins from environment variables or use safe defaults."""
    # Try new split format first (CORS_ORIGIN_1, CORS_ORIGIN_2, etc.)
    origins = []
    i = 1
    while True:
        origin_var = f"CORS_ORIGIN_{i}"
        origin = os.environ.get(origin_var, "")
        if origin:
            origins.append(origin.strip())
            i += 1
        else:
            break

    # If we found split origins, use them
    if origins:
        logger.info("CORS origins from split environment variables: %s", origins)
        return origins

    # Fall back to legacy format (CORS_ORIGINS comma-separated)
    origins_env = os.environ.get("CORS_ORIGINS", "")
    if origins_env:
        # Split CSV and strip whitespace
        origins = [
            origin.strip() for origin in origins_env.split(",") if origin.strip()
        ]
        logger.info("CORS origins from legacy environment variable: %s", origins)
        return origins

    # Safe development defaults when no config provided
    dev_origins = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://localhost:8082",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8082",
    ]
    logger.warning(
        "No CORS environment variables configured, using development defaults"
    )
    return dev_origins


def get_cors_origin_regex():
    """Get CORS origin regex pattern as single string for dynamic hosts."""
    regex_env = os.environ.get("CORS_ORIGIN_REGEX", "")

    if regex_env:
        # Use the provided regex pattern directly
        logger.info("CORS origin regex pattern: %s", regex_env)
        return regex_env
    # Combine default patterns into single regex with alternation (|)
    default_patterns = [
        r"https://.*\.vercel\.app$",  # Vercel deployments
        r"https://.*\.netlify\.app$",  # Netlify deployments
        r"https://.*\.github\.io$",  # GitHub Pages
        r"http://localhost:\d+$",  # Local development with any port
        r"http://127\.0\.0\.1:\d+$",  # Local development with any port
    ]
    # Join patterns with OR (|) to create single regex
    combined_pattern = "|".join(f"({pattern})" for pattern in default_patterns)
    logger.info("CORS combined regex pattern: %s", combined_pattern)
    return combined_pattern


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

# Global variables for pre-loaded models
emotion_model = None
summarization_model = None
whisper_model = None
models_loaded = False
startup_error = None


def run_emotion_analysis(text: str) -> dict:
    """Run emotion analysis in a separate thread to avoid blocking the event loop."""
    with torch.no_grad():
        inputs = emotion_model["tokenizer"](
            text, return_tensors="pt", truncation=True, max_length=512
        )
        outputs = emotion_model["model"](**inputs)
        predictions = outputs.logits.sigmoid()

    # Load emotion labels dynamically from model config
    # This ensures compatibility if the model is updated with different labels
    try:
        emotion_labels = list(emotion_model["model"].config.id2label.values())
    except (AttributeError, KeyError):
        # Fallback to hardcoded labels if model config doesn't have id2label
        logger.warning("Model config missing id2label, using fallback emotion labels")
        emotion_labels = [
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

    emotion_scores = predictions[0].tolist()
    return {
        "text": text,
        "emotions": dict(zip(emotion_labels, emotion_scores)),
        "predicted_emotion": emotion_labels[emotion_scores.index(max(emotion_scores))],
    }


def run_text_summarization(text: str) -> dict:
    """Run text summarization in a separate thread to avoid blocking the event loop."""
    with torch.no_grad():
        inputs = summarization_model["tokenizer"](
            f"summarize: {text}", return_tensors="pt", max_length=512, truncation=True
        )
        outputs = summarization_model["model"].generate(
            inputs["input_ids"],
            max_length=150,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )
        summary = summarization_model["tokenizer"].decode(
            outputs[0], skip_special_tokens=True
        )

    return {"original_text": text, "summary": summary}


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
        # CRITICAL: local_files_only=True prevents network downloads during Cloud Run startup
        # This is essential because:
        # 1. Cloud Run has strict startup timeouts (10 minutes max)
        # 2. Model downloads can take 5-10 minutes and would cause startup failures
        # 3. Models are pre-downloaded during Docker build phase
        # 4. Network downloads during runtime would cause 503 errors and service unavailability
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,  # Critical: prevent network downloads
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,  # Critical: prevent network downloads
        )

        # Set model to evaluation mode for deterministic inference
        model.eval()

        emotion_model = {"tokenizer": tokenizer, "model": model}
        logger.info("✅ DeBERTa-v3 emotion model loaded successfully")
        return True

    except Exception as e:
        logger.error("❌ Failed to load emotion model: %s", e)
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
        # CRITICAL: local_files_only=True prevents network downloads during Cloud Run startup
        # This is essential because:
        # 1. Cloud Run has strict startup timeouts (10 minutes max)
        # 2. Model downloads can take 5-10 minutes and would cause startup failures
        # 3. Models are pre-downloaded during Docker build phase
        # 4. Network downloads during runtime would cause 503 errors and service unavailability
        tokenizer = T5Tokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,  # Critical: prevent network downloads
        )
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,  # Critical: prevent network downloads
        )

        # Set model to evaluation mode for deterministic inference
        model.eval()

        summarization_model = {"tokenizer": tokenizer, "model": model}
        logger.info("✅ T5 summarization model loaded successfully")
        return True

    except Exception as e:
        logger.error("❌ Failed to load summarization model: %s", e)
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
        logger.error("❌ Failed to load Whisper model: %s", e)
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
            logger.info("Memory before loading: %.2fGB used / %.2fGB total",
                        memory_before.used / (1024**3), memory_before.total / (1024**3))
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
            logger.warning("⚠️ Whisper model failed to load (non-critical): %s", e)
            logger.info(
                "Continuing without Whisper - core emotion/summarization models "
                "loaded successfully"
            )

        # Log memory usage after loading (only if psutil available)
        try:
            import psutil  # re-import safely
            memory_after = psutil.virtual_memory()
            logger.info("Memory after loading: %.2fGB used / %.2fGB total",
                        memory_after.used / (1024**3), memory_after.total / (1024**3))
            if "memory_before" in locals():
                logger.info("Memory increase: %.2fGB",
                            (memory_after.used - memory_before.used) / (1024**3))
        except Exception:
            # psutil not available; skip memory logging
            pass

        models_loaded = True
        logger.info("🎉 CORE MODELS LOADED SUCCESSFULLY - CLOUD RUN DEPLOYMENT READY!")

    except Exception as e:
        startup_error = str(e)
        models_loaded = False
        logger.error("💥 CRITICAL STARTUP FAILURE: %s", e)
        logger.error(traceback.format_exc())
        # Don't raise here - let the app start but mark as not ready


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "SAMO Unified AI API",
        "status": "running",
        "models_loaded": models_loaded,
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
                detail=f"Models not loaded due to startup error: {startup_error}",
            )
        raise HTTPException(
            status_code=503, detail="Models still loading, please wait..."
        )

    return {
        "status": "ready",
        "models_loaded": True,
        "available_endpoints": ["/analyze/emotion", "/analyze/summarize"],
    }


@app.post("/analyze/emotion")
async def analyze_emotion(text: str = Body(..., embed=True)):
    """Analyze emotion in text using pre-loaded DeBERTa model."""
    # Verify model is loaded
    if not models_loaded or emotion_model is None:
        raise HTTPException(
            status_code=503, detail="Emotion model not loaded. Check /ready endpoint."
        )

    try:
        # Run emotion analysis in threadpool to avoid blocking event loop
        return await asyncio.to_thread(run_emotion_analysis, text)

    except Exception:
        logger.exception("Error in emotion analysis")
        raise HTTPException(status_code=500, detail="Analysis failed")


@app.post("/analyze/summarize")
async def summarize_text(text: str = Body(..., embed=True)):
    """Summarize text using pre-loaded T5 model."""
    # Verify model is loaded
    if not models_loaded or summarization_model is None:
        raise HTTPException(
            status_code=503,
            detail="Summarization model not loaded. Check /ready endpoint.",
        )

    try:
        # Run text summarization in threadpool to avoid blocking event loop
        return await asyncio.to_thread(run_text_summarization, text)

    except Exception:
        logger.exception("Error in summarization")
        raise HTTPException(status_code=500, detail="Summarization failed")


@app.post("/proxy/openai", response_model=OpenAIResponse)
async def proxy_openai(request: OpenAIRequest):
    """Proxy OpenAI API calls with server-side API key."""
    try:
        # Get API key from environment
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise HTTPException(
                status_code=500, detail="OpenAI API key not configured on server"
            )

        # Prepare OpenAI request
        openai_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a creative writing assistant that generates authentic, "
                        "emotionally rich personal journal entries. Write in first person, "
                        "include specific details and genuine emotions."
                    ),
                },
                {"role": "user", "content": request.prompt},
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        # Make async request to OpenAI using httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(
                openai_url, headers=headers, json=payload, timeout=httpx.Timeout(30.0)
            )

            if response.is_error:
                logger.error(
                    f"OpenAI API error: {response.status_code} - {response.text}"
                )
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"OpenAI API error: {response.text}",
                )

            data = response.json()

        if not data.get("choices") or not data["choices"][0].get("message"):
            raise HTTPException(
                status_code=500, detail="Invalid response format from OpenAI API"
            )

        return OpenAIResponse(
            text=data["choices"][0]["message"]["content"].strip(),
            model=data.get("model", "gpt-4o-mini"),
            usage=data.get("usage"),
        )

    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail="OpenAI API timeout")
    except httpx.RequestError as e:
        logger.error("OpenAI API request error: %s", e)
        raise HTTPException(status_code=502, detail="OpenAI API unavailable")
    except Exception as e:
        logger.exception("Error in OpenAI proxy: %s", e)
        raise HTTPException(status_code=500, detail="OpenAI proxy failed")


if __name__ == "__main__":
    # Use centralized security-first host binding configuration
    host, port = get_secure_host_binding(default_port=8080)

    # Validate the binding configuration
    validate_host_binding(host, port)

    # Log security summary
    security_summary = get_binding_security_summary(host, port)
    logger.info("Security Summary: %s", security_summary)

    logger.info("Starting server on %s:%s", host, port)
    uvicorn.run(app, host=host, port=port)
