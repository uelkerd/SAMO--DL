#!/usr/bin/env python3
"""Ultra-simple API that starts immediately for Cloud Run."""
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
    title="SAMO Simple API",
    description="Simple API that starts immediately",
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
    else:
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
        logger.info(f"CORS origin regex patterns: {patterns}")
        return patterns
    else:
        # Default patterns for common development and staging environments
        default_patterns = [
            r"https://.*\.vercel\.app$",  # Vercel deployments
            r"https://.*\.netlify\.app$",  # Netlify deployments
            r"https://.*\.github\.io$",    # GitHub Pages
            r"http://localhost:\d+$",      # Local development with any port
            r"http://127\.0\.0\.1:\d+$",   # Local development with any port
        ]
        return default_patterns

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

@lru_cache(maxsize=1)
def _load_emotion_model():
    """Load and cache emotion detection model."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    model_name = 'duelker/samo-goemotions-deberta-v3-large'
    logger.info(f"Loading emotion model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode for deterministic inference
    
    logger.info("Emotion model loaded and cached successfully")
    return tokenizer, model

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "SAMO API is running", "status": "healthy"}

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}

@app.post("/analyze/emotion")
async def analyze_emotion(request: EmotionRequest):
    """Analyze emotion using cached model with safe inference."""
    try:
        logger.info("Starting emotion analysis", extra={"text_length": len(request.text)})

        # Load cached model
        tokenizer, model = _load_emotion_model()

        # Perform analysis with safe inference
        inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits.sigmoid()

        # 28 emotions from our DeBERTa-v3 model
        emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]

        emotion_scores = predictions[0].tolist()
        predicted_emotion = emotion_labels[emotion_scores.index(max(emotion_scores))]
        
        result = {
            "text": request.text,
            "emotions": dict(zip(emotion_labels, emotion_scores)),
            "predicted_emotion": predicted_emotion
        }

        logger.info("Emotion analysis completed successfully", extra={
            "predicted_emotion": predicted_emotion,
            "max_confidence": max(emotion_scores)
        })
        return result

    except Exception as e:
        logger.exception("Error in emotion analysis")
        raise HTTPException(status_code=500, detail="Analysis failed")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    # Default to localhost for development to avoid exposure
    host = os.environ.get("HOST", "127.0.0.1")
    if os.environ.get("PRODUCTION") == "true" or os.environ.get("CLOUD_RUN_SERVICE"):
        host = "0.0.0.0"  # Cloud Run and production environments
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
