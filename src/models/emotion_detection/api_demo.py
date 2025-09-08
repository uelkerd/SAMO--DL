#!/usr/bin/env python3
"""SAMO Emotion Detection API Demo.

This demo showcases the emotion detection pipeline working with pre-trained
models and provides a preview of the API interface for Web Dev integration.
"""

import logging
import time
import traceback
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from transformers import AutoTokenizer

from ..api_rate_limiter import add_rate_limiting
from .bert_classifier import create_bert_emotion_classifier
from .labels import GOEMOTIONS_EMOTIONS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SAMO Emotion Detection API",
    description="""
    AI-powered emotion detection for journal entries.

    This API uses a BERT-based model fine-tuned on the GoEmotions dataset
    to detect emotions in text with high accuracy.

    Rate limiting: 100 requests per minute per user.
    """,
    version="0.1.0",
)

# Add rate limiting middleware (100 requests/minute per user)
add_rate_limiting(
    app,
    rate_limit=100,
    window_size=60,
    excluded_paths=["/health", "/docs", "/redoc", "/openapi.json"],
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, limit to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
model = None
tokenizer = None


class EmotionRequest(BaseModel):
    """Request model for emotion analysis."""

    text: str = Field(..., description="Text to analyze", min_length=1, max_length=2000)
    user_id: Optional[str] = Field(None, description="User ID for tracking")
    threshold: float = Field(0.5, description="Confidence threshold", ge=0.0, le=1.0)
    top_k: Optional[int] = Field(5, description="Number of top emotions to return", ge=1, le=28)

    class Config:
        schema_extra = {
            "example": {
                "text": "I'm feeling really excited about my new job and grateful for the opportunity.",
                "user_id": "user123",
                "threshold": 0.5,
                "top_k": 5,
            }
        }

    @validator("text")
    def validate_text(cls, text):
        """Validate that text is not empty."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        return text


class EmotionResponse(BaseModel):
    """Response model for emotion analysis."""

    primary_emotion: str = Field(..., description="Emotion with highest confidence", example="joy")
    confidence: float = Field(
        ..., description="Confidence score for primary emotion", ge=0.0, le=1.0, example=0.85
    )
    predicted_emotions: List[str] = Field(
        ..., description="Emotions above threshold", example=["joy", "gratitude", "optimism"]
    )
    emotion_scores: List[float] = Field(
        ..., description="Scores for predicted emotions", example=[0.85, 0.72, 0.64]
    )
    all_probabilities: List[float] = Field(
        ..., description="Probabilities for all emotions", example=[0.85, 0.72, 0.64, 0.0, 0.0]
    )
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds", example=42.5
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all exceptions with structured response."""
    logger.error("Unhandled exception: {exc}")
    logger.error(traceback.format_exc())

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
            "details": str(exc),
            "path": request.url.path,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail.lower().replace(" ", "_")
            if isinstance(exc.detail, str)
            else "http_error",
            "message": exc.detail,
            "path": request.url.path,
        },
        headers=exc.headers,
    )


@app.on_event("startup")
async def load_model() -> None:
    """Load emotion detection model on startup."""
    global model, tokenizer

    logger.info("Loading emotion detection model...")

    try:
        model, _ = create_bert_emotion_classifier(
            model_name="bert-base-uncased",
            freeze_bert_layers=0,  # Unfreeze for demo
        )

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        model.eval()

        logger.info("✅ Model loaded successfully!")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(traceback.format_exc())
        raise


@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "SAMO Emotion Detection API",
        "version": "0.1.0",
        "available_emotions": GOEMOTIONS_EMOTIONS,
        "endpoints": {
            "analyze": "/analyze - POST - Analyze emotion in text",
            "health": "/health - GET - Health check",
            "emotions": "/emotions - GET - List all supported emotions",
        },
        "rate_limit": "100 requests per minute per user",
    }


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "supported_emotions": len(GOEMOTIONS_EMOTIONS),
    }


@app.get("/emotions", tags=["Reference"])
async def list_emotions():
    """List all supported emotions."""
    return {
        "emotions": GOEMOTIONS_EMOTIONS,
        "count": len(GOEMOTIONS_EMOTIONS),
        "categories": {
            "positive": [
                "admiration",
                "amusement",
                "approval",
                "caring",
                "desire",
                "excitement",
                "gratitude",
                "joy",
                "love",
                "optimism",
                "pride",
                "relie",
            ],
            "negative": [
                "anger",
                "annoyance",
                "disappointment",
                "disapproval",
                "disgust",
                "embarrassment",
                "fear",
                "grie",
                "nervousness",
                "remorse",
                "sadness",
            ],
            "ambiguous": ["confusion", "curiosity", "realization", "surprise"],
            "neutral": ["neutral"],
        },
    }


@app.post(
    "/analyze",
    response_model=EmotionResponse,
    tags=["Analysis"],
    summary="Analyze emotions in text",
    description="Analyze the emotional content of text using BERT + GoEmotions",
)
async def analyze_emotion(
    request: EmotionRequest,
    x_api_key: Optional[str] = Header(None, description="API key for authentication"),  # noqa: ARG001
):
    """Analyze emotions in text.

    This endpoint detects emotions in the provided text using a BERT model
    fine-tuned on the GoEmotions dataset.

    Args:
        request: Emotion analysis request
        x_api_key: Optional API key for authentication

    Returns:
        Emotion analysis results with confidence scores

    Raises:
        HTTPException: If the model is not loaded or if processing fails
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Emotion detection model not available",
        )

    start_time = time.time()

    try:
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.sigmoid(outputs.logits)[0].tolist()

        sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        top_indices = sorted_indices[: request.top_k]

        filtered_indices = [i for i in top_indices if probs[i] >= request.threshold]

        primary_idx = sorted_indices[0]
        primary_emotion = GOEMOTIONS_EMOTIONS[primary_idx]
        primary_confidence = probs[primary_idx]

        predicted_emotions = [GOEMOTIONS_EMOTIONS[i] for i in filtered_indices]
        emotion_scores = [probs[i] for i in filtered_indices]

        processing_time = time.time() - start_time

        return EmotionResponse(
            primary_emotion=primary_emotion,
            confidence=primary_confidence,
            predicted_emotions=predicted_emotions,
            emotion_scores=emotion_scores,
            all_probabilities=probs,
            processing_time_ms=processing_time * 1000,
        )

    except Exception:
        logger.exception("Emotion analysis failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Emotion analysis processing failed. Please try again later.",
        )


@app.post(
    "/analyze/batch",
    tags=["Analysis"],
    summary="Batch analyze emotions in multiple texts",
    description="Analyze emotions in multiple texts in a single request",
)
async def analyze_emotions_batch(
    texts: List[str],
    threshold: float = 0.5,
    x_api_key: Optional[str] = Header(None, description="API key for authentication"),
):
    """Analyze emotions in multiple texts.

    This endpoint efficiently processes multiple texts in a single request.

    Args:
        texts: List of texts to analyze
        threshold: Confidence threshold (0.0-1.0)
        x_api_key: Optional API key for authentication

    Returns:
        List of emotion analysis results

    Raises:
        HTTPException: If the model is not loaded or processing fails
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Emotion detection model not available",
        )

    if not texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No texts provided for analysis",
        )

    if len(texts) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum batch size is 50 texts",
        )

    start_time = time.time()
    results = []

    try:
        for text in texts:
            request = EmotionRequest(text=text, threshold=threshold)
            result = await analyze_emotion(request, x_api_key=x_api_key)
            results.append(result)

        processing_time = time.time() - start_time

        return {
            "results": results,
            "count": len(results),
            "batch_processing_time_ms": processing_time * 1000,
            "average_processing_time_ms": (processing_time * 1000) / len(texts) if texts else 0,
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Batch emotion analysis failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch emotion analysis processing failed. Please try again later.",
        )


if __name__ == "__main__":
    logger.info("🚀 Starting SAMO Emotion Detection API...")
    uvicorn.run(
        "api_demo:app",
        host="127.0.0.1",  # Changed from 0.0.0.0 for security
        port=8001,
        reload=True,
        log_level="info",
    )
