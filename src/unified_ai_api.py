#!/usr/bin/env python3
"""
Unified AI API for SAMO Deep Learning.

This module provides a unified FastAPI interface for all AI models
in the SAMO Deep Learning pipeline.
"""

import logging
import tempfile
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Optional, Dict, List

import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .api_rate_limiter import add_rate_limiting

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global AI models (loaded on startup)
emotion_detector = None
text_summarizer = None
voice_transcriber = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage all AI models lifecycle - load on startup, cleanup on shutdown."""
    global emotion_detector, text_summarizer, voice_transcriber

    logger.info("🚀 Loading SAMO AI Pipeline...")
    start_time = time.time()

    try:
        logger.info("Loading emotion detection model...")
        try:
            # Import here to avoid issues if not available
            from src.models.emotion_detection.bert_classifier import (
                create_bert_emotion_classifier,
            )

            emotion_detector, _ = create_bert_emotion_classifier()
            logger.info("✅ Emotion detection model loaded")
        except Exception as exc:
            logger.warning(f"⚠️  Emotion detection model not available: {exc}")

        logger.info("Loading text summarization model...")
        try:
            from src.models.summarization.t5_summarizer import create_t5_summarizer

            text_summarizer = create_t5_summarizer("t5-small")
            logger.info("✅ Text summarization model loaded")
        except Exception as exc:
            logger.warning(f"⚠️  Text summarization model not available: {exc}")

        logger.info("Loading voice processing model...")
        try:
            from src.models.voice_processing.whisper_transcriber import (
                create_whisper_transcriber,
            )

            voice_transcriber = create_whisper_transcriber()
            logger.info("✅ Voice processing model loaded")
        except Exception as exc:
            logger.warning(f"⚠️  Voice processing model not available: {exc}")

        load_time = time.time() - start_time
        logger.info(f"✅ SAMO AI Pipeline loaded in {load_time:.2f} seconds")

    except Exception as exc:
        logger.error(f"❌ Failed to load SAMO AI Pipeline: {exc}")
        raise

    yield

    # Shutdown: Cleanup
    logger.info("🔄 Shutting down SAMO AI Pipeline...")
    try:
        # Cleanup any resources if needed
        logger.info("✅ SAMO AI Pipeline shutdown complete")
    except Exception as exc:
        logger.error(f"❌ Error during shutdown: {exc}")


# Initialize FastAPI with lifecycle management
app = FastAPI(
    title="SAMO AI Unified API",
    description="Complete Deep Learning Pipeline for Voice Journal Analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware (1000 requests/minute per user for testing)
add_rate_limiting(app, requests_per_minute=1000, burst_size=100, max_concurrent_requests=50, 
                 rapid_fire_threshold=100, sustained_rate_threshold=2000)


# Custom exception handler for all exceptions
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"❌ Unhandled exception: {exc}")
    logger.error(f"Request path: {request.url.path}")
    logger.error(f"Traceback: {traceback.format_exc()}")

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "type": type(exc).__name__,
        },
    )


# HTTP exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"⚠️  HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


# Request Models
class JournalEntryRequest(BaseModel):
    """Request model for journal entry analysis."""

    text: str = Field(
        ...,
        description="Journal text to analyze",
        min_length=5,
        max_length=5000,
        example="Today I received a promotion at work and I'm really excited about it.",
    )
    generate_summary: bool = Field(True, description="Whether to generate a summary")
    emotion_threshold: float = Field(0.1, description="Threshold for emotion detection", ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Today I received a promotion at work and I'm really excited about it.",
                "generate_summary": True,
                "emotion_threshold": 0.1,
            }
        }


# Unified Response Models
class EmotionAnalysis(BaseModel):
    """Emotion analysis results."""

    emotions: Dict[str, float] = Field(
        ..., description="Emotion probabilities", example={"joy": 0.75, "gratitude": 0.65}
    )
    primary_emotion: str = Field(..., description="Most confident emotion", example="joy")
    confidence: float = Field(
        ..., description="Primary emotion confidence", ge=0, le=1, example=0.75
    )
    emotional_intensity: str = Field(
        ..., description="Emotional intensity level", example="moderate"
    )


class TextSummary(BaseModel):
    """Text summarization results."""

    summary: str = Field(
        ...,
        description="Generated summary",
        example="User expressed joy about their recent promotion and gratitude toward their supportive team.",
    )
    key_emotions: List[str] = Field(
        ..., description="Key emotions identified", example=["joy", "gratitude"]
    )
    compression_ratio: float = Field(
        ..., description="Text compression ratio", ge=0, le=1, example=0.85
    )
    emotional_tone: str = Field(..., description="Overall emotional tone", example="positive")


class VoiceTranscription(BaseModel):
    """Voice transcription results."""

    text: str = Field(
        ...,
        description="Transcribed text",
        example="Today I received a promotion at work and I'm really excited about it.",
    )
    language: str = Field(..., description="Detected language", example="en")
    confidence: float = Field(..., description="Transcription confidence", ge=0, le=1, example=0.95)
    duration: float = Field(..., description="Audio duration in seconds", ge=0, example=15.4)
    word_count: int = Field(..., description="Number of words", ge=0, example=12)
    speaking_rate: float = Field(..., description="Words per minute", ge=0, example=120.5)
    audio_quality: str = Field(..., description="Audio quality assessment", example="excellent")


class CompleteJournalAnalysis(BaseModel):
    """Complete journal analysis combining all AI models."""

    transcription: Optional[VoiceTranscription] = Field(
        None, description="Voice transcription results"
    )
    emotion_analysis: EmotionAnalysis = Field(..., description="Emotion detection results")
    summary: TextSummary = Field(..., description="Text summarization results")
    processing_time_ms: float = Field(
        ..., description="Total processing time in milliseconds", ge=0, example=450.2
    )
    pipeline_status: Dict[str, bool] = Field(
        ...,
        description="Status of each AI component",
        example={"emotion_detection": True, "text_summarization": True, "voice_processing": False},
    )
    insights: Dict[str, Any] = Field(
        ..., description="Additional insights and metadata", example={"word_count": 12, "language": "en"}
    )


# Unified API Endpoints
@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "models": {
            "emotion_detection": {
                "loaded": emotion_detector is not None,
                "status": "available" if emotion_detector is not None else "unavailable"
            },
            "text_summarization": {
                "loaded": text_summarizer is not None,
                "status": "available" if text_summarizer is not None else "unavailable"
            },
            "voice_processing": {
                "loaded": voice_transcriber is not None,
                "status": "available" if voice_transcriber is not None else "unavailable"
            },
        },
    }


@app.post(
    "/analyze/journal",
    response_model=CompleteJournalAnalysis,
    tags=["Analysis"],
    summary="Analyze text journal entry",
    description="Analyze a text journal entry with emotion detection and summarization",
    response_description="Complete analysis results including emotion detection and text summarization",
)
async def analyze_journal_entry(
    request: JournalEntryRequest,
    x_api_key: Optional[str] = Header(None, description="API key for authentication"),
) -> CompleteJournalAnalysis:
    """Analyze a text journal entry with emotion detection and summarization."""
    start_time = time.time()

    try:
        # Validate input
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        # Emotion Analysis
        emotion_results = None
        if emotion_detector is not None:
            try:
                # Enhanced insights for voice processing
                emotion_results = emotion_detector.predict(request.text, threshold=request.emotion_threshold)
                logger.info(f"✅ Emotion analysis completed: {emotion_results['primary_emotion']}")
            except Exception as exc:
                logger.warning(f"⚠️  Emotion analysis failed: {exc}")
                emotion_results = {
                    "emotions": {"neutral": 1.0},
                    "primary_emotion": "neutral",
                    "confidence": 1.0,
                    "emotional_intensity": "neutral",
                }

        # Text Summarization
        summary_results = None
        if text_summarizer is not None and request.generate_summary:
            try:
                summary_results = text_summarizer.summarize(request.text)
                logger.info("✅ Text summarization completed")
            except Exception as exc:
                logger.warning(f"⚠️  Text summarization failed: {exc}")
                summary_results = {
                    "summary": request.text[:200] + "..." if len(request.text) > 200 else request.text,
                    "key_emotions": [emotion_results["primary_emotion"]] if emotion_results else ["neutral"],
                    "compression_ratio": 0.5,
                    "emotional_tone": "neutral",
                }

        # Fallback if models are not available
        if emotion_results is None:
            emotion_results = {
                "emotions": {"neutral": 1.0},
                "primary_emotion": "neutral",
                "confidence": 1.0,
                "emotional_intensity": "neutral",
            }

        if summary_results is None:
            summary_results = {
                "summary": request.text[:200] + "..." if len(request.text) > 200 else request.text,
                "key_emotions": [emotion_results["primary_emotion"]],
                "compression_ratio": 0.5,
                "emotional_tone": "neutral",
            }

        processing_time = (time.time() - start_time) * 1000

        return CompleteJournalAnalysis(
            transcription=None,
            emotion_analysis=EmotionAnalysis(**emotion_results),
            summary=TextSummary(**summary_results),
            processing_time_ms=processing_time,
            pipeline_status={
                "emotion_detection": emotion_detector is not None,
                "text_summarization": text_summarizer is not None,
                "voice_processing": False,
            },
            insights={
                "word_count": len(request.text.split()),
                "language": "en",  # Default assumption
                "text_length": len(request.text),
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"❌ Error in journal analysis: {exc}")
        raise HTTPException(status_code=500, detail="Analysis failed")


@app.post(
    "/analyze/voice-journal",
    response_model=CompleteJournalAnalysis,
    tags=["Analysis"],
    summary="Analyze voice journal entry",
    description="Complete voice journal analysis pipeline with transcription, emotion detection, and summarization",
    response_description="Complete analysis results including transcription, emotion detection, and text summarization",
)
async def analyze_voice_journal(
    audio_file: UploadFile = File(..., description="Audio file to transcribe and analyze"),
    language: Optional[str] = Form(
        None, description="Language code for transcription (auto-detect if not provided)"
    ),
    generate_summary: bool = Form(True, description="Whether to generate a summary"),
    emotion_threshold: float = Form(0.1, description="Threshold for emotion detection", ge=0, le=1),
    x_api_key: Optional[str] = Header(None, description="API key for authentication"),
) -> CompleteJournalAnalysis:
    """Complete voice journal analysis pipeline."""
    start_time = time.time()

    try:
        # Step 1: Voice Transcription
        transcription_results = None
        transcribed_text = ""
        if voice_transcriber is not None:
            try:
                # Create a temporary file for the audio
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    content = await audio_file.read()
                    temp_file.write(content)
                    temp_file_path = temp_file.name

                try:
                    transcription_results = voice_transcriber.transcribe(
                        temp_file_path, language=language
                    )
                    transcribed_text = transcription_results["text"]
                    logger.info(f"✅ Voice transcription completed: {len(transcribed_text)} characters")
                finally:
                    # Clean up temporary file
                    Path(temp_file_path).unlink(missing_ok=True)

            except Exception as exc:
                logger.warning(f"⚠️  Voice transcription failed: {exc}")
                # Continue in degraded mode
                transcribed_text = ""

        # Steps 2 & 3: Continue with text analysis using transcribed text
        if not transcribed_text.strip():
            raise HTTPException(
                status_code=400, detail="Failed to transcribe audio or audio is too short"
            )

        # Create a JournalEntryRequest for the text analysis
        text_request = JournalEntryRequest(
            text=transcribed_text,
            generate_summary=generate_summary,
            emotion_threshold=emotion_threshold,
        )

        # Delegate to text analysis
        text_analysis = await analyze_journal_entry(text_request, x_api_key)

        # Cross-model insights
        processing_time = (time.time() - start_time) * 1000

        return CompleteJournalAnalysis(
            transcription=VoiceTranscription(**transcription_results) if transcription_results else None,
            emotion_analysis=text_analysis.emotion_analysis,
            summary=text_analysis.summary,
            processing_time_ms=processing_time,
            pipeline_status={
                "emotion_detection": emotion_detector is not None,
                "text_summarization": text_summarizer is not None,
                "voice_processing": voice_transcriber is not None,
            },
            insights={
                **text_analysis.insights,
                "audio_duration": transcription_results.get("duration", 0) if transcription_results else 0,
                "audio_quality": transcription_results.get("audio_quality", "unknown") if transcription_results else "unknown",
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"❌ Error in voice journal analysis: {exc}")
        raise HTTPException(status_code=500, detail="Voice analysis failed")


@app.get(
    "/models/status",
    tags=["System"],
    summary="Get models status",
    description="Get detailed status information about all AI models in the pipeline",
)
async def get_models_status() -> Dict[str, Any]:
    """Get detailed status of all AI models."""
    return {
        "emotion_detector": {
            "loaded": emotion_detector is not None,
            "model_type": "BERT + GoEmotions",
            "capabilities": ["Multi-label emotion classification", "Emotion intensity analysis"],
            "available": emotion_detector is not None,
            "description": "Multi-label emotion classification",
        },
        "text_summarizer": {
            "loaded": text_summarizer is not None,
            "model_type": "T5",
            "capabilities": ["Text summarization", "Content compression"],
            "available": text_summarizer is not None,
            "description": "Text summarization and compression",
        },
        "voice_transcriber": {
            "loaded": voice_transcriber is not None,
            "model_type": "OpenAI Whisper",
            "capabilities": ["Speech-to-text transcription", "Language detection"],
            "available": voice_transcriber is not None,
            "description": "Speech-to-text transcription",
        },
        "pipeline": {
            "complete": all([emotion_detector, text_summarizer, voice_transcriber]),
            "partial": any([emotion_detector, text_summarizer, voice_transcriber]),
            "degraded_mode": not all([emotion_detector, text_summarizer, voice_transcriber]),
        },
    }


@app.get(
    "/",
    tags=["System"],
    summary="API information",
    description="Get information about the API endpoints and capabilities",
)
async def root() -> Dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "message": "SAMO AI Unified API is running",
        "name": "SAMO AI Unified API",
        "version": "1.0.0",
        "description": "Complete Deep Learning Pipeline for Voice Journal Analysis",
        "endpoints": {
            "health": "/health",
            "analyze_text": "/analyze/journal",
            "analyze_voice": "/analyze/voice-journal",
            "models_status": "/models/status",
        },
        "capabilities": [
            "Voice-to-text transcription",
            "Emotion detection and analysis",
            "Text summarization",
            "Complete journal processing pipeline",
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
