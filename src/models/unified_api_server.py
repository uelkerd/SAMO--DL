#!/usr/bin/env python3
"""
SAMO Unified API Server

This module provides a unified FastAPI server that integrates:
- T5 Summarization Model
- Whisper Transcription Model
- BERT Emotion Detection Model

The server provides individual endpoints for each model plus combined
endpoints that chain multiple models together for comprehensive journal
entry processing.

Key Features:
- RESTful API with OpenAPI documentation
- Individual model endpoints
- Combined processing pipelines
- Comprehensive error handling
- Request/response validation
- Health monitoring
- CORS support for web applications
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch

# Import SAMO models
from summarization.t5_summarizer import create_t5_summarizer
from voice_processing.whisper_transcriber import create_whisper_transcriber
from emotion_detection.samo_bert_emotion_classifier import (
    create_samo_bert_emotion_classifier
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Models
class SummarizationRequest(BaseModel):
    """Request model for text summarization."""
    text: str = Field(
        ..., min_length=10, max_length=10000, description="Text to summarize"
    )
    max_length: Optional[int] = Field(
        128, ge=30, le=512, description="Maximum summary length"
    )
    min_length: Optional[int] = Field(
        30, ge=10, le=100, description="Minimum summary length"
    )
    num_beams: Optional[int] = Field(
        4, ge=1, le=8, description="Beam search size"
    )

class SummarizationResponse(BaseModel):
    """Response model for summarization."""
    summary: str
    original_length: int
    summary_length: int
    processing_time: float
    model_info: Dict[str, Any]

class TranscriptionRequest(BaseModel):
    """Request model for audio transcription."""
    language: Optional[str] = Field(
        None, description="Language code (auto-detect if None)"
    )
    initial_prompt: Optional[str] = Field(
        None, description="Context prompt for better accuracy"
    )

class TranscriptionResponse(BaseModel):
    """Response model for transcription."""
    text: str
    language: str
    confidence: float
    duration: float
    processing_time: float
    audio_quality: str
    word_count: int
    speaking_rate: float
    no_speech_probability: float

class EmotionDetectionRequest(BaseModel):
    """Request model for emotion detection."""
    text: str = Field(
        ..., min_length=10, max_length=10000, description="Text to analyze"
    )
    threshold: Optional[float] = Field(
        0.5, ge=0.1, le=0.9, description="Prediction threshold"
    )
    top_k: Optional[int] = Field(
        None, ge=1, le=10, description="Return top-k emotions"
    )

class EmotionDetectionResponse(BaseModel):
    """Response model for emotion detection."""
    emotions: List[str]
    probabilities: List[float]
    predictions: List[int]
    processing_time: float
    model_info: Dict[str, Any]

class CombinedProcessingRequest(BaseModel):
    """Request model for combined audio-to-emotion analysis."""
    language: Optional[str] = Field(None, description="Language for transcription")
    summary_max_length: Optional[int] = Field(128, description="Max summary length")
    emotion_threshold: Optional[float] = Field(0.5, description="Emotion detection threshold")

class CombinedProcessingResponse(BaseModel):
    """Response model for combined processing."""
    transcription: TranscriptionResponse
    summary: SummarizationResponse
    emotions: EmotionDetectionResponse
    total_processing_time: float
    pipeline_steps: List[str]

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    models_loaded: Dict[str, bool]
    memory_usage: Dict[str, float]

# Unified API Server
class SAMOUnifiedAPIServer:
    """Unified API server for SAMO deep learning models."""

    def __init__(self):
        """Initialize the unified API server."""
        self.app = FastAPI(
            title="SAMO Unified API",
            description="Unified API server for SAMO deep learning models",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )

        # Configure CORS with environment-based origins
        import os
        allowed_origins = os.getenv("API_ALLOWED_ORIGINS", "https://your-production-domain.com")
        allowed_origins_list = [origin.strip() for origin in allowed_origins.split(",")]

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins_list,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Initialize models
        self.models = {}
        self._load_models()

        # Setup routes
        self._setup_routes()

        logger.info("✅ SAMO Unified API Server initialized")

    def _load_models(self):
        """Load all SAMO models."""
        try:
            logger.info("Loading T5 Summarization Model...")
            self.models["summarizer"] = create_t5_summarizer("t5-small")
            logger.info("✅ T5 Summarization Model loaded")

        except Exception as e:
            logger.error(f"❌ Failed to load T5 Summarization Model: {e}")
            self.models["summarizer"] = None

        try:
            logger.info("Loading Whisper Transcription Model...")
            self.models["transcriber"] = create_whisper_transcriber("base")
            logger.info("✅ Whisper Transcription Model loaded")

        except Exception as e:
            logger.error(f"❌ Failed to load Whisper Transcription Model: {e}")
            self.models["transcriber"] = None

        try:
            logger.info("Loading BERT Emotion Detection Model...")
            model, loss_fn = create_samo_bert_emotion_classifier()
            self.models["emotion_detector"] = model
            logger.info("✅ BERT Emotion Detection Model loaded")

        except Exception as e:
            logger.error(f"❌ Failed to load BERT Emotion Detection Model: {e}")
            self.models["emotion_detector"] = None

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return self._get_health_status()

        @self.app.post("/summarize", response_model=SummarizationResponse)
        async def summarize_text(request: SummarizationRequest):
            """Summarize text using T5 model."""
            if not self.models["summarizer"]:
                raise HTTPException(status_code=503, detail="Summarization model not available")

            start_time = time.time()
            try:
                summary = self.models["summarizer"].generate_summary(
                    request.text,
                    max_length=request.max_length,
                    min_length=request.min_length,
                    num_beams=request.num_beams
                )

                processing_time = time.time() - start_time

                return SummarizationResponse(
                    summary=summary,
                    original_length=len(request.text),
                    summary_length=len(summary),
                    processing_time=processing_time,
                    model_info=self.models["summarizer"].get_model_info()
                )

            except Exception as e:
                logger.error(f"Summarization error: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Summarization failed: {str(e)}"
                )

        @self.app.post("/transcribe", response_model=TranscriptionResponse)
        async def transcribe_audio(
            background_tasks: BackgroundTasks,
            file: UploadFile = File(...),
            language: Optional[str] = Form(None),
            initial_prompt: Optional[str] = Form(None)
        ):
            """Transcribe audio using Whisper model."""
            if not self.models["transcriber"]:
                raise HTTPException(
                    status_code=503, detail="Transcription model not available"
                )

            # Validate file type using MIME type
            import magic

            supported_mime_types = {
                "audio/mpeg",
                "audio/wav",
                "audio/x-wav",
                "audio/x-m4a",
                "audio/mp4",
                "audio/ogg",
                "audio/flac",
                "audio/x-flac",
            }

            file_content = await file.read()
            mime_type = magic.from_buffer(file_content, mime=True)
            await file.seek(0)  # Reset file pointer for downstream use

            if mime_type not in supported_mime_types:
                raise HTTPException(status_code=400, detail=f"Unsupported audio format: {mime_type}")

            import uuid

            try:
                # Save uploaded file temporarily with a unique name
                unique_suffix = uuid.uuid4().hex
                extension = file.filename.split('.')[-1] if '.' in file.filename else ''
                temp_path = f"/tmp/{unique_suffix}.{extension}" if extension else f"/tmp/{unique_suffix}"
                with open(temp_path, "wb") as buffer:
                    buffer.write(file_content)

                # Transcribe
                result = self.models["transcriber"].transcribe(
                    temp_path,
                    language=language,
                    initial_prompt=initial_prompt
                )

                # Cleanup temp file
                background_tasks.add_task(Path(temp_path).unlink, missing_ok=True)

                return TranscriptionResponse(
                    text=result.text,
                    language=result.language,
                    confidence=result.confidence,
                    duration=result.duration,
                    processing_time=result.processing_time,
                    audio_quality=result.audio_quality,
                    word_count=result.word_count,
                    speaking_rate=result.speaking_rate,
                    no_speech_probability=result.no_speech_probability
                )

            except Exception as e:
                logger.error(f"Transcription error: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Transcription failed: {str(e)}"
                )

        @self.app.post("/detect-emotions", response_model=EmotionDetectionResponse)
        async def detect_emotions(request: EmotionDetectionRequest):
            """Detect emotions using BERT model."""
            if not self.models["emotion_detector"]:
                raise HTTPException(
                    status_code=503, detail="Emotion detection model not available"
                )

            start_time = time.time()
            try:
                results = self.models["emotion_detector"].predict_emotions(
                    request.text,
                    threshold=request.threshold,
                    top_k=request.top_k
                )

                processing_time = time.time() - start_time

                return EmotionDetectionResponse(
                    emotions=results["emotions"][0] 
                    if results["emotions"] else [],
                    probabilities=results["probabilities"][0] 
                    if results["probabilities"] else [],
                    predictions=results["predictions"][0] 
                    if results["predictions"] else [],
                    processing_time=processing_time,
                    model_info={
                        "model_name": "SAMO BERT Emotion Classifier",
                        "num_emotions": 28,
                        "device": str(self.models["emotion_detector"].device)
                    }
                )

            except Exception as e:
                logger.error(f"Emotion detection error: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Emotion detection failed: {str(e)}"
                )

        @self.app.post("/process-audio", response_model=CombinedProcessingResponse)
        async def process_audio_completely(
            background_tasks: BackgroundTasks,
            file: UploadFile = File(...),
            language: Optional[str] = Form(None),
            summary_max_length: Optional[int] = Form(128),
            emotion_threshold: Optional[float] = Form(0.5)
        ):
            """Complete pipeline: Audio -> Transcription -> Summary -> Emotion Analysis."""
            pipeline_start = time.time()
            pipeline_steps = []

            try:
                # Step 1: Transcribe audio
                pipeline_steps.append("transcription")
                if not self.models["transcriber"]:
                    raise HTTPException(
                    status_code=503, detail="Transcription model not available"
                )

                # Generate unique temporary filename
                unique_suffix = uuid.uuid4().hex
                extension = file.filename.split('.')[-1] if '.' in file.filename else ''
                temp_path = f"/tmp/{unique_suffix}.{extension}" if extension else f"/tmp/{unique_suffix}"
                with open(temp_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)

                transcription_result = self.models["transcriber"].transcribe(
                    temp_path, language=language
                )

                transcription_response = TranscriptionResponse(
                    text=transcription_result.text,
                    language=transcription_result.language,
                    confidence=transcription_result.confidence,
                    duration=transcription_result.duration,
                    processing_time=transcription_result.processing_time,
                    audio_quality=transcription_result.audio_quality,
                    word_count=transcription_result.word_count,
                    speaking_rate=transcription_result.speaking_rate,
                    no_speech_probability=transcription_result.no_speech_probability
                )

                # Step 2: Summarize transcription
                pipeline_steps.append("summarization")
                if self.models["summarizer"]:
                    summary = self.models["summarizer"].generate_summary(
                        transcription_result.text,
                        max_length=summary_max_length
                    )

                    summary_response = SummarizationResponse(
                        summary=summary,
                        original_length=len(transcription_result.text),
                        summary_length=len(summary),
                        processing_time=0.0,  # Would need to track separately
                        model_info=self.models["summarizer"].get_model_info()
                    )
                else:
                    summary_response = SummarizationResponse(
                        summary=transcription_result.text[:200] + "...",
                        original_length=len(transcription_result.text),
                        summary_length=200,
                        processing_time=0.0,
                        model_info={"error": "Summarization model not available"}
                    )

                # Step 3: Detect emotions
                pipeline_steps.append("emotion_detection")
                if self.models["emotion_detector"]:
                    emotion_results = self.models["emotion_detector"].predict_emotions(
                        transcription_result.text,
                        threshold=emotion_threshold
                    )

                    emotion_response = EmotionDetectionResponse(
                        emotions=emotion_results["emotions"][0] 
                        if emotion_results["emotions"] else [],
                        probabilities=emotion_results["probabilities"][0] 
                        if emotion_results["probabilities"] else [],
                        predictions=emotion_results["predictions"][0] 
                        if emotion_results["predictions"] else [],
                        processing_time=0.0,
                        model_info={
                            "model_name": "SAMO BERT Emotion Classifier",
                            "num_emotions": 28,
                            "device": str(self.models["emotion_detector"].device)
                        }
                    )
                else:
                    emotion_response = EmotionDetectionResponse(
                        emotions=[],
                        probabilities=[],
                        predictions=[],
                        processing_time=0.0,
                        model_info={"error": "Emotion detection model not available"}
                    )

                # Cleanup
                background_tasks.add_task(Path(temp_path).unlink, missing_ok=True)

                total_time = time.time() - pipeline_start

                return CombinedProcessingResponse(
                    transcription=transcription_response,
                    summary=summary_response,
                    emotions=emotion_response,
                    total_processing_time=total_time,
                    pipeline_steps=pipeline_steps
                )

            except Exception as e:
                logger.error(f"Combined processing error: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Processing failed: {str(e)}"
                )

    def _get_health_status(self) -> HealthResponse:
        """Get comprehensive health status."""
        models_loaded = {
            "summarizer": self.models["summarizer"] is not None,
            "transcriber": self.models["transcriber"] is not None,
            "emotion_detector": self.models["emotion_detector"] is not None,
        }

        # Get memory usage if available
        memory_usage = {}
        if torch.cuda.is_available():
            memory_usage = {
                "gpu_allocated": torch.cuda.memory_allocated() / 1024**3,
                "gpu_reserved": torch.cuda.memory_reserved() / 1024**3,
            }

        return HealthResponse(
            status="healthy" if all(models_loaded.values()) else "degraded",
            timestamp=datetime.now(),
            models_loaded=models_loaded,
            memory_usage=memory_usage
        )

    def run(self, host: str = "0.0.0.0", port: int = 8000, workers: int = 1, reload: bool = False):
        """Run the API server."""
        logger.info(f"Starting SAMO Unified API Server on {host}:{port} with {workers} worker(s)")
        uvicorn.run(self.app, host=host, port=port, workers=workers, reload=reload)


# Global server instance
server = SAMOUnifiedAPIServer()

if __name__ == "__main__":
    # Start the server
    print("🚀 Starting SAMO Unified API Server")
    print("=" * 50)
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔄 ReDoc Documentation: http://localhost:8000/redoc")
    print("💚 Health Check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)

    server.run()
