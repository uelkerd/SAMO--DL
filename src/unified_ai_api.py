#!/usr/bin/env python3
"""Unified AI API for SAMO Deep Learning.

This module provides a unified FastAPI interface for all AI models in the SAMO Deep
Learning pipeline.
"""

from __future__ import annotations

import asyncio
import builtins
import inspect
import json
import logging
import os
import tempfile
import time
import traceback
from collections import defaultdict
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

# Defer FastAPI imports to avoid ModuleNotFoundError when console script imports this module
# These will be imported inside the main() function when the server actually starts

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Note: Avoid monkey-patching httpx internals. Tests should handle httpx.StreamConsumed
# or public APIs directly when dealing with closed file objects.

# Global AI models (loaded on startup) - will be initialized in main()
emotion_detector = None
text_summarizer = None
voice_transcriber = None

# Global variables for FastAPI app and metrics - will be initialized in main()
app = None
REQUEST_COUNT = None
REQUEST_LATENCY = None
app_start_time = None

# JWT Authentication - will be initialized in main()
jwt_manager = None
security = None

# Global WebSocket manager - will be initialized in main()
websocket_manager = None

# Authentication models - will be initialized in main()
UserLogin = None
UserRegister = None
UserProfile = None

# Authentication dependency - will be initialized in main()
get_current_user = None

# Permission dependency - will be initialized in main()
require_permission = None

# Lifespan function - will be initialized in main()
lifespan = None

# Request and Response Models - will be initialized in main()
JournalEntryRequest = None
EmotionAnalysis = None
TextSummary = None
VoiceTranscription = None
CompleteJournalAnalysis = None
RefreshTokenRequest = None
ChatMessage = None
ChatResponse = None

# All model classes and endpoints will be defined in main()
# This allows the module to be imported without FastAPI dependencies


def _tx_to_dict(result: Any) -> dict[str, Any]:
    """Normalize transcription result (dataclass or dict) to a plain dict."""
    if isinstance(result, dict):
        return result
    return {
        "text": getattr(result, "text", ""),
        "language": getattr(result, "language", "unknown"),
        "confidence": getattr(result, "confidence", 0.0),
        "duration": getattr(result, "duration", 0.0),
        "segments": getattr(result, "segments", []),
        "no_speech_prob": getattr(result, "no_speech_probability", 0.0),
    }


def _ensure_voice_transcriber_loaded() -> None:
    """Ensure voice_transcriber is available or raise 503 (avoid global statement)."""
    if voice_transcriber is not None:
        return
    try:
        from models.voice_processing.whisper_transcriber import (
            create_whisper_transcriber as _wcreate,
        )

        logger.info("Lazy-loading Whisper transcriber: small")
        globals()["voice_transcriber"] = _wcreate("small")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Voice transcriber lazy-load failed: %s", exc)
        # HTTPException will be available when main() is called
        raise RuntimeError("Voice transcription service unavailable")


def _write_temp_wav(content: bytes) -> str:
    """Persist uploaded audio bytes to a temporary WAV file and return its path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(content)
        temp_file.flush()
        return temp_file.name


def _normalize_transcription_dict(
    d: dict[str, Any],
) -> tuple[str, str, float, float, int, float, str]:
    """Normalize transcription attributes from a dict payload."""
    text_val = d.get("text", "")
    lang_val = d.get("language", "unknown")
    conf_val = float(d.get("confidence", 0.0) or 0.0)
    duration = float(d.get("duration", 0.0) or 0.0)
    word_count = int(d.get("word_count", 0) or 0)
    speaking_rate = float(d.get("speaking_rate", 0.0) or 0.0)
    audio_quality = d.get("audio_quality", "unknown")
    return (
        text_val,
        lang_val,
        conf_val,
        duration,
        word_count,
        speaking_rate,
        audio_quality,
    )


def _infer_quality_from_duration(duration: float) -> str:
    """Heuristic mapping from audio duration to a coarse quality label."""
    if duration < 1:
        return "poor"
    if duration < 5:
        return "fair"
    if duration < 15:
        return "good"
    return "excellent"


def _normalize_transcription_obj(
    obj: Any,
) -> tuple[str, str, float, float, int, float, str]:
    """Normalize attributes from an object-like transcription result."""
    text_val = getattr(obj, "text", "")
    lang_val = getattr(obj, "language", "unknown")
    conf_val = float(getattr(obj, "confidence", 0.0) or 0.0)
    duration = float(getattr(obj, "duration", 0.0) or 0.0)
    word_count = getattr(obj, "word_count", None)
    if word_count is None:
        word_count = len((text_val or "").split())
    speaking_rate = getattr(obj, "speaking_rate", None)
    if speaking_rate is None:
        speaking_rate = (word_count / duration * 60) if duration > 0 else 0.0
    audio_quality = getattr(obj, "audio_quality", None)
    if audio_quality is None:
        audio_quality = _infer_quality_from_duration(duration)
    return (
        text_val,
        lang_val,
        conf_val,
        duration,
        int(word_count),
        float(speaking_rate),
        audio_quality,
    )


def _normalize_transcription_attrs(
    result: Any,
) -> tuple[str, str, float, float, int, float, str]:
    """Extract common attributes from a transcription result object or dict."""
    if isinstance(result, dict):
        return _normalize_transcription_dict(result)
    return _normalize_transcription_obj(result)


def _ensure_summarizer_loaded() -> None:
    """Ensure text_summarizer is available or raise 503 (avoid global statement)."""
    if text_summarizer is not None:
        return
    try:
        from src.models.summarization.t5_summarizer import (
            create_t5_summarizer as _create,
        )

        logger.info("Lazy-loading summarizer model: t5-small")
        globals()["text_summarizer"] = _create("t5-small")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Summarizer lazy-load failed: %s", exc)
        raise RuntimeError("Text summarization service unavailable")


def _get_request_scoped_summarizer(model: str):
    """Return summarizer for requested model.

    If the requested model differs, attempt to create a request-scoped instance. On
    failure, raise ValueError/RuntimeError instead of HTTPException.
    """
    if hasattr(text_summarizer, "model_name") and text_summarizer.model_name != model:
        try:
            from src.models.summarization.t5_summarizer import (
                create_t5_summarizer as _create,
            )

            logger.info(
                (
                    "Requested summarizer model '%s' differs from default '%s'; "
                    "using request-scoped instance"
                ),
                model,
                getattr(text_summarizer, "model_name", "unknown"),
            )
            return _create(model)
        except ValueError as exc:  # invalid model name/config
            raise ValueError(f"Invalid summarizer model: {model}") from exc
        except Exception as exc:  # treat unknown models as bad request in tests
            raise RuntimeError(
                f"Requested summarizer model '{model}' unavailable"
            ) from exc
    return text_summarizer


def _derive_emotion(summary_text: str) -> tuple[str, list[str]]:
    """Infer emotional tone and key emotions from summary text."""
    if not summary_text or not emotion_detector:
        return "neutral", []
    try:
        # This would need to be implemented based on the emotion detection logic
        # For now, return neutral
        return "neutral", []
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning("Could not determine emotional tone from summary: %s", exc)
        return "neutral", []


def main():
    """Main entry point for the SAMO AI API server."""
    # Import FastAPI and related dependencies here to avoid ModuleNotFoundError
    # when console script imports this module
    try:
        import uvicorn
        from fastapi import (
            Depends,
            FastAPI,
            File,
            Form,
            Header,
            HTTPException,
            Query,
            Request,
            UploadFile,
            WebSocket,
            status,
        )
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse, Response
        from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
        from fastapi.websockets import WebSocketDisconnect
        from prometheus_client import (
            CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
        )
        from pydantic import BaseModel, Field

        from api_rate_limiter import add_rate_limiting
        from security.jwt_manager import JWTManager, TokenPayload, TokenResponse
    except ImportError as e:
        raise ImportError(
            f"Required dependencies not found: {e}. "
            "Please install the project dependencies with: pip install -e ."
        ) from e

    # Now that we have the imports, we can set up the global variables
    global (
        app, REQUEST_COUNT, REQUEST_LATENCY, app_start_time, jwt_manager, security,
        websocket_manager
    )
    global (
        UserLogin, UserRegister, UserProfile, emotion_detector, text_summarizer,
        voice_transcriber
    )
    global (
        JournalEntryRequest, EmotionAnalysis, TextSummary, VoiceTranscription,
        CompleteJournalAnalysis
    )
    global (
        RefreshTokenRequest, ChatMessage, ChatResponse, get_current_user,
        require_permission, lifespan
    )

    # Initialize global variables
    app_start_time = time.time()

    # Metrics
    REQUEST_COUNT = Counter(
        "samo_requests_total",
        "Total HTTP requests",
        ["endpoint", "method", "status"],
    )
    REQUEST_LATENCY = Histogram(
        "samo_request_latency_seconds",
        "Request latency (s)",
        ["endpoint", "method"],
    )

    # JWT Authentication
    jwt_manager = JWTManager()
    security = HTTPBearer()

    # Global WebSocket manager
    # websocket_manager = WebSocketConnectionManager()  # TODO: Define this class
    websocket_manager = None

    # Authentication models
    class UserLogin(BaseModel):
        """User login request model."""
        username: str = Field(description="Username")
        password: str = Field(description="Password", min_length=6)

    class UserRegister(BaseModel):
        """User registration request model."""
        username: str = Field(description="Username")
        email: str = Field(description="Email address")
        password: str = Field(description="Password", min_length=6)
        full_name: str = Field(description="Full name")

    class UserProfile(BaseModel):
        """User profile response model."""
        user_id: str = Field(description="User ID")
        username: str = Field(description="Username")
        email: str = Field(description="Email address")
        full_name: str = Field(description="Full name")
        permissions: list[str] = Field(
            default_factory=list, description="User permissions"
        )
        created_at: str = Field(description="Account creation date")

    # Set global references
    globals()['UserLogin'] = UserLogin
    globals()['UserRegister'] = UserRegister
    globals()['UserProfile'] = UserProfile

    # Authentication dependency
    async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> TokenPayload:
        """Get current authenticated user from JWT token."""
        token = credentials.credentials
        if payload := jwt_manager.verify_token(token):
            return payload
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Permission dependency
    def require_permission(permission: str):
        """Require specific permission for endpoint access."""
        async def permission_checker(
            request: Request,
            current_user: TokenPayload = Depends(get_current_user),
        ):
            if permission not in current_user.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission '{permission}' required",
                )
            return current_user
        return permission_checker

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
        """Manage all AI models lifecycle - load on startup, cleanup on shutdown."""
        global emotion_detector, text_summarizer, voice_transcriber

        logger.info("Loading SAMO AI Pipeline...")
        start_time = time.time()

        try:
            # Load AI models here (same logic as before)
            logger.info("Loading emotion detection model...")
            try:
                from models.emotion_detection.hf_loader import (
                    load_emotion_model_multi_source,
                )
                hf_model_id = os.getenv("EMOTION_MODEL_ID", "0xmnrv/samo")
                hf_token = os.getenv("HF_TOKEN")
                emotion_detector = load_emotion_model_multi_source(
                    model_id=hf_model_id,
                    token=hf_token,
                )
                logger.info("Loaded emotion model from HF Hub: %s", hf_model_id)
            except Exception as exc:
                logger.warning("Emotion detection model not available: %s", exc)

            logger.info("Loading text summarization model...")
            try:
                from models.summarization.t5_summarizer import create_t5_summarizer
                text_summarizer = create_t5_summarizer("t5-small")
                logger.info("Text summarization model loaded")
            except Exception as exc:
                logger.warning("Text summarization model not available: %s", exc)

            logger.info("Loading voice processing model...")
            try:
                from models.voice_processing.whisper_transcriber import (
                    create_whisper_transcriber,
                )
                voice_transcriber = create_whisper_transcriber()
                logger.info("Voice processing model loaded")
            except Exception as exc:
                logger.warning("Voice processing model not available: %s", exc)

            load_time = time.time() - start_time
            logger.info("SAMO AI Pipeline loaded in %.2f seconds", load_time)

        except Exception as exc:
            logger.exception("Failed to load SAMO AI Pipeline: %s", exc)
            raise

        yield

        # Shutdown: Cleanup
        logger.info("Shutting down SAMO AI Pipeline...")
        try:
            logger.info("SAMO AI Pipeline shutdown complete")
        except Exception as exc:
            logger.exception("Error during shutdown: %s", exc)

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

    # Add rate limiting middleware
    add_rate_limiting(
        app,
        requests_per_minute=1000,
        burst_size=100,
        max_concurrent_requests=50,
        rapid_fire_threshold=100,
        sustained_rate_threshold=2000,
    )

    # Define request/response models
    class JournalEntryRequest(BaseModel):
        """Request model for journal entry analysis."""
        text: str = Field(
            ..., description="Journal text to analyze", min_length=5, max_length=5000
        )
        generate_summary: bool = Field(
            default=True, description="Whether to generate a summary"
        )
        emotion_threshold: float = Field(
            0.1, description="Threshold for emotion detection", ge=0, le=1
        )

    # Add a simple health endpoint
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": time.time()}

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
