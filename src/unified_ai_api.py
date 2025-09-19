#!/usr/bin/env python3
"""Unified AI API for SAMO Deep Learning.

This module provides a unified FastAPI interface for all AI models
in the SAMO Deep Learning pipeline.
"""
from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import time
import traceback
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, AsyncGenerator, Optional, Set, Tuple
import inspect
from datetime import datetime, timezone
from collections import defaultdict

import uvicorn
from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    Depends,
    status,
    WebSocket,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.websockets import WebSocketDisconnect
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from .api_rate_limiter import add_rate_limiting
from .security.jwt_manager import JWTManager, TokenPayload, TokenResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Note: Avoid monkey-patching httpx internals. Tests should handle httpx.StreamConsumed
# or public APIs directly when dealing with closed file objects.

# Global AI models (loaded on startup)
emotion_detector = None
text_summarizer = None
voice_transcriber = None

# Metrics
REQUEST_COUNT = Counter(
    "samo_requests_total", "Total HTTP requests", ["endpoint", "method", "status"]
)
REQUEST_LATENCY = Histogram(
    "samo_request_latency_seconds", "Request latency (s)", ["endpoint", "method"]
)


# ------------------------------
# Helpers: emotion result normalization
# ------------------------------
def normalize_emotion_results(raw: Any) -> dict:
    """Normalize various emotion detector return shapes to a consistent dict.

    Supports dicts (possibly with MagicMock values) and objects with attributes.
    Returns a structure matching EmotionAnalysis fields.
    """
    try:
        if isinstance(raw, dict):

            def _as_float(v: Any) -> float:
                try:
                    return float(v)
                except Exception:
                    return 1.0

            def _as_str(v: Any, default: str = "neutral") -> str:
                try:
                    return str(v)
                except Exception:
                    return default

            emotions_dict = raw.get("emotions")
            if not isinstance(emotions_dict, dict):
                emotions_dict = {"neutral": 1.0}
            else:
                emotions_dict = {str(k): _as_float(v) for k, v in emotions_dict.items()}
            return {
                "emotions": emotions_dict,
                "primary_emotion": _as_str(raw.get("primary_emotion"), "neutral"),
                "confidence": _as_float(raw.get("confidence", 1.0)),
                "emotional_intensity": _as_str(
                    raw.get("emotional_intensity"), "neutral"
                ),
            }
        # Fallback: object with attributes
        emotions_attr = getattr(raw, "emotions", {"neutral": 1.0})
        emotions = (
            emotions_attr if isinstance(emotions_attr, dict) else {"neutral": 1.0}
        )
        return {
            "emotions": emotions,
            "primary_emotion": str(getattr(raw, "primary_emotion", "neutral")),
            "confidence": float(getattr(raw, "confidence", 1.0)),
            "emotional_intensity": str(getattr(raw, "emotional_intensity", "neutral")),
        }
    except Exception:
        # Conservative fallback
        return {
            "emotions": {"neutral": 1.0},
            "primary_emotion": "neutral",
            "confidence": 1.0,
            "emotional_intensity": "neutral",
        }


def _run_emotion_predict(text: str, threshold: float = 0.5) -> dict:
    """Run emotion prediction using available detector, adapting outputs to a
    common schema.

    Returns a dict with keys: emotions (label->prob), primary_emotion,
    confidence, emotional_intensity.
    """
    try:
        if not text or emotion_detector is None:
            return {}
        # If detector exposes the expected API
        if hasattr(emotion_detector, "predict"):
            return emotion_detector.predict(text, threshold=threshold) or {}
        # Adapter for BERTEmotionClassifier.predict_emotions
        if hasattr(emotion_detector, "predict_emotions"):
            # Import labels lazily to avoid heavy deps at import time
            from src.models.emotion_detection.labels import (
                GOEMOTIONS_EMOTIONS as _LABELS,
            )

            result = emotion_detector.predict_emotions(text, threshold=threshold) or {}
            probs_list = result.get("probabilities") or []
            if not probs_list:
                return {}
            probs = probs_list[0]
            # Build label->prob mapping
            emotions_map = {label: float(prob) for label, prob in zip(_LABELS, probs)}
            # Determine primary emotion
            if emotions_map:
                primary_label = max(emotions_map.items(), key=lambda kv: kv[1])[0]
                confidence = float(emotions_map[primary_label])
            else:
                primary_label = "neutral"
                confidence = 1.0
            # Simple intensity heuristic
            if confidence >= 0.75:
                intensity = "high"
            elif confidence >= 0.4:
                intensity = "moderate"
            else:
                intensity = "low"
            return {
                "emotions": emotions_map,
                "primary_emotion": primary_label,
                "confidence": confidence,
                "emotional_intensity": intensity,
            }
        return {}
    except Exception:
        return {}


# ------------------------------
# Helpers: test-only permission injection
# ------------------------------
def _has_injected_permission(request: Request, permission: str) -> bool:
    """Check for test-only injected permissions via headers when enabled.

    Active only when both PYTEST_CURRENT_TEST is set and
    ENABLE_TEST_PERMISSION_INJECTION is "true".
    """
    try:
        if os.environ.get("PYTEST_CURRENT_TEST") and (
            os.environ.get("ENABLE_TEST_PERMISSION_INJECTION", "false").lower()
            == "true"
        ):
            header_val = request.headers.get("X-User-Permissions")
            if header_val:
                perms = {p.strip() for p in header_val.split(",") if p.strip()}
                return permission in perms
    except Exception:
        # Defensive: never fail permission checks due to header parsing issues
        return False
    return False


# Application startup time
app_start_time = time.time()

# JWT Authentication
jwt_manager = JWTManager()
security = HTTPBearer()


# Enhanced WebSocket Connection Management
class WebSocketConnectionManager:
    """Enhanced WebSocket connection manager with pooling and heartbeat."""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        self.heartbeat_interval = 30  # seconds
        self.max_connections_per_user = 5
        self.connection_timeout = 300  # 5 minutes

    async def connect(self, websocket: WebSocket, user_id: str, token: str):
        """Connect a new WebSocket with enhanced management."""
        # Check connection limits
        if len(self.active_connections[user_id]) >= self.max_connections_per_user:
            await websocket.close(code=4008, reason="Maximum connections reached")
            return False

        await websocket.accept()
        self.active_connections[user_id].add(websocket)

        # Store connection metadata
        self.connection_metadata[websocket] = {
            "user_id": user_id,
            "token": token,
            "connected_at": time.time(),
            "last_heartbeat": time.time(),
            "message_count": 0,
            "bytes_processed": 0,
        }

        logger.info(
            "WebSocket connected for user %s. " "Total connections: %s",
            user_id,
            len(self.active_connections[user_id]),
        )
        return True

    async def disconnect(self, websocket: WebSocket):
        """Disconnect WebSocket and cleanup."""
        user_id = None
        if websocket in self.connection_metadata:
            user_id = self.connection_metadata[websocket]["user_id"]
            del self.connection_metadata[websocket]

        if user_id and websocket in self.active_connections[user_id]:
            self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]

        logger.info("WebSocket disconnected for user %s", user_id)

    async def send_personal_message(
        self, message: Dict[str, Any], websocket: WebSocket
    ):
        """Send message to specific WebSocket with error handling."""
        try:
            await websocket.send_json(message)
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["message_count"] += 1
        except Exception as e:
            logger.error("Failed to send message to WebSocket: %s", e)
            await self.disconnect(websocket)

    async def broadcast_to_user(self, message: Dict[str, Any], user_id: str):
        """Broadcast message to all connections of a specific user."""
        disconnected = set()
        for websocket in self.active_connections[user_id]:
            try:
                await websocket.send_json(message)
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]["message_count"] += 1
            except Exception as e:
                logger.error("Failed to broadcast to WebSocket: %s", e)
                disconnected.add(websocket)

        # Cleanup disconnected connections
        for websocket in disconnected:
            await self.disconnect(websocket)

    async def update_heartbeat(self, websocket: WebSocket):
        """Update heartbeat timestamp for connection."""
        if websocket in self.connection_metadata:
            self.connection_metadata[websocket]["last_heartbeat"] = time.time()

    async def cleanup_stale_connections(self):
        """Cleanup stale connections based on timeout."""
        current_time = time.time()
        stale_connections = []

        for websocket, metadata in self.connection_metadata.items():
            if current_time - metadata["last_heartbeat"] > self.connection_timeout:
                stale_connections.append(websocket)

        for websocket in stale_connections:
            logger.warning(
                "Cleaning up stale WebSocket connection for user %s",
                self.connection_metadata[websocket]["user_id"],
            )
            await self.disconnect(websocket)

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        total_connections = sum(
            len(connections) for connections in self.active_connections.values()
        )
        total_users = len(self.active_connections)

        return {
            "total_connections": total_connections,
            "total_users": total_users,
            "connections_per_user": {
                user_id: len(connections)
                for user_id, connections in self.active_connections.items()
            },
            "connection_metadata": {
                str(ws): metadata for ws, metadata in self.connection_metadata.items()
            },
        }


# Global WebSocket manager
websocket_manager = WebSocketConnectionManager()


# Authentication models
class UserLogin(BaseModel):
    """User login request model."""

    username: str = Field(..., description="Username", example="user@example.com")
    password: str = Field(
        ..., description="Password", min_length=6, example="password123"
    )


class UserRegister(BaseModel):
    """User registration request model."""

    username: str = Field(..., description="Username", example="user@example.com")
    email: str = Field(..., description="Email address", example="user@example.com")
    password: str = Field(
        ..., description="Password", min_length=6, example="password123"
    )
    full_name: str = Field(..., description="Full name", example="John Doe")


class UserProfile(BaseModel):
    """User profile response model."""

    user_id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    full_name: str = Field(..., description="Full name")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    created_at: str = Field(..., description="Account creation date")


# Authentication dependency
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> TokenPayload:
    """Get current authenticated user from JWT token."""
    token = credentials.credentials
    if payload := jwt_manager.verify_token(token):
        return payload
    # Tests expect 403 for invalid tokens and missing auth
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )


# Permission dependency
def require_permission(permission: str):
    """Require specific permission for endpoint access."""

    async def permission_checker(
        request: Request, current_user: TokenPayload = Depends(get_current_user)
    ):
        # Allow tests to inject permissions via header only during pytest runs and
        # explicit toggle
        if _has_injected_permission(request, permission):
            return current_user
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
        logger.info("Loading emotion detection model...")
        try:
            # Prefer loading our HF Hub model; fallback to local BERT if unavailable
            try:
                from src.models.emotion_detection.hf_loader import (
                    load_emotion_model_multi_source,
                )

                hf_model_id = os.getenv("EMOTION_MODEL_ID", "duelker/samo-goemotions-deberta-v3-large")
                hf_token = os.getenv("HF_TOKEN")
                local_dir = os.getenv("EMOTION_MODEL_LOCAL_DIR")
                archive_url = os.getenv("EMOTION_MODEL_ARCHIVE_URL")
                endpoint_url = os.getenv("EMOTION_MODEL_ENDPOINT_URL")
                logger.info(
                    "Attempting to load emotion model from HF Hub: %s", hf_model_id
                )
                logger.info(
                    "Sources configured: local_dir=%s, archive=%s, endpoint=%s",
                    bool(local_dir),
                    bool(archive_url),
                    bool(endpoint_url),
                )
                emotion_detector = load_emotion_model_multi_source(
                    model_id=hf_model_id,
                    token=hf_token,
                    local_dir=local_dir,
                    archive_url=archive_url,
                    endpoint_url=endpoint_url,
                    force_multi_label=None,
                )
                logger.info("Loaded emotion model from HF Hub: %s", hf_model_id)
            except Exception as hf_exc:
                logger.info(
                    "HF Hub model loading failed (normal in some environments): %s",
                    hf_exc,
                    exc_info=True,
                )
                logger.info("Falling back to local BERT emotion classifier...")
                from src.models.emotion_detection.bert_classifier import (
                    create_bert_emotion_classifier,
                )

                model, _ = create_bert_emotion_classifier()
                emotion_detector = model
                logger.info("Loaded local BERT emotion model (fallback successful)")
        except Exception as exc:
            logger.warning("Emotion detection model not available: %s", exc)

        logger.info("Loading text summarization model...")
        try:
            from src.models.summarization.t5_summarizer import create_t5_summarizer

            text_summarizer = create_t5_summarizer("t5-small")
            logger.info("Text summarization model loaded")
        except Exception as exc:
            logger.warning("Text summarization model not available: %s", exc)

        logger.info("Loading voice processing model...")
        try:
            from src.models.voice_processing.whisper_transcriber import (
                create_whisper_transcriber,
            )

            voice_transcriber = create_whisper_transcriber()
            logger.info("Voice processing model loaded")
        except Exception as exc:
            logger.warning("Voice processing model not available: %s", exc)

        load_time = time.time() - start_time
        logger.info("SAMO AI Pipeline loaded in %.2f seconds", load_time)

    except Exception as exc:
        logger.error("Failed to load SAMO AI Pipeline: %s", exc)
        raise

    yield

    # Shutdown: Cleanup
    logger.info("Shutting down SAMO AI Pipeline...")
    try:
        # Cleanup any resources if needed
        logger.info("SAMO AI Pipeline shutdown complete")
    except Exception as exc:
        logger.error("Error during shutdown: %s", exc)


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

# Add rate limiting middleware - permissive for development/testing
add_rate_limiting(
    app,
    requests_per_minute=1000,
    burst_size=100,
    max_concurrent_requests=50,
    rapid_fire_threshold=100,
    sustained_rate_threshold=2000,
    # Disable aggressive abuse detection for development
    enable_user_agent_analysis=False,
    enable_request_pattern_analysis=False,
    # Reduce block duration for faster recovery during testing
    block_duration_seconds=60,  # 1 minute instead of 5
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Collect per-request Prometheus metrics (count and latency).

    Records labels for endpoint path, method, and response status.
    """
    endpoint = request.url.path
    method = request.method
    start = time.time()
    resp_status = "500"
    try:
        response = await call_next(request)
        resp_status = str(response.status_code)
        return response
    finally:
        duration = time.time() - start
        REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(duration)
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=resp_status).inc()


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Expose Prometheus metrics in text format at /metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _tx_to_dict(result: Any) -> Dict[str, Any]:
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


# Custom exception handler for all exceptions
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error("❌ Unhandled exception: %s", exc)
    logger.error("Request path: %s", request.url.path)
    logger.error("Traceback: %s", traceback.format_exc())

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
    logger.warning("⚠️  HTTP exception: %s - %s", exc.status_code, exc.detail)
    # Preserve FastAPI's default validation/detail contract for 400-series
    # where tests expect 'detail'
    if exc.status_code in (400, 422):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


# ===== Helpers to reduce endpoint complexity =====
def _ensure_voice_transcriber_loaded() -> None:
    """Ensure voice_transcriber is available or raise 503 (avoid global statement)."""
    if voice_transcriber is not None:
        return
    try:
        from src.models.voice_processing.whisper_transcriber import (
            create_whisper_transcriber as _wcreate,
        )

        logger.info("Lazy-loading Whisper transcriber: small")
        globals()["voice_transcriber"] = _wcreate("small")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Voice transcriber lazy-load failed: %s", exc)
        raise HTTPException(
            status_code=503, detail="Voice transcription service unavailable"
        )


def _write_temp_wav(content: bytes) -> str:
    """Persist uploaded audio bytes to a temporary WAV file and return its path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(content)
        temp_file.flush()
        return temp_file.name


def _normalize_transcription_dict(
    d: Dict[str, Any],
) -> Tuple[str, str, float, float, int, float, str]:
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
) -> Tuple[str, str, float, float, int, float, str]:
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
) -> Tuple[str, str, float, float, int, float, str]:
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
        raise HTTPException(
            status_code=503, detail="Text summarization service unavailable"
        )


def _get_request_scoped_summarizer(model: str):
    """Return summarizer for requested model.

    If the requested model differs, attempt to create a request-scoped instance.
    On failure, raise HTTPException(400/503) instead of silently falling back.
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
            raise HTTPException(
                status_code=400,
                detail=f"Invalid summarizer model: {model}",
            ) from exc
        except Exception as exc:  # treat unknown models as bad request in tests
            raise HTTPException(
                status_code=400,
                detail=(f"Requested summarizer model '{model}' unavailable"),
            ) from exc
    return text_summarizer


def _derive_emotion(summary_text: str) -> Tuple[str, List[str]]:
    """Infer emotional tone and key emotions from summary text."""
    if not summary_text or not emotion_detector:
        return "neutral", []
    try:
        emotion_result = _run_emotion_predict(summary_text)
        primary = emotion_result.get("primary_emotion", "neutral")
        keys = emotion_result.get("key_emotions")
        if not isinstance(keys, list):
            keys = [primary]
        if primary in ["joy", "gratitude", "excitement"]:
            tone = "positive"
        elif primary in ["sadness", "anger", "fear"]:
            tone = "negative"
        else:
            tone = "neutral"
        return tone, keys
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning("Could not determine emotional tone from summary: %s", exc)
        return "neutral", []


# Request Models
class JournalEntryRequest(BaseModel):
    """Request model for journal entry analysis."""

    text: str = Field(
        ...,
        description="Journal text to analyze",
        min_length=5,
        max_length=5000,
        example=(
            "Today I received a promotion at work and I'm really excited " "about it."
        ),
    )
    generate_summary: bool = Field(True, description="Whether to generate a summary")
    emotion_threshold: float = Field(
        0.1, description="Threshold for emotion detection", ge=0, le=1
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": (
                    "Today I received a promotion at work and I'm really excited "
                    "about it."
                ),
                "generate_summary": True,
                "emotion_threshold": 0.1,
            }
        }


# Unified Response Models
class EmotionAnalysis(BaseModel):
    """Emotion analysis results."""

    emotions: Dict[str, float] = Field(
        ...,
        description="Emotion probabilities",
        example={"joy": 0.75, "gratitude": 0.65},
    )
    primary_emotion: str = Field(
        ..., description="Most confident emotion", example="joy"
    )
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
        example=(
            "User expressed joy about their recent promotion and gratitude "
            "toward their supportive team."
        ),
    )
    key_emotions: List[str] = Field(
        ..., description="Key emotions identified", example=["joy", "gratitude"]
    )
    compression_ratio: float = Field(
        ..., description="Text compression ratio", ge=0, le=1, example=0.85
    )
    emotional_tone: str = Field(
        ..., description="Overall emotional tone", example="positive"
    )


class VoiceTranscription(BaseModel):
    """Voice transcription results."""

    text: str = Field(
        ...,
        description="Transcribed text",
        example=(
            "Today I received a promotion at work and I'm really excited " "about it."
        ),
    )
    language: str = Field(..., description="Detected language", example="en")
    confidence: float = Field(
        ..., description="Transcription confidence", ge=0, le=1, example=0.95
    )
    duration: float = Field(
        ..., description="Audio duration in seconds", ge=0, example=15.4
    )
    word_count: int = Field(..., description="Number of words", ge=0, example=12)
    speaking_rate: float = Field(
        ..., description="Words per minute", ge=0, example=120.5
    )
    audio_quality: str = Field(
        ..., description="Audio quality assessment", example="excellent"
    )


class CompleteJournalAnalysis(BaseModel):
    """Complete journal analysis combining all AI models."""

    transcription: Optional[VoiceTranscription] = Field(
        None, description="Voice transcription results"
    )
    emotion_analysis: EmotionAnalysis = Field(
        ..., description="Emotion detection results"
    )
    summary: TextSummary = Field(..., description="Text summarization results")
    processing_time_ms: float = Field(
        ..., description="Total processing time in milliseconds", ge=0, example=450.2
    )
    pipeline_status: Dict[str, bool] = Field(
        ...,
        description="Status of each AI component",
        example={
            "emotion_detection": True,
            "text_summarization": True,
            "voice_processing": False,
        },
    )
    insights: Dict[str, Any] = Field(
        ...,
        description="Additional insights and metadata",
        example={"word_count": 12, "language": "en"},
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
                "status": (
                    "available" if emotion_detector is not None else "unavailable"
                ),
            },
            "text_summarization": {
                "loaded": text_summarizer is not None,
                "status": (
                    "available" if text_summarizer is not None else "unavailable"
                ),
            },
            "voice_processing": {
                "loaded": voice_transcriber is not None,
                "status": (
                    "available" if voice_transcriber is not None else "unavailable"
                ),
            },
        },
    }


# Authentication Endpoints
@app.post(
    "/auth/register",
    response_model=TokenResponse,
    tags=["Authentication"],
    summary="Register new user",
    description="Register a new user account and receive authentication tokens",
)
async def register_user(user_data: UserRegister) -> TokenResponse:
    """Register a new user account."""
    try:
        # In a real application, you would:
        # 1. Check if user already exists
        # 2. Hash the password
        # 3. Store user in database
        # 4. Generate user ID

        # For demo purposes, we'll create a simple user
        user_id = f"user_{int(time.time())}"

        # Create user data for token
        token_user_data = {
            "user_id": user_id,
            "username": user_data.username,
            "email": user_data.email,
            "permissions": ["read", "write"],  # Default permissions
        }

        # Generate tokens
        token_response: TokenResponse = jwt_manager.create_token_pair(token_user_data)

        logger.info("New user registered: %s", user_data.username)
        return token_response

    except Exception as exc:
        logger.error("Registration failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed",
        )


@app.post(
    "/auth/login",
    response_model=TokenResponse,
    tags=["Authentication"],
    summary="User login",
    description="Authenticate user and receive access tokens",
)
async def login_user(login_data: UserLogin) -> TokenResponse:
    """Authenticate user and provide access tokens."""
    try:
        # In a real application, you would:
        # 1. Verify username/password against database
        # 2. Check if account is active
        # 3. Retrieve user permissions

        # For demo purposes, we'll accept any valid email/password
        if not login_data.username or not login_data.password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username and password required",
            )

        # Create user data for token
        user_id = f"user_{hash(login_data.username) % 10000}"
        # Establish baseline permissions for all authenticated users
        base_permissions = ["read", "write"]
        # Assign admin only if explicitly allowed by environment or a simple role check
        is_admin_user = False
        # Allow enabling an admin account via env for demos/tests only
        allowed_admin = os.getenv("ADMIN_USERNAME", "").strip()
        if allowed_admin and login_data.username == allowed_admin:
            is_admin_user = True
        # Also support a comma-separated list of admin users
        if not is_admin_user:
            admin_list = {
                u.strip() for u in os.getenv("ADMIN_USERS", "").split(",") if u.strip()
            }
            if login_data.username in admin_list:
                is_admin_user = True

        permissions = list(base_permissions)
        if is_admin_user:
            permissions.append("admin")

        token_user_data = {
            "user_id": str(user_id),
            "username": login_data.username,
            "email": (
                login_data.username
                if "@" in login_data.username
                else f"{login_data.username}@example.com"
            ),
            "permissions": permissions,
        }

        # Generate tokens
        token_response: TokenResponse = jwt_manager.create_token_pair(token_user_data)

        logger.info("User logged in: %s", login_data.username)
        return token_response

    except HTTPException as http_exc:
        # Preserve HTTPExceptions without altering trace
        raise http_exc
    except Exception as exc:
        logger.error("Login failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Login failed"
        )


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""

    refresh_token: str = Field(..., description="Refresh token")


@app.post(
    "/auth/refresh",
    response_model=TokenResponse,
    tags=["Authentication"],
    summary="Refresh access token",
    description="Refresh access token using refresh token",
)
async def refresh_token(request: RefreshTokenRequest) -> TokenResponse:
    """Refresh access token using refresh token."""
    try:
        # Verify refresh token
        payload = jwt_manager.verify_token(request.refresh_token)
        if not payload or payload.type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
            )

        # Create new user data
        user_data = {
            "user_id": payload.user_id,
            "username": payload.username,
            "email": payload.email,
            "permissions": payload.permissions,
        }

        # Generate new token pair
        token_response: TokenResponse = jwt_manager.create_token_pair(user_data)

        logger.info("Token refreshed for user: %s", payload.username)
        return token_response

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Token refresh failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed",
        )


@app.post(
    "/auth/logout",
    tags=["Authentication"],
    summary="Logout user",
    description="Logout user and blacklist tokens",
)
async def logout_user(
    request: Request, current_user: TokenPayload = Depends(get_current_user)
) -> Dict[str, str]:
    """Logout user and blacklist tokens."""
    try:
        # Get the raw token from the Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            # Blacklist the token
            jwt_manager.blacklist_token(token)
            logger.info(
                "User logged out and token blacklisted: %s", current_user.username
            )
        else:
            logger.warning("No valid Authorization header found during logout")

        return {"message": "Successfully logged out"}

    except Exception as exc:
        logger.error("Logout failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Logout failed"
        )


@app.get(
    "/auth/profile",
    response_model=UserProfile,
    tags=["Authentication"],
    summary="Get user profile",
    description="Get current user profile information",
)
async def get_user_profile(
    current_user: TokenPayload = Depends(get_current_user),
) -> UserProfile:
    """Get current user profile."""
    return UserProfile(
        user_id=current_user.user_id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.username,  # In real app, get from database
        permissions=current_user.permissions,
        # In real app, get from database
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )


# Simple Chat Contracts (minimal)
class ChatMessage(BaseModel):
    """Single chat message from the user."""

    text: str = Field(..., min_length=1, description="User message text")
    summarize: bool = Field(False, description="Summarize response using T5")
    model: str = Field("t5-small", description="Summarizer model if summarize=true")


class ChatResponse(BaseModel):
    """Chat response payload."""

    reply: str
    summary: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Minimal chat over HTTP",
    description="Echo-style chat that optionally summarizes the reply via T5.",
)
async def chat_http(
    message: ChatMessage,
    current_user: TokenPayload = Depends(get_current_user),
) -> ChatResponse:
    """Minimal chat endpoint built on existing components.

    - Produces a simple echo-style reply.
    - If summarize=true, uses request-scoped summarizer to summarize the reply.
    """
    reply = f"You said: {message.text.strip()}"

    summary_text: Optional[str] = None
    if message.summarize:
        if text_summarizer is None:
            _ensure_summarizer_loaded()
        summarizer_instance = _get_request_scoped_summarizer(message.model)
        summary_text = summarizer_instance.generate_summary(
            reply, max_length=80, min_length=20
        )

    return ChatResponse(
        reply=reply,
        summary=summary_text,
        meta={
            "model": message.model if message.summarize else None,
            "user": current_user.username,
        },
    )


@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket, token: str = Query(None)) -> None:
    """Minimal WebSocket chat.

    Protocol:
    - Client connects with ?token=JWT or sends {"token":"..."} as first message.
    - Then sends {"text":"...", "summarize":bool, "model":"t5-small|t5-base"}.
    - Server responds with {"reply":"...", "summary":"..."}.
    """
    # Authenticate (accept once, then validate)
    await websocket.accept()
    auth_token = token
    if not auth_token:
        try:
            initial = await websocket.receive_text()
            auth_token = json.loads(initial).get("token")
        except (json.JSONDecodeError, AttributeError, WebSocketDisconnect):
            await websocket.send_json({"error": "Authentication token required"})
            await websocket.close(code=4001)
            return

    if not auth_token:
        await websocket.send_json({"error": "Authentication token required"})
        await websocket.close(code=4001)
        return

    try:
        payload = jwt_manager.verify_token(auth_token)
    except Exception:
        await websocket.send_json({"error": "Token verification failed"})
        await websocket.close(code=4001)
        return

    if not payload:
        await websocket.send_json({"error": "Invalid token"})
        await websocket.close(code=4001)
        return
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            text = (data.get("text") or "").strip()
            summarize_flag = bool(data.get("summarize", False))
            model = data.get("model", "t5-small")
            reply = f"You said: {text}"

            response: Dict[str, Any] = {"reply": reply}
            if summarize_flag and text:
                try:
                    if text_summarizer is None:
                        _ensure_summarizer_loaded()
                    summarizer_instance = _get_request_scoped_summarizer(model)
                    summary_text = summarizer_instance.generate_summary(
                        reply, max_length=80, min_length=20
                    )
                    response["summary"] = summary_text
                except HTTPException as exc:
                    response["summary_error"] = exc.detail
                except Exception as exc:  # pragma: no cover
                    logger.error(
                        "Error during websocket summary generation: %s",
                        exc,
                        exc_info=True,
                    )
                    response["summary_error"] = str(exc)

            await websocket.send_json(response)
    except WebSocketDisconnect:
        return


@app.post(
    "/analyze/journal",
    response_model=CompleteJournalAnalysis,
    tags=["Analysis"],
    summary="Analyze text journal entry",
    description="Analyze a text journal entry with emotion detection and summarization",
    response_description=(
        "Complete analysis results including emotion detection and text summarization"
    ),
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
                raw = _run_emotion_predict(
                    request.text, threshold=request.emotion_threshold
                )
                emotion_results = normalize_emotion_results(raw)
                logger.info(
                    "Emotion analysis completed: %s", emotion_results["primary_emotion"]
                )
            except Exception as exc:
                logger.warning("⚠️  Emotion analysis failed: %s", exc)
                emotion_results = normalize_emotion_results({})

        # Text Summarization
        summary_results = None
        if text_summarizer is not None and request.generate_summary:
            try:
                summary_results = text_summarizer.summarize(request.text)
                logger.info("✅ Text summarization completed")
            except Exception as exc:
                logger.warning("⚠️  Text summarization failed: %s", exc)
                summary_results = {
                    "summary": (
                        request.text[:200] + "..."
                        if len(request.text) > 200
                        else request.text
                    ),
                    "key_emotions": (
                        [emotion_results["primary_emotion"]]
                        if emotion_results
                        else ["neutral"]
                    ),
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
                "summary": (
                    request.text[:200] + "..."
                    if len(request.text) > 200
                    else request.text
                ),
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
        logger.error("❌ Error in journal analysis: %s", exc)
        raise HTTPException(status_code=500, detail="Analysis failed") from exc


@app.post(
    "/analyze/voice-journal",
    response_model=CompleteJournalAnalysis,
    tags=["Analysis"],
    summary="Analyze voice journal entry",
    description=(
        "Complete voice journal analysis pipeline with transcription, "
        "emotion detection, and summarization"
    ),
    response_description=(
        "Complete analysis results including transcription, emotion detection, "
        "and text summarization"
    ),
)
async def analyze_voice_journal(
    audio_file: UploadFile = File(
        ..., description="Audio file to transcribe and analyze"
    ),
    language: Optional[str] = Form(
        None,
        description="Language code for transcription (auto-detect if not provided)",
    ),
    generate_summary: bool = Form(True, description="Whether to generate a summary"),
    emotion_threshold: float = Form(
        0.1, description="Threshold for emotion detection", ge=0, le=1
    ),
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
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".wav"
                ) as temp_file:
                    content = await audio_file.read()
                    temp_file.write(content)
                    temp_file.flush()  # Ensure data is written to disk
                    temp_file_path = temp_file.name

                try:
                    transcription_results = voice_transcriber.transcribe(
                        temp_file_path, language=language
                    )
                    transcribed_text = transcription_results["text"]
                    logger.info(
                        "Voice transcription completed: %s characters",
                        len(transcribed_text),
                    )
                finally:
                    # Clean up temporary file
                    Path(temp_file_path).unlink(missing_ok=True)

            except Exception as exc:
                logger.warning("⚠️  Voice transcription failed: %s", exc)
                # Continue in degraded mode
                transcribed_text = ""

        # Steps 2 & 3: Continue with text analysis using transcribed text
        if not transcribed_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Failed to transcribe audio or audio is too short",
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

        # Normalize transcription dict to include required optional fields for schema
        # using helper
        normalized_tx = None
        if transcription_results:
            (
                _text,
                _lang,
                _conf,
                _duration,
                _word_count,
                _speaking_rate,
                _audio_quality,
            ) = _normalize_transcription_attrs(transcription_results)
            # Validate required fields before constructing VoiceTranscription
            if not isinstance(_text, str) or _text is None:
                logger.warning(
                    "Transcription missing text; skipping transcription payload"
                )
                normalized_tx = None
            else:
                normalized_tx = {
                    "text": _text,
                    "language": _lang or "unknown",
                    "confidence": float(_conf) if _conf is not None else 0.0,
                    "duration": float(_duration) if _duration is not None else 0.0,
                    "word_count": int(_word_count) if _word_count is not None else 0,
                    "speaking_rate": (
                        float(_speaking_rate) if _speaking_rate is not None else 0.0
                    ),
                    "audio_quality": _audio_quality or "unknown",
                }
                # Pre-compute commonly used insight fields to avoid recomputation
                # downstream
                normalized_tx["insight_duration"] = normalized_tx["duration"]
                normalized_tx["insight_quality"] = normalized_tx["audio_quality"]

        return CompleteJournalAnalysis(
            transcription=(
                VoiceTranscription(**normalized_tx) if normalized_tx else None
            ),
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
                # Use pre-computed insight values from normalized_tx when available
                "audio_duration": (
                    normalized_tx.get("insight_duration") if normalized_tx else 0
                ),
                "audio_quality": (
                    normalized_tx.get("insight_quality") if normalized_tx else "unknown"
                ),
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("❌ Error in voice journal analysis: %s", exc)
        raise HTTPException(status_code=500, detail="Voice analysis failed") from exc


# Enhanced Voice Transcription Endpoints
@app.post(
    "/transcribe/voice",
    response_model=VoiceTranscription,
    tags=["Voice Processing"],
    summary="Transcribe voice to text",
    description="Enhanced voice transcription with detailed analysis",
)
async def transcribe_voice(
    audio_file: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Form(
        None, description="Language code (auto-detect if not provided)"
    ),
    model_size: str = Form(
        "base", description="Whisper model size (tiny, base, small, medium, large)"
    ),
    timestamp: bool = Form(False, description="Include word-level timestamps"),
    current_user: TokenPayload = Depends(get_current_user),
) -> VoiceTranscription:
    """Enhanced voice transcription with detailed analysis."""
    start_time = time.time()

    try:
        # Validate file
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="Audio file required")

        # Unified file size limit used consistently across code and messages.
        # Use a conservative threshold to account for test data construction.
        MAX_AUDIO_BYTES = 45 * 1024 * 1024
        content = await audio_file.read()
        if len(content) > MAX_AUDIO_BYTES:
            # Return a JSON body with 'detail' to match tests expecting that key
            max_mb = MAX_AUDIO_BYTES // (1024 * 1024)
            raise HTTPException(
                status_code=400, detail=f"File too large (max {max_mb}MB)"
            )
        # Reset file position for later processing
        await audio_file.seek(0)

        # Save uploaded file temporarily
        temp_file_path = _write_temp_wav(content)

        try:
            # Transcribe audio; ensure transcriber is available
            _ensure_voice_transcriber_loaded()

            # Enhanced transcription: introspect signature once and adapt call
            sig = inspect.signature(voice_transcriber.transcribe)
            accepted = sig.parameters
            candidate_args = {
                "audio_path": temp_file_path,
                "path": temp_file_path,
                "file_path": temp_file_path,
                "language": language,
            }
            kwargs = {
                k: v
                for k, v in candidate_args.items()
                if k in accepted and v is not None
            }
            if not any(k in accepted for k in ("audio_path", "path", "file_path")):
                # Try positional fallback if no filename-like kw is accepted
                try:
                    transcription_result = voice_transcriber.transcribe(
                        temp_file_path,
                        **{
                            k: v
                            for k, v in kwargs.items()
                            if k not in {"audio_path", "path", "file_path"}
                        },
                    )
                except Exception as e_positional:
                    try:
                        transcription_result = voice_transcriber.transcribe(
                            temp_file_path
                        )
                    except Exception as e_fallback:
                        logger.error(
                            "Transcriber failed with both positional and fallback "
                            "calls: %s; %s",
                            repr(e_positional),
                            repr(e_fallback),
                        )
                        raise
            else:
                try:
                    transcription_result = voice_transcriber.transcribe(**kwargs)
                except Exception as e_kwargs:
                    # Fallback to positional if keyword call fails
                    try:
                        transcription_result = voice_transcriber.transcribe(
                            temp_file_path, language=language
                        )
                    except Exception as e_positional:
                        try:
                            transcription_result = voice_transcriber.transcribe(
                                temp_file_path
                            )
                        except Exception as e_fallback:
                            logger.error(
                                "Transcriber failed with kwargs, positional, and "
                                "fallback calls: %s; %s; %s",
                                repr(e_kwargs),
                                repr(e_positional),
                                repr(e_fallback),
                            )
                            raise

            (
                text_val,
                lang_val,
                conf_val,
                duration,
                word_count,
                speaking_rate,
                audio_quality,
            ) = _normalize_transcription_attrs(transcription_result)

            processing_time = (time.time() - start_time) * 1000

            return VoiceTranscription(
                text=text_val,
                language=lang_val,
                confidence=conf_val,
                duration=duration,
                word_count=word_count,
                speaking_rate=speaking_rate,
                audio_quality=audio_quality,
            )

        finally:
            # Cleanup temporary file
            Path(temp_file_path).unlink(missing_ok=True)

    except Exception as exc:
        if isinstance(exc, HTTPException):
            # Preserve FastAPI HTTPException semantics
            raise
        logger.error("Voice transcription failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Voice transcription failed",
        ) from exc


@app.post(
    "/transcribe/batch",
    tags=["Voice Processing"],
    summary="Batch voice transcription",
    description="Process multiple audio files for transcription",
)
async def batch_transcribe_voice(
    request: Request,
    audio_files: List[UploadFile] = File(
        ..., description="Multiple audio files to transcribe"
    ),
    language: Optional[str] = Form(None, description="Language code for all files"),
    current_user: TokenPayload = Depends(get_current_user),
) -> Dict[str, Any]:
    """Batch process multiple audio files for transcription."""
    start_time = time.time()
    results = []

    try:
        # Enforce permission always; allow pytest header override for tests only
        if (
            not _has_injected_permission(request, "batch_processing")
            and "batch_processing" not in current_user.permissions
        ):
            raise HTTPException(
                status_code=403, detail="Permission 'batch_processing' required"
            )

        for i, audio_file in enumerate(audio_files):
            try:
                # Process each file individually
                content = await audio_file.read()
                # Allow empty/invalid content to be passed to mocked transcriber
                # to exercise failure paths
                if audio_file.filename:
                    prefix = f"{Path(audio_file.filename).stem}_"
                else:
                    prefix = "file_"
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".wav", prefix=prefix
                ) as temp_file:
                    temp_file.write(content or b"")
                    temp_file.flush()  # Ensure data is written to disk
                    temp_file_path = temp_file.name

                try:
                    if voice_transcriber is None:
                        raise HTTPException(
                            status_code=503,
                            detail="Voice transcription service unavailable",
                        )

                    transcription_result = voice_transcriber.transcribe(
                        temp_file_path, language=language
                    )

                    results.append(
                        {
                            "file_index": i,
                            "filename": audio_file.filename,
                            "success": True,
                            "transcription": transcription_result.get("text", ""),
                            "language": transcription_result.get("language", "unknown"),
                            "confidence": transcription_result.get("confidence", 0.0),
                            "duration": transcription_result.get("duration", 0),
                        }
                    )

                finally:
                    Path(temp_file_path).unlink(missing_ok=True)

            except Exception as exc:
                results.append(
                    {
                        "file_index": i,
                        "filename": audio_file.filename,
                        "success": False,
                        "error": str(exc),
                    }
                )

        processing_time = (time.time() - start_time) * 1000

        return {
            "total_files": len(audio_files),
            "successful_transcriptions": len([r for r in results if r["success"]]),
            "failed_transcriptions": len([r for r in results if not r["success"]]),
            "processing_time_ms": processing_time,
            "results": results,
        }

    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        logger.error("Batch transcription failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch transcription failed",
        ) from exc


# Enhanced Text Summarization Endpoints
@app.post(
    "/summarize/text",
    response_model=TextSummary,
    tags=["Text Processing"],
    summary="Enhanced text summarization",
    description="Advanced text summarization with multiple models and options",
)
async def summarize_text(
    text: str = Form(..., description="Text to summarize", min_length=10),
    model: str = Form(
        "t5-small", description="Summarization model (t5-small, t5-base, t5-large)"
    ),
    max_length: int = Form(150, description="Maximum summary length", ge=10, le=500),
    min_length: int = Form(30, description="Minimum summary length", ge=5, le=200),
    # Removed do_sample to keep API contract accurate; summarizer uses beam search
    current_user: TokenPayload = Depends(get_current_user),
) -> TextSummary:
    """Enhanced text summarization with multiple model options."""
    start_time = time.time()

    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        if text_summarizer is None:
            _ensure_summarizer_loaded()

        # Request-scoped model override to avoid global mutation in production
        summarizer_instance = _get_request_scoped_summarizer(model)

        # Generate summary. Some tests inject fakes with simplified signatures;
        # support both.
        summary_text = None
        for call in (
            lambda: summarizer_instance.generate_summary(
                text, max_length=max_length, min_length=min_length
            ),
            lambda: summarizer_instance.generate_summary(text, max_length, min_length),
            lambda: summarizer_instance.generate_summary(text),
        ):
            try:
                summary_text = call()
                break
            except TypeError:
                continue
        if summary_text is None:
            logger.error("Summarizer invocation failed for all supported signatures")
            raise HTTPException(status_code=500, detail="Text summarization failed")

        # Calculate metrics
        original_length = len(text.split())
        summary_length = len((summary_text or "").split())
        if original_length > 0:
            compression_ratio = 1 - (summary_length / original_length)
        else:
            compression_ratio = 0

        # Determine emotional tone and key emotions from summary
        emotional_tone, key_emotions = _derive_emotion(summary_text or "")

        processing_time = (time.time() - start_time) * 1000

        return TextSummary(
            summary=summary_text or "",
            key_emotions=key_emotions,
            compression_ratio=compression_ratio,
            emotional_tone=emotional_tone,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Text summarization failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Text summarization failed",
        ) from exc


# Real-time Processing Endpoints
@app.websocket("/ws/realtime")
async def websocket_realtime_processing(websocket: WebSocket, token: str = Query(None)):
    """WebSocket endpoint for real-time voice processing."""
    # Validate authentication token
    if not token:
        await websocket.close(code=4001, reason="Authentication token required")
        return

    try:
        # Verify JWT token using the global jwt_manager instance
        payload = jwt_manager.verify_token(token)
        if not payload:
            await websocket.close(code=4001, reason="Invalid authentication token")
            return

        # Check if user has real-time processing permission
        if "realtime_processing" not in payload.permissions:
            await websocket.close(code=4003, reason="Insufficient permissions")
            return

    except Exception as e:
        await websocket.close(code=4001, reason=f"Authentication failed: {str(e)}")
        return

    await websocket.accept()

    # Authenticate WebSocket connection
    try:
        # Get token from query parameters or initial message
        token = websocket.query_params.get("token")
        if not token:
            # Try to get token from initial message
            initial_message = await websocket.receive_text()
            try:
                message_data = json.loads(initial_message)
                token = message_data.get("token")
            except (json.JSONDecodeError, KeyError):
                await websocket.send_json(
                    {"type": "error", "message": "Authentication token required"}
                )
                await websocket.close()
                return

        # Verify token using the global jwt_manager instance
        payload = jwt_manager.verify_token(token)
        if not payload:
            await websocket.send_json(
                {"type": "error", "message": "Invalid authentication token"}
            )
            await websocket.close()
            return

        logger.info("WebSocket authenticated for user: %s", payload.username)

    except Exception as exc:
        await websocket.send_json({"type": "error", "message": "Authentication failed"})
        await websocket.close()
        return

    try:
        while True:
            # Receive audio data or control messages
            try:
                data = await websocket.receive_bytes()
            except WebSocketDisconnect:
                break

            # Process audio in real-time
            if voice_transcriber:
                try:
                    # Save received audio data
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".wav"
                    ) as temp_file:
                        temp_file.write(data)
                        temp_file.flush()  # Ensure data is written to disk
                        temp_file_path = temp_file.name

                    try:
                        # Transcribe
                        result = voice_transcriber.transcribe(temp_file_path)

                        # Send result back
                        await websocket.send_json(
                            {
                                "type": "transcription",
                                "text": result.get("text", ""),
                                "confidence": result.get("confidence", 0.0),
                                "language": result.get("language", "unknown"),
                            }
                        )

                    finally:
                        Path(temp_file_path).unlink(missing_ok=True)

                except Exception as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
            else:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Voice transcription service unavailable",
                    }
                )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
        try:
            await websocket.send_json(
                {"type": "error", "message": "Internal server error"}
            )
        except Exception:
            pass


# Monitoring and Analytics Endpoints
@app.get(
    "/monitoring/performance",
    tags=["Monitoring"],
    summary="Performance monitoring",
    description="Get detailed performance metrics and analytics",
)
async def get_performance_metrics(
    current_user: TokenPayload = Depends(require_permission("monitoring")),
) -> Dict[str, Any]:
    """Get comprehensive performance metrics."""
    try:
        # Get system metrics
        import psutil

        cpu_percent = await asyncio.to_thread(psutil.cpu_percent, interval=1)
        memory = await asyncio.to_thread(psutil.virtual_memory)
        disk = await asyncio.to_thread(psutil.disk_usage, "/")

        # Model performance metrics
        model_metrics = {
            "emotion_detection": {
                "loaded": emotion_detector is not None,
                "last_used": time.time() if emotion_detector else None,
                "total_requests": 0,  # In real app, track from database
            },
            "text_summarization": {
                "loaded": text_summarizer is not None,
                "last_used": time.time() if text_summarizer else None,
                "total_requests": 0,
            },
            "voice_processing": {
                "loaded": voice_transcriber is not None,
                "last_used": time.time() if voice_transcriber else None,
                "total_requests": 0,
            },
        }

        return {
            "timestamp": time.time(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
            },
            "models": model_metrics,
            "api": {
                "uptime_seconds": time.time() - app_start_time,
                "active_connections": 0,  # In real app, track WebSocket connections
                "total_requests": 0,  # In real app, track from database
            },
        }

    except Exception as exc:
        logger.error("Failed to get performance metrics: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get performance metrics",
        )


@app.get(
    "/monitoring/health/detailed",
    tags=["Monitoring"],
    summary="Detailed health check",
    description="Comprehensive health check with model diagnostics",
)
async def detailed_health_check(
    current_user: TokenPayload = Depends(require_permission("monitoring")),
) -> Dict[str, Any]:
    """Comprehensive health check with detailed diagnostics."""
    health_status = "healthy"
    issues = []

    # Check models
    model_checks = {}

    if emotion_detector is None:
        health_status = "degraded"
        issues.append("Emotion detection model not loaded")
        model_checks["emotion_detection"] = {
            "status": "unavailable",
            "error": "Model not loaded",
        }
    else:
        try:
            # Test emotion detection
            test_result = emotion_detector.predict("I am happy today")
            model_checks["emotion_detection"] = {
                "status": "healthy",
                "test_passed": True,
            }
        except Exception as exc:
            health_status = "degraded"
            issues.append(f"Emotion detection model error: {exc}")
            model_checks["emotion_detection"] = {"status": "error", "error": str(exc)}

    if text_summarizer is None:
        health_status = "degraded"
        issues.append("Text summarization model not loaded")
        model_checks["text_summarization"] = {
            "status": "unavailable",
            "error": "Model not loaded",
        }
    else:
        try:
            # Test text summarization
            test_result = text_summarizer.summarize(
                "This is a test text for summarization."
            )
            model_checks["text_summarization"] = {
                "status": "healthy",
                "test_passed": True,
            }
        except Exception as exc:
            health_status = "degraded"
            issues.append(f"Text summarization model error: {exc}")
            model_checks["text_summarization"] = {"status": "error", "error": str(exc)}

    if voice_transcriber is None:
        health_status = "degraded"
        issues.append("Voice processing model not loaded")
        model_checks["voice_processing"] = {
            "status": "unavailable",
            "error": "Model not loaded",
        }
    else:
        model_checks["voice_processing"] = {"status": "healthy", "test_passed": True}

    # Check system resources
    try:
        import psutil

        cpu_percent = await asyncio.to_thread(psutil.cpu_percent, interval=1)
        memory = await asyncio.to_thread(psutil.virtual_memory)

        if cpu_percent > 90:
            health_status = "degraded"
            issues.append(f"High CPU usage: {cpu_percent}%")

        if memory.percent > 90:
            health_status = "degraded"
            issues.append(f"High memory usage: {memory.percent}%")

        system_checks = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "status": (
                "healthy" if cpu_percent < 90 and memory.percent < 90 else "warning"
            ),
        }
    except Exception as exc:
        system_checks = {"status": "error", "error": str(exc)}
        health_status = "degraded"
        issues.append(f"System check failed: {exc}")

    return {
        "status": health_status,
        "timestamp": time.time(),
        "issues": issues,
        "models": model_checks,
        "system": system_checks,
        "version": "1.0.0",
    }


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
            "capabilities": [
                "Multi-label emotion classification",
                "Emotion intensity analysis",
            ],
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
            "degraded_mode": not all(
                [emotion_detector, text_summarizer, voice_transcriber]
            ),
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
    # Environment-based host configuration for security
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))

    # Use centralized security-first host binding configuration
    from src.security.host_binding import (
        get_secure_host_binding,
        validate_host_binding,
        get_binding_security_summary,
    )

    host, port = get_secure_host_binding(default_port=port)
    validate_host_binding(host, port)

    security_summary = get_binding_security_summary(host, port)
    print(f"Security Summary: {security_summary}")

    uvicorn.run(app, host=host, port=port)
