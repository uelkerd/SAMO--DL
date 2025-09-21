#!/usr/bin/env python3
"""Unified AI API for SAMO Deep Learning.

This module provides a unified FastAPI interface for all AI models
in the SAMO Deep Learning pipeline.
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
                    raw.get("emotional_intensity"),
                    "neutral",
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
            "emotional_intensity": str(
                getattr(raw, "emotional_intensity", "neutral"),
            ),
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
            from models.emotion_detection.labels import (
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


# Application startup time - will be initialized in main()
app_start_time = None

# JWT Authentication - will be initialized in main()
jwt_manager = None
security = None


# Enhanced WebSocket Connection Management
class WebSocketConnectionManager:
    """Enhanced WebSocket connection manager with pooling and heartbeat."""

    def __init__(self):
        self.active_connections: dict[str, set[WebSocket]] = defaultdict(set)
        self.connection_metadata: dict[WebSocket, dict[str, Any]] = {}
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
            "WebSocket connected for user %s. Total connections: %s",
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
        self,
        message: dict[str, Any],
        websocket: WebSocket,
    ):
        """Send message to specific WebSocket with error handling."""
        try:
            await websocket.send_json(message)
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["message_count"] += 1
        except Exception as e:
            logger.exception("Failed to send message to WebSocket: %s", e)
            await self.disconnect(websocket)

    async def broadcast_to_user(self, message: dict[str, Any], user_id: str):
        """Broadcast message to all connections of a specific user."""
        disconnected = set()
        for websocket in self.active_connections[user_id]:
            try:
                await websocket.send_json(message)
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]["message_count"] += 1
            except Exception as e:
                logger.exception("Failed to broadcast to WebSocket: %s", e)
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

    def get_connection_stats(self) -> dict[str, Any]:
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


# Global WebSocket manager - will be initialized in main()
websocket_manager = None

# Authentication models - will be defined in main() when FastAPI is imported
UserLogin = None
UserRegister = None
UserProfile = None


# Authentication dependency - will be initialized in main()
get_current_user = None

# Permission dependency - will be initialized in main()
require_permission = None


# Lifespan function - will be initialized in main()
lifespan = None


# FastAPI app - will be initialized in main()
app = None


# Middleware and endpoints will be set up in main()


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


# Exception handlers will be set up in main()


# ===== Helpers to reduce endpoint complexity =====
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

    If the requested model differs, attempt to create a request-scoped instance.
    On failure, raise ValueError/RuntimeError instead of HTTPException.
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
            raise RuntimeError(f"Requested summarizer model '{model}' unavailable") from exc
    return text_summarizer


def _derive_emotion(summary_text: str) -> tuple[str, list[str]]:
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
