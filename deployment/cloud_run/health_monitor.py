"""Cloud Run Health Monitor - Phase 3 Optimization
Provides comprehensive health checks, graceful shutdown, and monitoring
"""

import logging
import os
import signal
import sys
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HealthMetrics:
    """Health check metrics."""

    status: str
    response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    active_requests: int
    timestamp: datetime
    error_message: Optional[str] = None


class HealthMonitor:
    """Comprehensive health monitoring for Cloud Run."""

    # Class constants for configuration
    MODULES_TO_CHECK = [
        "src.models.emotion_detection.bert_classifier",
        "src.models.summarization.t5_summarizer",
        "src.models.voice_processing.whisper_transcriber",
    ]

    # Resource thresholds
    MEMORY_THRESHOLD_MB = 1500  # 1.5GB threshold
    CPU_THRESHOLD_PERCENT = 80  # 80% CPU threshold

    # Metrics retention
    MAX_METRICS_COUNT = 100

    def __init__(self):
        self.start_time = datetime.now()
        self.is_shutting_down = False
        self.active_requests = 0
        self.health_metrics: OrderedDict[str, HealthMetrics] = OrderedDict()
        self.shutdown_timeout = int(
            os.getenv("GRACEFUL_SHUTDOWN_TIMEOUT", "30")
        )
        self.lock = threading.Lock()

        # Cache for model imports to avoid re-importing on every health check
        self._model_import_cache = {}
        self._model_import_cache_timestamp = None
        self._model_cache_ttl = 300  # 5 minutes

        # Cache for FastAPI app to avoid repeated imports
        self._flask_app = None
        self._flask_app_import_error = None
        self._test_client_class = None

        # Register graceful shutdown handlers
        # Note: Signal handling should be initialized in the main thread
        # In multi-threaded environments or non-main threads, this may not work as expected
        try:
            signal.signal(signal.SIGTERM, self._graceful_shutdown)
            signal.signal(signal.SIGINT, self._graceful_shutdown)
            logger.info("Signal handlers registered successfully")
        except ValueError as e:
            # Fallback for non-main thread environments
            logger.warning(
                f"Could not register signal handlers (likely not in main thread): {e}"
            )
            logger.warning(
                "Graceful shutdown may not work in this environment"
            )

        logger.info(
            f"Health monitor initialized with {self.shutdown_timeout}s shutdown timeout",
        )

    def _graceful_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info(f"Received shutdown signal {signum}, starting graceful shutdown...")
        self.is_shutting_down = True

        # Wait for active requests to complete
        start_wait = time.time()
        while (
            self.active_requests > 0
            and (time.time() - start_wait) < self.shutdown_timeout
        ):
            logger.info(
                f"Waiting for {self.active_requests} active requests to complete...",
            )
            time.sleep(1)

        if self.active_requests > 0:
            logger.warning(
                f"Force shutdown after {self.shutdown_timeout}s timeout with "
                f"{self.active_requests} active requests",
            )
        else:
            logger.info("Graceful shutdown completed successfully")

        sys.exit(0)

    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system resource usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            # Use interval parameter for accurate CPU measurement
            # First call may return 0.0, so we use a small interval
            cpu_percent = process.cpu_percent(interval=0.1)

            return {
                "memory_usage_mb": memory_info.rss / 1024 / 1024,
                "cpu_usage_percent": cpu_percent,
                "memory_percent": process.memory_percent(),
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {
                "memory_usage_mb": 0.0,
                "cpu_usage_percent": 0.0,
                "memory_percent": 0.0,
                "uptime_seconds": 0.0,
            }

    def check_model_health(self) -> Dict[str, Any]:
        """Check if ML models are loaded and responding."""
        try:
            # Test model loading
            start_time = time.time()

            # Check if cache is still valid
            current_time = time.time()
            if (self._model_import_cache_timestamp is None or
                    current_time - self._model_import_cache_timestamp >
                    self._model_cache_ttl):
                # Cache expired or doesn't exist, refresh it
                self._refresh_model_import_cache()
                self._model_import_cache_timestamp = current_time

            # Use cached results
            if not self._model_import_cache:
                return {
                    "status": "unhealthy",
                    "error": "No model modules could be imported",
                    "response_time_ms": (time.time() - start_time) * 1000,
                }

            return {
                "status": "healthy",
                "response_time_ms": (time.time() - start_time) * 1000,
                "models_loaded": len(self._model_import_cache),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": f"Model health check failed: {e}",
                "response_time_ms": 0,
            }

    def _refresh_model_import_cache(self):
        """Refresh the model import cache."""
        import importlib

        self._model_import_cache = {}

        for module_name in self.MODULES_TO_CHECK:
            try:
                module = importlib.import_module(module_name)
                self._model_import_cache[module_name] = module
            except ImportError as e:
                logger.warning(f"Model module {module_name} not available: {e}")
                # Don't fail the entire cache refresh for one module

    def _get_test_client(self):
        """Get FastAPI test client with lazy loading and caching."""
        if self._flask_app_import_error is not None:
            # Previous import failed, don't try again
            return None

        if self._flask_app is None:
            try:
                from secure_api_server import app
                from fastapi.testclient import TestClient
                self._flask_app = app
                self._test_client_class = TestClient
            except ImportError as e:
                self._flask_app_import_error = e
                logger.warning(f"Could not import FastAPI app: {e}")
                return None

        return self._test_client_class(self._flask_app)

    def check_api_health(self, test_client=None) -> Dict[str, Any]:
        """Check API endpoint health.

        Args:
            test_client: Optional FastAPI test client to use for testing.
                        If None, will attempt to import and create one.
        """
        try:
            start_time = time.time()

            # Use injected test client or create one
            if test_client is None:
                test_client = self._get_test_client()
                if test_client is None:
                    return {
                        "status": "unhealthy",
                        "error": (
                            "Could not create FastAPI test client for health check"
                        ),
                        "response_time_ms": 0,
                    }

            # Test internal health endpoint
            response = test_client.get("/health")
            response_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "response_time_ms": response_time,
                    "status_code": response.status_code,
                }
            return {
                "status": "unhealthy",
                "error": f"Health endpoint returned {response.status_code}",
                "response_time_ms": response_time,
                "status_code": response.status_code,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": f"API health check failed: {e}",
                "response_time_ms": 0,
            }

    def get_comprehensive_health(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        with self.lock:
            # Capture current time once for consistency
            now = datetime.now()

            if self.is_shutting_down:
                return {
                    "status": "shutting_down",
                    "message": "Service is shutting down gracefully",
                    "active_requests": self.active_requests,
                    "timestamp": now.isoformat(),
                }

            # Get system metrics
            system_metrics = self.get_system_metrics()

            # Check model health
            model_health = self.check_model_health()

            # Check API health
            api_health = self.check_api_health()

            # Determine overall health
            overall_status = "healthy"
            if model_health["status"] != "healthy" or api_health["status"] != "healthy":
                overall_status = "unhealthy"

            # Check resource thresholds (only if not already unhealthy)
            if overall_status == "healthy":
                if system_metrics["memory_usage_mb"] > self.MEMORY_THRESHOLD_MB:
                    overall_status = "degraded"

                if system_metrics["cpu_usage_percent"] > self.CPU_THRESHOLD_PERCENT:
                    overall_status = "degraded"

            health_data = {
                "status": overall_status,
                "timestamp": now.isoformat(),
                "uptime_seconds": system_metrics["uptime_seconds"],
                "system": {
                    "memory_usage_mb": round(system_metrics["memory_usage_mb"], 2),
                    "cpu_usage_percent": round(system_metrics["cpu_usage_percent"], 2),
                    "memory_percent": round(system_metrics["memory_percent"], 2),
                },
                "models": model_health,
                "api": api_health,
                "requests": {
                    "active": self.active_requests,
                    "historical_metrics_count": len(self.health_metrics),
                },
            }

            # Store metrics for trend analysis
            # Use microseconds to prevent timestamp collisions
            timestamp_key = now.isoformat(timespec="microseconds")
            self.health_metrics[timestamp_key] = HealthMetrics(
                status=overall_status,
                response_time_ms=api_health.get("response_time_ms", 0),
                memory_usage_mb=system_metrics["memory_usage_mb"],
                cpu_usage_percent=system_metrics["cpu_usage_percent"],
                active_requests=self.active_requests,
                timestamp=now,
                error_message=model_health.get("error") or api_health.get("error"),
            )

            # Keep only last MAX_METRICS_COUNT metrics using OrderedDict's popitem
            if len(self.health_metrics) > self.MAX_METRICS_COUNT:
                self.health_metrics.popitem(last=False)

            return health_data

    def request_started(self):
        """Track request start."""
        with self.lock:
            self.active_requests += 1

    def request_completed(self):
        """Track request completion."""
        with self.lock:
            self.active_requests = max(0, self.active_requests - 1)


# Global health monitor instance
health_monitor = HealthMonitor()


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance."""
    return health_monitor
