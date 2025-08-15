#!/usr/bin/env python3
"""SAMO Emotion Detection API - Production Entry Point."""
import os
import sys
import logging
from pathlib import Path

# Add project root to Python path for src package discovery
# Note: src/ has __init__.py, so this should work with PEP 420 implicit namespace packages
sys.path.insert(0, str(Path(__file__).parent))

# Configure module-level logger without altering global config
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# PRODUCTION: Import the unified API for Gunicorn
try:
    from src.unified_ai_api import app
    logger.info("✅ SAMO AI API loaded successfully")

except ModuleNotFoundError as e:
    logger.error("❌ Failed to import SAMO AI API: %s", e)

    # PRODUCTION: Fallback to simple health check API (FastAPI for consistency)
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI(
        title="SAMO AI API - Fallback Mode",
        description="Fallback mode when main API is unavailable",
        version="1.0.0"
    )

    @app.get("/health")
    async def health() -> JSONResponse:
        """Health check endpoint for fallback mode.

        Returns:
            JSONResponse: Degraded health status with model availability information.
        """
        return JSONResponse({
            "status": "degraded",
            "message": "SAMO API in fallback mode - core services unavailable",
            "models": {
                "emotion_detection": {"status": "unavailable"},
                "text_summarization": {"status": "unavailable"},
                "voice_processing": {"status": "unavailable"}
            }
        }, status_code=503)

    @app.get("/")
    async def root() -> JSONResponse:
        """Root endpoint for fallback mode API.

        Returns:
            JSONResponse: Basic API information and available endpoints.
        """
        return JSONResponse({
            "message": "SAMO AI API - Fallback Mode",
            "status": "degraded",
            "endpoints": {
                "health": "/health"
            }
        })

    logger.info("✅ SAMO AI API fallback mode loaded")

# PRODUCTION: For development/testing only
if __name__ == "__main__":
    import uvicorn

    # Development-only logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Get configuration from environment
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))

    logger.info("🚀 Starting SAMO AI API (development mode) on %s:%s", host, port)
    logger.warning("⚠️  Using development server - use Gunicorn for production!")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
