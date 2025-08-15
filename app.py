#!/usr/bin/env python3
"""
SAMO Emotion Detection API - Production Entry Point
"""
import os
import sys
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    # Import the unified API
    from src.unified_ai_api import app
    
    logger.info("✅ SAMO AI API loaded successfully")
    
    if __name__ == "__main__":
        import uvicorn
        
        # Get configuration from environment
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", 8000))
        
        logger.info(f"🚀 Starting SAMO AI API on {host}:{port}")
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
        
except ImportError as e:
    logger.error(f"❌ Failed to import SAMO AI API: {e}")
    
    # Fallback to simple health check API
    from flask import Flask, jsonify
    
    app = Flask(__name__)
    
    @app.route("/health")
    def health():
        return jsonify({
            "status": "healthy",
            "message": "SAMO API fallback mode",
            "models": {
                "emotion_detection": {"status": "unavailable"},
                "text_summarization": {"status": "unavailable"},
                "voice_processing": {"status": "unavailable"}
            }
        })
    
    @app.route("/")
    def root():
        return jsonify({
            "message": "SAMO AI API - Fallback Mode",
            "status": "degraded",
            "endpoints": {
                "health": "/health"
            }
        })
    
    if __name__ == "__main__":
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", 8000))
        
        logger.info(f"🚀 Starting SAMO AI API (fallback mode) on {host}:{port}")
        app.run(host=host, port=port, debug=False)