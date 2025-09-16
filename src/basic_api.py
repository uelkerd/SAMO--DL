#!/usr/bin/env python3
"""Ultra-basic API for testing Cloud Run deployment."""
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SAMO Basic API", version="1.0.0")

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

@app.get("/")
async def root():
    return {"message": "SAMO API is running", "status": "healthy"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/analyze/emotion")
async def analyze_emotion(request: EmotionRequest):
    """Mock emotion analysis for testing."""
    logger.info("Mock emotion analysis", extra={"text_length": len(request.text)})
    
    return {
        "text": request.text,
        "emotions": {
            "excitement": 0.9,
            "joy": 0.8,
            "optimism": 0.7
        },
        "predicted_emotion": "excitement"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    # Default to localhost for development to avoid exposure
    host = os.environ.get("HOST", "127.0.0.1")
    if os.environ.get("PRODUCTION") == "true" or os.environ.get("CLOUD_RUN_SERVICE"):
        host = "0.0.0.0"  # Cloud Run and production environments
    logger.info(f"Starting basic server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
