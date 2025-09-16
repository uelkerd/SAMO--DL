#!/usr/bin/env python3
"""Ultra-basic API for testing Cloud Run deployment."""
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SAMO Basic API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "SAMO API is running", "status": "healthy"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/analyze/emotion")
async def analyze_emotion(text: str):
    """Mock emotion analysis for testing."""
    return {
        "text": text,
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
