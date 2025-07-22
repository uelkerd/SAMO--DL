"""SAMO Emotion Detection API Demo.

This demo showcases the emotion detection pipeline working with pre-trained
models and provides a preview of the API interface for Web Dev integration.
"""

import logging
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from .dataset_loader import GOEMOTIONS_EMOTIONS
from .bert_classifier import BERTEmotionClassifier, create_bert_emotion_classifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SAMO Emotion Detection API",
    description="AI-powered emotion detection for journal entries",
    version="0.1.0"
)

# Global model storage
model = None
tokenizer = None


class EmotionRequest(BaseModel):
    """Request model for emotion analysis."""
    text: str
    user_id: Optional[str] = None
    threshold: float = 0.5
    top_k: Optional[int] = 5


class EmotionResponse(BaseModel):
    """Response model for emotion analysis."""
    primary_emotion: str
    confidence: float
    predicted_emotions: List[str]
    emotion_scores: List[float]
    all_probabilities: List[float]
    processing_time_ms: float


@app.on_event("startup")
async def load_model():
    """Load emotion detection model on startup."""
    global model, tokenizer
    
    logger.info("Loading emotion detection model...")
    
    try:
        # Create untrained model for demo (in production, load trained weights)
        model, _ = create_bert_emotion_classifier(
            model_name="bert-base-uncased",
            freeze_bert_layers=0  # Unfreeze for demo
        )
        
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Set to evaluation mode
        model.eval()
        
        logger.info("✅ Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "SAMO Emotion Detection API",
        "version": "0.1.0",
        "available_emotions": GOEMOTIONS_EMOTIONS,
        "endpoints": {
            "analyze": "/analyze - POST - Analyze emotion in text",
            "health": "/health - GET - Health check",
            "emotions": "/emotions - GET - List all supported emotions"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }


@app.get("/emotions")
async def list_emotions():
    """List all supported emotion categories."""
    return {
        "emotions": GOEMOTIONS_EMOTIONS,
        "total_count": len(GOEMOTIONS_EMOTIONS),
        "categories": {
            "positive": ["joy", "love", "gratitude", "optimism", "pride", "excitement", "amusement", "admiration"],
            "negative": ["sadness", "anger", "fear", "disgust", "grief", "disappointment", "annoyance"],
            "complex": ["confusion", "curiosity", "embarrassment", "nervousness", "realization", "surprise"],
            "social": ["caring", "approval", "disapproval", "desire", "relief", "remorse"],
            "neutral": ["neutral"]
        }
    }


@app.post("/analyze", response_model=EmotionResponse)
async def analyze_emotion(request: EmotionRequest):
    """Analyze emotion in text using BERT model."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        import time
        start_time = time.time()
        
        # Tokenize input text
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Get emotion predictions
        with torch.no_grad():
            predictions = model.predict_emotions(
                **inputs,
                threshold=request.threshold,
                top_k=request.top_k
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Format response
        response = EmotionResponse(
            primary_emotion=predictions["primary_emotion"],
            confidence=predictions["primary_confidence"],
            predicted_emotions=predictions["predicted_emotions"],
            emotion_scores=predictions["emotion_scores"],
            all_probabilities=predictions["all_probabilities"],
            processing_time_ms=processing_time
        )
        
        logger.info(f"Analyzed text: '{request.text[:50]}...' -> {predictions['primary_emotion']} "
                   f"({predictions['primary_confidence']:.3f}) in {processing_time:.1f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing emotion: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/batch")
async def analyze_emotions_batch(texts: List[str], threshold: float = 0.5):
    """Analyze emotions for multiple texts."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not texts or len(texts) == 0:
        raise HTTPException(status_code=400, detail="Text list cannot be empty")
    
    results = []
    
    for text in texts:
        if text and len(text.strip()) > 0:
            request = EmotionRequest(text=text, threshold=threshold)
            result = await analyze_emotion(request)
            results.append({
                "text": text,
                "analysis": result
            })
    
    return {"results": results, "total_analyzed": len(results)}


if __name__ == "__main__":
    print("🚀 Starting SAMO Emotion Detection API Demo...")
    print("📝 Example requests:")
    print("  POST /analyze with: {'text': 'I am so excited about this project!'}")
    print("  POST /analyze with: {'text': 'I feel overwhelmed and anxious today.'}")
    print("  GET /emotions to see all supported emotions")
    print()
    
    # Run the API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    ) 