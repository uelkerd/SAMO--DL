#!/usr/bin/env python3
"""Ultra-simple API that starts immediately for Cloud Run."""
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SAMO Simple API",
    description="Simple API that starts immediately",
    version="1.0.0"
)

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
    """Root endpoint."""
    return {"message": "SAMO API is running", "status": "healthy"}

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}

@app.post("/analyze/emotion")
async def analyze_emotion(text: str):
    """Analyze emotion - loads model on first request."""
    try:
        logger.info(f"Analyzing emotion for text: {text[:50]}...")

        # Import and load model only when needed
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_name = 'duelker/samo-goemotions-deberta-v3-large'
        logger.info(f"Loading model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Perform analysis
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        predictions = outputs.logits.sigmoid()

        # 28 emotions from our DeBERTa-v3 model
        emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]

        emotion_scores = predictions[0].tolist()
        result = {
            "text": text,
            "emotions": dict(zip(emotion_labels, emotion_scores)),
            "predicted_emotion": emotion_labels[emotion_scores.index(max(emotion_scores))]
        }

        logger.info(f"Analysis complete. Predicted emotion: {result['predicted_emotion']}")
        return result

    except Exception as e:
        logger.error(f"Error in emotion analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    host = "0.0.0.0"
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
