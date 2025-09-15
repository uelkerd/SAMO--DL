#!/usr/bin/env python3
"""Minimal Unified AI API for SAMO Deep Learning.

This is a simplified version that loads models on-demand to avoid startup timeout issues.
"""
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SAMO Unified AI API",
    description="Unified AI API for emotion detection, text summarization, and voice processing",
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

# Global variables for lazy loading
emotion_model = None
summarization_model = None
whisper_model = None

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "SAMO Unified AI API",
        "version": "1.0.0",
        "status": "running",
        "features": ["emotion_detection", "text_summarization", "voice_processing"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "API is running"}

@app.get("/api/health")
async def api_health():
    """API health check endpoint."""
    return {"status": "healthy", "message": "API is running"}

@app.post("/analyze/emotion")
async def analyze_emotion(text: str):
    """Analyze emotion in text."""
    try:
        # Lazy load emotion model
        global emotion_model
        if emotion_model is None:
            logger.info("Loading emotion model...")
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            model_name = 'duelker/samo-goemotions-deberta-v3-large'
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            emotion_model = {"tokenizer": tokenizer, "model": model}
            logger.info("Emotion model loaded successfully")

        # Perform emotion analysis
        inputs = emotion_model["tokenizer"](text, return_tensors="pt", truncation=True, max_length=512)
        outputs = emotion_model["model"](**inputs)
        predictions = outputs.logits.sigmoid()  # Use sigmoid for multi-label classification

        # Get emotion labels (28 emotions from our DeBERTa-v3 model)
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

        return result

    except Exception as e:
        logger.error(f"Error in emotion analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Emotion analysis failed: {str(e)}")

@app.post("/analyze/summarize")
async def summarize_text(text: str):
    """Summarize text using T5 model."""
    try:
        # Lazy load summarization model
        global summarization_model
        if summarization_model is None:
            logger.info("Loading summarization model...")
            try:
                import sentencepiece  # Required for T5 tokenizer
            except ImportError:
                logger.error("sentencepiece not installed. Please install it: pip install sentencepiece")
                raise HTTPException(status_code=500, detail="Summarization model requires sentencepiece. Please install it.")

            from transformers import T5Tokenizer, T5ForConditionalGeneration
            model_name = 't5-small'
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            summarization_model = {"tokenizer": tokenizer, "model": model}
            logger.info("Summarization model loaded successfully")

        # Perform summarization
        inputs = summarization_model["tokenizer"](f"summarize: {text}", return_tensors="pt", max_length=512, truncation=True)
        outputs = summarization_model["model"].generate(
            inputs["input_ids"],
            max_length=150,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = summarization_model["tokenizer"].decode(outputs[0], skip_special_tokens=True)

        return {
            "original_text": text,
            "summary": summary
        }

    except Exception as e:
        logger.error(f"Error in text summarization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text summarization failed: {str(e)}")

@app.post("/analyze/transcribe")
async def transcribe_audio(audio_file: bytes):
    """Transcribe audio using Whisper model."""
    try:
        # Lazy load Whisper model
        global whisper_model
        if whisper_model is None:
            logger.info("Loading Whisper model...")
            import whisper
            whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded successfully")

        # Save audio file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_file)
            tmp_file_path = tmp_file.name

        try:
            # Perform transcription
            result = whisper_model.transcribe(tmp_file_path)
            transcription = result["text"]

            return {
                "transcription": transcription,
                "language": result.get("language", "unknown")
            }
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)

    except Exception as e:
        logger.error(f"Error in audio transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio transcription failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
