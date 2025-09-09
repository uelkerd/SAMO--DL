import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import Optional
from pydantic import BaseModel
import logging
from src.models.emotion_detection.bert_classifier import BERTEmotionClassifier
from src.models.summarization.samo_t5_summarizer import create_samo_t5_summarizer
from src.models.voice_processing.whisper_transcriber import WhisperTranscriber
from src.data.validation import validate_text_input
from src.input_sanitizer import InputSanitizer, SanitizationConfig

app = FastAPI(title="SAMO-DL Unified AI API", version="1.0.0")

# Initialize models for complete analysis (lazy loading in production)
def get_emotion_classifier():
    """Get emotion classification model instance."""
    return BERTEmotionClassifier()

def get_summarizer():
    """Get SAMO-optimized text summarization model instance."""
    return create_samo_t5_summarizer()

def get_transcriber():
    """Get audio transcription model instance."""
    return WhisperTranscriber()

logger = logging.getLogger(__name__)

class AnalysisRequest(BaseModel):
    text: Optional[str] = None
    audio: Optional[UploadFile] = File(None)

class SummarizationRequest(BaseModel):
    text: str
    max_length: Optional[int] = 100
    min_length: Optional[int] = 20
    num_beams: Optional[int] = 4

@app.post("/complete-analysis/")
async def complete_analysis(request: AnalysisRequest):
    """Complete analysis endpoint integrating emotion detection, summarization,
    and transcription."""
    try:
        if not request.text and not request.audio:
            raise HTTPException(
                status_code=400, detail="At least text or audio input required"
            )

        analysis_result = {
            "emotion": None,
            "summary": None,
            "transcription": None,
            "analysis_complete": True
        }

        # Emotion detection
        if request.text:
            try:
                validated_text = validate_text_input(request.text)
                sanitized_text, warnings = InputSanitizer(
                    SanitizationConfig()
                ).sanitize_text(validated_text, "analysis")
                if warnings:
                    logger.warning("Sanitization warnings: %s", warnings)

                classifier = get_emotion_classifier()
                emotion_results = classifier.predict_emotions([sanitized_text])
                emotion_result = (
                    emotion_results["emotions"][0][0] if emotion_results["emotions"]
                    else {"label": "neutral", "score": 0.0}
                )
                analysis_result["emotion"] = emotion_result["label"]
                analysis_result["emotion_score"] = emotion_result["score"]
            except Exception as e:
                logger.error("Emotion detection failed: %s", e)
                analysis_result["emotion"] = "error"
                analysis_result["emotion_score"] = 0.0

        # SAMO-optimized summarization for journal entries
        if request.text and len(request.text.split()) > 20:  # Only summarize longer texts
            try:
                summarizer_instance = get_summarizer()
                summary_result = summarizer_instance.summarize_journal_entry(
                    request.text,
                    extract_emotions=True
                )
                
                # Extract summary and SAMO-specific metadata
                analysis_result["summary"] = summary_result["summary"]
                analysis_result["summary_confidence"] = summary_result.get("confidence", 0.0)
                analysis_result["summary_length"] = summary_result.get("summary_length", 0)
                analysis_result["input_length"] = summary_result.get("input_length", 0)
                analysis_result["processing_time"] = summary_result.get("processing_time", 0.0)
                
                # SAMO-specific metadata
                samo_metadata = summary_result.get("samo_metadata", {})
                analysis_result["emotional_keywords"] = summary_result.get("emotional_keywords", [])
                analysis_result["journal_mode"] = samo_metadata.get("journal_mode", False)
                analysis_result["emotional_context"] = samo_metadata.get("emotional_context", False)
                    
            except Exception as e:
                logger.error("SAMO summarization failed: %s", e)
                analysis_result["summary"] = "Summarization unavailable"
                analysis_result["summary_confidence"] = 0.0

        # Transcription
        if request.audio:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".wav"
                ) as temp_file:
                    temp_file.write(await request.audio.read())
                    temp_audio_path = temp_file.name

                transcriber_instance = get_transcriber()
                transcription_result = transcriber_instance.transcribe(temp_audio_path)
                analysis_result["transcription"] = transcription_result.text
                analysis_result["transcription_confidence"] = \
                    transcription_result.confidence

                # Clean up temp file
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.error("Transcription failed: %s", e)
                analysis_result["transcription"] = "Transcription unavailable"
                analysis_result["transcription_confidence"] = 0.0

        if not any([
            analysis_result["emotion"],
            analysis_result["summary"],
            analysis_result["transcription"]
        ]):
            raise HTTPException(
                status_code=400, detail="No valid input provided for analysis"
            )

        return analysis_result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Complete analysis error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/summarize/")
async def summarize_text(request: SummarizationRequest):
    """Dedicated text summarization endpoint with SAMO-optimized parameters."""
    try:
        # Validate input
        if not request.text or len(request.text.strip()) < 20:
            raise HTTPException(
                status_code=400, 
                detail="Text must be at least 20 words for meaningful summarization"
            )
        
        # Sanitize input
        sanitized_text, warnings = InputSanitizer(
            SanitizationConfig()
        ).sanitize_text(request.text, "summarization")
        
        if warnings:
            logger.warning("Sanitization warnings: %s", warnings)
        
        # Get SAMO-optimized summarizer and generate summary
        summarizer_instance = get_summarizer()
        summary_result = summarizer_instance.summarize_journal_entry(
            sanitized_text,
            extract_emotions=True
        )
        
        # Return structured response with SAMO-specific data
        return {
            "summary": summary_result["summary"],
            "confidence": summary_result.get("confidence", 0.0),
            "input_length": summary_result.get("input_length", 0),
            "summary_length": summary_result.get("summary_length", 0),
            "compression_ratio": summary_result.get("input_length", 0) / max(summary_result.get("summary_length", 1), 1),
            "processing_time": summary_result.get("processing_time", 0.0),
            "parameters": {
                "max_length": request.max_length,
                "min_length": request.min_length,
                "num_beams": request.num_beams
            },
            "samo_metadata": summary_result.get("samo_metadata", {}),
            "emotional_keywords": summary_result.get("emotional_keywords", []),
            "model_info": summarizer_instance.get_model_info()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Summarization error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/summarize/health")
async def summarization_health():
    """Health check for summarization service."""
    try:
        summarizer = get_summarizer()
        model_info = summarizer.get_model_info()
        
        return {
            "status": "healthy",
            "model_name": model_info["model_name"],
            "device": model_info["device"],
            "max_length": model_info["max_length"],
            "min_length": model_info["min_length"]
        }
    except Exception as e:
        logger.error("Health check failed: %s", e)
        raise HTTPException(status_code=500, detail="Service unavailable")

# Existing code would go here - this is appended for the new endpoint
# ... rest of existing unified_ai_api.py content ...

if __name__ == "__main__":
    import subprocess
    import tempfile
    import uvicorn

    # Log Python binary architecture info at startup
    result = subprocess.run(
        ['file', '/usr/local/bin/python'],
        capture_output=True, text=True, check=True
    )
    logger.info("Python binary info: %s", result.stdout)

    uvicorn.run(app, host="0.0.0.0", port=8000)
