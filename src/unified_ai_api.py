import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import Optional
from pydantic import BaseModel
import logging
from src.models.emotion_detection.bert_classifier import BERTEmotionClassifier
from src.models.summarization.t5_summarizer import T5SummarizationModel
from src.models.voice_processing.whisper_transcriber import WhisperTranscriber
from src.data.validation import validate_text_input
from src.input_sanitizer import InputSanitizer, SanitizationConfig

app = FastAPI(title="SAMO-DL Unified AI API", version="1.0.0")

# Initialize models for complete analysis (lazy loading in production)
def get_emotion_classifier():
    return BERTEmotionClassifier()

def get_summarizer():
    return T5SummarizationModel()

def get_transcriber():
    return WhisperTranscriber()

logger = logging.getLogger(__name__)

class AnalysisRequest(BaseModel):
    text: Optional[str] = None
    audio: Optional[UploadFile] = File(None)

@app.post("/complete-analysis/")
async def complete_analysis(request: AnalysisRequest):
    """Complete analysis endpoint integrating emotion detection, summarization,
    and transcription."""
    try:
        if not request.text and not request.audio:
            raise HTTPException(
                status_code=400, detail="At least text or audio input required"
            )

        result = {
            "emotion": None,
            "summary": None,
            "transcription": None,
            "analysis_complete": True
        }

        # Emotion detection
        if request.text:
            try:
                validated_text = validate_text_input(request.text)
                sanitized_text, warnings = InputSanitizer(SanitizationConfig()).sanitize_text(
                    validated_text, "analysis"
                )
                if warnings:
                    logger.warning("Sanitization warnings: %s", warnings)

                classifier = get_emotion_classifier()
                emotion_results = classifier.predict_emotions([sanitized_text])
                emotion_result = (
                    emotion_results["emotions"][0][0] if emotion_results["emotions"]
                    else {"label": "neutral", "score": 0.0}
                )
                result["emotion"] = emotion_result["label"]
                result["emotion_score"] = emotion_result["score"]
            except Exception as e:
                logger.error("Emotion detection failed: %s", e)
                result["emotion"] = "error"
                result["emotion_score"] = 0.0

        # Summarization
        if request.text and len(request.text) > 50:  # Only summarize longer texts
            try:
                summarizer_instance = get_summarizer()
                summary = summarizer_instance.generate_summary(request.text)
                result["summary"] = summary
            except Exception as e:
                logger.error("Summarization failed: %s", e)
                result["summary"] = "Summarization unavailable"

        # Transcription
        if request.audio:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    temp_file.write(await request.audio.read())
                    temp_audio_path = temp_file.name

                transcriber_instance = get_transcriber()
                transcription_result = transcriber_instance.transcribe(temp_audio_path)
                result["transcription"] = transcription_result.text
                result["transcription_confidence"] = transcription_result.confidence

                # Clean up temp file
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.error("Transcription failed: %s", e)
                result["transcription"] = "Transcription unavailable"
                result["transcription_confidence"] = 0.0

        if not any([
            result["emotion"], result["summary"], result["transcription"]
        ]):
            raise HTTPException(status_code=400, detail="No valid input provided for analysis")

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Complete analysis error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")

# Existing code would go here - this is appended for the new endpoint
# ... rest of existing unified_ai_api.py content ...

if __name__ == "__main__":
    import subprocess
    import tempfile
    import uvicorn

    # Log Python binary architecture info at startup
    result = subprocess.run(
        ['file', '/usr/local/bin/python'], capture_output=True, text=True, check=True
    )
    logger.info("Python binary info: %s", result.stdout)

    uvicorn.run(app, host="0.0.0.0", port=8000)
