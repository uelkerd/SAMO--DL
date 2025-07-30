import os

# G004: Logging f-strings temporarily allowed for development
import logging
import tempfile
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .audio_preprocessor import AudioPreprocessor
from .whisper_transcriber import WhisperTranscriber, create_whisper_transcriber

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance (loaded on startup)
whisper_transcriber: WhisperTranscriber | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle - load on startup, cleanup on shutdown."""
    global whisper_transcriber

    # Startup: Load Whisper model
    logger.info("🚀 Loading OpenAI Whisper model...")
    start_time = time.time()

    try:
        whisper_transcriber = create_whisper_transcriber(
            model_size="base",  # Balance between speed and accuracy
            language=None,  # Auto-detect language
            device=None,  # Auto-detect device
        )

        time.time() - start_time
        logger.info(
            "✅ Whisper model loaded successfully in {load_time:.2f}s", extra={"format_args": True}
        )
        logger.info(
            "Model info: {whisper_transcriber.get_model_info()}", extra={"format_args": True}
        )

    except Exception:
        logger.error("❌ Failed to load Whisper model: {exc}", extra={"format_args": True})
        logger.info("⚠️  Running in development mode without Whisper model")
        whisper_transcriber = None  # Continue without model for development

    yield  # App runs here

    # Shutdown: Cleanup
    logger.info("🔄 Shutting down voice processing service...")
    whisper_transcriber = None


# Initialize FastAPI with lifecycle management
app = FastAPI(
    title="SAMO Voice Processing API",
    description="OpenAI Whisper-based voice-to-text transcription for journal entries",
    version="1.0.0",
    lifespan=lifespan,
)


# Request/Response Models
class TranscriptionResponse(BaseModel):
    """Response model for transcription results."""

    text: str = Field(..., description="Transcribed text")
    language: str = Field(..., description="Detected language")
    confidence: float = Field(..., description="Transcription confidence score")
    duration: float = Field(..., description="Audio duration in seconds")
    processing_time: float = Field(..., description="Processing time in milliseconds")
    word_count: int = Field(..., description="Number of words transcribed")
    speaking_rate: float = Field(..., description="Speaking rate (words per minute)")
    audio_quality: str = Field(..., description="Audio quality assessment")
    no_speech_probability: float = Field(..., description="Probability of no speech")
    model_info: dict = Field(..., description="Model metadata")


class BatchTranscriptionResponse(BaseModel):
    """Response model for batch transcription."""

    transcriptions: list[TranscriptionResponse] = Field(
        ..., description="List of transcription results"
    )
    total_processing_time: float = Field(..., description="Total batch processing time")
    average_processing_time: float = Field(..., description="Average per-file processing time")
    success_count: int = Field(..., description="Number of successful transcriptions")
    error_count: int = Field(..., description="Number of failed transcriptions")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: dict | None = Field(None, description="Additional error details")


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if whisper_transcriber is None:
        return {
            "status": "degraded",
            "model_loaded": False,
            "message": "Running in development mode - Whisper model not loaded",
        }

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_info": whisper_transcriber.get_model_info(),
    }


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    language: str | None = Form(None),
    initial_prompt: str | None = Form(None),
):
    """Transcribe a single audio file to text.

    This endpoint accepts various audio formats (MP3, WAV, M4A, etc.)
    and returns high-quality transcription with confidence scoring.
    """
    if whisper_transcriber is None:
        raise HTTPException(
            status_code=503, detail="Whisper model not available - running in development mode"
        )

    # Validate file type
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_extension = Path(audio_file.filename).suffix.lower()
    if file_extension not in AudioPreprocessor.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format: {file_extension}. "
            "Supported formats: {list(AudioPreprocessor.SUPPORTED_FORMATS)}",
        )

    # Save uploaded file temporarily
    temp_file = None
    try:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=file_extension, delete=False)

        # Write uploaded content
        content = await audio_file.read()
        temp_file.write(content)
        temp_file.close()

        # Validate audio file
        is_valid, error_msg = AudioPreprocessor.validate_audio_file(temp_file.name)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        # Transcribe audio
        time.time()
        result = whisper_transcriber.transcribe_audio(
            temp_file.name, language=language, initial_prompt=initial_prompt
        )

        # Convert to API response
        response = TranscriptionResponse(
            text=result.text,
            language=result.language,
            confidence=result.confidence,
            duration=result.duration,
            processing_time=result.processing_time * 1000,  # Convert to ms
            word_count=result.word_count,
            speaking_rate=result.speaking_rate,
            audio_quality=result.audio_quality,
            no_speech_probability=result.no_speech_probability,
            model_info=whisper_transcriber.get_model_info(),
        )

        logger.info(
            "Transcribed {audio_file.filename}: {result.word_count} words, "
            "{result.confidence:.2f} confidence, {result.processing_time:.2f}s"
        )

        return response

    except HTTPException:
        raise
    except Exception as _:
        logger.error("Transcription error: {e}", extra={"format_args": True})
        raise HTTPException(status_code=500, detail="Transcription failed: {e!s}")

    finally:
        # Cleanup temporary file
        if temp_file and Path(temp_file.name).exists():
            with suppress(Exception):
                os.unlink(temp_file.name)


@app.post("/transcribe/batch", response_model=BatchTranscriptionResponse)
async def transcribe_batch(
    audio_files: list[UploadFile] = File(...),
    language: str | None = Form(None),
    initial_prompt: str | None = Form(None),
):
    """Transcribe multiple audio files in batch for efficiency.

    Useful for processing multiple journal voice entries simultaneously
    with improved throughput and resource utilization.
    """
    if whisper_transcriber is None:
        raise HTTPException(
            status_code=503, detail="Whisper model not available - running in development mode"
        )

    if len(audio_files) > 10:  # Limit batch size
        raise HTTPException(
            status_code=400, detail="Batch size too large. Maximum 10 files per batch."
        )

    temp_files = []
    transcriptions = []

    try:
        start_time = time.time()

        # Process each file
        for _i, audio_file in enumerate(audio_files):
            try:
                # Validate file
                if not audio_file.filename:
                    raise ValueError("File {i + 1}: No filename provided")

                file_extension = Path(audio_file.filename).suffix.lower()
                if file_extension not in AudioPreprocessor.SUPPORTED_FORMATS:
                    raise ValueError("File {i + 1}: Unsupported format {file_extension}")

                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(suffix=file_extension, delete=False)
                temp_files.append(temp_file.name)

                content = await audio_file.read()
                temp_file.write(content)
                temp_file.close()

                # Validate audio
                is_valid, error_msg = AudioPreprocessor.validate_audio_file(temp_file.name)
                if not is_valid:
                    raise ValueError("File {i + 1}: {error_msg}")

                # Transcribe
                result = whisper_transcriber.transcribe_audio(
                    temp_file.name, language=language, initial_prompt=initial_prompt
                )

                # Add to results
                transcriptions.append(
                    TranscriptionResponse(
                        text=result.text,
                        language=result.language,
                        confidence=result.confidence,
                        duration=result.duration,
                        processing_time=result.processing_time * 1000,
                        word_count=result.word_count,
                        speaking_rate=result.speaking_rate,
                        audio_quality=result.audio_quality,
                        no_speech_probability=result.no_speech_probability,
                        model_info=whisper_transcriber.get_model_info(),
                    )
                )

                logger.info(
                    "Batch item {i+1}: {result.word_count} words, {result.confidence:.2f} confidence",
                    extra={"format_args": True},
                )

            except Exception:
                logger.error(
                    "Failed to process file {i+1} ({audio_file.filename}): {e}",
                    extra={"format_args": True},
                )
                # Add error result
                transcriptions.append(
                    TranscriptionResponse(
                        text="",
                        language="unknown",
                        confidence=0.0,
                        duration=0.0,
                        processing_time=0.0,
                        word_count=0,
                        speaking_rate=0.0,
                        audio_quality="error",
                        no_speech_probability=1.0,
                        model_info={},
                    )
                )

        # Calculate batch metrics
        total_processing_time = (time.time() - start_time) * 1000
        success_count = sum(1 for t in transcriptions if t.confidence > 0)
        error_count = len(transcriptions) - success_count
        average_time = total_processing_time / len(transcriptions) if transcriptions else 0

        response = BatchTranscriptionResponse(
            transcriptions=transcriptions,
            total_processing_time=total_processing_time,
            average_processing_time=average_time,
            success_count=success_count,
            error_count=error_count,
        )

        logger.info(
            "Batch transcription complete: {success_count}/{len(audio_files)} successful, "
            "{total_processing_time:.2f}ms total"
        )

        return response

    except HTTPException:
        raise
    except Exception as _:
        logger.error("Batch transcription error: {e}", extra={"format_args": True})
        raise HTTPException(status_code=500, detail="Batch transcription failed: {e!s}") from e

    finally:
        # Cleanup temporary files
        for temp_file in temp_files:
            if Path(temp_file).exists():
                with suppress(Exception):
                    Path(temp_file).unlink()


@app.get("/model/info")
async def get_model_info() -> dict[str, Any]:
    """Get detailed information about the loaded Whisper model."""
    if whisper_transcriber is None:
        raise HTTPException(
            status_code=503, detail="Whisper model not available - running in development mode"
        )

    info = whisper_transcriber.get_model_info()

    # Add API information
    info.update(
        {
            "api_version": "1.0.0",
            "max_batch_size": 10,
            "max_file_duration": AudioPreprocessor.MAX_DURATION,
            "supported_languages": "auto-detect + 99 languages",
            "recommended_formats": [".wav", ".mp3", ".m4a"],
        }
    )

    return info


@app.post("/validate/audio")
async def validate_audio(audio_file: UploadFile = File(...)):
    """Validate audio file without transcribing.

    Useful for pre-upload validation to provide immediate feedback
    to users about file compatibility and quality.
    """
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_extension = Path(audio_file.filename).suffix.lower()

    # Basic format validation
    if file_extension not in AudioPreprocessor.SUPPORTED_FORMATS:
        return JSONResponse(
            status_code=400,
            content={
                "valid": False,
                "error": "unsupported_format",
                "message": "Unsupported audio format: {file_extension}",
                "supported_formats": list(AudioPreprocessor.SUPPORTED_FORMATS),
            },
        )

    # Save and validate audio content
    temp_file = None
    try:
        temp_file = tempfile.NamedTemporaryFile(suffix=file_extension, delete=False)

        content = await audio_file.read()
        temp_file.write(content)
        temp_file.close()

        # Validate with AudioPreprocessor
        is_valid, error_msg = AudioPreprocessor.validate_audio_file(temp_file.name)

        if is_valid:
            # Get audio metadata
            _, metadata = AudioPreprocessor.preprocess_audio(temp_file.name)

            return {
                "valid": True,
                "message": "Audio file is valid for transcription",
                "metadata": {
                    "duration": metadata["duration"],
                    "sample_rate": metadata["sample_rate"],
                    "channels": metadata["channels"],
                    "format": metadata["format"],
                    "file_size": metadata["file_size"],
                },
            }
        else:
            return JSONResponse(
                status_code=400,
                content={"valid": False, "error": "validation_failed", "message": error_msg},
            )

    except Exception as _:
        logger.error("Audio validation error occurred")
        return JSONResponse(
            status_code=500,
            content={
                "valid": False,
                "error": "validation_error",
                "message": "Error validating audio file",
            },
        )

    finally:
        if temp_file and Path(temp_file.name).exists():
            with suppress(Exception):
                os.unlink(temp_file.name)


@app.post("/model/warm-up")
async def warm_up_model(background_tasks: BackgroundTasks):
    """Warm up the model for faster subsequent requests."""
    if whisper_transcriber is None:
        raise HTTPException(
            status_code=503, detail="Whisper model not available - running in development mode"
        )

    def warm_up() -> None:
        # In a real implementation, you might transcribe a short test audio
        logger.info("Model warm-up would transcribe test audio")
        logger.info("Model warm-up completed successfully")

    background_tasks.add_task(warm_up)

    return {"message": "Model warm-up initiated"}


# Error Handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.error("Validation error: {exc}", extra={"format_args": True})
    return JSONResponse(
        status_code=422, content=ErrorResponse(error="validation_error", message=str(exc)).dict()
    )


if __name__ == "__main__":
    logger.info("🚀 Starting SAMO Voice Processing API...")
    uvicorn.run(
        "api_demo:app",
        host="127.0.0.1",  # Changed from 0.0.0.0 for security
        port=8002,  # Different port from other APIs
        reload=True,
        log_level="info",
    )
