# G004: Logging f-strings temporarily allowed for development
"""FastAPI Endpoints for T5/BART Summarization - SAMO Deep Learning.

This module provides production-ready API endpoints for text summarization
that integrate with the SAMO Web Development backend.

Key Features:
- Single text summarization endpoint
- Batch summarization for multiple entries
- Configurable summary parameters
- Error handling and validation
- Performance monitoring
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from .t5_summarizer import T5SummarizationModel, create_t5_summarizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance (loaded on startup)
summarization_model: T5SummarizationModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle - load on startup, cleanup on shutdown."""
    global summarization_model

    # Startup: Load model
    logger.info("🚀 Loading T5 summarization model...")
    start_time = time.time()

    try:
        summarization_model = create_t5_summarizer(
            model_name="t5-small",  # Start with small model for speed
            max_source_length=512,
            max_target_length=128,
        )

        time.time() - start_time
        logger.info("✅ Model loaded successfully in {load_time:.2f}s", extra={"format_args": True})
        logger.info(
            "Model info: {summarization_model.get_model_info()}", extra={"format_args": True}
        )

    except Exception as e:
        logger.error("❌ Failed to load summarization model: {e}", extra={"format_args": True})
        raise RuntimeError(f"Model loading failed: {e}")

    yield  # App runs here

    # Shutdown: Cleanup
    logger.info("🔄 Shutting down summarization service...")
    summarization_model = None


# Initialize FastAPI with lifecycle management
app = FastAPI(
    title="SAMO Summarization API",
    description="T5/BART-based text summarization for emotional journal analysis",
    version="1.0.0",
    lifespan=lifespan,
)


# Request/Response Models
class SummarizationRequest(BaseModel):
    """Request model for single text summarization."""

    text: str = Field(
        ...,
        description="Text to summarize (journal entry or conversation)",
        min_length=50,
        max_length=2000,
        example="Today was such a rollercoaster of emotions. I started feeling anxious about my job interview...",
    )
    max_length: int | None = Field(128, description="Maximum summary length", ge=30, le=256)
    min_length: int | None = Field(30, description="Minimum summary length", ge=10, le=100)
    focus_emotional: bool | None = Field(
        True, description="Whether to focus on emotional content in summary"
    )

    @validator("min_length")
    def validate_length_relationship(cls, min_length, values):
        max_length = values.get("max_length", 128)
        if min_length >= max_length:
            raise ValueError("min_length must be less than max_length")
        return min_length


class BatchSummarizationRequest(BaseModel):
    """Request model for batch summarization."""

    texts: list[str] = Field(
        ...,
        description="List of texts to summarize",
        min_items=1,
        max_items=10,  # Limit batch size
    )
    max_length: int | None = Field(128, ge=30, le=256)
    min_length: int | None = Field(30, ge=10, le=100)
    focus_emotional: bool | None = Field(True)

    @validator("texts")
    def validate_text_lengths(cls, texts):
        for i, text in enumerate(texts):
            if len(text) < 50:
                raise ValueError(f"Text {i + 1} too short (minimum 50 characters)")
            if len(text) > 2000:
                raise ValueError(f"Text {i + 1} too long (maximum 2000 characters)")
        return texts


class SummarizationResponse(BaseModel):
    """Response model for summarization results."""

    summary: str = Field(..., description="Generated summary")
    original_length: int = Field(..., description="Original text character count")
    summary_length: int = Field(..., description="Summary character count")
    compression_ratio: float = Field(..., description="Length reduction ratio")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_info: dict = Field(..., description="Model metadata")


class BatchSummarizationResponse(BaseModel):
    """Response model for batch summarization."""

    summaries: list[SummarizationResponse] = Field(..., description="List of summarization results")
    total_processing_time_ms: float = Field(..., description="Total batch processing time")
    average_processing_time_ms: float = Field(..., description="Average per-item processing time")


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if summarization_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_info": summarization_model.get_model_info(),
    }


@app.post("/summarize", response_model=SummarizationResponse)
async def summarize_text(request: SummarizationRequest):
    """Summarize a single journal entry or text.

    This endpoint generates an intelligent summary that preserves
    emotional context and key insights from the original text.
    """
    if summarization_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.time()

        # Generate summary
        summary = summarization_model.generate_summary(
            text=request.text, max_length=request.max_length, min_length=request.min_length
        )

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        # Calculate metrics
        original_length = len(request.text)
        summary_length = len(summary)
        compression_ratio = 1 - (summary_length / original_length) if original_length > 0 else 0

        # Log performance
        logger.info(
            "Summarized text: {original_length}→{summary_length} chars in {processing_time:.2f}ms",
            extra={"format_args": True},
        )

        return SummarizationResponse(
            summary=summary,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=compression_ratio,
            processing_time_ms=processing_time,
            model_info=summarization_model.get_model_info(),
        )

    except Exception as e:
        logger.error("Summarization error: {e}", extra={"format_args": True})
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e!s}")


@app.post("/summarize/batch", response_model=BatchSummarizationResponse)
async def summarize_batch(request: BatchSummarizationRequest):
    """Summarize multiple texts in batch for efficiency.

    Useful for processing multiple journal entries or conversation
    segments simultaneously with improved throughput.
    """
    if summarization_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.time()

        # Generate batch summaries
        summaries = summarization_model.generate_batch_summaries(
            texts=request.texts,
            batch_size=4,  # Process in smaller batches for memory efficiency
            max_length=request.max_length,
            min_length=request.min_length,
        )

        total_processing_time = (time.time() - start_time) * 1000

        # Create detailed responses
        detailed_responses = []
        for text, summary in zip(request.texts, summaries, strict=False):
            original_length = len(text)
            summary_length = len(summary)
            compression_ratio = 1 - (summary_length / original_length) if original_length > 0 else 0

            detailed_responses.append(
                SummarizationResponse(
                    summary=summary,
                    original_length=original_length,
                    summary_length=summary_length,
                    compression_ratio=compression_ratio,
                    processing_time_ms=total_processing_time / len(request.texts),  # Average
                    model_info=summarization_model.get_model_info(),
                )
            )

        average_time = total_processing_time / len(request.texts)

        logger.info(
            "Batch summarized {len(request.texts)} texts in {total_processing_time:.2f}ms (avg: {average_time:.2f}ms)",
            extra={"format_args": True},
        )

        return BatchSummarizationResponse(
            summaries=detailed_responses,
            total_processing_time_ms=total_processing_time,
            average_processing_time_ms=average_time,
        )

    except Exception as e:
        logger.error("Batch summarization error: {e}", extra={"format_args": True})
        raise HTTPException(status_code=500, detail=f"Batch summarization failed: {e!s}")


@app.get("/model/info")
async def get_model_info():
    """Get detailed information about the loaded model."""
    if summarization_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    info = summarization_model.get_model_info()

    # Add runtime information
    info.update(
        {
            "api_version": "1.0.0",
            "supported_formats": ["text/plain"],
            "max_batch_size": 10,
            "recommended_text_length": "50-1500 characters",
        }
    )

    return info


@app.post("/model/warm-up")
async def warm_up_model(background_tasks: BackgroundTasks):
    """Warm up the model with sample text for faster subsequent requests."""
    if summarization_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    def warm_up() -> None:
        sample_text = """Today was a great day filled with positive emotions and meaningful conversations.
        I felt grateful for the opportunities and connections in my life."""

        try:
            summarization_model.generate_summary(sample_text)
            logger.info("Model warm-up completed successfully")
        except Exception:
            logger.error("Model warm-up failed: {e}", extra={"format_args": True})

    background_tasks.add_task(warm_up)

    return {"message": "Model warm-up initiated"}


# Error Handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.error("Validation error: {exc}", extra={"format_args": True})
    return HTTPException(status_code=422, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting SAMO Summarization API...")
    uvicorn.run(
        "api_demo:app",
        host="127.0.0.1",  # Changed from 0.0.0.0 for security
        port=8001,  # Different port from main SAMO API
        reload=True,
        log_level="info",
    )
