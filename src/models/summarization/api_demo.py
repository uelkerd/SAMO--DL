#!/usr/bin/env python3
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
from typing import List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from .t5_summarizer import T5SummarizationModel, create_t5_summarizer

# Configure logging
# G004: Logging f-strings temporarily allowed for development
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
summarization_model: Optional[T5SummarizationModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle - load on startup, cleanup on shutdown."""
    global summarization_model

    logger.info("🚀 Loading T5 summarization model...")
    start_time = time.time()

    try:
        summarization_model = create_t5_summarizer(
            model_name="t5-small",  # Start with small model for speed
            max_source_length=512,
            max_target_length=128,
        )

        load_time = time.time() - start_time
        logger.info(f"✅ Model loaded successfully in {load_time:.2f}s")
        logger.info(f"Model info: {summarization_model.get_model_info()}")

    except Exception as e:
        logger.error(f"❌ Failed to load summarization model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

    yield  # App runs here

    logger.info("🔄 Shutting down summarization service...")
    summarization_model = None


# Initialize FastAPI with lifecycle management
app = FastAPI(
    title="SAMO Summarization API",
    description="T5/BART-based text summarization for emotional journal analysis",
    version="1.0.0",
    lifespan=lifespan,
)


class SummarizeRequest(BaseModel):
    """Request model for single text summarization."""

    text: str = Field(
        ..., description="Text to summarize", min_length=10, max_length=2000
    )
    max_length: Optional[int] = Field(
        128, description="Maximum summary length", ge=30, le=256
    )
    min_length: Optional[int] = Field(
        30, description="Minimum summary length", ge=10, le=100
    )
    focus_emotional: Optional[bool] = Field(
        False, description="Focus on emotional content in summary"
    )

    @validator("min_length")
    def validate_length_relationship(cls, min_length, values):
        max_length = values.get("max_length", 128)
        if min_length >= max_length:
            raise ValueError("min_length must be less than max_length")
        return min_length


class BatchSummarizationRequest(BaseModel):
    """Request model for batch summarization."""

    texts: List[str] = Field(..., description="List of texts to summarize")
    max_length: Optional[int] = Field(128, ge=30, le=256)
    min_length: Optional[int] = Field(30, ge=10, le=100)
    focus_emotional: Optional[bool] = Field(True)

    @validator("texts")
    def validate_text_lengths(cls, texts):
        for _i, text in enumerate(texts):
            if len(text) < 50:
                raise ValueError(f"Text {_i + 1} too short (minimum 50 characters)")
            if len(text) > 2000:
                raise ValueError(f"Text {_i + 1} too long (maximum 2000 characters)")
        return texts


class SummarizationResponse(BaseModel):
    """Response model for summarization results."""

    summary: str = Field(..., description="Generated summary")
    original_length: int = Field(..., description="Original text character count")
    summary_length: int = Field(..., description="Summary character count")
    compression_ratio: float = Field(..., description="Length reduction ratio")
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )
    model_info: dict = Field(..., description="Model metadata")


class BatchSummarizationResponse(BaseModel):
    """Response model for batch summarization."""

    summaries: List[SummarizationResponse] = Field(
        ..., description="List of summarization results"
    )
    total_processing_time_ms: float = Field(
        ..., description="Total batch processing time"
    )
    average_processing_time_ms: float = Field(
        ..., description="Average per-item processing time"
    )


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
async def summarize_text(request: SummarizeRequest):
    """Summarize a single journal entry or text.

    This endpoint generates an intelligent summary that preserves emotional context and
    key insights from the original text.
    """
    if summarization_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.time()

        summary = summarization_model.generate_summary(
            text=request.text,
            max_length=request.max_length,
            min_length=request.min_length,
        )

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        original_length = len(request.text)
        summary_length = len(summary)
        compression_ratio = (
            1 - (summary_length / original_length) if original_length > 0 else 0
        )

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

    except Exception:
        logger.exception("Summarization error")
        raise HTTPException(
            status_code=500,
            detail="Summarization processing failed. Please try again later.",
        )


@app.post("/summarize/batch", response_model=BatchSummarizationResponse)
async def summarize_batch(request: BatchSummarizationRequest):
    """Summarize multiple texts in batch for efficiency.

    Useful for processing multiple journal entries or conversation segments
    simultaneously with improved throughput.
    """
    if summarization_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.time()

        summaries = [
            summarization_model.generate_summary(text) for text in request.texts
        ]

        total_processing_time = (time.time() - start_time) * 1000

        detailed_responses = []
        for text, summary in zip(request.texts, summaries):
            original_length = len(text)
            summary_length = len(summary)
            compression_ratio = (
                1 - (summary_length / original_length) if original_length > 0 else 0
            )

            detailed_responses.append(
                SummarizationResponse(
                    summary=summary,
                    original_length=original_length,
                    summary_length=summary_length,
                    compression_ratio=compression_ratio,
                    processing_time_ms=total_processing_time
                    / len(request.texts),  # Average
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

    except Exception:
        logger.exception("Batch summarization error")
        raise HTTPException(
            status_code=500,
            detail="Batch summarization processing failed. Please try again later.",
        )


@app.get("/model/info")
async def get_model_info():
    """Get detailed information about the loaded model."""
    if summarization_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    info = summarization_model.get_model_info()

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
            logger.exception("Model warm-up failed")

    background_tasks.add_task(warm_up)

    return {"message": "Model warm-up initiated"}


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.error("Validation error: {exc}", extra={"format_args": True})
    return HTTPException(status_code=422, detail=str(exc))


if __name__ == "__main__":
    logger.info("🚀 Starting SAMO Summarization API...")
    uvicorn.run(
        "api_demo:app",
        host="127.0.0.1",  # Changed from 0.0.0.0 for security
        port=8001,  # Different port from main SAMO API
        reload=True,
        log_level="info",
    )
