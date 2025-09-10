"""
SAMO Whisper Transcription Results Module

This module defines the data structures for transcription results
and provides utilities for result processing and analysis.
"""

import logging
from typing import Dict, List
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result of audio transcription."""
    text: str
    language: str
    confidence: float
    duration: float
    processing_time: float
    segments: List[Dict]
    audio_quality: str
    word_count: int
    speaking_rate: float
    no_speech_probability: float

    def to_dict(self) -> Dict:
        """Convert result to dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        """String representation of the result."""
        return (
            f"TranscriptionResult(text='{self.text[:50]}...', "
            f"language='{self.language}', confidence={self.confidence:.3f})"
        )


class ResultProcessor:
    """Processes Whisper transcription results and extracts metrics."""

    @staticmethod
    def calculate_confidence(segments: List[Dict]) -> float:
        """Calculate overall confidence from segment data."""
        if not segments:
            return 0.0

        confidences = []
        for segment in segments:
            avg_logprob = segment.get("avg_logprob", -1.0)
            no_speech_prob = segment.get("no_speech_prob", 0.5)

            # Calculate segment confidence
            segment_confidence = min(
                1.0, max(0.0, np.exp(avg_logprob) * (1 - no_speech_prob))
            )
            confidences.append(segment_confidence)

        return float(np.mean(confidences)) if confidences else 0.5

    @staticmethod
    def calculate_no_speech_probability(segments: List[Dict]) -> float:
        """Calculate overall no_speech_probability from segment data."""
        if not segments:
            # No segments means silence - return high probability
            return 0.9

        # Get no_speech_prob from each segment and return the maximum
        no_speech_probs = []
        for segment in segments:
            no_speech_prob = segment.get("no_speech_prob", 0.0)
            no_speech_probs.append(no_speech_prob)

        # Return the maximum no_speech_prob across all segments
        return float(max(no_speech_probs)) if no_speech_probs else 0.9

    @staticmethod
    def calculate_speaking_rate(word_count: int, duration: float) -> float:
        """Calculate speaking rate in words per minute."""
        if duration <= 0:
            return 0.0
        return (word_count / duration) * 60

    @staticmethod
    def create_error_result(error_message: str) -> TranscriptionResult:
        """Create an error result for failed transcriptions."""
        return TranscriptionResult(
            text=f"[ERROR: {error_message}]",
            language="unknown",
            confidence=0.0,
            duration=0.0,
            processing_time=0.0,
            segments=[],
            audio_quality="error",
            word_count=0,
            speaking_rate=0.0,
            no_speech_probability=1.0,
        )
