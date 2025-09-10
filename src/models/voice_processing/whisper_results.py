#!/usr/bin/env python3
"""
SAMO Whisper Transcription Results Module

This module defines the data structures for transcription results
and provides utilities for result processing and analysis.
"""

import logging
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)


class TranscriptionResult:
    """Result of audio transcription."""
    
    def __init__(
        self,
        text: str,
        language: str,
        confidence: float,
        duration: float,
        processing_time: float,
        segments: List[Dict],
        audio_quality: str,
        word_count: int,
        speaking_rate: float,
        no_speech_probability: float
    ):
        self.text = text
        self.language = language
        self.confidence = confidence
        self.duration = duration
        self.processing_time = processing_time
        self.segments = segments
        self.audio_quality = audio_quality
        self.word_count = word_count
        self.speaking_rate = speaking_rate
        self.no_speech_probability = no_speech_probability
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary."""
        return {
            "text": self.text,
            "language": self.language,
            "confidence": self.confidence,
            "duration": self.duration,
            "processing_time": self.processing_time,
            "segments": self.segments,
            "audio_quality": self.audio_quality,
            "word_count": self.word_count,
            "speaking_rate": self.speaking_rate,
            "no_speech_probability": self.no_speech_probability,
        }
    
    def __str__(self) -> str:
        """String representation of the result."""
        return f"TranscriptionResult(text='{self.text[:50]}...', language='{self.language}', confidence={self.confidence:.3f})"


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
            segment_confidence = min(1.0, max(0.0, np.exp(avg_logprob) * (1 - no_speech_prob)))
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
