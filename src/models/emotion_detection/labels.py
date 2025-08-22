#!/usr/bin/env python3
# Centralized labels and mappings for GoEmotions

GOEMOTIONS_EMOTIONS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

EMOTION_ID_TO_LABEL = dict(enumerate(GOEMOTIONS_EMOTIONS))
EMOTION_LABEL_TO_ID = {emotion: i for i, emotion in enumerate(GOEMOTIONS_EMOTIONS)}
