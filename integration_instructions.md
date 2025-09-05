# Model Integration Instructions

## 1. Install Dependencies

```bash
pip install openai-whisper transformers torch
```

## 2. Update src/unified_ai_api.py

Replace the model loading section in the `lifespan` function with:

```python

# Add this to src/unified_ai_api.py in the startup function

# Replace the model loading section with:

        logger.info("Loading text summarization model...")
        try:
            # Try to load real T5 model first
            from scripts.integrate_real_models import RealT5Summarizer
            text_summarizer = RealT5Summarizer("t5-small")
            logger.info("Real T5 summarization model loaded")
        except Exception as exc:
            logger.warning(f"Real T5 not available: {exc}")
            # Fall back to existing implementation
            try:
                from src.models.summarization.t5_summarizer import create_t5_summarizer
                text_summarizer = create_t5_summarizer("t5-small")
                logger.info("Text summarization model loaded (fallback)")
            except Exception as exc2:
                logger.warning(f"Text summarization model not available: {exc2}")

        logger.info("Loading voice processing model...")
        try:
            # Try to load real Whisper model first
            from scripts.integrate_real_models import RealWhisperTranscriber
            voice_transcriber = RealWhisperTranscriber()
            logger.info("Real Whisper voice processing model loaded")
        except Exception as exc:
            logger.warning(f"Real Whisper not available: {exc}")
            # Fall back to existing implementation
            try:
                from src.models.voice_processing.whisper_transcriber import create_whisper_transcriber
                voice_transcriber = create_whisper_transcriber()
                logger.info("Voice processing model loaded (fallback)")
            except Exception as exc2:
                logger.warning(f"Voice processing model not available: {exc2}")
```

## 3. Test the Integration

```bash
python scripts/test_api_models.py
```
