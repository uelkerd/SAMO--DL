#!/usr/bin/env python3
"""
Script to ensure real Whisper and T5 models are properly integrated.
This script provides fallback implementations if the actual models can't load.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperFallback:
    """Fallback implementation if real Whisper can't load."""

    def __init__(self, model_size="base"):
        self.model_size = model_size
        logger.warning("Using Whisper fallback implementation (real model not available)")

    @staticmethod
    def transcribe(audio_path, language=None, **kwargs):
        """Provide mock transcription."""
        return {
            "text": f"[Mock transcription of audio file: {Path(audio_path).name}]",
            "language": language or "en",
            "confidence": 0.95,
            "duration": 5.0,
            "word_count": 10,
            "speaking_rate": 120.0,
            "audio_quality": "good"
        }

    def get_model_info(self):
        return {
            "model_size": self.model_size,
            "type": "fallback",
            "status": "mock_mode"
        }

class T5Fallback:
    """Fallback implementation if real T5 can't load."""

    def __init__(self, model_name="t5-small"):
        self.model_name = model_name
        logger.warning("Using T5 fallback implementation (real model not available)")

    @staticmethod
    def generate_summary(text, max_length=128, min_length=30):
        """Provide mock summary."""
        # Simple extractive summary - take first and last sentences
        sentences = text.split('. ')
        if len(sentences) > 2:
            summary = f"{sentences[0]}. {sentences[-1]}"
        else:
            summary = text[:min(len(text), max_length)]

        return summary[:max_length]

    def get_model_info(self):
        return {
            "model_name": self.model_name,
            "type": "fallback",
            "status": "mock_mode"
        }

def create_enhanced_whisper_transcriber(model_size="base", **kwargs):
    """Create Whisper transcriber with fallback support."""
    try:
        # Try to import and use real Whisper
        import whisper
        from src.models.voice_processing.whisper_transcriber import WhisperTranscriber, TranscriptionConfig

        logger.info("Loading real Whisper model...")
        config = TranscriptionConfig(model_size=model_size, **kwargs)
        transcriber = WhisperTranscriber(config)
        logger.info(f"✅ Real Whisper {model_size} model loaded successfully")
        return transcriber

    except ImportError as e:
        logger.warning(f"Whisper not installed: {e}")
        logger.info("Install with: pip install openai-whisper")
        return WhisperFallback(model_size)

    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        return WhisperFallback(model_size)

def create_enhanced_t5_summarizer(model_name="t5-small", **kwargs):
    """Create T5 summarizer with fallback support."""
    try:
        # Try to import and use real T5
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        from src.models.summarization.t5_summarizer import T5SummarizationModel, SummarizationConfig

        logger.info("Loading real T5 model...")
        config = SummarizationConfig(model_name=model_name, **kwargs)
        model = T5SummarizationModel(config)
        logger.info(f"✅ Real T5 {model_name} model loaded successfully")
        return model

    except ImportError as e:
        logger.warning(f"Transformers not installed: {e}")
        logger.info("Install with: pip install transformers torch")
        return T5Fallback(model_name)

    except Exception as e:
        logger.error(f"Failed to load T5 model: {e}")
        return T5Fallback(model_name)

def patch_unified_api():
    """Patch the unified API to use enhanced model loaders."""
    logger.info("Patching unified API to ensure model loading...")

    try:
        import src.unified_ai_api as api

        # Patch the model loading in the API
        original_startup = api.lifespan

        async def enhanced_lifespan(app):
            """Enhanced startup that ensures models load."""
            # Load models with fallbacks
            api.voice_transcriber = create_enhanced_whisper_transcriber()
            api.text_summarizer = create_enhanced_t5_summarizer()

            logger.info("Models loaded (with fallbacks if needed):")
            logger.info(f"- Voice Transcriber: {api.voice_transcriber.get_model_info()}")
            logger.info(f"- Text Summarizer: {api.text_summarizer.get_model_info()}")

            # Call original startup
            async for _ in original_startup(app):
                yield

        api.lifespan = enhanced_lifespan
        logger.info("✅ API patched successfully")

    except Exception as e:
        logger.error(f"Failed to patch API: {e}")

def test_models():
    """Test the enhanced model loaders."""
    logger.info("\n" + "="*60)
    logger.info("Testing Enhanced Model Loaders")
    logger.info("="*60)

    # Test Whisper
    logger.info("\nTesting Whisper loader...")
    whisper = create_enhanced_whisper_transcriber("base")
    logger.info(f"Whisper info: {whisper.get_model_info()}")

    # Test with dummy file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        try:
            result = whisper.transcribe(tmp.name)
            if isinstance(result, dict):
                logger.info(f"Transcription result: {result.get('text', 'N/A')}")
            else:
                logger.info(f"Transcription result: {result.text if hasattr(result, 'text') else result}")
        except Exception as e:
            logger.warning(f"Transcription test failed: {e}")

    # Test T5
    logger.info("\nTesting T5 loader...")
    t5 = create_enhanced_t5_summarizer("t5-small")
    logger.info(f"T5 info: {t5.get_model_info()}")

    # Test summarization
    test_text = "I had a wonderful day today. The weather was perfect and I spent time with my family. We went to the park and had a picnic."
    try:
        summary = t5.generate_summary(test_text)
        logger.info(f"Summary: {summary}")
    except Exception as e:
        logger.warning(f"Summarization test failed: {e}")

    logger.info("\n" + "="*60)
    logger.info("Test Complete")
    logger.info("="*60)

def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Ensure real models are loaded")
    parser.add_argument("--patch-api", action="store_true", help="Patch the unified API")
    parser.add_argument("--test", action="store_true", help="Test model loading")

    args = parser.parse_args()

    if args.test:
        test_models()

    if args.patch_api:
        patch_unified_api()

    if not args.test and not args.patch_api:
        # Default: test models
        test_models()
        logger.info("\nTo patch the API, run: python scripts/ensure_real_models.py --patch-api")

if __name__ == "__main__":
    main()
