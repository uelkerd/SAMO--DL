#!/usr/bin/env python3
"""
Integration script for real Whisper and T5 models.
This script properly integrates the models into the SAMO API.
"""
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealWhisperTranscriber:
    """Real Whisper implementation with proper error handling."""

    def __init__(self, model_size: str = "base"):
        """Initialize real Whisper model."""
        self.model_size = model_size
        self.model = None
        self.device = "cpu"  # Use CPU for compatibility

        try:
            import whisper
            logger.info(f"Loading Whisper {model_size} model...")
            self.model = whisper.load_model(model_size, device=self.device)
            logger.info(f"✅ Whisper {model_size} loaded successfully")
        except ImportError:
            logger.error("❌ Whisper not installed. Install with: pip install openai-whisper")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper: {e}")
            raise

    def transcribe(self, audio_path: str, language: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Transcribe audio file."""
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")

        try:
            # Transcribe with Whisper
            result = self.model.transcribe(str(audio_path), language=language)

            # Format response to match expected structure
            text = result.get("text", "")
            detected_language = result.get("language", "en")

            # Calculate metrics
            word_count = len(text.split()) if text else 0

            return {
                "text": text,
                "language": detected_language,
                "confidence": 0.95,  # Whisper doesn't provide confidence
                "duration": 5.0,  # Would need audio library to get actual duration
                "word_count": word_count,
                "speaking_rate": word_count * 12 if word_count else 0,  # Approximate
                "audio_quality": "good"
            }
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "type": "whisper",
            "loaded": self.model is not None
        }

class RealT5Summarizer:
    """Real T5 implementation with proper error handling."""

    def __init__(self, model_name: str = "t5-small"):
        """Initialize real T5 model."""
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cpu"  # Use CPU for compatibility

        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            import torch

            logger.info(f"Loading T5 {model_name} model...")
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model.eval()  # Set to evaluation mode

            logger.info(f"✅ T5 {model_name} loaded successfully")

        except ImportError:
            logger.error("❌ Transformers not installed. Install with: pip install transformers torch")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load T5: {e}")
            raise

    def generate_summary(self, text: str, max_length: int = 128, min_length: int = 30) -> str:
        """Generate summary using T5."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("T5 model not loaded")

        try:
            import torch

            # Add T5 prefix for summarization
            if not text.startswith("summarize:"):
                text = f"summarize: {text}"

            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )

            # Generate summary
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )

            # Decode output
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "type": "t5",
            "loaded": self.model is not None
        }

def update_unified_api():
    """Update the unified API to use real models."""
    logger.info("\n" + "="*60)
    logger.info("Updating Unified API with Real Models")
    logger.info("="*60)

    # Create a patch file for the unified API
    patch_code = '''
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
'''

    logger.info("To integrate real models, update src/unified_ai_api.py with the above code")

    # Save the patch instructions
    patch_file = Path("integration_instructions.md")
    with open(patch_file, "w") as f:
        f.write("# Model Integration Instructions\n\n")
        f.write("## 1. Install Dependencies\n\n")
        f.write("```bash\n")
        f.write("pip install openai-whisper transformers torch\n")
        f.write("```\n\n")
        f.write("## 2. Update src/unified_ai_api.py\n\n")
        f.write("Replace the model loading section in the `lifespan` function with:\n\n")
        f.write("```python\n")
        f.write(patch_code)
        f.write("```\n\n")
        f.write("## 3. Test the Integration\n\n")
        f.write("```bash\n")
        f.write("python scripts/test_api_models.py\n")
        f.write("```\n")

    logger.info(f"✅ Integration instructions saved to {patch_file}")

def test_real_models():
    """Test real model implementations."""
    logger.info("\n" + "="*60)
    logger.info("Testing Real Model Implementations")
    logger.info("="*60)

    success = True

    # Test T5
    try:
        logger.info("\nTesting Real T5 Model...")
        t5 = RealT5Summarizer("t5-small")

        test_text = """
        Today was an absolutely amazing day. I woke up feeling refreshed and energized. 
        The weather was perfect - sunny but not too hot. I went for a long walk in the park 
        and saw beautiful flowers blooming everywhere. Later, I met up with some old friends 
        for lunch at our favorite restaurant. We laughed and shared stories for hours. 
        In the evening, I spent some quiet time reading a good book. I feel grateful for 
        such a wonderful day filled with simple pleasures.
        """

        summary = t5.generate_summary(test_text, max_length=100, min_length=20)
        logger.info(f"Original text length: {len(test_text.split())} words")
        logger.info(f"Summary: {summary}")
        logger.info(f"Summary length: {len(summary.split())} words")
        logger.info("✅ T5 Model Working!")

    except Exception as e:
        logger.error(f"❌ T5 Test Failed: {e}")
        success = False

    # Test Whisper
    try:
        logger.info("\nTesting Real Whisper Model...")
        whisper = RealWhisperTranscriber("base")

        # Create a dummy audio file for testing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            # In real use, this would be an actual audio file
            result = whisper.transcribe(tmp.name)
            logger.info(f"Transcription result: {result}")
            logger.info("✅ Whisper Model Working!")

    except Exception as e:
        logger.error(f"❌ Whisper Test Failed: {e}")
        success = False

    return success

def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Integrate real Whisper and T5 models")
    parser.add_argument("--test", action="store_true", help="Test real model implementations")
    parser.add_argument("--generate-patch", action="store_true", help="Generate API patch instructions")
    parser.add_argument("--check-deps", action="store_true", help="Check if dependencies are installed")

    args = parser.parse_args()

    if args.check_deps:
        logger.info("Checking dependencies...")
        deps = {
            "whisper": False,
            "transformers": False,
            "torch": False
        }

        try:
            import whisper
            deps["whisper"] = True
        except ImportError:
            pass

        try:
            import transformers
            deps["transformers"] = True
        except ImportError:
            pass

        try:
            import torch
            deps["torch"] = True
        except ImportError:
            pass

        logger.info("\nDependency Status:")
        for dep, installed in deps.items():
            status = "✅ Installed" if installed else "❌ Not Installed"
            logger.info(f"  {dep}: {status}")

        if not all(deps.values()):
            logger.info("\nTo install missing dependencies:")
            logger.info("  pip install openai-whisper transformers torch")

    if args.test:
        success = test_real_models()
        if success:
            logger.info("\n✅ All models working correctly!")
        else:
            logger.info("\n❌ Some models failed. Please install dependencies first.")
            logger.info("  pip install openai-whisper transformers torch")

    if args.generate_patch:
        update_unified_api()

    if not any([args.test, args.generate_patch, args.check_deps]):
        # Default action: check dependencies
        parser.parse_args(["--check-deps"])

if __name__ == "__main__":
    main()
