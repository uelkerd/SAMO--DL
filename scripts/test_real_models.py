#!/usr/bin/env python3
"""Test script to verify real Whisper and T5 models are loading correctly."""

import sys
import logging
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_whisper_model():
    """Test loading and using real Whisper model."""
    logger.info("\n" + "="*60)
    logger.info("Testing Whisper Model Loading")
    logger.info("="*60)
    
    try:
        import whisper
        logger.info("✅ Whisper library imported successfully")
        
        # Load the model
        logger.info("Loading Whisper 'base' model...")
        model = whisper.load_model("base")
        logger.info("✅ Whisper model loaded successfully")
        
        # Test with dummy audio
        logger.info("Testing transcription with dummy audio...")
        # Create a short silent audio array (1 second at 16kHz)
        audio = np.zeros(16000, dtype=np.float32)
        
        # Transcribe
        result = model.transcribe(audio)
        logger.info("✅ Transcription completed")
        logger.info(f"Result text: '{result['text']}'")
        logger.info(f"Language: {result.get('language', 'unknown')}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import whisper: {e}")
        logger.error("Install with: pip install openai-whisper")
        return False
    except Exception as e:
        logger.error(f"❌ Whisper test failed: {e}")
        return False

def test_t5_model():
    """Test loading and using real T5 model."""
    logger.info("\n" + "="*60)
    logger.info("Testing T5 Model Loading")
    logger.info("="*60)
    
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        import torch
        logger.info("✅ Transformers library imported successfully")
        
        # Load the model and tokenizer
        logger.info("Loading T5-small model...")
        model_name = "t5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        logger.info("✅ T5 model and tokenizer loaded successfully")
        
        # Test with sample text
        logger.info("Testing summarization...")
        text = "summarize: The quick brown fox jumps over the lazy dog. This is a test sentence for summarization."
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate summary
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=50,
                min_length=10,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("✅ Summarization completed")
        logger.info(f"Summary: '{summary}'")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import transformers: {e}")
        logger.error("Install with: pip install transformers torch")
        return False
    except Exception as e:
        logger.error(f"❌ T5 test failed: {e}")
        return False

def test_model_integration():
    """Test the actual model creation functions from our codebase."""
    logger.info("\n" + "="*60)
    logger.info("Testing Model Integration")
    logger.info("="*60)
    
    success = True
    
    # Test Whisper integration
    try:
        logger.info("\nTesting Whisper integration...")
        from src.models.voice_processing.whisper_transcriber import create_whisper_transcriber
        
        transcriber = create_whisper_transcriber(model_size="base")
        logger.info("✅ Whisper transcriber created successfully")
        logger.info(f"Model info: {transcriber.get_model_info()}")
        
    except Exception as e:
        logger.error(f"❌ Whisper integration failed: {e}")
        success = False
    
    # Test T5 integration
    try:
        logger.info("\nTesting T5 integration...")
        from src.models.summarization.t5_summarizer import create_t5_summarizer
        
        summarizer = create_t5_summarizer(model_name="t5-small")
        logger.info("✅ T5 summarizer created successfully")
        logger.info(f"Model info: {summarizer.get_model_info()}")
        
        # Test actual summarization
        test_text = "I had a wonderful day today. The weather was perfect and I spent time with my family. We went to the park and had a picnic. The kids played on the swings while we relaxed on the blanket. It was one of those perfect moments that I want to remember forever."
        
        summary = summarizer.generate_summary(test_text)
        logger.info(f"Generated summary: '{summary}'")
        
    except Exception as e:
        logger.error(f"❌ T5 integration failed: {e}")
        success = False
    
    return success

def main():
    """Run all tests."""
    logger.info("🚀 Starting Real Model Tests")
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    results = {
        "Whisper": test_whisper_model(),
        "T5": test_t5_model(),
        "Integration": test_model_integration()
    }
    
    logger.info("\n" + "="*60)
    logger.info("Test Results Summary")
    logger.info("="*60)
    for test, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test}: {status}")
    
    if all(results.values()):
        logger.info("\n🎉 All tests passed! Models are ready for integration.")
    else:
        logger.info("\n⚠️ Some tests failed. Please install missing dependencies.")
        logger.info("\nTo install all dependencies:")
        logger.info("  pip install openai-whisper transformers torch")
    
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())