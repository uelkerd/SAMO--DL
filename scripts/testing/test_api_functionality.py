#!/usr/bin/env python3
"""
Test script to verify SAMO-DL API functionality without running a server.
This tests the core models and functions directly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from src.unified_ai_api import app
        print("✅ FastAPI app imports successfully")
        
        from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier
        print("✅ BERT emotion classifier imports successfully")
        
        from src.models.summarization.t5_summarizer import T5SummarizationModel
        print("✅ T5 summarizer imports successfully")
        
        from src.models.voice_processing.whisper_transcriber import WhisperTranscriber
        print("✅ Whisper transcriber imports successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_emotion_detection():
    """Test emotion detection functionality."""
    print("\n🧪 Testing emotion detection...")
    
    try:
        from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier
        
        # Create model
        model, tokenizer = create_bert_emotion_classifier()
        print("✅ BERT emotion model created successfully")
        
        # Test prediction
        test_text = "I am feeling absolutely wonderful and excited about today!"
        result = model.predict_emotions(test_text)
        print(f"✅ Emotion prediction successful: {result}")
        
        return True
    except Exception as e:
        print(f"❌ Emotion detection failed: {e}")
        return False

def test_text_summarization():
    """Test text summarization functionality."""
    print("\n🧪 Testing text summarization...")
    
    try:
        from src.models.summarization.t5_summarizer import T5SummarizationModel
        
        # Create summarizer
        summarizer = T5SummarizationModel()
        print("✅ T5 summarizer created successfully")
        
        # Test summarization
        test_text = "This is a very long text that should be properly summarized by the T5 model. " * 10
        summary = summarizer.generate_summary(test_text)
        print(f"✅ Text summarization successful: {summary[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Text summarization failed: {e}")
        return False

def test_voice_processing():
    """Test voice processing functionality."""
    print("\n🧪 Testing voice processing...")
    
    try:
        from src.models.voice_processing.whisper_transcriber import WhisperTranscriber
        
        # Create transcriber
        transcriber = WhisperTranscriber()
        print("✅ Whisper transcriber created successfully")
        
        # Note: We can't test actual transcription without audio files
        print("✅ Voice processing model loaded (transcription test skipped - no audio file)")
        
        return True
    except Exception as e:
        print(f"❌ Voice processing failed: {e}")
        return False

def test_api_routes():
    """Test that API routes are properly defined."""
    print("\n🧪 Testing API routes...")
    
    try:
        from src.unified_ai_api import app
        
        # Check for key routes
        routes = [route.path for route in app.routes if hasattr(route, 'path')]
        
        expected_routes = [
            '/health',
            '/analyze/journal',
            '/summarize/text',
            '/transcribe/voice',
            '/models/status'
        ]
        
        for route in expected_routes:
            if route in routes:
                print(f"✅ Route {route} found")
            else:
                print(f"❌ Route {route} missing")
                return False
        
        print("✅ All expected API routes found")
        return True
    except Exception as e:
        print(f"❌ API routes test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 SAMO-DL API Functionality Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_emotion_detection,
        test_text_summarization,
        test_voice_processing,
        test_api_routes
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API functionality is working correctly.")
        return 0
    print("⚠️ Some tests failed. Check the output above for details.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
