# Test script for SAMO T5 Summarization Model
"""
Standalone test for SAMO T5 Summarization Model

This script tests the T5 summarization model independently
to ensure it works correctly before integration.
"""

import sys
from pathlib import Path

# Add src to path - more robust approach
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / "src"))

try:
    from models.summarization.samo_t5_summarizer import create_samo_t5_summarizer
except ImportError:
    # Alternative import path
    from src.models.summarization.samo_t5_summarizer import create_samo_t5_summarizer

def test_summarizer_initialization():
    """Test summarizer initialization and model info."""
    print("1. Initializing SAMO T5 Summarizer...")
    cfg_path = str((Path(__file__).resolve().parent / "configs" / "samo_t5_config.yaml"))
    summarizer = create_samo_t5_summarizer(cfg_path)
    print("✅ Summarizer initialized successfully")
    
    # Test model info
    print("\n2. Checking model information...")
    model_info = summarizer.get_model_info()
    assert model_info['model_loaded'], "Model should be loaded"
    assert model_info['tokenizer_loaded'], "Tokenizer should be loaded"
    assert model_info['model_name'] == "t5-small", "Should use t5-small model"
    
    print(f"   Model: {model_info['model_name']}")
    print(f"   Device: {model_info['device']}")
    print(f"   Model loaded: {model_info['model_loaded']}")
    print(f"   Tokenizer loaded: {model_info['tokenizer_loaded']}")
    
    return summarizer


def test_single_summarization(summarizer):
    """Test single text summarization with assertions."""
    print("\n3. Testing text summarization...")
    test_text = """
    Today I had an amazing experience at the conference. I learned so much about
    AI and machine learning. The speakers were incredibly knowledgeable and the
    networking opportunities were fantastic. I met several people who share my
    passion for deep learning and we exchanged contact information. I'm feeling
    really excited about the future possibilities and can't wait to implement
    some of the techniques I learned. This has been one of the most productive
    days I've had in months.
    """
    
    result = summarizer.generate_summary(test_text)
    
    # Assertions instead of conditionals
    assert result["success"], f"Summarization failed: {result.get('error', 'Unknown error')}"
    assert "summary" in result, "Result should contain summary"
    assert isinstance(result["summary"], str), "Summary should be a string"
    assert result["summary"].strip(), "Summary should not be empty"
    assert result["original_length"] > 0, "Original length should be positive"
    assert result["summary_length"] > 0, "Summary length should be positive"
    assert 0 < result["compression_ratio"] < 1, "Compression ratio should be between 0 and 1"
    assert "emotional_keywords" in result, "Result should contain emotional keywords"
    assert isinstance(result["emotional_keywords"], list), "Emotional keywords should be a list"
    
    print("✅ Summarization successful!")
    print(f"   Original length: {result['original_length']} words")
    print(f"   Summary length: {result['summary_length']} words")
    print(f"   Compression ratio: {result['compression_ratio']:.2f}")
    print(f"   Processing time: {result['processing_time']:.2f}s")
    print(f"   Emotional keywords: {result['emotional_keywords']}")
    print(f"   Summary: {result['summary']}")


def test_batch_processing(summarizer):
    """Test batch processing with proper assertions."""
    print("\n4. Testing batch processing...")
    test_texts = [
        "I'm feeling really happy today because I accomplished my goals and I'm excited about the future possibilities that lie ahead.",
        "This has been a challenging week with many obstacles to overcome but I'm grateful for the lessons learned and the growth I've experienced.",
        "I'm grateful for all the support I've received from my friends and family during this difficult time and I know I can count on them."
    ]
    
    batch_results = summarizer.generate_batch_summaries(test_texts)
    
    # Assertions for batch processing
    assert len(batch_results) == len(test_texts), "Should return results for all inputs"
    
    successful_summaries = sum(r["success"] for r in batch_results)
    print(f"✅ Batch processing: {successful_summaries}/{len(test_texts)} successful")
    
    # Assert each summary is non-empty and emotional keywords are extracted
    for idx, result in enumerate(batch_results):
        assert result["success"], f"Batch summary {idx} failed"
        assert "summary" in result and isinstance(result["summary"], str) and result["summary"].strip(), f"Summary {idx} is empty"
        assert "emotional_keywords" in result and isinstance(result["emotional_keywords"], list), f"Emotional keywords missing for input {idx}"


def test_error_handling(summarizer):
    """Test error handling with individual test cases."""
    print("\n5. Testing error handling...")
    
    # Test empty text
    result = summarizer.generate_summary("")
    assert not result["success"], "Empty text should fail"
    assert "error" in result, "Error should be reported"
    print(f"   ✅ Empty text handled correctly: {result['error']}")
    
    # Test too short text
    result = summarizer.generate_summary("Short")
    assert not result["success"], "Short text should fail"
    assert "error" in result, "Error should be reported"
    print(f"   ✅ Short text handled correctly: {result['error']}")
    
    # Test too long text
    long_text = "word " * 1000  # Create text with 1000 words
    result = summarizer.generate_summary(long_text)
    assert not result["success"], "Long text should fail"
    assert "error" in result, "Error should be reported"
    print(f"   ✅ Long text handled correctly: {result['error']}")
    
    # Test wrong type
    result = summarizer.generate_summary(123)
    assert not result["success"], "Wrong type should fail"
    assert "error" in result, "Error should be reported"
    print(f"   ✅ Wrong type handled correctly: {result['error']}")


def test_samo_t5_summarizer():
    """Test the SAMO T5 summarizer functionality."""
    print("🧪 Testing SAMO T5 Summarization Model")
    print("=" * 50)
    
    try:
        # Run all test functions
        summarizer = test_summarizer_initialization()
        test_single_summarization(summarizer)
        test_batch_processing(summarizer)
        test_error_handling(summarizer)
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_samo_t5_summarizer()
    sys.exit(0 if success else 1)
