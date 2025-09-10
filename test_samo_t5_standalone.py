#!/usr/bin/env python3
"""
Standalone test for SAMO T5 Summarization Model

This script tests the T5 summarization model independently
to ensure it works correctly before integration.
"""

import sys
import os
from pathlib import Path

# Add src to path - more robust approach
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))
else:
    # Fallback for different directory structures
    sys.path.insert(0, str(current_dir))

try:
    from models.summarization.samo_t5_summarizer import create_samo_t5_summarizer
except ImportError:
    # Alternative import path
    from src.models.summarization.samo_t5_summarizer import create_samo_t5_summarizer

def test_samo_t5_summarizer():
    """Test the SAMO T5 summarizer functionality."""
    print("🧪 Testing SAMO T5 Summarization Model")
    print("=" * 50)
    
    try:
        # Initialize summarizer
        print("1. Initializing SAMO T5 Summarizer...")
        summarizer = create_samo_t5_summarizer("configs/samo_t5_config.yaml")
        print("✅ Summarizer initialized successfully")
        
        # Test model info
        print("\n2. Checking model information...")
        model_info = summarizer.get_model_info()
        print(f"   Model: {model_info['model_name']}")
        print(f"   Device: {model_info['device']}")
        print(f"   Model loaded: {model_info['model_loaded']}")
        print(f"   Tokenizer loaded: {model_info['tokenizer_loaded']}")
        
        # Test summarization
        print("\n3. Testing text summarization...")
        test_text = """
        Today I had an amazing experience at the conference. I learned so much about AI and machine learning.
        The speakers were incredibly knowledgeable and the networking opportunities were fantastic. I met
        several people who share my passion for deep learning and we exchanged contact information. I'm
        feeling really excited about the future possibilities and can't wait to implement some of the
        techniques I learned. This has been one of the most productive days I've had in months.
        """
        
        result = summarizer.generate_summary(test_text)
        
        if result["success"]:
            print("✅ Summarization successful!")
            print(f"   Original length: {result['original_length']} words")
            print(f"   Summary length: {result['summary_length']} words")
            print(f"   Compression ratio: {result['compression_ratio']:.2f}")
            print(f"   Processing time: {result['processing_time']:.2f}s")
            print(f"   Emotional keywords: {result['emotional_keywords']}")
            print(f"   Summary: {result['summary']}")
        else:
            print(f"❌ Summarization failed: {result['error']}")
            return False
        
        # Test batch processing
        print("\n4. Testing batch processing...")
        test_texts = [
            "I'm feeling really happy today because I accomplished my goals and I'm excited about the future possibilities that lie ahead.",
            "This has been a challenging week with many obstacles to overcome but I'm grateful for the lessons learned and the growth I've experienced.",
            "I'm grateful for all the support I've received from my friends and family during this difficult time and I know I can count on them."
        ]
        
        batch_results = summarizer.generate_batch_summaries(test_texts)
        successful_summaries = sum(r["success"] for r in batch_results)
        
        print(f"✅ Batch processing: {successful_summaries}/{len(test_texts)} successful")

        # Assert each summary is non-empty and emotional keywords are extracted
        for idx, result in enumerate(batch_results):
            assert result["success"], f"Batch summary {idx} failed"
            assert "summary" in result and isinstance(result["summary"], str) and result["summary"].strip(), f"Summary {idx} is empty"
            assert "emotional_keywords" in result and isinstance(result["emotional_keywords"], list) and result["emotional_keywords"], f"Emotional keywords missing or empty for input {idx}"
        
        # Test error handling
        print("\n5. Testing error handling...")
        error_cases = [
            "",  # Empty text
            "Short",  # Too short
            "A" * 2000,  # Too long
            123  # Wrong type
        ]
        
        for i, error_text in enumerate(error_cases):
            result = summarizer.generate_summary(error_text)
            if not result["success"]:
                print(f"   ✅ Error case {i+1} handled correctly: {result['error']}")
            else:
                print(f"   ❌ Error case {i+1} not handled properly")
        
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
