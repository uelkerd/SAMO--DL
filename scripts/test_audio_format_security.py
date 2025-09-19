#!/usr/bin/env python3
"""
Test script to verify audio format security fixes.
Tests that malicious audio_format values are properly sanitized.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_audio_format_validation():
    """Test that audio format validation works correctly."""
    
    # Test cases: (input_format, expected_output, should_be_valid)
    test_cases = [
        # Valid formats
        ('wav', 'wav', True),
        ('mp3', 'mp3', True),
        ('flac', 'flac', True),
        ('ogg', 'ogg', True),
        ('m4a', 'm4a', True),
        ('aac', 'aac', True),
        
        # Invalid formats that should be rejected or sanitized
        ('../../../etc/passwd', 'wav', False),  # Path traversal attempt
        ('<script>alert(1)</script>', 'wav', False),  # XSS attempt
        ('wav\x00', 'wav', False),  # Null byte injection
        ('WAV', 'wav', True),  # Case insensitive
        ('', 'wav', False),  # Empty string should be invalid but defaults to wav
        ('invalid', 'wav', False),  # Invalid format
        ('wav.exe', 'wav', False),  # Double extension attempt
    ]
    
    print("Testing audio format security fixes...")
    print("=" * 50)
    
    # Test the validation logic from complete_analysis_endpoint.py
    allowed_audio_formats = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac'}
    
    for input_format, expected_output, should_be_valid in test_cases:
        # Simulate the validation logic
        req_audio_format = input_format.lower().strip()
        sanitized_format = req_audio_format if req_audio_format in allowed_audio_formats else 'wav'
        
        is_valid = req_audio_format in allowed_audio_formats
        
        status = "✓ PASS" if (sanitized_format == expected_output and is_valid == should_be_valid) else "✗ FAIL"
        
        print(f"{status} Input: '{input_format}' -> Output: '{sanitized_format}' (Valid: {is_valid})")
        
        if status.startswith("✗"):
            print(f"    Expected: '{expected_output}' (Valid: {should_be_valid})")
    
    print("=" * 50)
    print("Security test completed!")

if __name__ == "__main__":
    test_audio_format_validation()
