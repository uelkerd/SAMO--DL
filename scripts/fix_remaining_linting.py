#!/usr/bin/env python3
"""
Quick fix for remaining critical linting issues.
"""

import re
from pathlib import Path

def fix_file(file_path: str, fixes: list):
    """Apply fixes to a file."""
    with open(file_path) as f:
        content = f.read()
    
    original_content = content
    
    for pattern, replacement, flags in fixes:
        content = re.sub(pattern, replacement, content, flags=flags)
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"✅ Fixed: {file_path}")
        return True
    return False

def main():
    """Fix remaining critical linting issues."""
    
    # Fix B007: Loop control variable not used
    fixes = [
        # Fix loop control variables
        (r'for i in range\((\d+)\):', r'for _ in range(\1):', re.MULTILINE),
        (r'for j in range\((\d+)\):', r'for _ in range(\1):', re.MULTILINE),
        (r'for k in range\((\d+)\):', r'for _ in range(\1):', re.MULTILINE),
        
        # Fix trailing whitespace
        (r'[ \t]+$', '', re.MULTILINE),
        
        # Fix print statements (replace with logging)
        (r'print\(([^)]+)\)', r'logger.info(\1)', re.MULTILINE),
    ]
    
    # Files to fix
    files_to_fix = [
        'src/models/emotion_detection/api_demo.py',
        'src/models/emotion_detection/bert_classifier.py',
        'src/models/emotion_detection/dataset_loader.py',
        'src/models/emotion_detection/training_pipeline.py',
        'src/models/summarization/api_demo.py',
        'src/models/summarization/dataset_loader.py',
        'src/models/summarization/t5_summarizer.py',
        'src/models/summarization/training_pipeline.py',
        'src/models/voice_processing/api_demo.py',
        'src/models/voice_processing/audio_preprocessor.py',
    ]
    
    fixed_count = 0
    for file_path in files_to_fix:
        if Path(file_path).exists():
            if fix_file(file_path, fixes):
                fixed_count += 1
    
    print(f"\n🎉 Fixed {fixed_count} files")

if __name__ == "__main__":
    main()
