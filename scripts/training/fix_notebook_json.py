#!/usr/bin/env python3
"""Fix JSON syntax error in expanded dataset training notebook."""

import re

def fix_notebook_json():
    """Fix JSON syntax errors in the notebook."""

    # Read the notebook as text
    with open('notebooks/expanded_dataset_training.ipynb', 'r') as f:
        content = f.read()

    # Fix unescaped quotes in strings
    # Replace "I'm" with "I\\'m" and similar patterns
    content = re.sub(r'"I\'m', r'"I\\\'m', content)
    content = re.sub(r'"I\'ve', r'"I\\\'ve', content)
    content = re.sub(r'"I\'ll', r'"I\\\'ll', content)
    content = re.sub(r'"I\'d', r'"I\\\'d', content)
    content = re.sub(r'"don\'t', r'"don\\\'t', content)
    content = re.sub(r'"can\'t', r'"can\\\'t', content)
    content = re.sub(r'"won\'t', r'"won\\\'t', content)
    content = re.sub(r'"isn\'t', r'"isn\\\'t', content)
    content = re.sub(r'"aren\'t', r'"aren\\\'t', content)
    content = re.sub(r'"doesn\'t', r'"doesn\\\'t', content)
    content = re.sub(r'"haven\'t', r'"haven\\\'t', content)
    content = re.sub(r'"hasn\'t', r'"hasn\\\'t', content)
    content = re.sub(r'"hadn\'t', r'"hadn\\\'t', content)
    content = re.sub(r'"wouldn\'t', r'"wouldn\\\'t', content)
    content = re.sub(r'"couldn\'t', r'"couldn\\\'t', content)
    content = re.sub(r'"shouldn\'t', r'"shouldn\\\'t', content)
    content = re.sub(r'"mightn\'t', r'"mightn\\\'t', content)
    content = re.sub(r'"mustn\'t', r'"mustn\\\'t', content)

    # Fix other common contractions
    content = re.sub(r'"(\w+)\'(\w+)"', r'"\\1\\\'\\2"', content)

    # Write the fixed content
    with open('notebooks/expanded_dataset_training_fixed.ipynb', 'w') as f:
        f.write(content)

    print("✅ Fixed notebook saved as 'notebooks/expanded_dataset_training_fixed.ipynb'")

    # Test if the JSON is valid
    try:
        import json
        with open('notebooks/expanded_dataset_training_fixed.ipynb', 'r') as f:
            json.load(f)
        print("✅ JSON syntax is now valid")
    except Exception as e:
        print(f"❌ JSON still has issues: {e}")

if __name__ == "__main__":
    fix_notebook_json()
