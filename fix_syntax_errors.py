#!/usr/bin/env python3
"""Automated syntax error fixer for SAMO codebase.
Fixes the most common syntax patterns found in DeepSource analysis.
"""

import os
import re


def fix_missing_parentheses(content):
    """Fix missing parentheses in function calls like logger.info"text" -> logger.info("text")"""
    # Pattern: word.word"text" or word.word'text'
    pattern = r'(\w+)\.(\w+)([\'"])'

    def replacement(match):
        object_name = match.group(1)
        method_name = match.group(2)
        quote = match.group(3)

        # Common function/method calls that need parentheses
        function_methods = [
            "info",
            "debug",
            "warning",
            "error",
            "critical",
            "log",
            "print",
            "append",
            "download",
            "filterwarnings",
        ]
        if method_name in function_methods:
            return f"{object_name}.{method_name}({quote}"
        return match.group(0)

    return re.sub(pattern, replacement, content)


def fix_unterminated_strings(content):
    """Fix basic unterminated string patterns"""
    lines = content.split("\n")
    fixed_lines = []

    for i, line in enumerate(lines):
        # Look for unterminated strings that end with a period at end of line
        if line.strip().endswith(".") and ('"' in line or "'" in line):
            # Check if this looks like a broken string literal
            if '= f"' in line and line.count('"') == 1:
                # This might be an f-string that got broken
                next_line_idx = i + 1
                if next_line_idx < len(lines) and lines[next_line_idx].strip().endswith('"'):
                    # Merge the lines
                    merged = line.rstrip(".") + lines[next_line_idx].strip()
                    fixed_lines.append(merged)
                    lines[next_line_idx] = ""  # Mark for skipping
                    continue

        if line:  # Only add non-empty lines (skip marked lines)
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_file(filepath):
    """Fix syntax errors in a single file"""
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Apply fixes
        content = fix_missing_parentheses(content)
        # content = fix_unterminated_strings(content)  # This one needs more care, skip for now

        if content != original_content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"❌ Error fixing {filepath}: {e}")
        return False


def main():
    # Focus on critical src files first
    critical_files = [
        "./src/models/secure_loader/integrity_checker.py",
        "./src/models/voice_processing/whisper_transcriber.py",
        "./src/models/voice_processing/transcription_api.py",
        "./src/monitoring/dashboard.py",
    ]

    fixed_count = 0
    for filepath in critical_files:
        if os.path.exists(filepath):
            if fix_file(filepath):
                print(f"✅ Fixed: {filepath}")
                fixed_count += 1
            else:
                print(f"⚪ No changes needed: {filepath}")
        else:
            print(f"⚠️  File not found: {filepath}")

    print(f"\n📊 Fixed {fixed_count} files")


if __name__ == "__main__":
    main()
