#!/usr/bin/env python3
"""
Quick quality check script to assess current code quality status.
This bypasses the complex pre-commit system for a simple assessment.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_simple_check():
    """Run a simple quality check using basic tools."""
    print("🔍 Quick Quality Check - Current Status")
    print("=" * 50)
    
    # Check Python syntax
    print("\n📝 Python Syntax Check:")
    python_files = list(Path(".").rglob("*.py"))
    syntax_errors = 0
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                compile(f.read(), str(py_file), 'exec')
        except SyntaxError as e:
            print(f"  ❌ {py_file}:{e.lineno} - {e.msg}")
            syntax_errors += 1
        except Exception as e:
            print(f"  ⚠️  {py_file}: Error reading file - {e}")
    
    if syntax_errors == 0:
        print("  ✅ All Python files have valid syntax")
    
    # Check for common issues
    print(f"\n📊 Files Analyzed: {len(python_files)}")
    print(f"🚨 Syntax Errors: {syntax_errors}")
    
    # Check for trailing whitespace and other simple issues
    print("\n🔍 Common Formatting Issues:")
    trailing_whitespace = 0
    missing_newlines = 0
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines, 1):
                    if line.rstrip() != line and line.strip():
                        trailing_whitespace += 1
                
                if lines and not lines[-1].endswith('\n'):
                    missing_newlines += 1
        except Exception:
            continue
    
    print(f"  Trailing whitespace: {trailing_whitespace}")
    print(f"  Missing newlines: {missing_newlines}")
    
    # Summary
    print("\n📋 SUMMARY:")
    if syntax_errors == 0 and trailing_whitespace == 0 and missing_newlines == 0:
        print("  🎉 EXCELLENT! All basic quality checks passed!")
    elif syntax_errors == 0:
        print("  ✅ Good! Syntax is clean, minor formatting issues remain")
    else:
        print("  ❌ Critical syntax errors need immediate attention")
    
    return syntax_errors == 0


if __name__ == "__main__":
    success = run_simple_check()
    sys.exit(0 if success else 1)
