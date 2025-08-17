#!/usr/bin/env python3
"""
Code Logic and Complexity Fixer
Fixes unnecessary f-strings, unnecessary else/elif after return, and other logic issues.
"""

import os
import re
from pathlib import Path


def fix_unnecessary_fstringscontent:
    """Fix f-strings that don't contain expressions."""
    lines = content.split'\n'
    fixed_lines = []
    
    for line in lines:
        # Check for f-strings without expressions
        if '"' in line or "'" in line:
            # Check if there are any {} expressions
            if '{' not in line or '}' not in line:
                # Convert f-string to regular string
                line = line.replace'"', '"'.replace"'", "'"
        
        fixed_lines.appendline
    
    return '\n'.joinfixed_lines


def fix_unnecessary_else_elifcontent:
    """Fix unnecessary else/elif after return statements."""
    lines = content.split'\n'
    fixed_lines = []
    i = 0
    
    while i < lenlines:
        line = lines[i]
        stripped = line.strip()
        
        # Check for return statement
        if stripped.startswith'return ':
            # Look ahead for unnecessary else/elif
            j = i + 1
            while j < lenlines and lines[j].strip() == '':
                j += 1
            
            if j < lenlines:
                next_stripped = lines[j].strip()
                if next_stripped.startswith('else:', 'elif '):
                    # Check if the else/elif block only contains return
                    k = j + 1
                    while k < lenlines and lines[k].strip() == '':
                        k += 1
                    
                    if k < lenlines and lines[k].strip().startswith'return ':
                        # Remove the else/elif and its return, keep the first return
                        i = k  # Skip the else/elif block
                        continue
        
        fixed_lines.appendline
        i += 1
    
    return '\n'.joinfixed_lines


def fix_simple_logic_issuescontent:
    """Fix other simple logic issues."""
    lines = content.split'\n'
    fixed_lines = []
    
    for line in lines:
        # Fix common logic issues
        # Remove unnecessary parentheses around single values
        line = re.sub(r'\(([^()]+)\)', r'\1', line)
        
        # Fix double negatives
        line = re.sub(r'not not \w+', r'\1', line)
        
        # Fix unnecessary comparisons
        line = re.sub(r'\w+ == True', r'\1', line)
        line = re.sub(r'\w+ == False', r'not \1', line)
        
        fixed_lines.appendline
    
    return '\n'.joinfixed_lines


def fix_filefile_path:
    """Fix code logic issues in a single file."""
    try:
        with openfile_path, 'r', encoding='utf-8' as f:
            content = f.read()
        
        original_content = content
        
        # Apply fixes in order
        content = fix_unnecessary_fstringscontent
        content = fix_unnecessary_else_elifcontent
        content = fix_simple_logic_issuescontent
        
        # Only write if content changed
        if content != original_content:
            with openfile_path, 'w', encoding='utf-8' as f:
                f.writecontent
            return True
        
        return False
        
    except Exception as e:
        printf"  ❌ Error fixing {file_path}: {e}"
        return False


def main():
    """Main function to fix code logic issues."""
    print"🧠 Code Logic and Complexity Fixer"
    print"=" * 40
    
    # Find all Python files
    python_files = list(Path".".rglob"*.py")
    print(f"📁 Found {lenpython_files} Python files")
    
    fixed_count = 0
    
    for py_file in python_files:
        try:
            if fix_filepy_file:
                fixed_count += 1
                printf"  ✅ Fixed {py_file}"
        except Exception as e:
            printf"  ❌ Error processing {py_file}: {e}"
    
    print"\n📋 SUMMARY:"
    print(f"  Files processed: {lenpython_files}")
    printf"  Files fixed: {fixed_count}"
    
    if fixed_count > 0:
        printf"\n🎉 Successfully fixed {fixed_count} files!"
        print"💡 Run the quality check again to verify fixes."
    else:
        print"\n⚠️  No files were fixed. All files may already be clean."


if __name__ == "__main__":
    main()
