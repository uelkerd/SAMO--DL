#!/usr/bin/env python3
"""
Line Length and Style Fixer
Fixes line length violations and ensures consistent code style.
"""

import os
import re
from pathlib import Path


def fix_long_lines(content, max_length=88):
    """Fix lines that exceed the maximum length."""
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        if len(line) > max_length:
            # Try to break long lines intelligently
            fixed_line = break_long_line(line, max_length)
            fixed_lines.extend(fixed_line)
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def break_long_line(line, max_length):
    """Break a long line into multiple lines."""
    if len(line) <= max_length:
        return [line]
    
    # Check for different break patterns
    # 1. Function calls with parentheses
    if '(' in line and ')' in line:
        return break_function_call(line, max_length)
    
    # 2. Long string literals
    if '"' in line or "'" in line:
        return break_string_literal(line, max_length)
    
    # 3. Long assignments
    if '=' in line:
        return break_assignment(line, max_length)
    
    # 4. Long method chains
    if '.' in line:
        return break_method_chain(line, max_length)
    
    # 5. Default: break at spaces
    return break_at_spaces(line, max_length)


def break_function_call(line, max_length):
    """Break long function calls."""
    # Find the opening parenthesis
    open_paren = line.find('(')
    if open_paren == -1:
        return break_at_spaces(line, max_length)
    
    # Check if we can break after the function name
    if open_paren < max_length:
        # Break after function name
        func_part = line[:open_paren + 1]
        args_part = line[open_paren + 1:]
        
        if args_part.endswith(')'):
            args_part = args_part[:-1]
            closing = ')'
        else:
            closing = ''
        
        # Split arguments by comma
        args = [arg.strip() for arg in args_part.split(',')]
        
        result = [func_part]
        current_line = ' ' * (open_paren + 1)  # Indent to match opening
        
        for i, arg in enumerate(args):
            if i == 0:
                result.append(current_line + arg)
            else:
                result.append(current_line + arg)
            
            if i < len(args) - 1:
                result[-1] += ','
        
        if closing:
            result.append(' ' * open_paren + closing)
        
        return result
    
    return break_at_spaces(line, max_length)


def break_string_literal(line, max_length):
    """Break long string literals."""
    # Simple approach: break at word boundaries
    return break_at_spaces(line, max_length)


def break_assignment(line, max_length):
    """Break long assignments."""
    if '=' not in line:
        return [line]
    
    equal_pos = line.find('=')
    var_part = line[:equal_pos].strip()
    value_part = line[equal_pos + 1:].strip()
    
    if len(var_part) + 3 < max_length:  # 3 for " = "
        # Can fit variable and = on first line
        result = [var_part + ' =']
        # Break value part
        value_lines = break_at_spaces(value_part, max_length - 4)  # 4 for indentation
        for i, val_line in enumerate(value_lines):
            if i == 0:
                result.append(' ' * 4 + val_line)
            else:
                result.append(' ' * 4 + val_line)
        return result
    else:
        # Need to break variable part too
        return break_at_spaces(line, max_length)


def break_method_chain(line, max_length):
    """Break long method chains."""
    # Split by dots
    parts = line.split('.')
    if len(parts) <= 1:
        return [line]
    
    result = [parts[0]]
    current_line = ' ' * 4  # Indent continuation lines
    
    for part in parts[1:]:
        if len(current_line + '.' + part) <= max_length:
            current_line += '.' + part
        else:
            result.append(current_line)
            current_line = ' ' * 4 + '.' + part
    
    if current_line.strip():
        result.append(current_line)
    
    return result


def break_at_spaces(line, max_length):
    """Break line at word boundaries."""
    if len(line) <= max_length:
        return [line]
    
    words = line.split()
    result = []
    current_line = ''
    
    for word in words:
        if len(current_line + ' ' + word) <= max_length:
            if current_line:
                current_line += ' ' + word
            else:
                current_line = word
        else:
            if current_line:
                result.append(current_line)
            current_line = word
    
    if current_line:
        result.append(current_line)
    
    return result


def fix_file(file_path):
    """Fix line length and style issues in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply fixes
        content = fix_long_lines(content)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"  ❌ Error fixing {file_path}: {e}")
        return False


def main():
    """Main function to fix line length and style issues."""
    print("📏 Line Length and Style Fixer")
    print("=" * 40)
    
    # Find all Python files
    python_files = list(Path(".").rglob("*.py"))
    print(f"📁 Found {len(python_files)} Python files")
    
    fixed_count = 0
    
    for py_file in python_files:
        try:
            if fix_file(py_file):
                fixed_count += 1
                print(f"  ✅ Fixed {py_file}")
        except Exception as e:
            print(f"  ❌ Error processing {py_file}: {e}")
    
    print(f"\n📋 SUMMARY:")
    print(f"  Files processed: {len(python_files)}")
    print(f"  Files fixed: {fixed_count}")
    
    if fixed_count > 0:
        print(f"\n🎉 Successfully fixed {fixed_count} files!")
        print("💡 Run the quality check again to verify fixes.")
    else:
        print(f"\n⚠️  No files were fixed. All files may already be clean.")


if __name__ == "__main__":
    main()
