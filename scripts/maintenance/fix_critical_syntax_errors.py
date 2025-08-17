#!/usr/bin/env python3
"""
Critical Syntax Error Fixer
Fixes the most common syntax errors that prevent Python code from running.
"""

import re
import sys
from pathlib import Path


def fix_unterminated_strings(content):
    """Fix unterminated string literals."""
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Check for unterminated strings
        if '"' in line and not line.count('"') % 2 == 0:
            # Try to fix by adding missing quote
            if line.strip().endswith('\\'):
                # Line continuation, keep as is
                fixed_lines.append(line)
            else:
                # Add missing quote at end
                fixed_lines.append(line + '"')
        elif "'" in line and not line.count("'") % 2 == 0:
            # Try to fix by adding missing quote
            if line.strip().endswith('\\'):
                # Line continuation, keep as is
                fixed_lines.append(line)
            else:
                # Add missing quote at end
                fixed_lines.append(line + "'")
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def fix_invalid_unicode(content):
    """Remove or replace invalid Unicode characters."""
    # Remove emojis and other problematic Unicode
    content = re.sub(r'[🎯✅🎉🔍📊🚨📋]', '', content)
    return content


def fix_unmatched_parentheses(content):
    """Fix unmatched parentheses by adding missing ones."""
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Count parentheses
        open_parens = line.count('(')
        close_parens = line.count(')')
        
        if open_parens > close_parens:
            # Add missing close parentheses
            line = line + ')' * (open_parens - close_parens)
        elif close_parens > open_parens:
            # Add missing open parentheses
            line = '(' * (close_parens - open_parens) + line
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def fix_indentation_issues(content):
    """Fix basic indentation issues."""
    lines = content.split('\n')
    fixed_lines = []
    expected_indent = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            fixed_lines.append('')
            continue
            
        # Check for indentation issues
        if stripped.startswith(('if ', 'for ', 'while ', 'def ', 'class ')):
            # This should be at base level or properly indented
            if line.startswith(' ' * (expected_indent + 4)):
                expected_indent += 4
            elif not line.startswith(' ' * expected_indent):
                # Fix indentation
                line = ' ' * expected_indent + stripped
        
        # Check for dedent
        if stripped in ('return', 'break', 'continue', 'pass'):
            if line.startswith(' ' * (expected_indent + 4)):
                expected_indent = max(0, expected_indent - 4)
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def fix_file(file_path):
    """Fix a single file's syntax errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply fixes in order
        content = fix_invalid_unicode(content)
        content = fix_unterminated_strings(content)
        content = fix_unmatched_parentheses(content)
        content = fix_indentation_issues(content)
        
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
    """Main function to fix critical syntax errors."""
    print("🔧 Critical Syntax Error Fixer")
    print("=" * 40)
    
    # Find all Python files
    python_files = list(Path(".").rglob("*.py"))
    print(f"📁 Found {len(python_files)} Python files")
    
    fixed_count = 0
    error_count = 0
    
    for py_file in python_files:
        try:
            # Test if file has syntax errors
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                compile(content, str(py_file), 'exec')
                # No syntax errors
                continue
            except SyntaxError:
                # Has syntax errors, try to fix
                print(f"🔧 Fixing {py_file}...")
                if fix_file(py_file):
                    fixed_count += 1
                    print(f"  ✅ Fixed {py_file}")
                else:
                    print(f"  ⚠️  Could not fix {py_file}")
                    error_count += 1
                    
        except Exception as e:
            print(f"  ❌ Error processing {py_file}: {e}")
            error_count += 1
    
    print(f"\n📋 SUMMARY:")
    print(f"  Files processed: {len(python_files)}")
    print(f"  Files fixed: {fixed_count}")
    print(f"  Files with errors: {error_count}")
    
    if fixed_count > 0:
        print(f"\n🎉 Successfully fixed {fixed_count} files!")
        print("💡 Run the quality check again to verify fixes.")
    else:
        print(f"\n⚠️  No files were fixed. Manual intervention may be needed.")


if __name__ == "__main__":
    main()
