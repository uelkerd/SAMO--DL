#!/usr/bin/env python3
"""
Enhanced Syntax Error Fixer
Fixes complex syntax errors that the basic fixer couldn't handle.
"""

import re
import sys
from pathlib import Path


def fix_mixed_quotes(content):
    """Fix mixed quote types in strings."""
    # Fix cases like: "text {variable" or 'text {variable'
    content = re.sub(r'["\']([^"\']*)\{([^"\']*)["\']', r'"\1{\2"', content)

    # Fix cases with mismatched quotes
    content = re.sub(r'(["\'])([^"\']*)\{([^"\']*)\1([^"\']*)\1', r'"\2{\3}\4"', content)

    return content


def fix_missing_colons(content):
    """Fix missing colons after if/for/while/def/class statements."""
    lines = content.split('\n')
    fixed_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Check for missing colons after control structures
        if re.match(r'^(if|for|while|def|class|elif|else|try|except|finally|with)\s', stripped):
            if not stripped.endswith(':'):
                # Add missing colon
                line = line.rstrip() + ':'

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)


def fix_malformed_function_calls(content):
    """Fix malformed function calls with incorrect parentheses."""
    # Fix cases like: function()\n    param=value\n)
    content = re.sub(r'(\w+\(\))\s*\n\s*([^)]+)\n\s*\)', r'\1(\2)', content)

    # Fix cases with extra parentheses
    content = re.sub(r'\(\s*\)\s*\)', r')', content)

    return content


def fix_indentation_issues(content):
    """Fix indentation problems."""
    lines = content.split('\n')
    fixed_lines = []
    expected_indent = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            fixed_lines.append('')
            continue

        # Check for control structures that should be at base level
        if re.match(r'^(if|for|while|def|class|elif|else|try|except|finally|with)\s', stripped):
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
        content = fix_mixed_quotes(content)
        content = fix_missing_colons(content)
        content = fix_malformed_function_calls(content)
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
    """Main function to fix complex syntax errors."""
    print("🔧 Enhanced Syntax Error Fixer")
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
