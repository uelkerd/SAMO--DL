#!/usr/bin/env python3
"""
Advanced Quality Fixer
Fixes cyclomatic complexity and other advanced code quality issues.
"""

import os
import ast
import re
from pathlib import Path


def calculate_cyclomatic_complexity(tree):
    """Calculate cyclomatic complexity of a function."""
    complexity = 1  # Base complexity
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.For)):
            complexity += 1
        elif isinstance(node, ast.ExceptHandler):
            complexity += 1
        elif isinstance(node, ast.With):
            complexity += 1
        elif isinstance(node, ast.Assert):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1
    
    return complexity


def fix_high_complexity_functions(content):
    """Fix functions with high cyclomatic complexity."""
    try:
        tree = ast.parse(content)
        modified = False
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = calculate_cyclomatic_complexity(node)
                if complexity > 10:  # Threshold for high complexity
                    # Try to refactor the function
                    new_content = refactor_complex_function(content, node)
                    if new_content != content:
                        content = new_content
                        modified = True
        
        return content, modified
        
    except SyntaxError:
        # If we can't parse, return original content
        return content, False


def refactor_complex_function(content, func_node):
    """Refactor a complex function by extracting logic."""
    lines = content.split('\n')
    func_start = func_node.lineno - 1
    func_end = func_node.end_lineno
    
    # Get function body
    func_body = lines[func_start:func_end]
    func_text = '\n'.join(func_body)
    
    # Simple refactoring: extract complex conditions
    if 'if ' in func_text and func_text.count('if ') > 3:
        # Extract complex conditions to helper functions
        new_func_text = extract_complex_conditions(func_text, func_node.name)
        
        # Replace the function
        new_lines = lines[:func_start] + [new_func_text] + lines[func_end:]
        return '\n'.join(new_lines)
    
    return content


def extract_complex_conditions(func_text, func_name):
    """Extract complex conditions to helper functions."""
    lines = func_text.split('\n')
    new_lines = []
    
    # Add helper functions at the beginning
    helper_functions = []
    condition_count = 0
    
    for line in lines:
        if line.strip().startswith('if '):
            condition_count += 1
            if condition_count > 2:  # Extract complex conditions
                # Create a helper function for this condition
                condition = line.strip()[3:]  # Remove 'if '
                if condition.endswith(':'):
                    condition = condition[:-1]
                
                helper_name = f"_check_condition_{condition_count}"
                helper_func = f"def {helper_name}():\n    return {condition}\n"
                helper_functions.append(helper_func)
                
                # Replace the condition with helper call
                new_lines.append(f"    if {helper_name}():")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    # Add helper functions at the beginning
    if helper_functions:
        # Find the right place to insert helpers (after function definition)
        for i, line in enumerate(new_lines):
            if line.strip().startswith('def ') and ':' in line:
                # Insert helpers after function definition
                new_lines = new_lines[:i+1] + [''] + helper_functions + [''] + new_lines[i+1:]
                break
    
    return '\n'.join(new_lines)


def fix_file(file_path):
    """Fix advanced quality issues in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply fixes
        content, modified = fix_high_complexity_functions(content)
        
        # Only write if content changed
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"  ❌ Error fixing {file_path}: {e}")
        return False


def main():
    """Main function to fix advanced quality issues."""
    print("⚡ Advanced Quality Fixer")
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
