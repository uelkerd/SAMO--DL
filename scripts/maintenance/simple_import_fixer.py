#!/usr/bin/env python3
"""
Simple Import Fixer
Fixes basic import issues without complex parsing.
"""


import os
import re
from pathlib import Path



def fix_imports_in_file(file_path):
    """Fix imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Simple fixes: remove obvious unused imports
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip obviously unused imports
            if stripped.startswith('import ') and any(unused in stripped for unused in [
                'unused', 'deprecated', 'legacy', 'old_'
            ]):
                continue
            
            # Keep the line
            fixed_lines.append(line)
        
        # Ensure proper spacing between import groups
        final_lines = []
        in_imports = False
        
        for line in fixed_lines:
            if line.strip().startswith(('import ', 'from ')):
                if not in_imports:
                    final_lines.append('')  # Add blank line before imports
                in_imports = True
                final_lines.append(line)
            elif in_imports and line.strip() == '':
                final_lines.append(line)
            elif in_imports and not line.strip().startswith(('import ', 'from ')):
                in_imports = False
                final_lines.append('')  # Add blank line after imports
                final_lines.append(line)
            else:
                final_lines.append(line)
        
        new_content = '\n'.join(final_lines)
        
        # Only write if content changed
        if new_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
        return False
        
    except Exception as e:
        print(f"  ❌ Error fixing {file_path}: {e}")
        return False


def main():
    """Main function to fix import issues."""
    print("📦 Simple Import Fixer")
    print("=" * 30)
    
    # Find all Python files
    python_files = list(Path(".").rglob("*.py"))
    print(f"📁 Found {len(python_files)} Python files")
    
    fixed_count = 0
    
    for py_file in python_files:
        try:
            if fix_imports_in_file(py_file):
                fixed_count += 1
                print(f"  ✅ Fixed {py_file}")
        except Exception as e:
            print(f"  ❌ Error processing {py_file}: {e}")
    
    print(f"\n📋 SUMMARY:")
    print(f"  Files processed: {len(python_files)}")
    print(f"  Files fixed: {fixed_count}")
    
    if fixed_count > 0:
        print(f"\n🎉 Successfully fixed {fixed_count} files!")
    else:
        print(f"\n⚠️  No files were fixed.")


if __name__ == "__main__":
    main()
