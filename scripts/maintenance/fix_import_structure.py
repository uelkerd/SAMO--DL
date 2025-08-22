#!/usr/bin/env python3
"""
Import and Structure Fixer
Fixes import organization and unused imports without changing logic.
"""


import os
import ast
import re
from pathlib import Path



def find_unused_imports(file_path):
    """Find unused imports in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the file
        tree = ast.parse(content)
        
        # Find all imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    if alias.name == '*':
                        imports.append(f"{module}.*")
                    else:
                        imports.append(f"{module}.{alias.name}")
        
        # Find all names used in the code
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
        
        # Find unused imports
        unused = []
        for imp in imports:
            if '.' in imp:
                base_name = imp.split('.')[0]
                if base_name not in used_names:
                    unused.append(imp)
            else:
                if imp not in used_names:
                    unused.append(imp)
        
        return unused
        
    except Exception as e:
        print(f"  ⚠️  Could not parse {file_path}: {e}")
        return []


def organize_imports(content):
    """Organize imports alphabetically and group them."""
    lines = content.split('\n')
    import_lines = []
    other_lines = []
    
    in_imports = False
    
    for line in lines:
        stripped = line.strip()
        
        # Check if we're in import section
        if stripped.startswith(('import ', 'from ')):
            in_imports = True
            import_lines.append(line)
        elif in_imports and stripped == '':
            # Empty line after imports
            import_lines.append(line)
        elif in_imports and not stripped.startswith(('import ', 'from ')):
            # End of imports
            in_imports = False
            other_lines.append(line)
        else:
            other_lines.append(line)
    
    if import_lines:
        # Sort import lines alphabetically
        import_lines.sort()
        
        # Group standard library, third party, and local imports
        std_imports = []
        third_party_imports = []
        local_imports = []
        
        for line in import_lines:
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                if line.strip().startswith(('from .', 'import .')):
                    local_imports.append(line)
                elif any(pkg in line for pkg in ['os', 'sys', 'json', 'pathlib', 're', 'tempfile', 'datetime', 'time']):
                    std_imports.append(line)
                else:
                    third_party_imports.append(line)
            else:
                # Empty lines
                std_imports.append(line)
                third_party_imports.append(line)
                local_imports.append(line)
        
        # Combine organized imports
        organized_imports = std_imports + third_party_imports + local_imports
        
        # Remove duplicate empty lines
        final_imports = []
        prev_empty = False
        for line in organized_imports:
            if line.strip() == '':
                if not prev_empty:
                    final_imports.append(line)
                prev_empty = True
            else:
                final_imports.append(line)
                prev_empty = False
        
        return final_imports + other_lines
    
    return lines


def fix_file(file_path):
    """Fix import structure in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Find unused imports
        unused_imports = find_unused_imports(file_path)
        
        # Remove unused imports
        for unused in unused_imports:
            if '.' in unused:
                base_name = unused.split('.')[0]
                # Remove import lines containing this base name
                lines = content.split('\n')
                filtered_lines = []
                for line in lines:
                    if not (line.strip().startswith(f'import {base_name}') or 
                           line.strip().startswith(f'from {base_name}')):
                        filtered_lines.append(line)
                content = '\n'.join(filtered_lines)
        
        # Organize remaining imports
        content = '\n'.join(organize_imports(content.split('\n')))
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, len(unused_imports)
        
        return False, 0
        
    except Exception as e:
        print(f"  ❌ Error fixing {file_path}: {e}")
        return False, 0


def main():
    """Main function to fix import and structure issues."""
    print("📦 Import and Structure Fixer")
    print("=" * 40)
    
    # Find all Python files
    python_files = list(Path(".").rglob("*.py"))
    print(f"📁 Found {len(python_files)} Python files")
    
    fixed_count = 0
    total_unused_removed = 0
    
    for py_file in python_files:
        try:
            fixed, unused_count = fix_file(py_file)
            if fixed:
                fixed_count += 1
                print(f"  ✅ Fixed {py_file} (removed {unused_count} unused imports)")
                total_unused_removed += unused_count
        except Exception as e:
            print(f"  ❌ Error processing {py_file}: {e}")
    
    print(f"\n📋 SUMMARY:")
    print(f"  Files processed: {len(python_files)}")
    print(f"  Files fixed: {fixed_count}")
    print(f"  Total unused imports removed: {total_unused_removed}")
    
    if fixed_count > 0:
        print(f"\n🎉 Successfully fixed {fixed_count} files!")
        print("💡 Run the quality check again to verify fixes.")
    else:
        print(f"\n⚠️  No files were fixed. All files may already be clean.")


if __name__ == "__main__":
    main()
