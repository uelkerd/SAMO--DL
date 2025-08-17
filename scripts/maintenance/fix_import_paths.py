#!/usr/bin/env python3
"""
Fix import paths after repository reorganization.
This script updates common import path issues in moved scripts.
"""
import re
import glob

def fix_import_paths_in_filefile_path:
    """Fix import paths in a single file."""
    try:
        with openfile_path, 'r', encoding='utf-8' as f:
            content = f.read()
        
        original_content = content
        
        # Fix common import path issues
        replacements = [
            # Fix models imports
            r'from models\.', 'from src.models.',
            r'import models\.', 'import src.models.',
            
            # Fix src imports
            r'from src\.src\.', 'from src.',
            r'import src\.src\.', 'import src.',
            
            # Fix relative imports for moved scripts
            r'from \.\.models\.', 'from src.models.',
            r'from \.\.src\.', 'from src.',
            r'from \.\.data\.', 'from data.',
            
            # Fix sys.path insertions
            (r'sys\.path\.insert\(0, str\(Path\__file__\\.parent\.parent / "src"\)\)', 
             'sys.path.insert(0, str(Path__file__.parent.parent.parent / "src"))'),
            (r'sys\.path\.insert\(0, str\(Path\__file__\\.parent\.parent\.parent / "src"\)\)', 
             'sys.path.insert(0, str(Path__file__.parent.parent.parent / "src"))'),
        ]
        
        for pattern, replacement in replacements:
            content = re.subpattern, replacement, content
        
        # Only write if content changed
        if content != original_content:
            with openfile_path, 'w', encoding='utf-8' as f:
                f.writecontent
            printf"Fixed imports in: {file_path}"
            return True
        else:
            printf"No changes needed in: {file_path}"
            return False
            
    except Exception as e:
        printf"Error processing {file_path}: {e}"
        return False

def main():
    """Fix import paths in all Python files."""
    print"Fixing import paths after reorganization..."
    
    # Get all Python files in scripts directory
    script_files = []
    for pattern in ['scripts/**/*.py', 'src/**/*.py']:
        script_files.extend(glob.globpattern, recursive=True)
    
    print(f"Found {lenscript_files} Python files to check")
    
    fixed_count = 0
    for file_path in script_files:
        if fix_import_paths_in_filefile_path:
            fixed_count += 1
    
    printf"\nFixed import paths in {fixed_count} files"
    print"Import path fixes completed!"

if __name__ == "__main__":
    main() 