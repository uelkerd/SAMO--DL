#!/usr/bin/env python3
"""
Fix Remaining Python 3.8 Compatibility Issues

This script uses regex patterns to fix remaining type hint issues:
- list[T] -> List[T]
- dict[K, V] -> Dict[K, V]
- set[T] -> Set[T]
- tuple[T, ...] -> Tuple[T, ...]
- A | B -> Union[A, B] or Optional[A] for A | None
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Any


def fix_file(file_path: Path, dry_run: bool = False) -> Dict[str, Any]:
    """Fix Python 3.8 compatibility issues in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        original_content = content
        changes_made = []
        imports_to_add = set()
        
        # Fix list[T] -> List[T]
        list_pattern = r'\blist\[([^\]]+)\]'
        list_matches = re.findall(list_pattern, content)
        if list_matches:
            content = re.sub(list_pattern, r'List[\1]', content)
            imports_to_add.add('List')
            changes_made.append(f"list[T] -> List[T] ({len(list_matches)} instances)")
            
        # Fix dict[K, V] -> Dict[K, V]
        dict_pattern = r'\bdict\[([^\]]+)\]'
        dict_matches = re.findall(dict_pattern, content)
        if dict_matches:
            content = re.sub(dict_pattern, r'Dict[\1]', content)
            imports_to_add.add('Dict')
            changes_made.append(f"dict[T] -> Dict[T] ({len(dict_matches)} instances)")
            
        # Fix set[T] -> Set[T]
        set_pattern = r'\bset\[([^\]]+)\]'
        set_matches = re.findall(set_pattern, content)
        if set_matches:
            content = re.sub(set_pattern, r'Set[\1]', content)
            imports_to_add.add('Set')
            changes_made.append(f"set[T] -> Set[T] ({len(set_matches)} instances)")
            
        # Fix tuple[T, ...] -> Tuple[T, ...]
        tuple_pattern = r'\btuple\[([^\]]+)\]'
        tuple_matches = re.findall(tuple_pattern, content)
        if tuple_matches:
            content = re.sub(tuple_pattern, r'Tuple[\1]', content)
            imports_to_add.add('Tuple')
            changes_made.append(
                f"tuple[T] -> Tuple[T] ({len(tuple_matches)} instances)"
            )
            
        # Fix A | None -> Optional[A] (most common case)
        optional_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\|\s*None'
        optional_matches = re.findall(optional_pattern, content)
        if optional_matches:
            content = re.sub(optional_pattern, r'Optional[\1]', content)
            imports_to_add.add('Optional')
            changes_made.append(
                f"A | None -> Optional[A] ({len(optional_matches)} instances)"
            )
            
        # Fix None | A -> Optional[A]
        optional_pattern2 = r'None\s*\|\s*([a-zA-Z_][a-zA-Z0-9_]*)'
        optional_matches2 = re.findall(optional_pattern2, content)
        if optional_matches2:
            content = re.sub(optional_pattern2, r'Optional[\1]', content)
            imports_to_add.add('Optional')
            changes_made.append(
                f"None | A -> Optional[A] ({len(optional_matches2)} instances)"
            )
            
        # Fix general A | B -> Union[A, B] 
        # (but be careful not to catch bitwise operations)
        # Look for type annotations specifically
        union_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\|\s*([a-zA-Z_][a-zA-Z0-9_]*)'
        union_matches = re.findall(union_pattern, content)
        if union_matches:
            # Filter out matches that are likely not type annotations
            filtered_matches = []
            for left, right in union_matches:
                # Skip if it looks like a bitwise operation in code
                type_names = [
                    'None', 'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple'
                ]
                if not (left in type_names or right in type_names):
                    continue
                filtered_matches.append((left, right))
            
            if filtered_matches:
                # Replace the filtered matches
                for left, right in filtered_matches:
                    if left != 'None' and right != 'None':
                        pattern = f'\\b{re.escape(left)}\\s*\\|\\s*{re.escape(right)}\\b'
                        replacement = f'Union[{left}, {right}]'
                        content = re.sub(pattern, replacement, content)
                        imports_to_add.add('Union')
                        changes_made.append(
                            f"{left} | {right} -> Union[{left}, {right}]"
                        )
        
        # Add missing imports
        if imports_to_add and not dry_run:
            # Find existing typing imports
            typing_import_match = re.search(r'from typing import ([^\\n]+)', content)
            if typing_import_match:
                existing_imports = typing_import_match.group(1).strip()
                new_imports = ', '.join(sorted(imports_to_add))
                if existing_imports:
                    # Add to existing import
                    content = re.sub(
                        r'from typing import ([^\\n]+)',
                        f'from typing import {existing_imports}, {new_imports}',
                        content
                    )
                else:
                    # Replace empty import
                    content = re.sub(
                        r'from typing import', 
                        f'from typing import {new_imports}', 
                        content
                    )
            else:
                # Find last import line
                lines = content.splitlines()
                last_import_line = -1
                for i, line in enumerate(lines):
                    if (line.strip().startswith('import ') or
                        line.strip().startswith('from ')):
                        last_import_line = i
                
                if last_import_line >= 0:
                    import_line = (
                        f"from typing import {', '.join(sorted(imports_to_add))}"
                    )
                    lines.insert(last_import_line + 1, import_line)
                else:
                    import_line = (
                        f"from typing import {', '.join(sorted(imports_to_add))}"
                    )
                    lines.insert(0, import_line)
                content = '\n'.join(lines)
        
        # Write back to file if changes were made
        if content != original_content and not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        return {
            'file': str(file_path),
            'changes': changes_made,
            'imports_added': list(imports_to_add),
            'modified': content != original_content
        }
        
    except Exception as e:
        return {'file': str(file_path), 'error': str(e), 'modified': False}


def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in a directory recursively."""
    python_files = []
    for item in directory.rglob('*.py'):
        if not any(part.startswith('.') for part in item.parts):
            python_files.append(item)
    return python_files


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python fix_remaining_py38_types.py <directory> [--dry-run]")
        sys.exit(1)
        
    directory = Path(sys.argv[1])
    dry_run = '--dry-run' in sys.argv
    
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        sys.exit(1)
        
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory")
        sys.exit(1)
        
    print(f"Processing directory: {directory}")
    if dry_run:
        print("DRY RUN MODE - No changes will be made")
    print()
    
    # Find Python files
    python_files = find_python_files(directory)
    print(f"Found {len(python_files)} Python files")
    print()
    
    # Process files
    results = []
    total_changes = 0
    
    for file_path in python_files:
        print(f"Processing: {file_path}")
        result = fix_file(file_path, dry_run=dry_run)
        results.append(result)
        
        if 'error' in result:
            print(f"  ❌ Error: {result['error']}")
        elif result['modified']:
            print(f"  ✅ Modified: {', '.join(result['changes'])}")
            print(f"     Imports added: {', '.join(result['imports_added'])}")
            total_changes += len(result['changes'])
        else:
            print(f"  ⏭️  No changes needed")
        print()
    
    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    modified = [r for r in results if r.get('modified', False)]
    errors = [r for r in results if 'error' in r]
    no_changes = [
        r for r in results 
        if not r.get('modified', False) and 'error' not in r
    ]
    
    print(f"Files processed: {len(results)}")
    print(f"Modified: {len(modified)}")
    print(f"Errors: {len(errors)}")
    print(f"No changes needed: {len(no_changes)}")
    print(f"Total changes: {total_changes}")
    
    if errors:
        print("\nFiles with errors:")
        for result in errors:
            print(f"  {result['file']}: {result['error']}")
            
    if dry_run and total_changes > 0:
        print(f"\nTo apply these changes, run without --dry-run")
        
    print()


if __name__ == '__main__':
    main()
