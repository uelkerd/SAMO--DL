#!/usr/bin/env python3
"""
Script to automatically fix line length issues (FLK-E501) across the codebase.
This script will break long lines into multiple lines while preserving functionality.
"""
import sys
from pathlib import Path

def fix_line_length_issues(file_path: str, max_length: int = 88) -> bool:
    """Fix line length issues in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        modified = False
        new_lines = []

        for i, line in enumerate(lines):
            if len(line.rstrip()) > max_length:
                # Try to break the line intelligently
                fixed_line = break_long_line(line.rstrip(), max_length)
                if fixed_line != line.rstrip():
                    new_lines.append(fixed_line + '\n')
                    modified = True
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            print(f"✅ Fixed line length issues in {file_path}")
            return True
        return False
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def break_long_line(line: str, max_length: int) -> str:
    """Break a long line into multiple lines."""
    # Don't break if it's a comment or docstring
    if line.strip().startswith('#') or line.strip().startswith('"""') or line.strip().startswith("'''"):
        return line

    # Don't break if it's a string literal that shouldn't be broken
    if line.strip().startswith('"') and line.strip().endswith('"'):
        return line
    if line.strip().startswith("'") and line.strip().endswith("'"):
        return line

    # Try to break at logical points
    indent = len(line) - len(line.lstrip())

    # Break at commas in function calls
    if ',' in line and '(' in line:
        parts = line.split(',')
        if len(parts) > 1:
            result = parts[0]
            for part in parts[1:]:
                if len(result + ',' + part) <= max_length:
                    result += ',' + part
                else:
                    result += ',\n' + ' ' * (indent + 4) + part.strip()
            return result

    # Break at logical operators
    for op in [' and ', ' or ', ' if ', ' else ']:
        if op in line and len(line) > max_length:
            parts = line.split(op)
            if len(parts) > 1:
                result = parts[0]
                for part in parts[1:]:
                    if len(result + op + part) <= max_length:
                        result += op + part
                    else:
                        result += op + '\n' + ' ' * (indent + 4) + part.strip()
                return result

    # Break at assignment operators
    if ' = ' in line and len(line) > max_length:
        parts = line.split(' = ', 1)
        if len(parts) == 2:
            left, right = parts
            if len(left + ' = ' + right) > max_length:
                return left + ' = (\n' + ' ' * (indent + 4) + right + '\n' + ' ' * indent + ')'

    # Break at method chaining
    if '.' in line and len(line) > max_length:
        parts = line.split('.')
        if len(parts) > 1:
            result = parts[0]
            for part in parts[1:]:
                if len(result + '.' + part) <= max_length:
                    result += '.' + part
                else:
                    result += '.\n' + ' ' * (indent + 4) + part.strip()
            return result

    # If we can't break intelligently, just return the original line
    return line

def main():
    """Main function to fix line length issues across the codebase."""
    project_root = Path(__file__).parent.parent.parent
    src_dir = project_root / "src"

    if not src_dir.exists():
        print("❌ src directory not found")
        return 1

    fixed_files = 0
    total_files = 0

    # Process Python files in src directory
    for py_file in src_dir.rglob("*.py"):
        total_files += 1
        if fix_line_length_issues(str(py_file)):
            fixed_files += 1

    print("\n📊 Summary:")
    print(f"   Total files processed: {total_files}")
    print(f"   Files modified: {fixed_files}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
