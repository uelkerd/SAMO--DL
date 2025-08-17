#!/usr/bin/env python3
"""
SAMO-DL Auto-Fix Code Quality Issues

This script automatically fixes most common code quality issues:
- Trailing whitespace (FLK-W291)
- Missing newlines (FLK-W292)
- Blank line whitespace (FLK-W293)
- Line length violations (FLK-E501) - basic fixes
- Import sorting and cleanup
- Basic formatting issues

Usage: python auto_fix_code_quality.py [--dry-run] [directory]
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any


class AutoCodeQualityFixer:
    """Automatically fixes common code quality issues."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.files_processed = 0
        self.files_modified = 0
        self.total_fixes = 0
        
    def fix_file(self, file_path: Path) -> Dict[str, Any]:
        """Fix quality issues in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                original_content = content
            
            lines = content.splitlines()
            fixes_applied = []
            
            # 1. Fix trailing whitespace (FLK-W291)
            if self._fix_trailing_whitespace(lines):
                fixes_applied.append('trailing_whitespace')
            
            # 2. Fix blank line whitespace (FLK-W293)
            if self._fix_blank_line_whitespace(lines):
                fixes_applied.append('blank_line_whitespace')
            
            # 3. Fix missing newline (FLK-W292)
            if self._fix_missing_newline(lines):
                fixes_applied.append('missing_newline')
            
            # 4. Fix basic line length issues (FLK-E501)
            if self._fix_basic_line_length(lines):
                fixes_applied.append('line_length')
            
            # 5. Fix import formatting
            if self._fix_import_formatting(lines):
                fixes_applied.append('import_formatting')
            
            # 6. Fix basic indentation issues (FLK-E128)
            if self._fix_basic_indentation(lines):
                fixes_applied.append('indentation')
            
            # Apply fixes if any were made
            if fixes_applied:
                new_content = '\n'.join(lines)
                if not self.dry_run:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                
                self.files_modified += 1
                self.total_fixes += len(fixes_applied)
                
                return {
                    'status': 'modified',
                    'fixes': fixes_applied,
                    'file': str(file_path)
                }
            else:
                return {
                    'status': 'no_changes',
                    'file': str(file_path)
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'file': str(file_path),
                'error': str(e)
            }
    
    def _fix_trailing_whitespace(self, lines: List[str]) -> bool:
        """Remove trailing whitespace from lines."""
        modified = False
        for i in range(len(lines)):
            original = lines[i]
            cleaned = original.rstrip()
            if original != cleaned:
                lines[i] = cleaned
                modified = True
        return modified
    
    def _fix_blank_line_whitespace(self, lines: List[str]) -> bool:
        """Remove whitespace from blank lines."""
        modified = False
        for i in range(len(lines)):
            if not lines[i].strip() and lines[i] != '':
                lines[i] = ''
                modified = True
        return modified
    
    def _fix_missing_newline(self, lines: List[str]) -> bool:
        """Ensure file ends with newline."""
        if lines and lines[-1] != '':
            lines.append('')
            return True
        return False
    
    def _fix_basic_line_length(self, lines: List[str]) -> bool:
        """Fix basic line length issues (simple cases)."""
        modified = False
        max_length = 88
        
        for i in range(len(lines)):
            line = lines[i]
            if len(line) > max_length:
                # Try to fix common patterns
                new_line = self._try_fix_long_line(line, max_length)
                if new_line != line:
                    lines[i] = new_line
                    modified = True
        
        return modified
    
    def _try_fix_long_line(self, line: str, max_length: int) -> str:
        """Try to fix a long line by breaking it intelligently."""
        # Skip comments and strings
        if line.strip().startswith('#'):
            return line
        
        # Try to break long function calls
        if '(' in line and ')' in line and len(line) > max_length:
            # Find the opening parenthesis
            open_paren = line.find('(')
            if open_paren < max_length - 20:  # Only if there's room
                # Try to break after the opening parenthesis
                indent = len(line) - len(line.lstrip())
                if line.count(',') > 0:
                    # Multiple parameters, break after first comma
                    first_comma = line.find(',', open_paren)
                    if first_comma > 0:
                        before_comma = line[:first_comma + 1]
                        after_comma = line[first_comma + 1:].lstrip()
                        if len(before_comma) <= max_length:
                            return f"{before_comma}\n{' ' * (indent + 4)}{after_comma}"
        
        # Try to break long assignments
        if ' = ' in line and len(line) > max_length:
            parts = line.split(' = ', 1)
            if len(parts) == 2:
                var_name = parts[0]
                value = parts[1]
                indent = len(line) - len(line.lstrip())
                if len(var_name) + 3 <= max_length:
                    return f"{var_name} = \\\n{' ' * (indent + 4)}{value}"
        
        # Try to break long strings
        if '"' in line and "'" in line and len(line) > max_length:
            # Look for string concatenation opportunities
            if ' + ' in line:
                parts = line.split(' + ')
                if len(parts) > 1:
                    indent = len(line) - len(line.lstrip())
                    result = parts[0]
                    for part in parts[1:]:
                        if len(result + ' + ' + part) > max_length:
                            result += ' + \\\n' + ' ' * (indent + 4) + part
                        else:
                            result += ' + ' + part
                    return result
        
        return line
    
    def _fix_import_formatting(self, lines: List[str]) -> bool:
        """Fix basic import formatting issues."""
        modified = False
        
        # Group imports by type
        import_lines = []
        other_lines = []
        in_import_section = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')):
                if not in_import_section:
                    in_import_section = True
                import_lines.append(line)
            else:
                if in_import_section and stripped == '':
                    # Empty line in import section
                    import_lines.append(line)
                elif in_import_section and not stripped.startswith('#'):
                    # Non-import, non-comment line - end import section
                    in_import_section = False
                    other_lines.append(line)
                else:
                    other_lines.append(line)
        
        # Sort import lines
        if import_lines:
            # Separate standard library, third-party, and local imports
            std_imports = []
            third_party_imports = []
            local_imports = []
            
            for line in import_lines:
                stripped = line.strip()
                if stripped.startswith('from .') or stripped.startswith('import .'):
                    local_imports.append(line)
                elif any(pkg in stripped for pkg in ['numpy', 'pandas', 'torch', 'transformers', 'sklearn']):
                    third_party_imports.append(line)
                else:
                    std_imports.append(line)
            
            # Reconstruct with proper grouping
            new_import_lines = []
            if std_imports:
                new_import_lines.extend(sorted(std_imports))
                new_import_lines.append('')
            if third_party_imports:
                new_import_lines.extend(sorted(third_party_imports))
                new_import_lines.append('')
            if local_imports:
                new_import_lines.extend(sorted(local_imports))
                new_import_lines.append('')
            
            # Remove trailing empty line
            if new_import_lines and new_import_lines[-1] == '':
                new_import_lines.pop()
            
            # Check if anything changed
            if new_import_lines != import_lines:
                modified = True
                # Reconstruct the file
                lines.clear()
                lines.extend(new_import_lines)
                lines.append('')  # Add blank line after imports
                lines.extend(other_lines)
        
        return modified
    
    def _fix_basic_indentation(self, lines: List[str]) -> bool:
        """Fix basic indentation issues."""
        modified = False
        
        for i in range(len(lines) - 1):
            current_line = lines[i]
            next_line = lines[i + 1]
            
            # Check for continuation line indentation
            if current_line.strip() and current_line.strip().endswith(('(', '[', '{')):
                # This line opens a bracket, next line should be indented
                expected_indent = len(current_line) - len(current_line.lstrip()) + 4
                actual_indent = len(next_line) - len(next_line.lstrip())
                
                if next_line.strip() and not next_line.startswith('#') and actual_indent < expected_indent:
                    # Fix indentation
                    lines[i + 1] = ' ' * expected_indent + next_line.lstrip()
                    modified = True
        
        return modified
    
    def run_fixes(self, directory: Path) -> Dict[str, Any]:
        """Run auto-fixes on all Python files in directory."""
        print(f"🔧 Running automatic code quality fixes on {directory}")
        if self.dry_run:
            print("DRY RUN MODE - No changes will be made")
        
        python_files = list(directory.rglob('*.py'))
        python_files = [f for f in python_files if not any(part.startswith('.') for part in f.parts)]
        
        results = []
        
        for file_path in python_files:
            result = self.fix_file(file_path)
            results.append(result)
            self.files_processed += 1
            
            if result['status'] == 'modified':
                print(f"  ✅ {file_path.name}: {', '.join(result['fixes'])}")
            elif result['status'] == 'error':
                print(f"  ❌ {file_path.name}: {result['error']}")
        
        # Print summary
        print(f"\n📊 AUTO-FIX SUMMARY")
        print(f"=" * 40)
        print(f"Files processed: {self.files_processed}")
        print(f"Files modified: {self.files_modified}")
        print(f"Total fixes applied: {self.total_fixes}")
        
        if self.dry_run:
            print(f"\n💡 This was a dry run. Run without --dry-run to apply fixes.")
        else:
            print(f"\n✅ Auto-fixes completed!")
        
        return {
            'files_processed': self.files_processed,
            'files_modified': self.files_modified,
            'total_fixes': self.total_fixes,
            'results': results
        }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Automatically fix common code quality issues'
    )
    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='Directory to process (default: current directory)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without making changes'
    )
    
    args = parser.parse_args()
    directory = Path(args.directory)
    
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        sys.exit(1)
    
    fixer = AutoCodeQualityFixer(dry_run=args.dry_run)
    result = fixer.run_fixes(directory)
    
    if result['files_modified'] > 0:
        print(f"\n💡 Next steps:")
        print(f"  1. Review the changes: git diff")
        print(f"  2. Test that everything still works")
        print(f"  3. Run the code quality enforcer: python scripts/maintenance/code_quality_enforcer.py .")
        print(f"  4. Commit the fixes: git add . && git commit -m 'Auto-fix code quality issues'")


if __name__ == "__main__":
    main()
