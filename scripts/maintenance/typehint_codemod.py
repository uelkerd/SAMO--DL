#!/usr/bin/env python3
"""
TypeHint Codemod for Python 3.8 Compatibility

This script converts Python 3.9+ type hints to Python 3.8 compatible syntax:
- list[T] -> List[T]
- dict[K, V] -> Dict[K, V]
- set[T] -> Set[T]
- tuple[T, ...] -> Tuple[T, ...]
- A | B -> Union[A, B] or Optional[A] for A | None

Usage:
    python scripts/maintenance/typehint_codemod.py <directory> [--dry-run] [--verbose]
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import List, Dict, Any

# Try to import astor for Python 3.8 compatibility
try:
    import astor
    def ast_to_source(node):
        return astor.to_source(node)
except ImportError:
    # Fallback for Python 3.8 without astor
    def ast_to_source(node):
        # Simple fallback - this won't be perfect but will work for basic cases
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Subscript):
            value = ast_to_source(node.value)
            slice_str = ast_to_source(node.slice)
            return f"{value}[{slice_str}]"
        elif isinstance(node, ast.Tuple):
            elts = [ast_to_source(elt) for elt in node.elts]
            return f"({', '.join(elts)})"
        elif isinstance(node, ast.Constant):
            if node.value is None:
                return "None"
            return str(node.value)
        else:
            return str(node)


class TypeHintVisitor(ast.NodeVisitor):
    """AST visitor to find and replace type hints."""

    def __init__(self):
        self.changes = []
        self.imports_to_add = set()

    def visit_AnnAssign(self, node):
        """Visit annotated assignments."""
        if node.annotation:
            new_annotation = self._convert_type_hint(node.annotation)
            if new_annotation != node.annotation:
                self.changes.append({
                    'type': 'annotation',
                    'node': node,
                    'old': node.annotation,
                    'new': new_annotation
                })
        self.generic_visit(node)

    def visit_arg(self, node):
        """Visit function arguments."""
        if node.annotation:
            new_annotation = self._convert_type_hint(node.annotation)
            if new_annotation != node.annotation:
                self.changes.append({
                    'type': 'arg',
                    'node': node,
                    'old': node.annotation,
                    'new': new_annotation
                })
        self.generic_visit(node)


    def visit_FunctionDef(self, node):
        """Visit function definitions."""
        if node.returns:
            new_returns = self._convert_type_hint(node.returns)
            if new_returns != node.returns:
                self.changes.append({
                    'type': 'returns',
                    'node': node,
                    'old': node.returns,
                    'new': new_returns
                })
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        """Visit async function definitions."""
        if node.returns:
            new_returns = self._convert_type_hint(node.returns)
            if new_returns != node.returns:
                self.changes.append({
                    'type': 'returns',
                    'node': node,
                    'old': node.returns,
                    'new': new_returns
                })
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        """Visit class definitions."""
        for base in node.bases:
            new_base = self._convert_type_hint(base)
            if new_base != base:
                self.changes.append({
                    'type': 'base',
                    'node': node,
                    'old': base,
                    'new': new_base
                })
        self.generic_visit(node)

    def _convert_type_hint(self, node):
        """Convert a type hint node to Python 3.8 compatible syntax."""
        if isinstance(node, ast.Subscript):
            return self._convert_subscript(node)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return self._convert_union(node)
        return node

    def _convert_subscript(self, node):
        """Convert subscript type hints (list[T], dict[K, V], etc.)."""
        if isinstance(node.value, ast.Name):
            name = node.value.id
            if name in ['list', 'dict', 'set', 'tuple']:
                # Convert to typing module equivalent
                if name == 'list':
                    new_name = ast.Name(id='List', ctx=ast.Load())
                elif name == 'dict':
                    new_name = ast.Name(id='Dict', ctx=ast.Load())
                elif name == 'set':
                    new_name = ast.Name(id='Set', ctx=ast.Load())
                elif name == 'tuple':
                    new_name = ast.Name(id='Tuple', ctx=ast.Load())

                # Add to imports
                self.imports_to_add.add(name.capitalize())

                # Create new subscript node
                return ast.Subscript(
                    value=new_name,
                    slice=node.slice,
                    ctx=node.ctx
                )
        return node

    def _convert_union(self, node):
        """Convert union type hints (A | B -> Union[A, B])."""
        # Handle A | None -> Optional[A] case
        if isinstance(node.right, ast.Constant) and node.right.value is None:
            self.imports_to_add.add('Optional')
            return ast.Subscript(
                value=ast.Name(id='Optional', ctx=ast.Load()),
                slice=node.left,
                ctx=ast.Load()
            )
        elif isinstance(node.left, ast.Constant) and node.left.value is None:
            self.imports_to_add.add('Optional')
            return ast.Subscript(
                value=ast.Name(id='Optional', ctx=ast.Load()),
                slice=node.right,
                ctx=ast.Load()
            )
        else:
            # General union case
            self.imports_to_add.add('Union')
            return ast.Subscript(
                value=ast.Name(id='Union', ctx=ast.Load()),
                slice=ast.Tuple(
                    elts=[node.left, node.right],
                    ctx=ast.Load()
                ),
                ctx=ast.Load()
            )


def process_file(
    file_path: Path, dry_run: bool = False, verbose: bool = False
) -> Dict[str, Any]:
    """Process a single Python file for type hint conversions."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse the file
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            if verbose:
                print(f"  ⚠️  Syntax error in {file_path}: {e}")
            return {'file': str(file_path), 'status': 'syntax_error', 'error': str(e)}

        # Visit the AST
        visitor = TypeHintVisitor()
        visitor.visit(tree)

        if not visitor.changes:
            return {'file': str(file_path), 'status': 'no_changes', 'changes': 0}

        # Apply changes
        if not dry_run:
            # Sort changes by line number (reverse order to avoid offset issues)
            visitor.changes.sort(
                key=lambda x: getattr(x['node'], 'lineno', 0), reverse=True
            )

            # Convert content to lines for easier manipulation
            lines = content.splitlines()

            for change in visitor.changes:
                if change['type'] == 'annotation':
                    old_code = ast_to_source(change['old'])
                    new_code = ast_to_source(change['new'])
                    if verbose:
                        print(f"    Annotation: {old_code} -> {new_code}")

                elif change['type'] == 'arg':
                    old_code = ast_to_source(change['old'])
                    new_code = ast_to_source(change['new'])
                    if verbose:
                        print(f"    Argument: {old_code} -> {new_code}")

                elif change['type'] == 'returns':
                    old_code = ast_to_source(change['old'])
                    new_code = ast_to_source(change['new'])
                    if verbose:
                        print(f"    Returns: {old_code} -> {new_code}")

                elif change['type'] == 'base':
                    old_code = ast_to_source(change['old'])
                    new_code = ast_to_source(change['new'])
                    if verbose:
                        print(f"    Base: {old_code} -> {new_code}")

            # Add missing imports
            if visitor.imports_to_add:
                import_lines = []
                for import_name in sorted(visitor.imports_to_add):
                    import_lines.append(f"from typing import {import_name}")

                # Find the last typing import or add after existing imports
                lines_with_imports = []
                typing_import_found = False
                last_import_line = -1

                for i, line in enumerate(lines):
                    lines_with_imports.append(line)
                    if line.strip().startswith('from typing import'):
                        typing_import_found = True
                        last_import_line = i
                    elif (line.strip().startswith('import ') or
                          line.strip().startswith('from ')):
                        last_import_line = i

                if typing_import_found:
                    # Add to existing typing import
                    for i, line in enumerate(lines):
                        if line.strip().startswith('from typing import'):
                            existing_imports = line.replace('from typing import ', '').strip()
                            new_imports = ', '.join(sorted(visitor.imports_to_add))
                            if existing_imports:
                                new_import_line = (
                                f"from typing import {existing_imports}, {new_imports}"
                            )
                            lines[i] = new_import_line
                        else:
                            lines[i] = f"from typing import {new_imports}"
                            break
                else:
                    # Add new typing import after last import
                    if last_import_line >= 0:
                        import_line = (
                            f"from typing import {', '.join(sorted(visitor.imports_to_add))}"
                        )
                        lines.insert(last_import_line + 1, import_line)
                    else:
                        import_line = (
                            f"from typing import {', '.join(sorted(visitor.imports_to_add))}"
                        )
                        lines.insert(0, import_line)

                # Write back to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))

        return {
            'file': str(file_path),
            'status': 'success',
            'changes': len(visitor.changes),
            'imports_added': list(visitor.imports_to_add)
        }

    except Exception as e:
        return {'file': str(file_path), 'status': 'error', 'error': str(e)}


def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in a directory recursively."""
    python_files = []
    for item in directory.rglob('*.py'):
        if not any(part.startswith('.') for part in item.parts):
            python_files.append(item)
    return python_files


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description=(
            'Convert Python 3.9+ type hints to Python 3.8 compatible syntax'
        )
    )
    parser.add_argument('directory', help='Directory to process')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without making changes')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')

    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        sys.exit(1)

    if not directory.is_dir():
        print(f"Error: {directory} is not a directory")
        sys.exit(1)

    print(f"Processing directory: {directory}")
    if args.dry_run:
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
        if args.verbose:
            print(f"Processing: {file_path}")

        result = process_file(file_path, dry_run=args.dry_run, verbose=args.verbose)
        results.append(result)

        if result['status'] == 'success' and result['changes'] > 0:
            total_changes += result['changes']
            if args.verbose:
                print(
                    f"  ✅ {result['changes']} changes, "
                    f"imports: {result['imports_added']}"
                )
        elif result['status'] == 'no_changes':
            if args.verbose:
                print(f"  ⏭️  No changes needed")
        elif result['status'] == 'error':
            print(f"  ❌ Error: {result['error']}")
        elif result['status'] == 'syntax_error':
            print(f"  ⚠️  Syntax error: {result['error']}")

        if args.verbose:
            print()

    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)

    successful = [r for r in results if r['status'] == 'success']
    errors = [r for r in results if r['status'] == 'error']
    syntax_errors = [r for r in results if r['status'] == 'syntax_error']
    no_changes = [r for r in results if r['status'] == 'no_changes']

    print(f"Files processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Errors: {len(errors)}")
    print(f"Syntax errors: {len(syntax_errors)}")
    print(f"No changes needed: {len(no_changes)}")
    print(f"Total changes: {total_changes}")

    if errors:
        print("\nFiles with errors:")
        for result in errors:
            print(f"  {result['file']}: {result['error']}")

    if syntax_errors:
        print("\nFiles with syntax errors:")
        for result in syntax_errors:
            print(f"  {result['file']}: {result['error']}")

    if args.dry_run and total_changes > 0:
        print(f"\nTo apply these changes, run without --dry-run")

    print()


if __name__ == '__main__':
    main()
