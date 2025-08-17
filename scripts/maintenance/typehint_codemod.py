#!/usr/bin/env python3
""""
TypeHint Codemod for Python 3.8 Compatibility

This script converts Python 3.9+ type hints to Python 3.8 compatible syntax:
- list[T] -> List[T]
- dict[K, V] -> Dict[K, V]
- set[T] -> Set[T]
- tuple[T, ...] -> Tuple[T, ...]
- A | B -> Union[A, B] or Optional[A] for A | None

Usage:
    python scripts/maintenance/typehint_codemod.py <directory> [--dry-run] [--verbose]
""""

import argparse
import ast
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Try to import astor for Python 3.8 compatibility
try:
    import astor
    def ast_to_source(node):
        """Convert AST node to source code using astor library."

        Args:
            node: AST node to convert

        Returns:
            str: Source code representation of the node
        """"
        return astor.to_source(node)
except ImportError:
    # Fallback for Python 3.8 without astor
    def ast_to_source(node):
        """Convert AST node to source code using simple fallback."

        This is a basic fallback when astor is not available.
        Only handles simple cases like ast.Name nodes.

        Args:
            node: AST node to convert

        Returns:
            str: Source code representation of the node (basic cases only)
        """"
        # Simple fallback - this won't be perfect but will work for basic cases'
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Subscript):
            value = ast_to_source(node.value)
            slice_str = ast_to_source(node.slice)
            return f"{value}[{slice_str}]"
        if isinstance(node, ast.Tuple):
            elts = [ast_to_source(elt) for elt in node.elts]
            return "({", '.join(elts)})""
        if isinstance(node, ast.Constant):
            if node.value is None:
                return "None"
            return str(node.value)
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
                self.changes.append({)
                    'type': 'annotation',
                    'node': node,
                    'old': node.annotation,
                    'new': new_annotation
(                })
        self.generic_visit(node)

            def visit_arg(self, node):
        """Visit function arguments."""
            if node.annotation:
            new_annotation = self._convert_type_hint(node.annotation)
            if new_annotation != node.annotation:
                self.changes.append({)
                    'type': 'arg',
                    'node': node,
                    'old': node.annotation,
                    'new': new_annotation
(                })
        self.generic_visit(node)


            def visit_FunctionDef(self, node):
        """Visit function definitions."""
            if node.returns:
            new_returns = self._convert_type_hint(node.returns)
            if new_returns != node.returns:
                self.changes.append({)
                    'type': 'returns',
                    'node': node,
                    'old': node.returns,
                    'new': new_returns
(                })
        self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node):
        """Visit async function definitions."""
            if node.returns:
            new_returns = self._convert_type_hint(node.returns)
            if new_returns != node.returns:
                self.changes.append({)
                    'type': 'returns',
                    'node': node,
                    'old': node.returns,
                    'new': new_returns
(                })
        self.generic_visit(node)

            def visit_ClassDef(self, node):
        """Visit class definitions."""
            for base in node.bases:
            new_base = self._convert_type_hint(base)
            if new_base != base:
                self.changes.append({)
                    'type': 'base',
                    'node': node,
                    'old': base,
                    'new': new_base
(                })
        self.generic_visit(node)

            def _convert_type_hint(self, node):
        """Convert a type hint node to Python 3.8 compatible syntax."""
            if isinstance(node, ast.Subscript):
            return self._convert_subscript(node)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
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
                return ast.Subscript()
                    value=new_name,
                    slice=node.slice,
                    ctx=node.ctx
(                )
        return node

                def _convert_union(self, node):
        """Convert union type hints (A | B -> Union[A, B])."""
        # Handle A | None -> Optional[A] case
                if isinstance(node.right, ast.Constant) and node.right.value is None:
            self.imports_to_add.add('Optional')
            return ast.Subscript()
                value=ast.Name(id='Optional', ctx=ast.Load()),
                slice=node.left,
                ctx=ast.Load()
(            )
                if isinstance(node.left, ast.Constant) and node.left.value is None:
            self.imports_to_add.add('Optional')
            return ast.Subscript()
                value=ast.Name(id='Optional', ctx=ast.Load()),
                slice=node.right,
                ctx=ast.Load()
(            )
        # General union case
        self.imports_to_add.add('Union')
        return ast.Subscript()
            value=ast.Name(id='Union', ctx=ast.Load()),
            slice=ast.Tuple()
                elts=[node.left, node.right],
                ctx=ast.Load()
(            ),
            ctx=ast.Load()
(        )


                def _log_changes(visitor: TypeHintVisitor, verbose: bool) -> None:
    """Log AST changes for debugging purposes."""
                for change in visitor.changes:
        old_code = ast_to_source(change['old'])
        new_code = ast_to_source(change['new'])

                if verbose:
            print("    {change["type'].title()}: {old_code} -> {new_code}")"


                def _add_typing_imports_to_lines(lines: List[str], imports_to_add: set) -> None:
    """Add missing typing imports to the lines."""
                if not imports_to_add:
        return

    # Find the last typing import or add after existing imports
    typing_import_found = False
    last_import_line = -1

                for i, line in enumerate(lines):
                if line.strip().startswith('from typing import'):
            typing_import_found = True
            last_import_line = i
        elif (line.strip().startswith('import ') or)
(              line.strip().startswith('from ')):
            last_import_line = i

                if typing_import_found:
        # Add to existing typing import
                for i, line in enumerate(lines):
                if line.strip().startswith('from typing import'):
                existing_imports = line.replace('from typing import ', '').strip()
                new_imports = ', '.join(sorted(imports_to_add))
                if existing_imports:
                    new_import_line = ()
                        f"from typing import {existing_imports}, {new_imports}"
(                    )
                    lines[i] = new_import_line
                else:
                    lines[i] = f"from typing import {new_imports}"
                break
    else:
        # Add new typing import after last import
                if last_import_line >= 0:
            import_line = ()
                "from typing import {", '.join(sorted(imports_to_add))}""
(            )
            lines.insert(last_import_line + 1, import_line)
        else:
            import_line = ()
                "from typing import {", '.join(sorted(imports_to_add))}""
(            )
            lines.insert(0, import_line)


                def _read_file_content(file_path: Path) -> str:
    """Read file content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


                def _parse_ast_safely(content: str, file_path: Path, verbose: bool) -> Optional[ast.AST]:
    """Parse AST safely, returning None on syntax error."""
    try:
        return ast.parse(content)
    except SyntaxError as e:
                if verbose:
            print(f"  ⚠️  Syntax error in {file_path}: {e}")
        return None


                def _apply_changes_and_save(file_path: Path, content: str, visitor: TypeHintVisitor, verbose: bool) -> None:
    """Apply changes and save the file."""
    # Sort changes by line number (reverse order to avoid offset issues)
    visitor.changes.sort()
        key=lambda x: getattr(x['node'], 'lineno', 0), reverse=True
(    )

    # Convert content to lines for easier manipulation
    lines = content.splitlines()

    # Log AST changes for debugging
    _log_changes(visitor, verbose)

    # Add missing imports
    _add_typing_imports_to_lines(lines, visitor.imports_to_add)

    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


                def _create_success_result(file_path: Path, visitor: TypeHintVisitor) -> Dict[str, Any]:
    """Create success result dictionary."""
    return {
        'file': str(file_path),
        'status': 'success',
        'changes': len(visitor.changes),
        'imports_added': list(visitor.imports_to_add)
    }


                def _create_error_result(file_path: Path, error: str) -> Dict[str, Any]:
    """Create error result dictionary."""
    return {'file': str(file_path), 'status': 'error', 'error': error}


                def _create_syntax_error_result(file_path: Path, error: str) -> Dict[str, Any]:
    """Create syntax error result dictionary."""
    return {'file': str(file_path), 'status': 'syntax_error', 'error': error}


                def _create_no_changes_result(file_path: Path) -> Dict[str, Any]:
    """Create no changes result dictionary."""
    return {'file': str(file_path), 'status': 'no_changes', 'changes': 0}


                def process_file()
    file_path: Path, dry_run: bool = False, verbose: bool = False
() -> Dict[str, Any]:
    """Process a single Python file for type hint conversions."""
    try:
        # Read file content
        content = _read_file_content(file_path)

        # Parse the file
        tree = _parse_ast_safely(content, file_path, verbose)
                if tree is None:
            return _create_syntax_error_result(file_path, "Syntax error during parsing")

        # Visit the AST
        visitor = TypeHintVisitor()
        visitor.visit(tree)

                if not visitor.changes:
            return _create_no_changes_result(file_path)

        # Apply changes if not dry run
                if not dry_run:
            _apply_changes_and_save(file_path, content, visitor, verbose)

        return _create_success_result(file_path, visitor)

    except Exception as e:
        return _create_error_result(file_path, str(e))


                def _process_single_file(file_path: Path, dry_run: bool, verbose: bool) -> Dict[str, Any]:
    """Process a single file and return the result."""
                if verbose:
        print(f"Processing: {file_path}")

    result = process_file(file_path, dry_run=dry_run, verbose=verbose)

                if result['status'] == 'success' and result['changes'] > 0:
                if verbose:
            print()
                "   {result["changes']} changes, ""
                "imports: {result["imports_added']}""
(            )
    elif result['status'] == 'no_changes':
                if verbose:
            print("  ⏭️  No changes needed")
    elif result['status'] == 'error':
        print("  ❌ Error: {result["error']}")"
    elif result['status'] == 'syntax_error':
        print("  ⚠️  Syntax error: {result["error']}")"

                if verbose:
        print()

    return result


                def _print_summary(results: List[Dict[str, Any]], total_changes: int, dry_run: bool) -> None:
    """Print summary of processing results."""
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
            print("  {result["file']}: {result['error']}")"

                if syntax_errors:
        print("\nFiles with syntax errors:")
                for result in syntax_errors:
            print("  {result["file']}: {result['error']}")"

                if dry_run and total_changes > 0:
        print("\nTo apply these changes, run without --dry-run")


                def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in a directory recursively."""
    python_files = []
                for item in directory.rglob('*.py'):
                if not any(part.startswith('.') for part in item.parts):
            python_files.append(item)
    return python_files


                def _parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
        description=()
            'Convert Python 3.9+ type hints to Python 3.8 compatible syntax'
(        )
(    )
    parser.add_argument('directory', help='Directory to process')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without making changes')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    return parser.parse_args()


                def _validate_directory(directory: Path) -> None:
    """Validate that the directory exists and is a directory."""
                if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        sys.exit(1)

                if not directory.is_dir():
        print(f"Error: {directory} is not a directory")
        sys.exit(1)


                def _print_processing_info(directory: Path, dry_run: bool) -> None:
    """Print processing information."""
    print(f"Processing directory: {directory}")
                if dry_run:
        print("DRY RUN MODE - No changes will be made")
    print()


                def _process_all_files(python_files: List[Path], dry_run: bool, verbose: bool) -> Tuple[List[Dict[str, Any]], int]:
    """Process all Python files and return results and total changes."""
    results = []
    total_changes = 0

                for file_path in python_files:
        result = _process_single_file(file_path, dry_run, verbose)
        results.append(result)

                if result['status'] == 'success' and result['changes'] > 0:
            total_changes += result['changes']

    return results, total_changes


                def main():
    """Main function."""
    # Parse and validate arguments
    args = _parse_arguments()
    directory = Path(args.directory)
    _validate_directory(directory)

    # Print processing info
    _print_processing_info(directory, args.dry_run)

    # Find and process Python files
    python_files = find_python_files(directory)
    print(f"Found {len(python_files)} Python files")
    print()

    results, total_changes = _process_all_files(python_files, args.dry_run, args.verbose)

    # Print summary
    _print_summary(results, total_changes, args.dry_run)
    print()


                if __name__ == '__main__':
    main()
