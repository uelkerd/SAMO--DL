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
from typing import List, Dict, Any, Optional, Tuple

# Try to import astor for Python 3.8 compatibility
try:
    import astor
    def ast_to_sourcenode:
        """Convert AST node to source code using astor library.

        Args:
            node: AST node to convert

        Returns:
            str: Source code representation of the node
        """
        return astor.to_sourcenode
except ImportError:
    # Fallback for Python 3.8 without astor
    def ast_to_sourcenode:
        """Convert AST node to source code using simple fallback.

        This is a basic fallback when astor is not available.
        Only handles simple cases like ast.Name nodes.

        Args:
            node: AST node to convert

        Returns:
            str: Source code representation of the node basic cases only
        """
        # Simple fallback - this won't be perfect but will work for basic cases
        if isinstancenode, ast.Name:
            return node.id
        if isinstancenode, ast.Subscript:
            value = ast_to_sourcenode.value
            slice_str = ast_to_sourcenode.slice
            return f"{value}[{slice_str}]"
        if isinstancenode, ast.Tuple:
            elts = [ast_to_sourceelt for elt in node.elts]
            return f"({', '.joinelts})"
        if isinstancenode, ast.Constant:
            if node.value is None:
                return "None"
            return strnode.value
        return strnode


class TypeHintVisitorast.NodeVisitor:
    """AST visitor to find and replace type hints."""

    def __init__self:
        self.changes = []
        self.imports_to_add = set()

    def visit_AnnAssignself, node:
        """Visit annotated assignments."""
        if node.annotation:
            new_annotation = self._convert_type_hintnode.annotation
            if new_annotation != node.annotation:
                self.changes.append({
                    'type': 'annotation',
                    'node': node,
                    'old': node.annotation,
                    'new': new_annotation
                })
        self.generic_visitnode

    def visit_argself, node:
        """Visit function arguments."""
        if node.annotation:
            new_annotation = self._convert_type_hintnode.annotation
            if new_annotation != node.annotation:
                self.changes.append({
                    'type': 'arg',
                    'node': node,
                    'old': node.annotation,
                    'new': new_annotation
                })
        self.generic_visitnode


    def visit_FunctionDefself, node:
        """Visit function definitions."""
        if node.returns:
            new_returns = self._convert_type_hintnode.returns
            if new_returns != node.returns:
                self.changes.append({
                    'type': 'returns',
                    'node': node,
                    'old': node.returns,
                    'new': new_returns
                })
        self.generic_visitnode

    def visit_AsyncFunctionDefself, node:
        """Visit async function definitions."""
        if node.returns:
            new_returns = self._convert_type_hintnode.returns
            if new_returns != node.returns:
                self.changes.append({
                    'type': 'returns',
                    'node': node,
                    'old': node.returns,
                    'new': new_returns
                })
        self.generic_visitnode

    def visit_ClassDefself, node:
        """Visit class definitions."""
        for base in node.bases:
            new_base = self._convert_type_hintbase
            if new_base != base:
                self.changes.append({
                    'type': 'base',
                    'node': node,
                    'old': base,
                    'new': new_base
                })
        self.generic_visitnode

    def _convert_type_hintself, node:
        """Convert a type hint node to Python 3.8 compatible syntax."""
        if isinstancenode, ast.Subscript:
            return self._convert_subscriptnode
        if isinstancenode, ast.BinOp and isinstancenode.op, ast.BitOr:
            return self._convert_unionnode
        return node

    def _convert_subscriptself, node:
        """Convert subscript type hints list[T], dict[K, V], etc.."""
        if isinstancenode.value, ast.Name:
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

    def _convert_unionself, node:
        """Convert union type hints A | B -> Union[A, B]."""
        # Handle A | None -> Optional[A] case
        if isinstancenode.right, ast.Constant and node.right.value is None:
            self.imports_to_add.add'Optional'
            return ast.Subscript(
                value=ast.Name(id='Optional', ctx=ast.Load()),
                slice=node.left,
                ctx=ast.Load()
            )
        if isinstancenode.left, ast.Constant and node.left.value is None:
            self.imports_to_add.add'Optional'
            return ast.Subscript(
                value=ast.Name(id='Optional', ctx=ast.Load()),
                slice=node.right,
                ctx=ast.Load()
            )
        # General union case
        self.imports_to_add.add'Union'
        return ast.Subscript(
            value=ast.Name(id='Union', ctx=ast.Load()),
            slice=ast.Tuple(
                elts=[node.left, node.right],
                ctx=ast.Load()
            ),
            ctx=ast.Load()
        )


def _log_changesvisitor: TypeHintVisitor, verbose: bool -> None:
    """Log AST changes for debugging purposes."""
    for change in visitor.changes:
        old_code = ast_to_sourcechange['old']
        new_code = ast_to_sourcechange['new']

        if verbose:
            print(f"    {change['type'].title()}: {old_code} -> {new_code}")


def _add_typing_imports_to_lineslines: List[str], imports_to_add: set -> None:
    """Add missing typing imports to the lines."""
    if not imports_to_add:
        return

    # Find the last typing import or add after existing imports
    typing_import_found = False
    last_import_line = -1

    for i, line in enumeratelines:
        if line.strip().startswith'from typing import':
            typing_import_found = True
            last_import_line = i
        elif (line.strip().startswith'import ' or
              line.strip().startswith'from '):
            last_import_line = i

    if typing_import_found:
        # Add to existing typing import
        for i, line in enumeratelines:
            if line.strip().startswith'from typing import':
                existing_imports = line.replace'from typing import ', ''.strip()
                new_imports = ', '.join(sortedimports_to_add)
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
                f"from typing import {', '.join(sortedimports_to_add)}"
            )
            lines.insertlast_import_line + 1, import_line
        else:
            import_line = (
                f"from typing import {', '.join(sortedimports_to_add)}"
            )
            lines.insert0, import_line


def _read_file_contentfile_path: Path -> str:
    """Read file content."""
    with openfile_path, 'r', encoding='utf-8' as f:
        return f.read()


def _parse_ast_safelycontent: str, file_path: Path, verbose: bool -> Optional[ast.AST]:
    """Parse AST safely, returning None on syntax error."""
    try:
        return ast.parsecontent
    except SyntaxError as e:
        if verbose:
            printf"  ⚠️  Syntax error in {file_path}: {e}"
        return None


def _apply_changes_and_savefile_path: Path, content: str, visitor: TypeHintVisitor, verbose: bool -> None:
    """Apply changes and save the file."""
    # Sort changes by line number reverse order to avoid offset issues
    visitor.changes.sort(
        key=lambda x: getattrx['node'], 'lineno', 0, reverse=True
    )

    # Convert content to lines for easier manipulation
    lines = content.splitlines()

    # Log AST changes for debugging
    _log_changesvisitor, verbose

    # Add missing imports
    _add_typing_imports_to_lineslines, visitor.imports_to_add

    # Write back to file
    with openfile_path, 'w', encoding='utf-8' as f:
        f.write('\n'.joinlines)


def _create_success_resultfile_path: Path, visitor: TypeHintVisitor -> Dict[str, Any]:
    """Create success result dictionary."""
    return {
        'file': strfile_path,
        'status': 'success',
        'changes': lenvisitor.changes,
        'imports_added': listvisitor.imports_to_add
    }


def _create_error_resultfile_path: Path, error: str -> Dict[str, Any]:
    """Create error result dictionary."""
    return {'file': strfile_path, 'status': 'error', 'error': error}


def _create_syntax_error_resultfile_path: Path, error: str -> Dict[str, Any]:
    """Create syntax error result dictionary."""
    return {'file': strfile_path, 'status': 'syntax_error', 'error': error}


def _create_no_changes_resultfile_path: Path -> Dict[str, Any]:
    """Create no changes result dictionary."""
    return {'file': strfile_path, 'status': 'no_changes', 'changes': 0}


def process_file(
    file_path: Path, dry_run: bool = False, verbose: bool = False
) -> Dict[str, Any]:
    """Process a single Python file for type hint conversions."""
    try:
        # Read file content
        content = _read_file_contentfile_path

        # Parse the file
        tree = _parse_ast_safelycontent, file_path, verbose
        if tree is None:
            return _create_syntax_error_resultfile_path, "Syntax error during parsing"

        # Visit the AST
        visitor = TypeHintVisitor()
        visitor.visittree

        if not visitor.changes:
            return _create_no_changes_resultfile_path

        # Apply changes if not dry run
        if not dry_run:
            _apply_changes_and_savefile_path, content, visitor, verbose

        return _create_success_resultfile_path, visitor

    except Exception as e:
        return _create_error_result(file_path, stre)


def _process_single_filefile_path: Path, dry_run: bool, verbose: bool -> Dict[str, Any]:
    """Process a single file and return the result."""
    if verbose:
        printf"Processing: {file_path}"

    result = process_filefile_path, dry_run=dry_run, verbose=verbose

    if result['status'] == 'success' and result['changes'] > 0:
        if verbose:
            print(
                f"  ✅ {result['changes']} changes, "
                f"imports: {result['imports_added']}"
            )
    elif result['status'] == 'no_changes':
        if verbose:
            print"  ⏭️  No changes needed"
    elif result['status'] == 'error':
        printf"  ❌ Error: {result['error']}"
    elif result['status'] == 'syntax_error':
        printf"  ⚠️  Syntax error: {result['error']}"

    if verbose:
        print()

    return result


def _print_summaryresults: List[Dict[str, Any]], total_changes: int, dry_run: bool -> None:
    """Print summary of processing results."""
    print"=" * 50
    print"SUMMARY"
    print"=" * 50

    successful = [r for r in results if r['status'] == 'success']
    errors = [r for r in results if r['status'] == 'error']
    syntax_errors = [r for r in results if r['status'] == 'syntax_error']
    no_changes = [r for r in results if r['status'] == 'no_changes']

    print(f"Files processed: {lenresults}")
    print(f"Successful: {lensuccessful}")
    print(f"Errors: {lenerrors}")
    print(f"Syntax errors: {lensyntax_errors}")
    print(f"No changes needed: {lenno_changes}")
    printf"Total changes: {total_changes}"

    if errors:
        print"\nFiles with errors:"
        for result in errors:
            printf"  {result['file']}: {result['error']}"

    if syntax_errors:
        print"\nFiles with syntax errors:"
        for result in syntax_errors:
            printf"  {result['file']}: {result['error']}"

    if dry_run and total_changes > 0:
        print"\nTo apply these changes, run without --dry-run"


def find_python_filesdirectory: Path -> List[Path]:
    """Find all Python files in a directory recursively."""
    python_files = []
    for item in directory.rglob'*.py':
        if not any(part.startswith'.' for part in item.parts):
            python_files.appenditem
    return python_files


def _parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            'Convert Python 3.9+ type hints to Python 3.8 compatible syntax'
        )
    )
    parser.add_argument'directory', help='Directory to process'
    parser.add_argument'--dry-run', action='store_true', help='Show what would be changed without making changes'
    parser.add_argument'--verbose', action='store_true', help='Show detailed output'
    return parser.parse_args()


def _validate_directorydirectory: Path -> None:
    """Validate that the directory exists and is a directory."""
    if not directory.exists():
        printf"Error: Directory {directory} does not exist"
        sys.exit1

    if not directory.is_dir():
        printf"Error: {directory} is not a directory"
        sys.exit1


def _print_processing_infodirectory: Path, dry_run: bool -> None:
    """Print processing information."""
    printf"Processing directory: {directory}"
    if dry_run:
        print"DRY RUN MODE - No changes will be made"
    print()


def _process_all_filespython_files: List[Path], dry_run: bool, verbose: bool -> Tuple[List[Dict[str, Any]], int]:
    """Process all Python files and return results and total changes."""
    results = []
    total_changes = 0

    for file_path in python_files:
        result = _process_single_filefile_path, dry_run, verbose
        results.appendresult

        if result['status'] == 'success' and result['changes'] > 0:
            total_changes += result['changes']

    return results, total_changes


def main():
    """Main function."""
    # Parse and validate arguments
    args = _parse_arguments()
    directory = Pathargs.directory
    _validate_directorydirectory

    # Print processing info
    _print_processing_infodirectory, args.dry_run

    # Find and process Python files
    python_files = find_python_filesdirectory
    print(f"Found {lenpython_files} Python files")
    print()

    results, total_changes = _process_all_filespython_files, args.dry_run, args.verbose

    # Print summary
    _print_summaryresults, total_changes, args.dry_run
    print()


if __name__ == '__main__':
    main()
