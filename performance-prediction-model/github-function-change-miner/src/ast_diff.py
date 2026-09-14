from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class FunctionInfo:
    qualified_name: str
    start_line: int
    end_line: int
    code: str
    body_fingerprint: str


@dataclass(frozen=True)
class FunctionBodyChange:
    qualified_name: str
    old: FunctionInfo
    new: FunctionInfo


class _FunctionCollector(ast.NodeVisitor):
    def __init__(self, source: str) -> None:
        """Initialize the collector for a specific source file.

        Args:
            source: Full Python source text to analyze.
        """
        self._source = source
        self._lines = source.splitlines(keepends=True)
        self._stack: List[str] = []
        self.functions: Dict[str, FunctionInfo] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class definition and collect functions within it.

        Pushes the class name onto the qualification stack while visiting the
        class body, then pops it afterward.

        Args:
            node: The class definition AST node.
        """
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a function definition and collect it (and nested functions).

        Adds the current function to `self.functions`, then visits its body while
        pushing the function name onto the qualification stack.

        Args:
            node: The function definition AST node.
        """
        self._add_function(node)
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit an async function definition and collect it (and nested functions).

        Adds the current function to `self.functions`, then visits its body while
        pushing the function name onto the qualification stack.

        Args:
            node: The async function definition AST node.
        """
        self._add_function(node)
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def _add_function(self, node: ast.AST) -> None:
        """Add a function-like node to `self.functions` if it has line metadata.

        Builds a qualified name from the current stack, extracts the source code
        for the node's line span, computes a body fingerprint, and stores a
        `FunctionInfo` record.

        If the node does not have a usable `lineno`, it is ignored. If `end_lineno`
        is missing, it falls back to `lineno`.

        Args:
            node: A function node (typically `ast.FunctionDef` or `ast.AsyncFunctionDef`).
        """
        name = getattr(node, "name", None)
        if not name:
            return
        qualified = ".".join([*self._stack, name]) if self._stack else name

        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if not isinstance(start, int):
            return
        if not isinstance(end, int):
            end = start

        code = "".join(self._lines[start - 1 : end])
        body_fp = _fingerprint_function_body(node)
        self.functions[qualified] = FunctionInfo(
            qualified_name=qualified,
            start_line=start,
            end_line=end,
            code=code,
            body_fingerprint=body_fp,
        )


def _fingerprint_function_body(node: ast.AST) -> str:
    """Compute a stable fingerprint for a function's body AST.

    The fingerprint is produced by dumping each statement in the node's `body`
    using `ast.dump(..., include_attributes=False)` and joining the results.
    Line/column metadata is excluded so formatting-only changes do not affect
    the fingerprint.

    Args:
        node: A function-like AST node with a `body` attribute.

    Returns:
        A fingerprint string, or an empty string if no body is available.
    """
    body = getattr(node, "body", None)
    if not isinstance(body, list):
        return ""

    return "\n".join(ast.dump(stmt, include_attributes=False) for stmt in body)


def extract_functions(source: str) -> Dict[str, FunctionInfo]:
    """Extract functions from Python source using the AST.

    Functions are keyed by a qualified name that reflects nesting (e.g.,
    `MyClass.method`, `outer.inner`). Each value includes source code, line
    range, and a body fingerprint.

    Args:
        source: Full Python source text to parse.

    Returns:
        A mapping of qualified function name to `FunctionInfo`. Returns an empty
        dict if the source cannot be parsed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    collector = _FunctionCollector(source)
    collector.visit(tree)

    return collector.functions


def find_function_body_changes(old_source: str, new_source: str) -> List[FunctionBodyChange]:
    """Find existing functions whose bodies changed between two source versions.

    Only functions present in both versions (same qualified name) are compared.
    A change is recorded when the function body fingerprint differs.

    Args:
        old_source: Source text from the older revision.
        new_source: Source text from the newer revision.

    Returns:
        A list of `FunctionBodyChange` records for functions whose bodies changed.
        Returns an empty list if either source has no parsable functions.
    """
    old_funcs = extract_functions(old_source)
    new_funcs = extract_functions(new_source)
    if not old_funcs or not new_funcs:
        return []

    changes: List[FunctionBodyChange] = []
    for qualified_name, old_info in old_funcs.items():
        new_info = new_funcs.get(qualified_name)
        if not new_info:
            continue
        if old_info.body_fingerprint != new_info.body_fingerprint:
            changes.append(FunctionBodyChange(qualified_name=qualified_name, old=old_info, new=new_info))

    return changes
