"""Convert old Python functions into typed graphs without executing them."""

import ast
import re
import textwrap
from typing import Any

from .config import ROOT_FIELD

TOKEN_PATTERN = re.compile(
    r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[0-9]+|[A-Z]+"
)


def tokenize_identifier(text: str) -> list[str]:
    normalized = text.replace("_", " ")
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(normalized)]


def _node_lexemes(node: ast.AST) -> list[str]:
    value: Any = None
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        value = node.name
    elif isinstance(node, ast.Name):
        value = node.id
    elif isinstance(node, ast.Attribute):
        value = node.attr
    elif isinstance(node, ast.arg):
        value = node.arg
    elif isinstance(node, ast.alias):
        value = node.name
    elif isinstance(node, ast.Constant):
        value = node.value

    if isinstance(value, str):
        return tokenize_identifier(value)
    if value is None:
        return []
    if isinstance(value, bool):
        return [str(value).lower()]
    if isinstance(value, (int, float, complex)):
        return [str(value)]
    return [type(value).__name__.lower()]


def _parse(source: str) -> ast.AST:
    try:
        return ast.parse(source)
    except (SyntaxError, IndentationError):
        return ast.parse(textwrap.dedent(source))


def build_ast_graph(source: str) -> dict:
    """Build typed, bidirectional parent-child and sibling AST edges."""
    tree = _parse(source)
    node_types: list[str] = []
    fields: list[str] = []
    depths: list[int] = []
    lexemes: list[list[str]] = []
    edges: list[list[int | str]] = []

    def visit(node: ast.AST, parent: int | None, field: str, depth: int) -> int:
        node_index = len(node_types)
        node_types.append(type(node).__name__)
        fields.append(field)
        depths.append(depth)
        lexemes.append(_node_lexemes(node))

        if parent is not None:
            edges.append([parent, node_index, "ast_parent_to_child"])
            edges.append([node_index, parent, "ast_child_to_parent"])

        for child_field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                visit(value, node_index, child_field, depth + 1)
            elif isinstance(value, list):
                sibling_indices = [
                    visit(child, node_index, child_field, depth + 1)
                    for child in value
                    if isinstance(child, ast.AST)
                ]
                for left, right in zip(sibling_indices, sibling_indices[1:]):
                    edges.append([left, right, "ast_next_sibling"])
                    edges.append([right, left, "ast_previous_sibling"])
        return node_index

    visit(tree, None, ROOT_FIELD, 0)
    return {
        "node_types": node_types,
        "fields": fields,
        "depths": depths,
        "lexemes": lexemes,
        "edges": edges,
    }
