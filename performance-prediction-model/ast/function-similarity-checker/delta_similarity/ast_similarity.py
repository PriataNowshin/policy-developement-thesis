"""AST edit-script similarity for Python function updates."""

from __future__ import annotations

import ast
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, Counter as CounterType, Dict, Iterable, List, Optional, Tuple


Operation = Tuple[str, str, str, str, str, str, str]
AstFeature = Tuple[str, str, str, str, str, str]


@dataclass(frozen=True)
class ParsedAst:
    """Parsed Python AST plus the mode needed to parse it."""

    tree: ast.AST
    mode: str


def parse_python_ast(code: str) -> ParsedAst:
    """Parse function code or a function-body fragment into a Python AST."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            return ParsedAst(tree=ast.parse(code), mode="module")
        except SyntaxError as module_error:
            wrapped = "def __ast_similarity_wrapper__():\n"
            wrapped += "".join(f"    {line}\n" for line in code.splitlines())
            try:
                return ParsedAst(tree=ast.parse(wrapped), mode="wrapped_function_body")
            except SyntaxError:
                raise module_error


class SemanticAstNormalizer(ast.NodeTransformer):
    """Normalize common equivalent Python forms before AST comparison.

    This does not rewrite dataset code. It only standardizes the in-memory AST
    representation used for similarity scoring.
    """

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if call_name(node.value) in {"Optional", "typing.Optional"}:
            normalized = ast.Call(
                func=ast.Name(id="__optional_type__", ctx=ast.Load()),
                args=[node.slice],
                keywords=[],
            )
            return ast.copy_location(normalized, node)
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.BitOr):
            left_is_none = is_none_constant(node.left)
            right_is_none = is_none_constant(node.right)
            if left_is_none != right_is_none:
                value = node.right if left_is_none else node.left
                normalized = ast.Call(
                    func=ast.Name(id="__optional_type__", ctx=ast.Load()),
                    args=[value],
                    keywords=[],
                )
                return ast.copy_location(normalized, node)
        return node

    def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.AST:
        self.generic_visit(node)
        if len(node.values) == 1 and isinstance(node.values[0], ast.FormattedValue):
            formatted = node.values[0]
            if formatted.conversion == -1:
                format_spec = fstring_format_spec(formatted.format_spec)
                if format_spec is not None:
                    normalized = ast.Call(
                        func=ast.Name(id="__format_value__", ctx=ast.Load()),
                        args=[formatted.value, ast.Constant(value=format_spec)],
                        keywords=[],
                    )
                    return ast.copy_location(normalized, node)

        template = fstring_template(node)
        if template is None:
            return node

        args = [value.value for value in node.values if isinstance(value, ast.FormattedValue)]
        normalized = ast.Call(
            func=ast.Name(id="__format_template__", ctx=ast.Load()),
            args=[ast.Constant(value=template), *args],
            keywords=[],
        )
        return ast.copy_location(normalized, node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if call_name(node.func) == "format" and len(node.args) >= 2 and not node.keywords:
            normalized = ast.Call(
                func=ast.Name(id="__format_value__", ctx=ast.Load()),
                args=[node.args[0], node.args[1]],
                keywords=[],
            )
            return ast.copy_location(normalized, node)

        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "format"
            and isinstance(node.func.value, ast.Constant)
            and isinstance(node.func.value.value, str)
            and not node.keywords
        ):
            normalized = ast.Call(
                func=ast.Name(id="__format_template__", ctx=ast.Load()),
                args=[ast.Constant(value=node.func.value.value), *node.args],
                keywords=[],
            )
            return ast.copy_location(normalized, node)
        return node


def normalize_ast(tree: ast.AST) -> ast.AST:
    """Return an AST normalized for similarity comparison."""

    normalized = SemanticAstNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    return normalized


def is_none_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def fstring_format_spec(node: ast.AST | None) -> str | None:
    if node is None:
        return ""
    if not isinstance(node, ast.JoinedStr):
        return None

    parts: List[str] = []
    for value in node.values:
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            return None
        parts.append(value.value)
    return "".join(parts)


def fstring_template(node: ast.JoinedStr) -> str | None:
    parts: List[str] = []
    found_formatted_value = False
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
            continue
        if isinstance(value, ast.FormattedValue) and value.conversion == -1 and value.format_spec is None:
            found_formatted_value = True
            parts.append("{}")
            continue
        return None
    return "".join(parts) if found_formatted_value else None


def node_detail(node: ast.AST) -> str:
    """Return a compact semantic detail for AST nodes where the value matters."""

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.arg):
        return node.arg
    if isinstance(node, ast.FunctionDef):
        return node.name
    if isinstance(node, ast.AsyncFunctionDef):
        return node.name
    if isinstance(node, ast.Call):
        return call_name(node.func)
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.alias):
        return node.name
    if isinstance(node, ast.keyword):
        return node.arg or ""
    if isinstance(node, ast.operator):
        return type(node).__name__
    if isinstance(node, ast.unaryop):
        return type(node).__name__
    if isinstance(node, ast.boolop):
        return type(node).__name__
    if isinstance(node, ast.cmpop):
        return type(node).__name__
    if isinstance(node, ast.comprehension):
        return "async" if node.is_async else "sync"
    return ""


def call_name(node: ast.AST) -> str:
    """Extract a dotted call/attribute/name string when possible."""

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return call_name(node.func)
    if isinstance(node, ast.Subscript):
        return call_name(node.value)
    return type(node).__name__


def ast_features(tree: ast.AST) -> CounterType[AstFeature]:
    """Collect a multiset of structural AST features with richer context."""

    features: CounterType[AstFeature] = Counter()

    def visit(
        node: ast.AST,
        parent_type: str,
        field_name: str,
        ancestors: List[str],
        statement_context: str,
    ) -> None:
        current_type = type(node).__name__
        current_statement = statement_context
        if isinstance(node, ast.stmt):
            current_statement = statement_signature(node)

        ancestor_path = ">".join(ancestors[-4:])
        features[(current_type, parent_type, field_name, ancestor_path, current_statement, node_detail(node))] += 1

        for child_field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                visit(value, current_type, child_field, ancestors + [current_type], current_statement)
            elif isinstance(value, list):
                for idx, child in enumerate(value):
                    if isinstance(child, ast.AST):
                        indexed_field = f"{child_field}[{statement_bucket(idx)}]"
                        visit(child, current_type, indexed_field, ancestors + [current_type], current_statement)

    visit(tree, "ROOT", "root", [], "ROOT")
    return features


def statement_signature(node: ast.stmt) -> str:
    """Return a compact statement-level context string."""

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return f"{type(node).__name__}:{node.name}"
    if isinstance(node, ast.Assign):
        return f"Assign:{','.join(expr_name(target) for target in node.targets)}"
    if isinstance(node, ast.AnnAssign):
        return f"AnnAssign:{expr_name(node.target)}"
    if isinstance(node, ast.AugAssign):
        return f"AugAssign:{expr_name(node.target)}"
    if isinstance(node, ast.Return):
        return "Return"
    if isinstance(node, ast.If):
        return f"If:{expr_name(node.test)}"
    if isinstance(node, (ast.For, ast.AsyncFor)):
        return f"{type(node).__name__}:{expr_name(node.target)}"
    if isinstance(node, ast.While):
        return f"While:{expr_name(node.test)}"
    if isinstance(node, (ast.With, ast.AsyncWith)):
        return type(node).__name__
    if isinstance(node, ast.Raise):
        return "Raise"
    if isinstance(node, ast.Assert):
        return "Assert"
    if isinstance(node, ast.Expr):
        return f"Expr:{expr_name(node.value)}"
    return type(node).__name__


def expr_name(node: ast.AST | None) -> str:
    """Return a compact expression identifier for statement context."""

    if node is None:
        return ""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return call_name(node)
    if isinstance(node, ast.Call):
        return call_name(node.func)
    if isinstance(node, ast.Subscript):
        return call_name(node.value)
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Compare):
        return expr_name(node.left)
    if isinstance(node, ast.BoolOp):
        return type(node.op).__name__
    if isinstance(node, ast.BinOp):
        return type(node.op).__name__
    return type(node).__name__


def statement_bucket(index: int) -> str:
    """Keep coarse list-position context without overfitting to exact line moves."""

    if index < 3:
        return str(index)
    if index < 10:
        return "3-9"
    return "10+"


def canonical_operations(old_tree: ast.AST, new_tree: ast.AST) -> List[Operation]:
    """Build a simple AST edit script from multiset feature differences."""

    old_features = ast_features(old_tree)
    new_features = ast_features(new_tree)

    deletes: CounterType[AstFeature] = old_features - new_features
    inserts: CounterType[AstFeature] = new_features - old_features

    operations: List[Operation] = []
    used_insert_keys: CounterType[AstFeature] = Counter()

    for deleted, delete_count in list(deletes.items()):
        node_type, parent_type, field_name, ancestor_path, statement_context, old_detail = deleted
        candidate: Optional[AstFeature] = None
        for inserted, insert_count in inserts.items():
            if used_insert_keys[inserted] >= insert_count:
                continue
            ins_node_type, ins_parent_type, ins_field_name, ins_ancestor_path, ins_statement_context, new_detail = inserted
            if (
                node_type == ins_node_type
                and parent_type == ins_parent_type
                and field_name == ins_field_name
                and ancestor_path == ins_ancestor_path
                and statement_context == ins_statement_context
                and old_detail != new_detail
            ):
                candidate = inserted
                break

        if candidate is None:
            for _ in range(delete_count):
                operations.append(
                    ("Delete", node_type, parent_type, field_name, ancestor_path, statement_context, old_detail)
                )
            continue

        update_count = min(delete_count, inserts[candidate] - used_insert_keys[candidate])
        _, _, _, _, _, new_detail = candidate
        for _ in range(update_count):
            operations.append(
                (
                    "Update",
                    node_type,
                    parent_type,
                    field_name,
                    ancestor_path,
                    statement_context,
                    f"{old_detail}->{new_detail}",
                )
            )
        used_insert_keys[candidate] += update_count

        for _ in range(delete_count - update_count):
            operations.append(
                ("Delete", node_type, parent_type, field_name, ancestor_path, statement_context, old_detail)
            )

    for inserted, insert_count in inserts.items():
        remaining = insert_count - used_insert_keys[inserted]
        node_type, parent_type, field_name, ancestor_path, statement_context, detail = inserted
        for _ in range(remaining):
            operations.append(
                ("Insert", node_type, parent_type, field_name, ancestor_path, statement_context, detail)
            )

    return sorted(operations)


def multiset_f1(left: Iterable[Any], right: Iterable[Any]) -> Dict[str, float]:
    """Compute multiset precision, recall, and F1."""

    left_counter = Counter(left)
    right_counter = Counter(right)
    left_count = sum(left_counter.values())
    right_count = sum(right_counter.values())

    if left_count == 0 and right_count == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "matched": 0.0}
    if left_count == 0 or right_count == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matched": 0.0}

    matched = sum((left_counter & right_counter).values())
    precision = matched / right_count
    recall = matched / left_count
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "matched": float(matched),
    }


def ast_structural_similarity(left_tree: ast.AST, right_tree: ast.AST) -> float:
    """Compare two final ASTs with a normalized structural multiset score."""

    left_features = ast_features(left_tree)
    right_features = ast_features(right_tree)
    left_count = sum(left_features.values())
    right_count = sum(right_features.values())

    if left_count == 0 and right_count == 0:
        return 1.0
    if left_count == 0 or right_count == 0:
        return 0.0

    matched = sum((left_features & right_features).values())
    return float((2 * matched) / (left_count + right_count))


def compare_ast_deltas(old_code: str, human_code: str, llm_code: str) -> Dict[str, Any]:
    """Compare human and LLM updates as AST edit scripts."""

    old_parsed = parse_python_ast(old_code)
    human_parsed = parse_python_ast(human_code)
    llm_parsed = parse_python_ast(llm_code)

    old_tree = normalize_ast(old_parsed.tree)
    human_tree = normalize_ast(human_parsed.tree)
    llm_tree = normalize_ast(llm_parsed.tree)

    human_operations = canonical_operations(old_tree, human_tree)
    llm_operations = canonical_operations(old_tree, llm_tree)
    operation_scores = multiset_f1(human_operations, llm_operations)

    return {
        "ast_delta_similarity": operation_scores["f1"],
        "operation_precision": operation_scores["precision"],
        "operation_recall": operation_scores["recall"],
        "matched_edit_count": int(operation_scores["matched"]),
        "human_edit_count": len(human_operations),
        "llm_edit_count": len(llm_operations),
        "final_ast_similarity": ast_structural_similarity(human_tree, llm_tree),
        "old_parse_mode": old_parsed.mode,
        "human_parse_mode": human_parsed.mode,
        "llm_parse_mode": llm_parsed.mode,
        "human_ast_edit_script": operation_to_dicts(human_operations),
        "llm_ast_edit_script": operation_to_dicts(llm_operations),
    }


def operation_to_dicts(operations: List[Operation]) -> List[Dict[str, str]]:
    """Convert operation tuples into JSON-friendly dictionaries."""

    return [
        {
            "operation": operation,
            "node_type": node_type,
            "parent_type": parent_type,
            "field": field_name,
            "ancestor_path": ancestor_path,
            "statement_context": statement_context,
            "detail": detail,
        }
        for operation, node_type, parent_type, field_name, ancestor_path, statement_context, detail in operations
    ]
