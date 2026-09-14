"""Deterministic, single-edit Python AST mutations for controlled validation."""

from __future__ import annotations

import ast
import copy
import hashlib
import random
import textwrap
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    taxonomy_category: str
    description: str
    applicability: str


@dataclass(frozen=True)
class Mutation:
    operator_id: str
    taxonomy_category: str
    code: str
    requirement: str
    location: dict[str, int]
    old_fragment: str
    new_fragment: str


OPERATOR_SPECS = {
    "modify_docstring_text": OperatorSpec(
        "modify_docstring_text", "documentation_edit",
        "Append a fixed clarification sentence to an existing function docstring.",
        "The function has an existing non-empty docstring.",
    ),
    "remove_output_statement": OperatorSpec(
        "remove_output_statement", "statement_removal",
        "Remove one standalone print(...) or .debug(...) output statement.",
        "A block contains an output statement and at least one other statement.",
    ),
    "replace_literal_value": OperatorSpec(
        "replace_literal_value", "literal_path_url_or_config",
        "Replace one non-docstring numeric or Boolean literal with a deterministic alternative.",
        "The function contains a numeric or Boolean literal outside its docstring.",
    ),
    "modify_return_expression": OperatorSpec(
        "modify_return_expression", "return_statement_edit",
        "Replace one non-empty return expression with None.",
        "The function contains return <expression> where the expression is not None.",
    ),
    "modify_comparison_operator": OperatorSpec(
        "modify_comparison_operator", "condition_or_comparison_edit",
        "Replace one comparison operator with its controlled counterpart.",
        "The function contains a supported comparison operator.",
    ),
    "modify_exception_handler": OperatorSpec(
        "modify_exception_handler", "exception_handling_edit",
        "Replace one bare or specific exception handler type with Exception.",
        "The function contains an except handler that is not already except Exception.",
    ),
    "modernize_string_formatting": OperatorSpec(
        "modernize_string_formatting", "string_formatting_edit",
        "Convert one simple f-string into an equivalent .format(...) expression.",
        "The function contains an f-string without conversions or format specifications.",
    ),
    "modify_type_annotation": OperatorSpec(
        "modify_type_annotation", "type_annotation_edit",
        "Add the built-in object annotation to one unannotated parameter.",
        "The function has an unannotated parameter other than self or cls.",
    ),
    "modernize_super_call": OperatorSpec(
        "modernize_super_call", "super_call_modernization",
        "Replace super(ClassName, self_or_cls) with zero-argument super().",
        "The function contains a two-positional-argument super call without keywords.",
    ),
    "rename_local_identifier": OperatorSpec(
        "rename_local_identifier", "identifier_rename",
        "Rename one function-local assigned identifier consistently.",
        "The root function assigns a safe local name that has no updated-name collision.",
    ),
}


def _source(code: str) -> str:
    source = textwrap.dedent(code).strip("\n") + "\n"
    ast.parse(source)
    return source


def _function(tree: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    return next(
        (node for node in getattr(tree, "body", []) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )


def _pick(source: str, operator_id: str, values: list):
    if not values:
        return None
    seed = int.from_bytes(
        hashlib.sha256(f"fusion-guided-v1\0{operator_id}\0{source}".encode()).digest()[:8],
        "big",
    )
    return random.Random(seed).choice(values)


def _fragment(node: ast.AST) -> str:
    return ast.unparse(node)


def _clean(text: str) -> str:
    return text


def _finish(
    source: str,
    tree: ast.AST,
    spec: OperatorSpec,
    target: ast.AST,
    old_fragment: str,
    new_fragment: str,
    instruction: str,
) -> Mutation:
    ast.fix_missing_locations(tree)
    code = ast.unparse(tree).strip() + "\n"
    ast.parse(code)
    if ast.dump(ast.parse(source), include_attributes=False) == ast.dump(ast.parse(code), include_attributes=False):
        raise ValueError(f"{spec.operator_id} produced no AST-visible change")
    line = int(getattr(target, "lineno", 1))
    column = int(getattr(target, "col_offset", 0))
    requirement = (
        f"In the old function, {instruction} at function line {line}, column {column}. "
        f"Change the exact fragment <before>{_clean(old_fragment)}</before> "
        f"to <after>{_clean(new_fragment)}</after>. "
        "Leave everything else unchanged."
    )
    return Mutation(
        operator_id=spec.operator_id,
        taxonomy_category=spec.taxonomy_category,
        code=code,
        requirement=requirement,
        location={"line": line, "column": column},
        old_fragment=old_fragment,
        new_fragment=new_fragment,
    )


def _docstring(source: str, tree: ast.AST, function, spec) -> Mutation | None:
    if not function.body:
        return None
    statement = function.body[0]
    if not (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
        and statement.value.value.strip()
        and len(repr(statement.value.value)) <= 300
    ):
        return None
    target = statement.value
    old = target.value
    suffix = " Synthetic clarification added."
    new = old.rstrip() + suffix
    target.value = new
    return _finish(source, tree, spec, target, repr(old), repr(new), "update the function docstring text")


def _is_debug_output(node: ast.AST) -> bool:
    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
        return False
    called = node.value.func
    return (
        isinstance(called, ast.Name) and called.id == "print"
    ) or (
        isinstance(called, ast.Attribute) and called.attr == "debug"
    )


def _debug_output(source: str, tree: ast.AST, function, spec) -> Mutation | None:
    candidates = []
    for parent in ast.walk(function):
        for _field, value in ast.iter_fields(parent):
            if isinstance(value, list) and len(value) > 1:
                for index, child in enumerate(value):
                    if _is_debug_output(child):
                        if len(_fragment(child)) <= 300:
                            candidates.append((value, index, child))
    picked = _pick(source, spec.operator_id, candidates)
    if picked is None:
        return None
    body, index, target = picked
    old = _fragment(target)
    del body[index]
    return _finish(source, tree, spec, target, old, "<deleted>", "remove the standalone output statement")


def _docstring_constant_ids(function) -> set[int]:
    if not function.body:
        return set()
    first = function.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        return {id(first.value)}
    return set()


def _literal(source: str, tree: ast.AST, function, spec) -> Mutation | None:
    docs = _docstring_constant_ids(function)
    candidates = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Constant)
        and id(node) not in docs
        and isinstance(node.value, (int, float, bool))
        and node.value is not None
        and len(repr(node.value)) <= 300
    ]
    target = _pick(source, spec.operator_id, candidates)
    if target is None:
        return None
    old = repr(target.value)
    if isinstance(target.value, bool):
        target.value = not target.value
    elif isinstance(target.value, int):
        target.value += 1
    elif isinstance(target.value, float):
        target.value += 1.0
    else:
        target.value = target.value + "_updated"
    new = repr(target.value)
    return _finish(source, tree, spec, target, old, new, "replace the selected literal value")


def _return(source: str, tree: ast.AST, function, spec) -> Mutation | None:
    candidates = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Return)
        and node.value is not None
        and not (isinstance(node.value, ast.Constant) and node.value.value is None)
        and len(_fragment(node.value)) <= 300
    ]
    target = _pick(source, spec.operator_id, candidates)
    if target is None:
        return None
    old = _fragment(target.value)
    target.value = ast.copy_location(ast.Constant(value=None), target.value)
    return _finish(source, tree, spec, target, old, "None", "replace the return expression")


COMPARISON_REPLACEMENTS: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Is: ast.IsNot, ast.IsNot: ast.Is,
    ast.In: ast.NotIn, ast.NotIn: ast.In,
}


def _comparison(source: str, tree: ast.AST, function, spec) -> Mutation | None:
    candidates = [
        (node, index)
        for node in ast.walk(function) if isinstance(node, ast.Compare)
        for index, operator in enumerate(node.ops)
        if type(operator) in COMPARISON_REPLACEMENTS
        and len(_fragment(node)) <= 300
    ]
    picked = _pick(source, spec.operator_id, candidates)
    if picked is None:
        return None
    target, index = picked
    old = _fragment(target)
    operator = target.ops[index]
    target.ops[index] = COMPARISON_REPLACEMENTS[type(operator)]()
    new = _fragment(target)
    return _finish(source, tree, spec, target, old, new, "modify one comparison operator")


def _exception(source: str, tree: ast.AST, function, spec) -> Mutation | None:
    candidates = [
        node for node in ast.walk(function)
        if isinstance(node, ast.ExceptHandler)
        and (node.type is None or _fragment(node.type) != "Exception")
    ]
    target = _pick(source, spec.operator_id, candidates)
    if target is None:
        return None
    old = "except" if target.type is None else f"except {_fragment(target.type)}"
    reference = target.type if target.type is not None else target
    target.type = ast.copy_location(ast.Name(id="Exception", ctx=ast.Load()), reference)
    return _finish(source, tree, spec, target, old, "except Exception", "modify the exception handler type")


def _simple_fstring(node: ast.JoinedStr) -> bool:
    return any(isinstance(value, ast.FormattedValue) for value in node.values) and all(
        isinstance(value, ast.Constant)
        or (
            isinstance(value, ast.FormattedValue)
            and value.conversion == -1
            and value.format_spec is None
        )
        for value in node.values
    )


class _ReplaceNode(ast.NodeTransformer):
    def __init__(self, target: ast.AST, replacement: ast.AST):
        self.target = target
        self.replacement = replacement

    def generic_visit(self, node):
        if node is self.target:
            return ast.copy_location(self.replacement, node)
        return super().generic_visit(node)


def _string_format(source: str, tree: ast.AST, function, spec) -> Mutation | None:
    candidates = [
        node for node in ast.walk(function)
        if isinstance(node, ast.JoinedStr)
        and _simple_fstring(node)
        and len(_fragment(node)) <= 300
    ]
    target = _pick(source, spec.operator_id, candidates)
    if target is None:
        return None
    pieces = []
    arguments = []
    for value in target.values:
        if isinstance(value, ast.Constant):
            pieces.append(str(value.value).replace("{", "{{").replace("}", "}}"))
        else:
            pieces.append("{}")
            arguments.append(copy.deepcopy(value.value))
    template = "".join(pieces)
    replacement = ast.Call(
        func=ast.Attribute(value=ast.Constant(value=template), attr="format", ctx=ast.Load()),
        args=arguments,
        keywords=[],
    )
    old = _fragment(target)
    new = _fragment(replacement)
    tree = _ReplaceNode(target, replacement).visit(tree)
    return _finish(source, tree, spec, target, old, new, "convert the selected f-string to equivalent .format syntax")


def _annotation(source: str, tree: ast.AST, function, spec) -> Mutation | None:
    candidates = [
        argument
        for argument in [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
        if argument.annotation is None and argument.arg not in {"self", "cls"}
    ]
    target = _pick(source, spec.operator_id, candidates)
    if target is None:
        return None
    old = target.arg
    target.annotation = ast.copy_location(ast.Name(id="object", ctx=ast.Load()), target)
    new = f"{target.arg}: object"
    return _finish(source, tree, spec, target, old, new, "add a built-in type annotation to the parameter")


def _super(source: str, tree: ast.AST, function, spec) -> Mutation | None:
    candidates = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "super"
        and len(node.args) == 2
        and not node.keywords
        and isinstance(node.args[0], ast.Name)
        and isinstance(node.args[1], ast.Name)
        and node.args[1].id in {"self", "cls"}
    ]
    target = _pick(source, spec.operator_id, candidates)
    if target is None:
        return None
    old = _fragment(target)
    target.args = []
    return _finish(source, tree, spec, target, old, "super()", "modernize the explicit super call")


class _RootScopeNames(ast.NodeVisitor):
    def __init__(self, root):
        self.root = root
        self.stored = set()
        self.all_names = set()

    def visit_FunctionDef(self, node):
        if node is self.root:
            self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        if node is self.root:
            self.generic_visit(node)

    def visit_ClassDef(self, node):
        return

    def visit_Lambda(self, node):
        return

    def visit_Name(self, node):
        self.all_names.add(node.id)
        if isinstance(node.ctx, ast.Store):
            self.stored.add(node.id)


class _RenameRootScope(ast.NodeTransformer):
    def __init__(self, root, old: str, new: str):
        self.root = root
        self.old = old
        self.new = new

    def visit_FunctionDef(self, node):
        if node is self.root:
            return self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node):
        if node is self.root:
            return self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        return node

    def visit_Lambda(self, node):
        return node

    def visit_Name(self, node):
        if node.id == self.old:
            node.id = self.new
        return node


def _names_in_nested_scopes(root: ast.AST) -> set[str]:
    """Names whose binding could be changed by crossing a Python lexical scope."""
    unsafe: set[str] = set()
    scope_types = (
        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
        ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
    )
    for node in ast.walk(root):
        if node is root or not isinstance(node, scope_types):
            continue
        unsafe.update(
            child.id for child in ast.walk(node)
            if isinstance(child, ast.Name)
        )
    return unsafe


def _rename(source: str, tree: ast.AST, function, spec) -> Mutation | None:
    names = _RootScopeNames(function)
    names.visit(function)
    unsafe_scoped_names = _names_in_nested_scopes(function)
    candidates = sorted(
        name for name in names.stored
        if name not in {"self", "cls"}
        and not name.startswith("__")
        and f"{name}_updated" not in names.all_names
        and name not in unsafe_scoped_names
    )
    old = _pick(source, spec.operator_id, candidates)
    if old is None:
        return None
    target = next(
        node for node in ast.walk(function)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == old
    )
    new = f"{old}_updated"
    _RenameRootScope(function, old, new).visit(function)
    return _finish(source, tree, spec, target, old, new, "rename the selected local identifier consistently")


IMPLEMENTATIONS: dict[str, Callable] = {
    "modify_docstring_text": _docstring,
    "remove_output_statement": _debug_output,
    "replace_literal_value": _literal,
    "modify_return_expression": _return,
    "modify_comparison_operator": _comparison,
    "modify_exception_handler": _exception,
    "modernize_string_formatting": _string_format,
    "modify_type_annotation": _annotation,
    "modernize_super_call": _super,
    "rename_local_identifier": _rename,
}


def apply_operator(code: str, operator_id: str) -> Mutation | None:
    if operator_id not in OPERATOR_SPECS:
        raise KeyError(f"Unknown operator: {operator_id}")
    source = _source(code)
    tree = ast.parse(source)
    function = _function(tree)
    if function is None:
        return None
    spec = OPERATOR_SPECS[operator_id]
    try:
        return IMPLEMENTATIONS[operator_id](source, tree, function, spec)
    except (SyntaxError, ValueError):
        return None
