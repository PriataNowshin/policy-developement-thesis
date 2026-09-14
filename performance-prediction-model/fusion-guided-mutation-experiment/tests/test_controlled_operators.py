import ast
import unittest

from controlled_mutation import OPERATOR_SPECS, apply_operator


SAMPLES = {
    "modify_docstring_text": '''
def example(value):
    """Return a value."""
    return value
''',
    "remove_output_statement": '''
def example(value):
    print(value)
    return value
''',
    "replace_literal_value": '''
def example():
    value = 10
    return value
''',
    "modify_return_expression": '''
def example(value):
    return value
''',
    "modify_comparison_operator": '''
def example(value):
    if value < 10:
        return True
    return False
''',
    "modify_exception_handler": '''
def example(value):
    try:
        return int(value)
    except ValueError:
        return 0
''',
    "modernize_string_formatting": '''
def example(value):
    return f"value={value}"
''',
    "modify_type_annotation": '''
def example(value):
    return value
''',
    "modernize_super_call": '''
def example(self):
    return super(Example, self).run()
''',
    "rename_local_identifier": '''
def example(value):
    result = value + 1
    return result
''',
}


class ControlledOperatorTests(unittest.TestCase):
    def test_all_ten_operators_are_covered(self):
        self.assertEqual(set(OPERATOR_SPECS), set(SAMPLES))
        self.assertEqual(len(OPERATOR_SPECS), 10)

    def test_each_operator_is_deterministic_parseable_and_ast_visible(self):
        for operator_id, source in SAMPLES.items():
            with self.subTest(operator=operator_id):
                first = apply_operator(source, operator_id)
                second = apply_operator(source, operator_id)
                self.assertIsNotNone(first)
                self.assertEqual(first, second)
                ast.parse(first.code)
                self.assertNotEqual(
                    ast.dump(ast.parse(source), include_attributes=False),
                    ast.dump(ast.parse(first.code), include_attributes=False),
                )
                self.assertIn("Leave everything else unchanged.", first.requirement)
                self.assertTrue(first.old_fragment)
                self.assertTrue(first.new_fragment)
                self.assertIn(
                    f"<before>{first.old_fragment}</before>", first.requirement
                )
                self.assertIn(
                    f"<after>{first.new_fragment}</after>", first.requirement
                )
                for delimiter in ("<before>", "</before>", "<after>", "</after>"):
                    self.assertEqual(first.requirement.count(delimiter), 1)

    def test_inapplicable_operator_returns_none(self):
        source = "def example():\n    pass\n"
        for operator_id in OPERATOR_SPECS:
            with self.subTest(operator=operator_id):
                self.assertIsNone(apply_operator(source, operator_id))

    def test_debug_removal_keeps_nonempty_block(self):
        source = "def example():\n    print('only statement')\n"
        self.assertIsNone(apply_operator(source, "remove_output_statement"))

    def test_literal_operator_does_not_create_arbitrary_string_values(self):
        source = "def example():\n    return 'get'\n"
        self.assertIsNone(apply_operator(source, "replace_literal_value"))

    def test_super_modernization_requires_self_or_cls_receiver(self):
        source = "def example(other):\n    return super(Example, other).run()\n"
        self.assertIsNone(apply_operator(source, "modernize_super_call"))

    def test_string_modernization_requires_interpolation(self):
        source = "def example():\n    return f'constant text'\n"
        self.assertIsNone(apply_operator(source, "modernize_string_formatting"))

    def test_rename_does_not_enter_nested_function(self):
        source = '''
def outer(value):
    outer_value = value
    def inner():
        result = 10
        return result
    return outer_value
'''
        mutation = apply_operator(source, "rename_local_identifier")
        self.assertIsNotNone(mutation)
        tree = ast.parse(mutation.code)
        inner = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "inner")
        self.assertIn("result", {node.id for node in ast.walk(inner) if isinstance(node, ast.Name)})

    def test_rename_rejects_name_shared_with_nested_scope(self):
        source = '''
def outer():
    result = 1
    def inner():
        return result
    return result
'''
        self.assertIsNone(apply_operator(source, "rename_local_identifier"))

    def test_rename_rejects_comprehension_binding(self):
        source = "def example(values):\n    return [item for item in values]\n"
        self.assertIsNone(apply_operator(source, "rename_local_identifier"))


if __name__ == "__main__":
    unittest.main()
