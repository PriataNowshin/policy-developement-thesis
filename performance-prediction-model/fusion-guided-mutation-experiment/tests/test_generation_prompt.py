"""Regression checks for controlled function-generation instructions."""
from __future__ import annotations

import runpy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "fusion-guided-mutation-experiment"))

from generate_llm_updates import validate_generated_code


class GenerationPromptTests(unittest.TestCase):
    def test_deletion_marker_is_explicitly_instruction_only(self):
        values = runpy.run_path(str(ROOT / "llm-function-generation/src/prompts.py"))
        prompt = values["FUNCTION_GENERATION_USER_PROMPT"]
        self.assertIn("<after><deleted></after> means remove", prompt)
        self.assertIn("Never output the literal text <deleted>", prompt)
        self.assertIn("use pass", prompt)

    def test_deletion_fix_has_a_new_prompt_version(self):
        values = runpy.run_path(str(ROOT / "llm-function-generation/src/prompts.py"))
        self.assertEqual(
            values["FUNCTION_GENERATION_PROMPT_VERSION"],
            "controlled_exact_edit_v2_deletion_marker",
        )

    def test_generated_code_cannot_contain_deletion_marker(self):
        with self.assertRaisesRegex(ValueError, "deletion instruction marker"):
            validate_generated_code("def example():\n    return '<deleted>'\n")


if __name__ == "__main__":
    unittest.main()
