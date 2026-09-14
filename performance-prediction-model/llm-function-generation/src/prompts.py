"""Prompt and message construction for the LLM calls."""


FUNCTION_GENERATION_PROMPT_VERSION: str = "controlled_exact_edit_v2_deletion_marker"
 

REQUIREMENT_GENERATION_SYSTEM_PROMPT: str = """
You are an expert in software requirement analysis.

You will be given two versions of the same function:
1. Old function code
2. Updated function code

Your task is to infer the requirement that would reasonably lead a developer
to update the old function into the updated function. You write the requirement 
as a standalone task request or acceptance criterion, instead of summarizing code differences.
"""


REQUIREMENT_GENERATION_USER_PROMPT: str = """

Old function code:
---
{old_code}
---

Updated function code:
---
{new_code}
---

Write the requirement using the following rules:

1. Write from the user's perspective.
   The requirement should sound like a task given to a developer.

2. Focus on what the function should do after the update. Describe only observable behavior.

3. Avoid describing how the old code changed into the new code.

4. Do not mention implementation details.

5. If the update is non-behavioral, such as formatting, cleanup, renaming,
   or refactoring without behavior change, return:
   REFACTOR_CHANGE: <brief maintenance-oriented requirement>

EXAMPLE OF GOOD REQUIREMENT TEXT:
1. Good behavioral requirement:
"The function should return the total score of all cards in the hand."

2. Good validation requirement:
"The function should raise a ValueError when the input is empty."

Output format:
- Return only the requirement text.
- Use plain English.
- Do not include markdown, bullet points, code, or explanations.
"""


FUNCTION_GENERATION_SYSTEM_PROMPT: str = """
You are an expert software developer.

You will be given:
1. an old function code
2. A developer-facing requirement text

Your task is to modify the old function to an updated function so that it satisfies the given developer-facing requirement.
"""


FUNCTION_GENERATION_USER_PROMPT: str = """
Old function code:
---
{old_code}
---

Requirement:
---
{requirement}
---

Follow these rules while generating the updated function code:

1. Return the complete updated function only.
   Do not return a patch, explanation, markdown, or code fence.

2. Preserve the original function signature unless the requirement clearly
   requires a signature change.

3. Preserve the original indentation, docstring, comments, and coding style
   as much as possible.

4. Make the smallest necessary change to satisfy the requirement.

5. Do not add unrelated logic, unrelated refactoring, or top-level code.

6. You may add local logic inside the function body if needed.

7. If the requirement starts with REFACTOR_CHANGE, apply only the described
   maintenance change and do not change behavior.

8. The tags <before> and <after> delimit exact fragments in a requirement. They
   are instruction markup only; never copy these tags into the updated code.

9. In particular, <after><deleted></after> means remove the fragment identified
   by <before> completely. Never output the literal text <deleted>. If removing
   the statement would otherwise leave a required Python suite empty, use pass
   to keep the function syntactically valid.

Output format:
- Return only the complete updated function code.
- Do not include explanations.
- Do not include markdown.
- Do not include code fences.
- Do not include any comments inside the updated function that were not present in the old function.
"""
