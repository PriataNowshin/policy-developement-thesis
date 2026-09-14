# function-similarity-checker

Computes AST edit-script similarity between human and LLM code changes:

- human update: `old_function_code -> new_function_code`
- LLM update: `old_function_code -> new_function_code_by_llm`

The main score, `ast_delta_similarity`, compares the two AST edit scripts with
operation precision, recall, and F1. This focuses on whether the LLM made the
same structural code change as the human update, rather than comparing CodeBERT
embeddings of text deltas.


## Setup

The AST implementation uses only the Python standard library.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Full dataset:

```bash
python3 ast_delta_similarity.py \
  --input data/llm_generated_dataset.json \
  --output output_ast/ast_delta_similarity_results.json
```

Quick limited run:

```bash
python3 ast_delta_similarity.py \
  --input data/llm_generated_dataset.json \
  --limit 50 \
  --output output_ast/ast_delta_similarity_first_50_results.json
```

## Output

Each processed item includes:

- `ast_delta_similarity`: F1 similarity between human and LLM AST edit scripts
- `operation_precision`: share of LLM edit operations that match human edits
- `operation_recall`: share of human edit operations captured by LLM edits
- `matched_edit_count`: number of matched AST edit operations
- `human_edit_count`: number of AST edit operations in the human update
- `llm_edit_count`: number of AST edit operations in the LLM update
- `final_ast_similarity`: structural similarity between the final human and LLM ASTs
- `parse_success` / parse error records
- `human_ast_edit_script` and `llm_ast_edit_script`
