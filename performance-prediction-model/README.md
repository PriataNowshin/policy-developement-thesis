# Performance Prediction Model

Code for Chapter 5 Part 2 (RQ5): predicting, before generation, whether an
LLM's update to a function will structurally match what a human actually
did. Includes a synthetic follow-up that repeats the evaluation on
deterministic mutations.

## Pipeline

```
github-function-change-miner -> llm-function-generation -> ast/function-similarity-checker
    -> ast/llm-code-update-risk-predictor -> fusion-guided-mutation-experiment
```

1. `github-function-change-miner`: mines GitHub commits where a function body changed.
2. `llm-function-generation`: writes a requirement per change, asks an LLM to redo it independently.
3. `ast/function-similarity-checker`: AST edit-script similarity between the LLM's update and the human's.
4. `ast/llm-code-update-risk-predictor`: turns that score into low/mid/high labels, trains three pathways to predict it pre-generation.
5. `fusion-guided-mutation-experiment`: same evaluation on a synthetic, deterministic mutation dataset.

Each folder has its own README with full detail. Stages 1-2 need API keys
(`.env`: `GITHUB_TOKEN`, `API_KEY` for OpenRouter); the rest run offline.

Every `data/` and `output/` folder below is gitignored, so a fresh clone
starts without the mined functions, LLM-generated updates, similarity
scores, splits, or trained checkpoints; the commands below regenerate all
of it. Stage 1 pulls from GitHub's live repository list and stage 2 calls
an LLM, so a full re-run reproduces the pipeline, not necessarily the exact
numbers reported in Chapter 5.

## Running it

```bash
cd github-function-change-miner && source venv/bin/activate
python3 main.py

cd ../llm-function-generation && source venv/bin/activate
cp ../github-function-change-miner/output/changed_functions.json data/changed_functions.json
python3 main.py --llm gpt-oss

cd ../ast/function-similarity-checker && source .venv/bin/activate
cp ../../llm-function-generation/output/llm_generated_dataset.json data/llm_generated_dataset.json
python3 ast_delta_similarity.py \
  --input data/llm_generated_dataset.json \
  --output output_ast/ast_delta_similarity_results.json

cd ../llm-code-update-risk-predictor && source venv/bin/activate
cp ../function-similarity-checker/output_ast/ast_delta_similarity_results.json data/

python baseline_ast_gnn_classifier/train.py \
  --input data/ast_delta_similarity_results.json \
  --output_dir output/ast_gnn_classifier_output

python requirement_only_codebert_classifier/train.py \
  --split_dir output/ast_gnn_classifier_output \
  --output_dir output/requirement_only_codebert_output \
  --epochs 5 --batch_size 8 --learning_rate 0.00002 --max_length 256 \
  --patience 2 --random_state 42

python ast_only_gnn_classifier/train.py \
  --split_dir output/ast_gnn_classifier_output \
  --output_dir output/ast_only_gnn_output --device cpu

python codebert_ast_fusion_classifier/train.py \
  --split_mode fixed \
  --split_dir output/ast_gnn_classifier_output \
  --codebert_model output/requirement_only_codebert_output/model \
  --output_dir output/codebert_ast_fusion_top2
```

`baseline_ast_gnn_classifier` produces the shared split and thresholds
(`ast_gnn_classifier_output/{train,validation,test}_data.json`,
`percentile_thresholds.json`) the three pathways above read via
`--split_dir`; it isn't one of the reported pathways itself.

Topic analysis on Pathway 3: `python posthoc_pathway3_topic_analysis.py --topics 8`.

Synthetic mutation experiment: `fusion-guided-mutation-experiment/README.md`.
It has no venv of its own; use `llm-function-generation`'s for everything
in it except `evaluate_and_label.py`, which needs `numpy` and should run
under `ast/llm-code-update-risk-predictor`'s venv instead. Fine-tuning the
three pathways on the resulting synthetic split is documented in
`ast/llm-code-update-risk-predictor/README.md`.
