# LLM Code-Update Risk Predictor

Three pathways predicting an LLM's update-capability class (low/mid/high
`ast_delta_similarity`) before generation, plus a post-hoc topic analysis and
a synthetic-mutation re-evaluation.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Split

All three pathways use the same 3,050/436/872 train/validation/test split
and train-only percentile thresholds (p33 = 0.3280, p66 = 0.9123), stored in
`output/ast_gnn_classifier_output/{train,validation,test}_data.json` and
`percentile_thresholds.json`, read via `--split_dir`.

## Packages

- `requirement_only_codebert_classifier/`: Pathway 1, requirement text only.
  Output: `output/requirement_only_codebert_output/`.
- `ast_only_gnn_classifier/`: Pathway 2, old-function AST only (4 directed
  AST relations, 2 message-passing layers, mean+max pooling). Output:
  `output/ast_only_gnn_output/`.
- `codebert_ast_fusion_classifier/`: Pathway 3, requirement + AST fusion
  (reuses Pathway 1's CodeBERT with the lower 10 layers frozen; separate AST
  branch, same design as Pathway 2). Also has `predict_external.py`,
  `prepare_synthetic_splits.py`, and the post-hoc topic analysis script.
  Output: `output/codebert_ast_fusion_top2/` (2 = unfrozen upper CodeBERT
  layers); topics in `output/pathway3_posthoc_topics_k8/`.
- `baseline_ast_gnn_classifier/`: earlier prototype that mixed requirement
  text into what it called an AST-only model. Not a reported pathway; kept
  because its `splitting.py` produced the split above.

## Synthetic re-evaluation

Fine-tunes each pathway's saved checkpoint on the 15,569-example synthetic
split from `fusion-guided-mutation-experiment/output/step15_full_imbalanced_fusion_splits/`
(10,844/1,580/3,143). Same `train.py` / `train_synthetic.py` as above,
pointed at the synthetic split and the saved checkpoint instead of a fresh
model:

```bash
# Pathway 1
python requirement_only_codebert_classifier/train.py \
  --model_name output/requirement_only_codebert_output/model \
  --split_dir ../../fusion-guided-mutation-experiment/output/step15_full_imbalanced_fusion_splits \
  --output_dir output/requirement_only_codebert_synthetic_full_imbalanced_15569 \
  --epochs 5 --batch_size 8 --learning_rate 0.00002 --max_length 256 \
  --patience 2 --random_state 42

# Pathway 2
python ast_only_gnn_classifier/train_synthetic.py \
  --split_dir ../../fusion-guided-mutation-experiment/output/step15_full_imbalanced_fusion_splits \
  --initial_checkpoint output/ast_only_gnn_output/model.pt \
  --output_dir output/ast_only_gnn_synthetic_full_imbalanced_15569 \
  --device cpu

# Pathway 3
python codebert_ast_fusion_classifier/train.py \
  --split_mode fixed \
  --split_dir ../../fusion-guided-mutation-experiment/output/step15_full_imbalanced_fusion_splits \
  --codebert_model output/requirement_only_codebert_output/model \
  --initial_model_dir output/codebert_ast_fusion_top2 \
  --output_dir output/codebert_ast_fusion_fusion_guided_full_imbalanced_15569
```

## Data

`data/ast_delta_similarity_results.json` comes from
`ast/function-similarity-checker/`, which scores LLM updates (from
`llm-function-generation/`) against human updates on functions mined by
`github-function-change-miner/`.
