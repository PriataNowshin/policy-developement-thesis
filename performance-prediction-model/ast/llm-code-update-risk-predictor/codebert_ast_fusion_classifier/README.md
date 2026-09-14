# CodeBERT + AST-GNN Fusion Classifier

This package is separate from the existing baselines. It predicts the
train-only percentile class of `ast_delta_similarity` using exactly:

- `requirement` through CodeBERT
- the complete AST of `old_function_code` through a relational GNN

It never uses updated code, deltas, AST edit scripts, or the target score as
model features.

## Recommended architecture

```text
requirement
  -> CodeBERT
  -> first-token representation (768)
  -> projection (128)
                                  \
                                   concatenate (256)
                                  /                 \
old_function_code                                   MLP (256 -> 128 -> 3)
  -> Python AST                                    /
  -> 2-layer relational GNN                       /
  -> mean + max pooling (256)                    /
  -> projection (128) --------------------------/
```

Defaults reflect the current dataset audit:

- requirement `max_length=256` (7 of 4,358 current examples exceed it)
- CodeBERT embeddings and layers 1-10 frozen
- CodeBERT layers 11-12 trainable at `5e-6`
- two GNN layers, hidden size 128, dropout 0.3
- GNN/projection/MLP learning rate `3e-4`
- no AST node truncation
- complete graphs grouped under a 4,096-node batch budget
- gradient accumulation of 2
- early stopping on validation macro-F1

The source CodeBERT defaults to the saved requirement-only checkpoint. Its old
classification head is not used; the fusion model loads the encoder and trains
a new fusion classifier.

## Recommended fixed-split run

Run from `llm-code-update-risk-predictor/` with the virtual environment active:

```bash
python codebert_ast_fusion_classifier/train.py \
  --split_mode fixed \
  --split_dir output/ast_gnn_classifier_output \
  --codebert_model output/requirement_only_codebert_output/model \
  --output_dir output/codebert_ast_fusion_top2
```

Always use a new output directory. The trainer refuses to overwrite a nonempty
experiment folder.

## Layer-freezing ablations

Use the same fixed splits and change only the freezing configuration:

```bash
# Frozen CodeBERT encoder
python codebert_ast_fusion_classifier/train.py \
  --output_dir output/codebert_ast_fusion_frozen \
  --unfrozen_codebert_layers 0

# Recommended: train top two layers
python codebert_ast_fusion_classifier/train.py \
  --output_dir output/codebert_ast_fusion_top2 \
  --unfrozen_codebert_layers 2

# Train top four layers
python codebert_ast_fusion_classifier/train.py \
  --output_dir output/codebert_ast_fusion_top4 \
  --unfrozen_codebert_layers 4

# Fully unfreeze encoder layers and token/position embeddings
python codebert_ast_fusion_classifier/train.py \
  --output_dir output/codebert_ast_fusion_full \
  --unfrozen_codebert_layers 12 \
  --train_codebert_embeddings
```

Choose among these using validation macro-F1. Do not select a configuration
using test accuracy.

## Repository-disjoint robustness run

The fixed split is necessary for direct comparison with the 59.40%
requirement-only result, but most fixed-split test repositories also occur in
training. This optional mode creates repository-disjoint 70/10/20 splits and
computes percentile thresholds from its training partition only:

```bash
python codebert_ast_fusion_classifier/train.py \
  --split_mode repo_grouped \
  --input data/ast_delta_similarity_results.json \
  --codebert_model output/requirement_only_codebert_output/model \
  --output_dir output/codebert_ast_fusion_top2_repo_grouped
```

This result answers a different question—generalization to unseen
repositories—so it should be reported separately from the fixed-split result.

## Large AST handling

Graphs are never cut. `max_total_nodes_per_batch` controls batching, not graph
content. If one graph exceeds the budget, it is retained intact and processed
alone. Reduce `--max_graphs_per_batch` or
`--max_total_nodes_per_batch` if memory is tight; increase
`--gradient_accumulation_steps` to retain a useful effective batch size.

## Output contract

Each run saves:

- `evaluation_report.json`: metrics, parameters, length audits, and overlap audit
- `training_history.json`: epoch loss and validation results
- `test_predictions.json`: probabilities and predictions
- `*_manifest.json`: auditable split identities without expanded ASTs
- `model_trainable.pt`: trainable layers plus reconstruction configuration
- `vocabularies.json`: train-only AST vocabularies
- `tokenizer/`: tokenizer used by the run
- `run_config.json`, label/edge mappings, thresholds, and parse errors

The model checkpoint stores only trainable weights. Frozen CodeBERT weights are
reloaded from `codebert_source`, which is recorded in the checkpoint.

## Tests

From `llm-code-update-risk-predictor/`:

```bash
python -m unittest discover \
  -s codebert_ast_fusion_classifier/tests \
  -p "test_*.py"
```
