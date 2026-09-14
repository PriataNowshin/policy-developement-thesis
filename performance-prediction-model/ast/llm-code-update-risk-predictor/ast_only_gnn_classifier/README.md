# True AST-Only GNN Classifier

This experiment measures whether the old function's AST independently predicts
the low, mid, or high `ast_delta_similarity` class.

The only model input is:

```text
AST(old_function_code)
```

The requirement, updated code, deltas, edit scripts, and target score are never
model features. The package uses the exact fixed splits and train-only
thresholds used by the requirement-only and fusion experiments.

## Architecture

```text
old_function_code
  -> complete Python AST
  -> node type + field + depth + identifier features
  -> two relational message-passing layers
  -> graph mean and max pooling
  -> MLP classifier
  -> low / mid / high
```

Relations are parent-to-child, child-to-parent, next-sibling, and
previous-sibling. ASTs are not truncated. Node-budget batching processes an
oversized graph alone rather than removing nodes.

## Recommended run

From `llm-code-update-risk-predictor/`:

```bash
python ast_only_gnn_classifier/train.py \
  --split_dir output/ast_gnn_classifier_output \
  --output_dir output/ast_only_gnn_output \
  --device cpu
```

Defaults:

- 2 GNN layers
- hidden dimension 128
- lexical dimension 64
- dropout 0.3
- learning rate `1e-3`
- weight decay `1e-4`
- at most 30 epochs
- early-stopping patience 5 based on validation macro-F1
- maximum 16 graphs and 4,096 ordinary AST nodes per batch

## Outputs

`evaluation_report.json` records the strict input contract, fixed splits,
class distributions, graph sizes, metrics, confusion matrices, best epoch,
runtime, and shuffled-AST diagnostic.

The reported test accuracy uses the best validation-macro-F1 checkpoint.
