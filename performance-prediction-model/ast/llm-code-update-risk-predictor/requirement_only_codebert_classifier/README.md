# Requirement-Only CodeBERT Ablation

This isolated experiment tests whether requirement wording alone can predict
the low, mid, or high AST edit-script similarity class.

## Comparison contract

The experiment reads the fixed split artifacts produced by the original AST-GNN
run. This guarantees that CodeBERT and the GNN use the same 3,050 training, 436
validation, and 872 test examples, labels, and training-derived percentile
thresholds.

The CodeBERT dataset contains only requirement token IDs, attention masks, and
the target label. The old function, updated functions, deltas, edit scripts, and
similarity score are excluded from model inputs. The similarity score remains
available only as the source of the already-created target label.

## Run

From the repository root with the virtual environment active:

```bash
python requirement_only_codebert_classifier/train.py \
  --split_dir output/ast_gnn_classifier_output \
  --output_dir output/requirement_only_codebert_output \
  --epochs 5 \
  --batch_size 8 \
  --learning_rate 0.00002 \
  --max_length 256 \
  --patience 2 \
  --random_state 42
```

The first run may download `microsoft/codebert-base` if it is not cached.
Training automatically uses CUDA when available and otherwise uses CPU.

## Outputs

The separate output directory contains the saved model, run configuration,
training history, thresholds, test predictions, and `evaluation_report.json`.
The report uses the same accuracy, balanced accuracy, macro-F1, weighted-F1,
per-class scores, and confusion-matrix format as the GNN reports.
