# Baseline AST-GNN Classifier

This package preserves the original AST-GNN used to produce
`output/ast_gnn_classifier_output`.

The model uses only `old_function_code`, parsed into an undirected AST graph,
and `requirement`, represented by mean-pooled train-vocabulary embeddings.
The AST similarity score is used only to derive target labels.

The data is split first. Percentile thresholds are computed from the training
split only and then applied unchanged to validation and test.

```bash
python baseline_ast_gnn_classifier/train.py \
  --input data/ast_delta_similarity_results.json \
  --output_dir output/ast_gnn_baseline_rerun \
  --epochs 50 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --hidden_dim 128 \
  --lexical_dim 64 \
  --gnn_layers 3 \
  --dropout 0.2 \
  --patience 8
```
