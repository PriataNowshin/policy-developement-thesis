# Fusion-guided controlled mutation experiment

Builds a mutation taxonomy from requirements the Pathway 3 fusion model
classified correctly, then applies ten deterministic mutation operators at
full scale to build the synthetic capability-prediction dataset. Separate
from the Roy-based mutation experiments.

Bound to `../ast/llm-code-update-risk-predictor/output/codebert_ast_fusion_top2`
(accuracy 0.5883, macro-F1 0.5876, validation macro-F1 0.6112) via
`experiment_config.json`; `verify_study_model.py` checks this before
anything else runs.

```bash
python3 verify_study_model.py
python3 extract_correct_predictions.py   # 513 correct held-out predictions -> output/step1_correct_predictions/
python3 build_requirement_taxonomy.py    # keyword-rule taxonomy over those 513 -> output/step2_requirement_taxonomy/
```

`output/step3_author_validation/author_validated_taxonomy.csv` is the
manually reviewed taxonomy: 322 of 513 retained (264 single-operation, 58
multiple-operation).

The ten operators live in `controlled_mutation/operators.py`
(`python3 -m unittest discover -s tests -v`): docstrings, output-statement
removal, literals, returns, comparisons, exception handlers, string
formatting, type annotations, `super()` calls, local identifier renaming.

```bash
python3 generate_full_cohort.py \
  --train-sources 3050 --validation-sources 436 --test-sources 872 \
  --output output/step6_full_cohort/mutations_all_4358_sources.json
```

All 10 operators against all 4,358 functions: 15,569 valid mutations from
4,240 sources.

```bash
python3 select_llm_batch.py \
  --input output/step6_full_cohort/mutations_all_4358_sources.json \
  --output-dir output/step8_selected_5000
```

Splits into a batch of 5,000 (`mutations_5000.json`) and the remaining
10,569 (`mutations_remaining.json`).

```bash
python3 generate_llm_updates.py --input output/step8_selected_5000/mutations_5000.json \
  --output output/step9_llm_5000/gpt4o_mini_5000.json --llm gpt-4o-mini
python3 generate_llm_updates.py --input output/step8_selected_5000/mutations_remaining.json \
  --output output/step12_llm_remaining/gpt4o_mini_remaining_10569.json --llm gpt-4o-mini
```

GPT-4o-mini gets only the original function and the generated requirement;
15,567 of 15,569 produced valid Python.

```bash
python3 merge_llm_batches.py \
  --expected output/step6_full_cohort/mutations_all_4358_sources.json \
  --batch output/step9_llm_5000/gpt4o_mini_5000.json \
  --batch output/step12_llm_remaining/gpt4o_mini_remaining_10569.json \
  --output output/step13_llm_full/gpt4o_mini_full_15569.json

python3 evaluate_and_label.py --input output/step13_llm_full/gpt4o_mini_full_15569.json \
  --output_dir output/step14_similarity_labels_full
```

AST-delta similarity against the expected mutation; thresholds p33 = 0.1000,
p66 = 0.9908 from the training split.

```bash
python3 prepare_imbalanced_fusion_splits.py --label-dir output/step14_similarity_labels_full \
  --output-dir output/step15_full_imbalanced_fusion_splits
```

Final: 10,844 train / 1,580 validation / 3,143 test. This is what the three
pathway models in `ast/llm-code-update-risk-predictor/` are fine-tuned and
re-evaluated on.
