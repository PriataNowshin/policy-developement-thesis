"""Shared constants for the requirement-only experiment."""

MODEL_NAME = "microsoft/codebert-base"
DEFAULT_RANDOM_STATE = 42
LABEL_TO_ID = {"low": 0, "mid": 1, "high": 2}
ID_TO_LABEL = {0: "low", 1: "mid", 2: "high"}
LABEL_NAMES = ["low", "mid", "high"]
REQUIRED_SPLIT_FIELDS = {
    "repo_full_name",
    "file_path",
    "function_name",
    "requirement",
    "ast_delta_similarity",
    "similarity_label",
    "label_id",
}
