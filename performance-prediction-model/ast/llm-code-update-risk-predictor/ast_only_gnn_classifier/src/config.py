"""Constants for the AST-only GNN experiment."""

DEFAULT_RANDOM_STATE = 42

LABEL_TO_ID = {"low": 0, "mid": 1, "high": 2}
ID_TO_LABEL = {0: "low", 1: "mid", 2: "high"}
LABEL_NAMES = ["low", "mid", "high"]

PAD_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unk>"
ROOT_FIELD = "<root>"

EDGE_TYPE_TO_ID = {
    "ast_parent_to_child": 0,
    "ast_child_to_parent": 1,
    "ast_next_sibling": 2,
    "ast_previous_sibling": 3,
}

REQUIRED_FIXED_FIELDS = {
    "repo_full_name",
    "file_path",
    "function_name",
    "old_function_code",
    "ast_delta_similarity",
    "similarity_label",
    "label_id",
}
