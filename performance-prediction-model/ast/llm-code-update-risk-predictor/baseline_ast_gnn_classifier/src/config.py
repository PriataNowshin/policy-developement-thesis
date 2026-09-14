"""Constants shared by the AST-GNN pipeline."""

DEFAULT_RANDOM_STATE = 42

LABEL_TO_ID = {"low": 0, "mid": 1, "high": 2}
ID_TO_LABEL = {0: "low", 1: "mid", 2: "high"}
LABEL_NAMES = ["low", "mid", "high"]

REQUIRED_FIELDS = [
    "repo_full_name",
    "file_path",
    "function_name",
    "old_function_code",
    "requirement",
    "ast_delta_similarity",
]

PAD_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unk>"
ROOT_FIELD = "<root>"
