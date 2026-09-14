"""Project-wide constants.

Keep configuration-like values here so they can be imported from other modules.
"""

from typing import Dict

DATA_DIR: str = "data"
OUTPUT_DIR: str = "output"

INPUT_FILENAME: str = "changed_functions.json"
OUTPUT_FILENAME: str = "llm_generated_dataset.json"

BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_TIMEOUT_SEC: int = 60
DEFAULT_MAX_RETRIES: int = 5
DEFAULT_RETRY_BACKOFF_SEC: float = 1.5

MODEL_MAP: Dict[str, str] = {
    "gpt-oss": "openai/gpt-oss-20b:free",
    "gpt-4o-mini": "openai/gpt-4o-mini",
}
