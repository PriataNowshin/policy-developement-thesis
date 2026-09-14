"""I/O helpers for loading JSON inputs and writing JSON outputs."""

import json
import os
from typing import Any, Dict, List


def load_data_from_json(path: str) -> List[Dict[str, Any]]:
    """Load JSON objects from a JSON file.

    Reads the file and returns the parsed JSON object.

    Args:
        path: Path to the input JSON file.

    Returns:
        A list of dictionaries parsed from the JSON file.
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    return []


def load_processed_function_ids(path: str) -> set:
    """Load the set of processed function names from an existing output JSON file.

    This is used to avoid re-processing functions when appending to an existing
    output file.

    Args:
        path: Path to the existing output JSON file.

    Returns:
        A set of function names that have already been processed.
    """
    function_ids = set()

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
            for record in data:
                function_ids.add(record.get("id"))

    return function_ids


def append_record_to_json(path: str, record: Dict[str, Any]) -> None:
    """Append or update a single record in the JSON array file.
    
    If the file doesn't exist, creates it with the record as the first element.
    If the file exists, loads the current array, appends the new record, and 
    writes back.
    
    Args:
        path: Path to the output JSON file.
        record: Dictionary to append.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    data = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

    data.append(record)
    
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
