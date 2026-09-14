from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FileRecord:
    repo_full_name: str
    file_path: str
    old_sha: str
    new_sha: str
    old_date: str
    new_date: str
    old_message: str
    new_message: str
    changed_function_count: int


@dataclass(frozen=True)
class FunctionRecord:
    repo_full_name: str
    file_path: str
    function_name: str
    old_sha: str
    new_sha: str
    old_date: str
    new_date: str
    old_message: str
    new_message: str
    old_code: str
    new_code: str
    old_start_line: int
    old_end_line: int
    new_start_line: int
    new_end_line: int
    change_type: str = "function_body_updated"
