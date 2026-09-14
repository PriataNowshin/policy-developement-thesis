from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class Settings:
    # Dataset target
    target_repository_count: int = 1000
    min_valid_files_per_repo: int = 5

    # Output Directory
    output_dir: str = "output"

    # Cutoff: ignore anything after this date (inclusive cutoff at end-of-day UTC)
    cutoff_utc: datetime = datetime(2022, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    # Repo discovery (rank by stars desc)
    search_per_page: int = 100
    max_search_pages: int = 10

    # File selection
    max_python_files_to_check_per_repo: int = 200

    # Commit selection per file
    max_commits_to_consider_per_file: int = 50
    commits_per_page: int = 100
    max_commit_pages_per_file: int = 10

    # Content constraints
    max_file_bytes: int = 750_000  # skip giant files to avoid API/parse issues

    # GitHub API
    github_api_base_url: str = "https://api.github.com"
    user_agent: str = "github-function-change-miner"


SETTINGS = Settings()
