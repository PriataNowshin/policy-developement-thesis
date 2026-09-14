from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.ast_diff import find_function_body_changes
from src.config import SETTINGS
from src.github_client import GitHubApiError, GitHubClient, parse_github_datetime
from src.json_writer import save_data_as_json
from src.records import FileRecord, FunctionRecord


def run() -> None:
    client = GitHubClient()

    selected_repos: List[Tuple[Dict[str, Any], List[FileRecord], List[FunctionRecord]]] = []

    print("Searching for candidate repositories...")
    for page in range(1, SETTINGS.max_search_pages + 1):
        results = client.search_repositories(page=page, per_page=SETTINGS.search_per_page)
        items = results.get("items", [])
        if not items:
            break

        for item in items:
            if len(selected_repos) >= SETTINGS.target_repository_count:
                break

            repo_full_name = item.get("full_name")
            if not repo_full_name:
                continue

            try:
                repo = client.get_repo(repo_full_name)
            except GitHubApiError as e:
                print(f"Skipping {repo_full_name}: repo fetch failed: {e}")
                continue

            if not _repo_passes_rules(repo):
                continue

            try:
                file_records, function_records = _collect_repo_changes(client, repo)
            except GitHubApiError as e:
                print(f"Skipping {repo_full_name}: API error while collecting changes: {e}")
                continue
            except Exception as e:
                print(f"Skipping {repo_full_name}: unexpected error: {e}")
                continue

            if len(file_records) < SETTINGS.min_valid_files_per_repo:
                continue

            selected_repos.append((repo, file_records, function_records))
            print(f"Selected {repo_full_name} ({len(file_records)} valid files). Total selected: {len(selected_repos)}")

        if len(selected_repos) >= SETTINGS.target_repository_count:
            break

    if len(selected_repos) < SETTINGS.target_repository_count:
        print(
            f"Only selected {len(selected_repos)} repositories; increase SETTINGS.max_search_pages or relax limits.",
            file=sys.stderr,
        )

    repos_out: List[Dict[str, Any]] = []
    files_out: List[Dict[str, Any]] = []
    funcs_out: List[Dict[str, Any]] = []

    # Rank final selection by stars descending
    selected_repos_sorted = sorted(selected_repos, key=lambda t: int(t[0].get("stargazers_count", 0)), reverse=True)

    function_id_counter = 1
    for pos, (repo, file_records, function_records) in enumerate(selected_repos_sorted, start=1):
        repos_out.append(
            {
                "repo_full_name": repo["full_name"],
                "repo_url": repo["html_url"],
                "creation_date": repo["created_at"],
                "stars": int(repo.get("stargazers_count", 0)),
                "forks": int(repo.get("forks_count", 0)),
                "default_branch": repo.get("default_branch", ""),
                "ranking_position": pos,
                "number_of_valid_files": len(file_records),
            }
        )

        for fr in file_records:
            files_out.append(
                {
                    "repo_full_name": fr.repo_full_name,
                    "file_path": fr.file_path,
                    "old_commit_sha": fr.old_sha,
                    "new_commit_sha": fr.new_sha,
                    "old_commit_date": fr.old_date,
                    "new_commit_date": fr.new_date,
                    "old_commit_message": fr.old_message,
                    "new_commit_message": fr.new_message,
                    "number_of_changed_functions": fr.changed_function_count,
                }
            )

        for func in function_records:
            funcs_out.append(
                {
                    "id": f"function_{function_id_counter}",
                    "repo_full_name": func.repo_full_name,
                    "file_path": func.file_path,
                    "function_name": func.function_name,
                    "old_commit_sha": func.old_sha,
                    "new_commit_sha": func.new_sha,
                    "old_commit_date": func.old_date,
                    "new_commit_date": func.new_date,
                    "old_commit_message": func.old_message,
                    "new_commit_message": func.new_message,
                    "full_old_function_code": func.old_code,
                    "full_new_function_code": func.new_code,
                    "old_function_start_line": func.old_start_line,
                    "old_function_end_line": func.old_end_line,
                    "new_function_start_line": func.new_start_line,
                    "new_function_end_line": func.new_end_line,
                    "change_type": func.change_type,
                }
            )

            function_id_counter += 1

    output_dir = Path(SETTINGS.output_dir)
    selected_repos_path = output_dir / "selected_repositories.json"
    changed_files_path = output_dir / "changed_files.json"
    changed_functions_path = output_dir / "changed_functions.json"

    save_data_as_json(selected_repos_path, repos_out)
    save_data_as_json(changed_files_path, files_out)
    save_data_as_json(changed_functions_path, funcs_out)

    print("Done.")
    print(f"Wrote {selected_repos_path}")
    print(f"Wrote {changed_files_path}")
    print(f"Wrote {changed_functions_path}")


def _repo_passes_rules(repo: Dict[str, Any]) -> bool:
    """Check whether a repository meets the selection rules for mining.

    Enforces the project constraints:
    - public (not private),
    - not archived,
    - not a fork,
    - created on or before `SETTINGS.cutoff_utc`, and
    - primary language is Python.

    Args:
        repo: Repository metadata dict returned by the GitHub API.

    Returns:
        `True` if the repository passes all rules, otherwise `False`.
    """    
    if repo.get("private") is True or repo.get("archived") is True or repo.get("fork") is True:
        return False

    created_at = repo.get("created_at")
    if not created_at:
        return False

    created_dt = parse_github_datetime(created_at)
    if created_dt > SETTINGS.cutoff_utc:
        return False

    if repo.get("language") != "Python":
        return False

    return True


def _collect_repo_changes(client: GitHubClient, repo: Dict[str, Any]) -> Tuple[List[FileRecord], List[FunctionRecord]]:
    """Collect function-body change records for a single repository.

    Enumerates Python files from the repository's default-branch tree (up to
    `SETTINGS.max_python_files_to_check_per_repo`). For each file, searches commit
    history up to `SETTINGS.cutoff_utc` and selects the latest adjacent commit pair
    where at least one function body changed (via `find_function_body_changes`).

    The scan stops early once `SETTINGS.min_valid_files_per_repo` valid files have
    been found.

    Args:
        client: GitHub API client.
        repo: Repository metadata dict returned by the GitHub API.

    Returns:
        A tuple `(file_records, function_records)`. If the repository has no
        default branch, returns `([], [])`.
    """
    full_name = repo["full_name"]
    default_branch = repo.get("default_branch")
    if not default_branch:
        return ([], [])

    tree = _get_default_branch_tree(client, full_name, default_branch)
    python_files = [
        item["path"]
        for item in tree
        if item.get("type") == "blob" and isinstance(item.get("path"), str) and item["path"].endswith(".py")
    ]

    python_files = python_files[: SETTINGS.max_python_files_to_check_per_repo]

    valid_files: List[FileRecord] = []
    changed_functions: List[FunctionRecord] = []

    for path in python_files:
        if len(valid_files) >= SETTINGS.min_valid_files_per_repo:
            break

        file_result = _find_latest_valid_commit_pair_for_file(
            client=client, repo_full_name=full_name, default_branch=default_branch, file_path=path
        )
        if not file_result:
            continue

        file_record, func_records = file_result
        valid_files.append(file_record)
        changed_functions.extend(func_records)

    return (valid_files, changed_functions)


def _get_default_branch_tree(client: GitHubClient, full_name: str, branch: str) -> List[Dict[str, Any]]:
    """Fetch the recursive git tree for a repository's default branch tip.

    Resolves the branch ref to a commit SHA, fetches the commit to obtain its tree SHA,
    then retrieves the full recursive tree via the GitHub Git Trees API.

    Args:
        client: GitHub API client.
        full_name: Repository in "owner/name" form.
        branch: Branch name to resolve (typically the repository's default branch).

    Returns:
        A list of tree items (dicts) from the API response. Returns an empty list if
        the branch ref, commit SHA, or tree SHA cannot be resolved.
    """
    ref = client.get_branch_ref(full_name, branch)
    commit_sha = ref.get("object", {}).get("sha")
    if not commit_sha:
        return []

    commit = client.get_git_commit(full_name, commit_sha)
    tree_sha = commit.get("tree", {}).get("sha")
    if not tree_sha:
        return []

    tree = client.get_tree_recursive(full_name, tree_sha)

    return tree.get("tree", []) or []


def _find_latest_valid_commit_pair_for_file(
    *, 
    client: GitHubClient, 
    repo_full_name: str, 
    default_branch: str, 
    file_path: str
) -> Optional[Tuple[FileRecord, List[FunctionRecord]]]:
    """Find the most recent pre-cutoff commit pair for a file with a valid function-body change.

    Walks the commit history for `file_path` on `default_branch`, ignoring commits
    after `SETTINGS.cutoff_utc`. It then compares adjacent commits (newest to older)
    and returns the first pair where:
    - the file contents are retrievable and decodable as UTF-8,
    - both versions are under `SETTINGS.max_file_bytes`, and
    - `find_function_body_changes(old_source, new_source)` reports at least one
      function whose body changed.

    Args:
        client: GitHub API client.
        repo_full_name: Repository in "owner/name" form.
        default_branch: Branch name to query commit history from.
        file_path: Path to the Python file within the repository.

    Returns:
        A tuple of `(file_record, function_records)` for the selected commit pair,
        or `None` if no suitable commit pair is found.
    """
    commits: List[Dict[str, Any]] = []

    for page in range(1, SETTINGS.max_commit_pages_per_file + 1):
        page_items = client.list_commits_for_path(
            full_name=repo_full_name,
            path=file_path,
            branch=default_branch,
            per_page=SETTINGS.commits_per_page,
            page=page,
        )

        if not page_items:
            break

        for c in page_items:
            commit_dt = _commit_datetime_utc(c)

            if commit_dt is None:
                continue

            if commit_dt <= SETTINGS.cutoff_utc:
                commits.append(c)

            if len(commits) >= SETTINGS.max_commits_to_consider_per_file:
                break

        if len(commits) >= SETTINGS.max_commits_to_consider_per_file:
            break

    if len(commits) < 2:
        return None

    for i in range(0, len(commits) - 1):
        new_c = commits[i]
        old_c = commits[i + 1]
        new_sha = new_c.get("sha")
        old_sha = old_c.get("sha")

        if not new_sha or not old_sha:
            continue

        new_bytes = client.get_contents(full_name=repo_full_name, path=file_path, ref=new_sha)
        old_bytes = client.get_contents(full_name=repo_full_name, path=file_path, ref=old_sha)

        if not new_bytes or not old_bytes:
            continue

        if len(new_bytes) > SETTINGS.max_file_bytes or len(old_bytes) > SETTINGS.max_file_bytes:
            continue

        try:
            new_source = new_bytes.decode("utf-8")
            old_source = old_bytes.decode("utf-8")
        except UnicodeDecodeError:
            continue

        changes = find_function_body_changes(old_source, new_source)
        if not changes:
            continue

        old_dt = _commit_datetime_utc(old_c)
        new_dt = _commit_datetime_utc(new_c)
        if old_dt is None or new_dt is None:
            continue

        old_msg = (old_c.get("commit") or {}).get("message") or ""
        new_msg = (new_c.get("commit") or {}).get("message") or ""

        func_records = [
            FunctionRecord(
                repo_full_name=repo_full_name,
                file_path=file_path,
                function_name=ch.qualified_name,
                old_sha=old_sha,
                new_sha=new_sha,
                old_date=old_dt.isoformat(),
                new_date=new_dt.isoformat(),
                old_message=old_msg,
                new_message=new_msg,
                old_code=ch.old.code,
                new_code=ch.new.code,
                old_start_line=ch.old.start_line,
                old_end_line=ch.old.end_line,
                new_start_line=ch.new.start_line,
                new_end_line=ch.new.end_line,
            )
            for ch in changes
        ]

        file_record = FileRecord(
            repo_full_name=repo_full_name,
            file_path=file_path,
            old_sha=old_sha,
            new_sha=new_sha,
            old_date=old_dt.isoformat(),
            new_date=new_dt.isoformat(),
            old_message=old_msg,
            new_message=new_msg,
            changed_function_count=len(func_records),
        )

        return (file_record, func_records)

    return None


def _commit_datetime_utc(commit_item: Dict[str, Any]) -> Optional[datetime]:
    """Extract the commit author timestamp as a UTC-aware datetime.

    Args:
        commit_item: A commit item dict from the GitHub API (e.g., an element of
            the list returned by `list_commits_for_path`).

    Returns:
        The commit author datetime in UTC, or `None` if the expected date field
        is missing or not a string.
    """
    commit = commit_item.get("commit") or {}
    author = commit.get("author") or {}
    date_str = author.get("date")

    if not isinstance(date_str, str):
        return None

    dt = parse_github_datetime(date_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)
