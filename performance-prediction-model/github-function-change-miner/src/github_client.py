from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

import requests
from dotenv import load_dotenv

from .config import SETTINGS


class GitHubApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class RateLimit:
    remaining: int
    reset_epoch_seconds: int


class GitHubClient:
    def __init__(self) -> None:
        load_dotenv()
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise GitHubApiError("Missing environment variable GITHUB_TOKEN")

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": SETTINGS.user_agent,
            }
        )

    def _maybe_sleep_for_rate_limit(self, response: requests.Response) -> None:
        """Sleep until the GitHub rate limit resets when the limit is exhausted.

        Reads `X-RateLimit-Remaining` and `X-RateLimit-Reset` from the response.
        If remaining requests are 0, sleeps until the reset time (plus a small
        buffer). Otherwise, returns immediately.
        """
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset = response.headers.get("X-RateLimit-Reset")

        if remaining is None or reset is None:
            return
        try:
            rl = RateLimit(remaining=int(remaining), reset_epoch_seconds=int(reset))
        except ValueError:
            return

        if rl.remaining > 0:
            return

        sleep_seconds = max(0, rl.reset_epoch_seconds - int(time.time()) + 2)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Send a GitHub API request and return the decoded JSON response.

        Builds the full URL from the API base + `path`, sends the request with the
        configured session, and handles basic rate-limit waiting/retry for 403s.
        Non-2xx responses raise `GitHubApiError`.

        Args:
            method: HTTP method (e.g., "GET").
            path: API path starting with "/" (e.g., "/repos/owner/repo").
            params: Optional query parameters.

        Returns:
            The parsed JSON response.
        """
        url = f"{SETTINGS.github_api_base_url}{path}"
        resp = self._session.request(method, url, params=params, timeout=60)

        if resp.status_code == 403:
            # Could be rate limit OR content restrictions.
            self._maybe_sleep_for_rate_limit(resp)

            # retry once after potential sleep
            resp = self._session.request(method, url, params=params, timeout=60)

        if resp.status_code < 200 or resp.status_code >= 300:
            raise GitHubApiError(f"GitHub API error {resp.status_code} for {method} {path}: {resp.text}")

        self._maybe_sleep_for_rate_limit(resp)

        return resp.json()

    def search_repositories(self, *, page: int, per_page: int) -> Dict[str, Any]:
        """Search for candidate Python repositories using the GitHub Search API.

        The query is fixed to match this project’s selection rules:
        public Python repos created on or before 2023-12-31, excluding forks and
        archived repos, sorted by stars (descending).

        Args:
            page: Page number (1-based).
            per_page: Number of results per page.

        Returns:
            The raw Search API JSON response as a dict (includes an `items` list).
        """
        q = "language:Python created:<=2023-12-31 fork:false archived:false"
        
        return self._request(
            "GET",
            "/search/repositories",
            params={"q": q, "sort": "stars", "order": "desc", "per_page": per_page, "page": page},
        )

    def get_repo(self, full_name: str) -> Dict[str, Any]:
        """Fetch repository metadata from the GitHub REST API.

        This wraps `GET /repos/{owner}/{repo}` and returns fields such as
        `created_at`, `stargazers_count`, `fork`, `archived`, `language`,
        and `default_branch`.

        Args:
            full_name: Repository full name, e.g. ``"owner/repo"``.

        Returns:
            The repository JSON as a dict.
        """
        return self._request("GET", f"/repos/{full_name}")

    def get_branch_ref(self, full_name: str, branch: str) -> Dict[str, Any]:
        """Fetch the git ref object for a branch on the repository.

        This wraps `GET /repos/{owner}/{repo}/git/ref/heads/{branch}` and is
        typically used to resolve a branch name to its current commit SHA.

        Args:
            full_name: Repository full name, e.g. ``"owner/repo"``.
            branch: Branch name (e.g. ``"main"``).

        Returns:
            The ref JSON as a dict (includes an `object.sha` field).
        """        
        owner, repo = full_name.split("/", 1)

        return self._request("GET", f"/repos/{owner}/{repo}/git/ref/heads/{branch}")

    def get_git_commit(self, full_name: str, sha: str) -> Dict[str, Any]:
        """Fetch a git commit object by SHA via the GitHub Git Data API.

        This wraps `GET /repos/{owner}/{repo}/git/commits/{sha}` and returns the
        raw JSON response (which includes fields like `tree`).

        Args:
            full_name: Repository full name, e.g. ``"owner/repo"``.
            sha: Commit SHA.

        Returns:
            The commit JSON as a dict.
        """
        owner, repo = full_name.split("/", 1)

        return self._request("GET", f"/repos/{owner}/{repo}/git/commits/{sha}")

    def get_tree_recursive(self, full_name: str, tree_sha: str) -> Dict[str, Any]:
        """Fetch a git tree recursively (files and directories) for a given tree SHA.

        This wraps the GitHub REST endpoint `GET /repos/{owner}/{repo}/git/trees/{tree_sha}`
        with `recursive=1`, returning the API response which includes a `tree` list
        of entries (blobs/files and trees/directories).

        Args:
            full_name: Repository full name, e.g. ``"owner/repo"``.
            tree_sha: SHA of the git tree object to fetch (usually from a commit).

        Returns:
            The JSON response as a dict (typically containing a `tree` key).
        """
        owner, repo = full_name.split("/", 1)

        return self._request("GET", f"/repos/{owner}/{repo}/git/trees/{tree_sha}", params={"recursive": "1"})

    def list_commits_for_path(
        self, 
        *, 
        full_name: str, 
        path: str, 
        branch: str, 
        per_page: int, 
        page: int
    ) -> List[Dict[str, Any]]:
        """List commits on a branch that touched a specific file path.

        This wraps the GitHub REST endpoint `GET /repos/{owner}/{repo}/commits`
        with the `path` filter so you get a file-specific commit history.

        Args:
            full_name: Repository full name, e.g. ``"owner/repo"``.
            path: File path within the repository to filter commits by.
            branch: Branch name or commit SHA to query from (passed as `sha`).
            per_page: Number of results per page (GitHub max is typically 100).
            page: Page number.

        Returns:
            A list of commit objects as returned by the GitHub API.
        """
        owner, repo = full_name.split("/", 1)
        
        return self._request(
            "GET",
            f"/repos/{owner}/{repo}/commits",
            params={"path": path, "sha": branch, "per_page": per_page, "page": page},
        )

    def get_contents(self, *, full_name: str, path: str, ref: str) -> Optional[bytes]:
        """Fetch a file's contents at a specific git ref and return raw bytes.
        Uses the GitHub REST Contents API for the given repository, path, and ref
        (branch name, tag, or commit SHA).

        Args:
            full_name: Repository full name, e.g. ``"owner/repo"``.
            path: File path within the repository.
            ref: Git reference (commit SHA, branch, or tag).

        Returns:
            The decoded file content as bytes, or ``None`` if unavailable.
        """        
        owner, repo = full_name.split("/", 1)
        url_path = f"/repos/{owner}/{repo}/contents/{path}"
        
        try:
            data = self._request("GET", url_path, params={"ref": ref})
        except GitHubApiError as e:
            # Missing file at that ref, or too large.
            if "404" in str(e) or "409" in str(e) or "too large" in str(e).lower():
                return None
            raise

        if isinstance(data, dict) and data.get("encoding") == "base64" and "content" in data:
            try:
                return base64.b64decode(data["content"], validate=False)
            except Exception:
                return None

        return None


def parse_github_datetime(dt: str) -> datetime:
    """Parse a GitHub ISO-8601 timestamp string into a timezone-aware datetime.

    Args:
        dt: Timestamp from GitHub (e.g., "2020-01-01T00:00:00Z").

    Returns:
        A timezone-aware datetime (UTC).
    """    
    return datetime.fromisoformat(dt.replace("Z", "+00:00"))
