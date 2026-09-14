# GitHub Function Change Miner

Python research data collection pipeline that uses the GitHub REST API to:
- select public Python repositories (created on or before 2022-12-31, non-archived, non-fork), ranked by stars, up to `target_repository_count` in `src/config.py` (currently 1,000)
- find at least 5 Python files per repo where the **latest** pre-2023 adjacent commit pair contains an **existing function body change**
- write JSON datasets (one indented array per file) with repository, file, and function-level change records

## Setup

1. Provide a GitHub token:

Option A: Put it in `.env`

```bash
# .env
GITHUB_TOKEN=...
```

```bash
pip install -r requirements.txt
```

## Run

```bash
python3 main.py
```

Outputs are written to:
- `output/selected_repositories.json`
- `output/changed_files.json`
- `output/changed_functions.json`

## Notes
- This project uses only the GitHub REST API (no HTML scraping).
- Filtering and selection settings live in `src/config.py` as constants (no CLI flags).
