import datetime
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple


@dataclass
class CommitAuthor:
    date: datetime.datetime


@dataclass
class CommitMeta:
    author: CommitAuthor


@dataclass
class CommitStats:
    additions: int
    deletions: int


@dataclass
class CommitRecord:
    commit: CommitMeta
    stats: CommitStats


def _run_git_command(args: list[str], error_label: str) -> None:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{error_label}: {stderr}")


def ensure_repo_clone(
    repo_owner: str,
    repo_name: str,
    cache_dir: Path,
    status_callback=None,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_path = cache_dir / f"{repo_owner}_{repo_name}"
    repo_url = f"https://github.com/{repo_owner}/{repo_name}.git"

    if (repo_path / ".git").exists():
        if status_callback:
            status_callback("Updating local repository cache...")
        _run_git_command(
            ["git", "-C", str(repo_path), "fetch", "--prune", "--tags"],
            "Failed to fetch updates",
        )
    else:
        if status_callback:
            status_callback("Cloning repository locally...")
        _run_git_command(
            [
                "git",
                "clone",
                "--no-tags",
                "--filter=blob:none",
                repo_url,
                str(repo_path),
            ],
            "Failed to clone repository",
        )

    return repo_path


def refetch_repo_objects(repo_path: Path, status_callback=None) -> None:
    if status_callback:
        status_callback("Refetching missing git objects...")
    _run_git_command(
        ["git", "-C", str(repo_path), "fetch", "--refetch", "--prune", "--tags"],
        "Failed to refetch missing objects",
    )


def _parse_iso_datetime(value: str) -> datetime.datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.datetime.fromisoformat(normalized)


def get_local_commit_headers(
    repo_path: Path,
    max_count: Optional[int] = None,
) -> list[Tuple[str, datetime.datetime]]:
    args = [
        "git",
        "-C",
        str(repo_path),
        "log",
        "--no-merges",
        "--date=iso-strict",
        "--pretty=format:%H|%ad",
    ]
    if max_count is not None:
        args.extend(["-n", str(max_count)])

    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if not process.stdout:
        return []

    headers: list[Tuple[str, datetime.datetime]] = []
    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("|", 1)
        if len(parts) != 2:
            logging.warning(f"Unexpected git log header format: {line}")
            continue
        commit_sha, date_raw = parts
        try:
            commit_date = _parse_iso_datetime(date_raw)
        except ValueError:
            logging.warning(f"Unexpected git log date format: {line}")
            continue
        headers.append((commit_sha, commit_date))
        if max_count is not None and len(headers) >= max_count:
            break

    process.stdout.close()
    stderr = ""
    if process.stderr:
        stderr = process.stderr.read().strip()
        process.stderr.close()
    process.wait()
    if process.returncode not in (0, None):
        if stderr:
            logging.warning(f"git log incomplete: {stderr}")
        else:
            logging.warning(f"git log incomplete: exit code {process.returncode}")

    return headers


def iter_local_commit_stats(
    repo_path: Path,
    max_count: Optional[int] = None,
) -> Iterator[CommitRecord]:
    headers = get_local_commit_headers(repo_path, max_count=max_count)
    for commit_sha, commit_date in headers:
        stats = get_commit_stats(repo_path, commit_sha)
        if stats is None:
            continue
        yield CommitRecord(
            commit=CommitMeta(author=CommitAuthor(date=commit_date)),
            stats=stats,
        )


def get_commit_stats(repo_path: Path, commit_sha: str) -> Optional[CommitStats]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_path),
            "show",
            "--numstat",
            "--format=",
            "-1",
            commit_sha,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        logging.warning(f"Skipping commit {commit_sha}: {stderr}")
        return None

    additions = 0
    deletions = 0
    for raw_line in result.stdout.splitlines():
        columns = raw_line.split("\t")
        if len(columns) < 2:
            continue
        added_raw, deleted_raw = columns[0], columns[1]
        try:
            added = int(added_raw)
        except ValueError:
            added = 0
        try:
            deleted = int(deleted_raw)
        except ValueError:
            deleted = 0
        additions += added
        deletions += deleted

    return CommitStats(additions=additions, deletions=deletions)
