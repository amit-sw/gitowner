import logging
from typing import List

import streamlit as st
from github import Github
import concurrent.futures


def get_repo_candidates(repo_owner: str) -> List[str]:
    """Return a list of repository names owned by ``repo_owner``."""
    try:
        access_token = st.secrets["GITHUB_API_KEY"]
        g = Github(access_token)
        try:
            user = g.get_user(repo_owner)
            repos = user.get_repos()
        except Exception:
            org = g.get_organization(repo_owner)
            repos = org.get_repos()
        return [repo.name for repo in repos]
    except Exception as e:
        logging.error(f"Error fetching repos for {repo_owner}: {e}")
        return []


def get_commits(
    repo_owner: str,
    repo_name: str,
    max_count: int | None = None,
    degree_of_parallelism: int = 4,
):
    """Return a list of non-merge commits for ``repo_owner/repo_name``."""
    try:
        access_token = st.secrets["GITHUB_API_KEY"]
        g = Github(access_token)
        logging.info(f"Fetching repo: {repo_owner}/{repo_name}")
        repo = g.get_repo(f"{repo_owner}/{repo_name}")
        logging.info(f"Fetching commits for {repo_owner}/{repo_name}")

        commit_shas = []
        for commit in repo.get_commits():
            if len(commit.parents) > 1:
                continue
            commit_shas.append(commit.sha)
            if max_count is not None and len(commit_shas) >= max_count:
                break

        def fetch_single_commit(sha):
            try:
                return repo.get_commit(sha)
            except Exception as e:
                logging.error(f"Error fetching commit {sha}: {e}")
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=degree_of_parallelism) as ex:
            commits = list(filter(None, ex.map(fetch_single_commit, commit_shas)))
        return commits
    except Exception as e:
        logging.error(f"Error fetching commits for {repo_owner}/{repo_name}: {e}")
        st.error(
            f"Failed to fetch commits for {repo_owner}/{repo_name}. Please check the repository details and your GitHub API key. Error: {e}"
        )
        return []


def _process_single_commit(commit) -> str:
    info = (
        f"\n\nCommit: {commit.sha}"
        f"\n  Comment: {commit.commit.message}"
        f"\n  Author: {commit.commit.author.name}"
        f"\n  Date: {commit.commit.author.date}"
    )
    files = commit.files
    info += "\n  Files:"
    for file in files:
        info += f"\n    - {file.filename}"
    info += "\n-------------------------\n"
    return info


def extract_commit_info(commits, max_count: int, degree_of_parallelism: int) -> str:
    logging.info(
        f"Starting commit info extraction for max {max_count} commits with {degree_of_parallelism} workers."
    )
    commits_to_process = []
    for count, commit in enumerate(commits):
        if count >= max_count:
            break
        commits_to_process.append(commit)

    if not commits_to_process:
        logging.warning("No commits to process in extract_commit_info.")
        return ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=degree_of_parallelism) as ex:
        try:
            commit_strings = list(ex.map(_process_single_commit, commits_to_process))
        except Exception as e:
            logging.error(f"Error during parallel commit processing: {e}")
            st.error(f"An error occurred while processing commit data: {e}")
            return ""
    logging.info("Finished parallel extraction of commit info.")
    return "".join(commit_strings)


def compute_daily_stats_table(commits, max_count: int) -> str:
    from collections import defaultdict

    daily = defaultdict(lambda: {"checkins": 0, "additions": 0, "deletions": 0})
    for count, commit in enumerate(commits):
        if count >= max_count:
            break
        date = commit.commit.author.date.date()
        stats = commit.stats
        daily[date]["checkins"] += 1
        daily[date]["additions"] += stats.additions
        daily[date]["deletions"] += stats.deletions

    header = "| Date | Check-ins | Lines Added | Lines Deleted |"
    separator = "|------|-----------|-------------|---------------|"
    lines = [header, separator]

    totals = {"checkins": 0, "additions": 0, "deletions": 0}
    for day in sorted(daily):
        stats = daily[day]
        lines.append(
            f"| {day} | {stats['checkins']} | {stats['additions']} | {stats['deletions']} |"
        )
        for key in totals:
            totals[key] += stats[key]

    lines.append(
        f"| **Total** | **{totals['checkins']}** | **{totals['additions']}** | **{totals['deletions']}** |"
    )
    return "\n".join(lines)


def compute_weekly_stats_table(commits, max_count: int) -> str:
    from collections import defaultdict

    weekly = defaultdict(lambda: {"checkins": 0, "additions": 0, "deletions": 0})
    for count, commit in enumerate(commits):
        if count >= max_count:
            break
        dt = commit.commit.author.date
        year, week, _ = dt.isocalendar()
        key = f"{year}-W{week:02d}"
        stats = commit.stats
        weekly[key]["checkins"] += 1
        weekly[key]["additions"] += stats.additions
        weekly[key]["deletions"] += stats.deletions

    header = "| Week | Check-ins | Lines Added | Lines Deleted |"
    separator = "|------|-----------|-------------|---------------|"
    lines = [header, separator]

    totals = {"checkins": 0, "additions": 0, "deletions": 0}
    for week in sorted(weekly):
        stats = weekly[week]
        lines.append(
            f"| {week} | {stats['checkins']} | {stats['additions']} | {stats['deletions']} |"
        )
        for key in totals:
            totals[key] += stats[key]

    lines.append(
        f"| **Total** | **{totals['checkins']}** | **{totals['additions']}** | **{totals['deletions']}** |"
    )
    return "\n".join(lines)


def compute_monthly_stats_table(commits, max_count: int) -> str:
    from collections import defaultdict

    monthly = defaultdict(lambda: {"checkins": 0, "additions": 0, "deletions": 0})
    for count, commit in enumerate(commits):
        if count >= max_count:
            break
        dt = commit.commit.author.date
        key = dt.strftime("%Y-%m")
        stats = commit.stats
        monthly[key]["checkins"] += 1
        monthly[key]["additions"] += stats.additions
        monthly[key]["deletions"] += stats.deletions

    header = "| Month | Check-ins | Lines Added | Lines Deleted |"
    separator = "|------|-----------|-------------|---------------|"
    lines = [header, separator]

    totals = {"checkins": 0, "additions": 0, "deletions": 0}
    for month in sorted(monthly):
        stats = monthly[month]
        lines.append(
            f"| {month} | {stats['checkins']} | {stats['additions']} | {stats['deletions']} |"
        )
        for key in totals:
            totals[key] += stats[key]

    lines.append(
        f"| **Total** | **{totals['checkins']}** | **{totals['additions']}** | **{totals['deletions']}** |"
    )
    return "\n".join(lines)
