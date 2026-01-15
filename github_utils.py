import logging
from typing import List

import streamlit as st
import pandas as pd
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


def iter_commits_serial(repo_owner: str, repo_name: str):
    """Yield non-merge commits serially with stats for ``repo_owner/repo_name``."""
    try:
        access_token = st.secrets["GITHUB_API_KEY"]
        g = Github(access_token)
        logging.info(f"Fetching repo: {repo_owner}/{repo_name}")
        repo = g.get_repo(f"{repo_owner}/{repo_name}")
        logging.info(f"Streaming commits for {repo_owner}/{repo_name}")
        for commit in repo.get_commits():
            if len(commit.parents) > 1:
                continue
            try:
                yield repo.get_commit(commit.sha)
            except Exception as e:
                logging.error(f"Error fetching commit {commit.sha}: {e}")
                continue
    except Exception as e:
        logging.error(f"Error fetching commits for {repo_owner}/{repo_name}: {e}")
        st.error(
            f"Failed to fetch commits for {repo_owner}/{repo_name}. Please check the repository details and your GitHub API key. Error: {e}"
        )


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

    header = "| Date | Check-ins | Lines Added | Lines Deleted | Lines Added/Check-in | Lines Deleted/Check-in |"
    separator = "|------|-----------|-------------|---------------|----------------------|------------------------|"
    lines = [header, separator]

    totals = {"checkins": 0, "additions": 0, "deletions": 0}
    for day in sorted(daily):
        stats = daily[day]
        checkin_denominator = stats["checkins"] or 1
        lines.append(
            f"| {day} | {stats['checkins']} | {stats['additions']} | {stats['deletions']} | "
            f"{stats['additions'] / checkin_denominator:.2f} | "
            f"{stats['deletions'] / checkin_denominator:.2f} |"
        )
        for key in totals:
            totals[key] += stats[key]

    periods = len(daily) if daily else 1
    checkins_per_period = totals["checkins"] / periods
    checkin_denominator = totals["checkins"] or 1
    averages = {
        "checkins": checkins_per_period,
        "additions": totals["additions"] / checkin_denominator,
        "deletions": totals["deletions"] / checkin_denominator,
    }
    lines.append(
        f"| **Total** | **{totals['checkins']}** | **{totals['additions']}** | **{totals['deletions']}** |"
    )
    lines.append(
        f"| **Average** | **{averages['checkins']:.2f}** | **{averages['additions']:.2f}** | **{averages['deletions']:.2f}** |"
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

    header = "| Week | Check-ins | Lines Added | Lines Deleted | Lines Added/Check-in | Lines Deleted/Check-in |"
    separator = "|------|-----------|-------------|---------------|----------------------|------------------------|"
    lines = [header, separator]

    totals = {"checkins": 0, "additions": 0, "deletions": 0}
    for week in sorted(weekly):
        stats = weekly[week]
        checkin_denominator = stats["checkins"] or 1
        lines.append(
            f"| {week} | {stats['checkins']} | {stats['additions']} | {stats['deletions']} | "
            f"{stats['additions'] / checkin_denominator:.2f} | "
            f"{stats['deletions'] / checkin_denominator:.2f} |"
        )
        for key in totals:
            totals[key] += stats[key]

    periods = len(weekly) if weekly else 1
    checkins_per_period = totals["checkins"] / periods
    checkin_denominator = totals["checkins"] or 1
    averages = {
        "checkins": checkins_per_period,
        "additions": totals["additions"] / checkin_denominator,
        "deletions": totals["deletions"] / checkin_denominator,
    }
    lines.append(
        f"| **Total** | **{totals['checkins']}** | **{totals['additions']}** | **{totals['deletions']}** |"
    )
    lines.append(
        f"| **Average** | **{averages['checkins']:.2f}** | **{averages['additions']:.2f}** | **{averages['deletions']:.2f}** |"
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

    header = "| Month | Check-ins | Lines Added | Lines Deleted | Lines Added/Check-in | Lines Deleted/Check-in |"
    separator = "|------|-----------|-------------|---------------|----------------------|------------------------|"
    lines = [header, separator]

    totals = {"checkins": 0, "additions": 0, "deletions": 0}
    for month in sorted(monthly):
        stats = monthly[month]
        checkin_denominator = stats["checkins"] or 1
        lines.append(
            f"| {month} | {stats['checkins']} | {stats['additions']} | {stats['deletions']} | "
            f"{stats['additions'] / checkin_denominator:.2f} | "
            f"{stats['deletions'] / checkin_denominator:.2f} |"
        )
        for key in totals:
            totals[key] += stats[key]

    periods = len(monthly) if monthly else 1
    checkins_per_period = totals["checkins"] / periods
    checkin_denominator = totals["checkins"] or 1
    averages = {
        "checkins": checkins_per_period,
        "additions": totals["additions"] / checkin_denominator,
        "deletions": totals["deletions"] / checkin_denominator,
    }
    lines.append(
        f"| **Total** | **{totals['checkins']}** | **{totals['additions']}** | **{totals['deletions']}** |"
    )
    lines.append(
        f"| **Average** | **{averages['checkins']:.2f}** | **{averages['additions']:.2f}** | **{averages['deletions']:.2f}** |"
    )
    return "\n".join(lines)


def compute_daily_stats_df(commits, max_count: int) -> pd.DataFrame:
    """Return a DataFrame with daily check-in statistics."""
    from collections import defaultdict

    daily = defaultdict(lambda: {"Check-ins": 0, "Lines Added": 0, "Lines Deleted": 0})
    for count, commit in enumerate(commits):
        if count >= max_count:
            break
        date = commit.commit.author.date.date()
        stats = commit.stats
        daily[date]["Check-ins"] += 1
        daily[date]["Lines Added"] += stats.additions
        daily[date]["Lines Deleted"] += stats.deletions

    records = []
    for day in sorted(daily):
        rec = {"Date": str(day)}
        rec.update(daily[day])
        checkin_denominator = daily[day]["Check-ins"] or 1
        rec["Lines Added/Check-in"] = daily[day]["Lines Added"] / checkin_denominator
        rec["Lines Deleted/Check-in"] = daily[day]["Lines Deleted"] / checkin_denominator
        records.append(rec)
    return pd.DataFrame(records)


def compute_weekly_stats_df(commits, max_count: int) -> pd.DataFrame:
    """Return a DataFrame with weekly check-in statistics."""
    from collections import defaultdict

    weekly = defaultdict(lambda: {"Check-ins": 0, "Lines Added": 0, "Lines Deleted": 0})
    for count, commit in enumerate(commits):
        if count >= max_count:
            break
        dt = commit.commit.author.date
        year, week, _ = dt.isocalendar()
        key = f"{year}-W{week:02d}"
        stats = commit.stats
        weekly[key]["Check-ins"] += 1
        weekly[key]["Lines Added"] += stats.additions
        weekly[key]["Lines Deleted"] += stats.deletions

    records = []
    for week in sorted(weekly):
        rec = {"Week": week}
        rec.update(weekly[week])
        checkin_denominator = weekly[week]["Check-ins"] or 1
        rec["Lines Added/Check-in"] = weekly[week]["Lines Added"] / checkin_denominator
        rec["Lines Deleted/Check-in"] = weekly[week]["Lines Deleted"] / checkin_denominator
        records.append(rec)
    return pd.DataFrame(records)


def compute_monthly_stats_df(commits, max_count: int) -> pd.DataFrame:
    """Return a DataFrame with monthly check-in statistics."""
    from collections import defaultdict

    monthly = defaultdict(lambda: {"Check-ins": 0, "Lines Added": 0, "Lines Deleted": 0})
    for count, commit in enumerate(commits):
        if count >= max_count:
            break
        dt = commit.commit.author.date
        key = dt.strftime("%Y-%m")
        stats = commit.stats
        monthly[key]["Check-ins"] += 1
        monthly[key]["Lines Added"] += stats.additions
        monthly[key]["Lines Deleted"] += stats.deletions

    records = []
    for month in sorted(monthly):
        rec = {"Month": month}
        rec.update(monthly[month])
        checkin_denominator = monthly[month]["Check-ins"] or 1
        rec["Lines Added/Check-in"] = monthly[month]["Lines Added"] / checkin_denominator
        rec["Lines Deleted/Check-in"] = monthly[month]["Lines Deleted"] / checkin_denominator
        records.append(rec)
    return pd.DataFrame(records)


def get_repo_contents(repo_owner: str, repo_name: str):
    """Return the contents of a repository."""
    try:
        access_token = st.secrets["GITHUB_API_KEY"]
        g = Github(access_token)
        repo = g.get_repo(f"{repo_owner}/{repo_name}")
        contents = repo.get_contents("")
        return contents
    except Exception as e:
        logging.error(f"Error fetching contents for {repo_owner}/{repo_name}: {e}")
        st.error(
            f"Failed to fetch contents for {repo_owner}/{repo_name}. Please check the repository details and your GitHub API key. Error: {e}"
        )
        return []
