import logging
import time
from pathlib import Path
from typing import List
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import streamlit as st
import pandas as pd

from github_utils import (
    get_repo_candidates,
    iter_commits_serial,
    compute_daily_stats_table,
    compute_weekly_stats_table,
    compute_monthly_stats_table,
    compute_daily_stats_df,
    compute_weekly_stats_df,
    compute_monthly_stats_df,
)
from report_generator import process_commits_and_generate_report
from git_local_utils import (
    ensure_repo_clone,
    iter_local_commit_stats,
    refetch_repo_objects,
    get_local_commit_headers,
    get_commit_stats,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_REPO_OWNER = "amit-sw"
DEFAULT_REPO_NAME = "gitowner"
DEFAULT_COMMIT_COUNT = 2000

DEFAULT_PROMPT = """
You are a code analysis expoert.

Based on these check-ins, please identify the key code areas, as well as the best two experts on each.
"""
DEFAULT_INITIAL_ANALYSIS_PROMPT = DEFAULT_PROMPT

DEFAULT_FINAL_SUMMARY_PROMPT = """
You are a code summarization expert.
Based on the following individual analyses of commit chunks, please synthesize a single, coherent summary.
Identify key overall themes, common areas of change, and the most impacted experts.
Ensure the final output is a unified report.
"""


@st.cache_data(show_spinner=False)
def cached_repo_candidates(owner: str) -> List[str]:
    return get_repo_candidates(owner)


def run_line_count_analysis(
    repo_owner: str,
    repo_name: str,
    commit_count: int,
    parallelism: int,
    use_local_git: bool,
):
    st.title(f"Line Count Report for {repo_owner}/{repo_name}")
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    progress_text = st.empty()
    start_time = time.time()
    display_placeholder = st.empty()

    def update_status(label, state="running"):
        elapsed = time.time() - start_time
        if state == "error":
            status_placeholder.error(f"{label} ({elapsed:.1f}s elapsed)")
        elif state == "complete":
            status_placeholder.success(f"{label} ({elapsed:.1f}s elapsed)")
        else:
            status_placeholder.info(f"{label} ({elapsed:.1f}s elapsed)")

    with st.spinner("Running line count analysis..."):
        if commit_count <= 0:
            update_status("Commit count must be greater than 0.", state="error")
            return

        if use_local_git:
            try:
                cache_dir = Path(".gitowner_cache")
                repo_path = ensure_repo_clone(repo_owner, repo_name, cache_dir, update_status)
                update_status("Streaming commits from local git clone...")
            except Exception as exc:
                update_status(f"Local git setup failed: {exc}", state="error")
                return
        else:
            update_status("Fetching commits serially via GitHub API...")
        progress_bar = progress_placeholder.progress(0)
        progress_text.info(f"Fetched 0 / {commit_count} commits")

        if use_local_git and parallelism > 1:
            headers = get_local_commit_headers(repo_path, max_count=commit_count)
            if not headers:
                update_status("No commits found for the selected repository.", state="error")
                return

            total_commits = min(len(headers), commit_count)
            update_status(f"Processing commits in parallel ({parallelism} workers)...")

            daily = defaultdict(lambda: {"checkins": 0, "additions": 0, "deletions": 0})
            weekly = defaultdict(lambda: {"checkins": 0, "additions": 0, "deletions": 0})
            monthly = defaultdict(lambda: {"checkins": 0, "additions": 0, "deletions": 0})
            lock = Lock()

            counters = {"processed": 0}
            completed_count = 0
            skipped_count = 0

            def _update_aggregates(commit_date, stats):
                date_key = commit_date.date()
                year, week, _ = commit_date.isocalendar()
                week_key = f"{year}-W{week:02d}"
                month_key = commit_date.strftime("%Y-%m")
                daily[date_key]["checkins"] += 1
                daily[date_key]["additions"] += stats.additions
                daily[date_key]["deletions"] += stats.deletions
                weekly[week_key]["checkins"] += 1
                weekly[week_key]["additions"] += stats.additions
                weekly[week_key]["deletions"] += stats.deletions
                monthly[month_key]["checkins"] += 1
                monthly[month_key]["additions"] += stats.additions
                monthly[month_key]["deletions"] += stats.deletions

            def _snapshot_periods(source):
                return {key: dict(value) for key, value in source.items()}

            def _build_stats_table(periods, label, include_per_checkin=False):
                if include_per_checkin:
                    header = (
                        f"| {label} | Check-ins | Lines Added | Lines Deleted | "
                        "Lines Added/Check-in | Lines Deleted/Check-in |"
                    )
                    separator = (
                        "|------|-----------|-------------|---------------|"
                        "----------------------|------------------------|"
                    )
                else:
                    header = f"| {label} | Check-ins | Lines Added | Lines Deleted |"
                    separator = "|------|-----------|-------------|---------------|"
                lines = [header, separator]
                totals = {"checkins": 0, "additions": 0, "deletions": 0}
                for period in sorted(periods):
                    stats = periods[period]
                    row = (
                        f"| {period} | {stats['checkins']} | "
                        f"{stats['additions']} | {stats['deletions']} |"
                    )
                    if include_per_checkin:
                        checkin_denominator = stats["checkins"] or 1
                        row = (
                            f"{row} {stats['additions'] / checkin_denominator:.2f} | "
                            f"{stats['deletions'] / checkin_denominator:.2f} |"
                        )
                    lines.append(row)
                    for key in totals:
                        totals[key] += stats[key]
                periods_count = len(periods) if periods else 1
                checkins_per_period = totals["checkins"] / periods_count
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

            def _build_stats_df(periods, label, include_per_checkin=False):
                records = []
                for period in sorted(periods):
                    stats = periods[period]
                    record = {
                        label: str(period),
                        "Check-ins": stats["checkins"],
                        "Lines Added": stats["additions"],
                        "Lines Deleted": stats["deletions"],
                    }
                    if include_per_checkin:
                        checkin_denominator = stats["checkins"] or 1
                        record["Lines Added/Check-in"] = stats["additions"] / checkin_denominator
                        record["Lines Deleted/Check-in"] = stats["deletions"] / checkin_denominator
                    records.append(record)
                return pd.DataFrame(records)

            def _process_header(commit_sha, commit_date):
                stats = get_commit_stats(repo_path, commit_sha)
                if stats is None:
                    return False
                with lock:
                    _update_aggregates(commit_date, stats)
                    counters["processed"] += 1
                return True

            with ThreadPoolExecutor(max_workers=parallelism) as executor:
                futures = [
                    executor.submit(_process_header, commit_sha, commit_date)
                    for commit_sha, commit_date in headers[:total_commits]
                ]
                for future in as_completed(futures):
                    completed_count += 1
                    try:
                        result = future.result()
                    except Exception as exc:
                        skipped_count += 1
                        logging.warning(f"Skipping commit due to error: {exc}")
                        result = None

                    with lock:
                        daily_snapshot = _snapshot_periods(daily)
                        weekly_snapshot = _snapshot_periods(weekly)
                        monthly_snapshot = _snapshot_periods(monthly)
                        processed_count = counters["processed"]

                    if result is False:
                        skipped_count += 1

                    progress_bar.progress(completed_count / total_commits)
                    progress_text.info(
                        f"Processed {processed_count} / {total_commits} check-ins"
                    )
                    update_status(f"Computed stats for last {processed_count} check-ins.")
                    daily_table = _build_stats_table(daily_snapshot, "Date", include_per_checkin=True)
                    weekly_table = _build_stats_table(weekly_snapshot, "Week", include_per_checkin=True)
                    monthly_table = _build_stats_table(monthly_snapshot, "Month", include_per_checkin=True)
                    daily_df = _build_stats_df(daily_snapshot, "Date", include_per_checkin=True)
                    weekly_df = _build_stats_df(weekly_snapshot, "Week", include_per_checkin=True)
                    monthly_df = _build_stats_df(monthly_snapshot, "Month", include_per_checkin=True)
                    with display_placeholder.container():
                        st.subheader(f"Stats for last {processed_count} check-ins")
                        tabs = st.tabs(["Tables", "Graphs"])

                        with tabs[1]:
                            if not daily_df.empty:
                                st.subheader("Daily")
                                st.line_chart(
                                    daily_df.set_index(daily_df.columns[0])[["Lines Added", "Lines Deleted"]]
                                )

                            if not weekly_df.empty:
                                st.subheader("Weekly")
                                st.line_chart(
                                    weekly_df.set_index(weekly_df.columns[0])[["Lines Added", "Lines Deleted"]]
                                )

                            if not monthly_df.empty:
                                st.subheader("Monthly")
                                st.line_chart(
                                    monthly_df.set_index(monthly_df.columns[0])[["Lines Added", "Lines Deleted"]]
                                )

                        with tabs[0]:
                            if daily_table:
                                st.subheader("Daily")
                                st.markdown(daily_table)

                            if weekly_table:
                                st.subheader("Weekly")
                                st.markdown(weekly_table)

                            if monthly_table:
                                st.subheader("Monthly")
                                st.markdown(monthly_table)

            if processed_count == 0:
                update_status("No commits found for the selected repository.", state="error")
                return

            if skipped_count:
                update_status(f"Skipped {skipped_count} commits due to missing objects.")

            update_status("Line count analysis complete!", state="complete")
            progress_bar.progress(1.0)
            total_time = time.time() - start_time
            status_placeholder.success(f"Total time taken: {total_time:.2f}s")
            return

        commits = []
        if use_local_git:
            commit_iter = iter_local_commit_stats(repo_path, max_count=commit_count)
        else:
            commit_iter = iter_commits_serial(repo_owner, repo_name)

        attempts = 0
        while True:
            try:
                for commit in commit_iter:
                    commits.append(commit)
                    current_count = len(commits)
                    progress = min(current_count / commit_count, 1.0)
                    progress_bar.progress(progress)
                    progress_text.info(f"Fetched {min(current_count, commit_count)} / {commit_count} commits")
                    if current_count >= commit_count:
                        current_count = commit_count
                        commits = commits[:commit_count]
                    daily_table = compute_daily_stats_table(commits, current_count)
                    weekly_table = compute_weekly_stats_table(commits, current_count)
                    monthly_table = compute_monthly_stats_table(commits, current_count)
                    daily_df = compute_daily_stats_df(commits, current_count)
                    weekly_df = compute_weekly_stats_df(commits, current_count)
                    monthly_df = compute_monthly_stats_df(commits, current_count)
                    update_status(f"Computed stats for last {current_count} check-ins.")
                    with display_placeholder.container():
                        st.subheader(f"Stats for last {current_count} check-ins")
                        tabs = st.tabs(["Tables", "Graphs"])

                        with tabs[1]:
                            if daily_df is not None:
                                st.subheader("Daily")
                                st.line_chart(
                                    daily_df.set_index(daily_df.columns[0])[["Lines Added", "Lines Deleted"]]
                                )

                            if weekly_df is not None:
                                st.subheader("Weekly")
                                st.line_chart(
                                    weekly_df.set_index(weekly_df.columns[0])[["Lines Added", "Lines Deleted"]]
                                )

                            if monthly_df is not None:
                                st.subheader("Monthly")
                                st.line_chart(
                                    monthly_df.set_index(monthly_df.columns[0])[["Lines Added", "Lines Deleted"]]
                                )

                        with tabs[0]:
                            if daily_table:
                                st.subheader("Daily")
                                st.markdown(daily_table)

                            if weekly_table:
                                st.subheader("Weekly")
                                st.markdown(weekly_table)

                            if monthly_table:
                                st.subheader("Monthly")
                                st.markdown(monthly_table)
                    if len(commits) >= commit_count:
                        break
                break
            except RuntimeError as exc:
                if not use_local_git or attempts >= 1:
                    update_status(f"Local git log failed: {exc}", state="error")
                    return
                update_status("Local git cache corrupted. Attempting repair...")
                try:
                    refetch_repo_objects(repo_path, update_status)
                except Exception as repair_exc:
                    update_status(f"Local git repair failed: {repair_exc}", state="error")
                    return
                attempts += 1
                commits = []
                progress_bar.progress(0)
                progress_text.info(f"Fetched 0 / {commit_count} commits")
                commit_iter = iter_local_commit_stats(repo_path, max_count=commit_count)

        if not commits:
            update_status("No commits found for the selected repository.", state="error")
            return

        update_status("Line count analysis complete!", state="complete")
        progress_bar.progress(1.0)
    total_time = time.time() - start_time
    status_placeholder.success(f"Total time taken: {total_time:.2f}s")


def run_contributor_analysis(
    repo_owner: str,
    repo_name: str,
    commit_count: int,
    parallelism: int,
    max_context: int,
):
    st.title(f"Git Analysis Report for {repo_owner}/{repo_name}")
    status_placeholder = st.empty()
    start_time = time.time()

    def update_status(label, state="running"):
        elapsed = time.time() - start_time
        if state == "error":
            status_placeholder.error(f"{label} ({elapsed:.1f}s elapsed)")
        elif state == "complete":
            status_placeholder.success(f"{label} ({elapsed:.1f}s elapsed)")
        else:
            status_placeholder.info(f"{label} ({elapsed:.1f}s elapsed)")

    with st.spinner("Running contributor analysis..."):
        stats_report, analysis_report = process_commits_and_generate_report(
            repo_owner=repo_owner,
            repo_name=repo_name,
            commit_count=commit_count,
            degree_of_parallelism=parallelism,
            max_context_window=max_context,
            initial_analysis_prompt=DEFAULT_INITIAL_ANALYSIS_PROMPT,
            final_summary_prompt=DEFAULT_FINAL_SUMMARY_PROMPT,
            status_ui_update_callback=update_status,
        )
        if "Error:" in analysis_report or not analysis_report:
            update_status("Analysis encountered an error.", state="error")
        elif "No analysis performed." in analysis_report:
            update_status("Analysis complete.", state="complete")
        else:
            update_status("Analysis complete!", state="complete")

        tabs = st.tabs(["Check-in Stats", "Contributor Analysis"])
        with tabs[0]:
            if stats_report:
                st.markdown(stats_report)
            else:
                st.write("No statistics available.")
        with tabs[1]:
            st.markdown(analysis_report)
    total_time = time.time() - start_time
    status_placeholder.success(f"Total time taken: {total_time:.2f}s")


def main():
    st.sidebar.title("🎈Who dun it?")
    use_local_git = st.sidebar.checkbox(
        "Use local git clone for line count analysis",
        value=True,
    )
    st.sidebar.caption(
        "Local cache stored in `.gitowner_cache`. Delete this folder to refresh."
    )
    repo_owner = st.sidebar.text_input("acct", value=DEFAULT_REPO_OWNER)
    repo_candidates = cached_repo_candidates(repo_owner)
    repo_name = st.sidebar.selectbox(
        "repo",
        options=repo_candidates if repo_candidates else [DEFAULT_REPO_NAME],
    )
    commit_count = st.sidebar.number_input("commits", value=DEFAULT_COMMIT_COUNT)
    parallelism = st.sidebar.number_input("Parallelism", value=20)
    max_context = st.sidebar.number_input("Max LLM Context (Tokens)", value=20000)
    run_stats = st.sidebar.button("Run Line Count Analysis")
    run_contrib = st.sidebar.button("Run Contributor Analysis")

    if run_stats:
        run_line_count_analysis(repo_owner, repo_name, commit_count, parallelism, use_local_git)
    elif run_contrib:
        run_contributor_analysis(repo_owner, repo_name, commit_count, parallelism, max_context)


if __name__ == "__main__":
    main()
