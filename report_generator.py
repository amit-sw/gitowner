import logging
from typing import Tuple

import streamlit as st

from github_utils import (
    get_commits,
    extract_commit_info,
    compute_daily_stats_table,
    compute_weekly_stats_table,
    compute_monthly_stats_table,
    compute_daily_stats_df,
    compute_weekly_stats_df,
    compute_monthly_stats_df,
)
from llm_utils import analyze_text_chunks_parallel, summarize_intermediate_results


def fetch_commit_data(
    repo_owner: str,
    repo_name: str,
    commit_count: int,
    degree_of_parallelism: int,
    status_ui_update_callback,
):
    status_ui_update_callback(label="Fetching commits from repository...")
    commits = get_commits(
        repo_owner,
        repo_name,
        max_count=commit_count,
        degree_of_parallelism=degree_of_parallelism,
    )
    if not commits:
        status_ui_update_callback(
            label="Failed to fetch commits. Please check repository details and API key.",
            state="error",
        )
        logging.error("Failed to fetch commits.")
        return None, "", "", "", []

    commit_text = extract_commit_info(commits, commit_count, degree_of_parallelism)
    daily_stats = compute_daily_stats_table(commits, commit_count)
    weekly_stats = compute_weekly_stats_table(commits, commit_count)
    monthly_stats = compute_monthly_stats_table(commits, commit_count)
    return commit_text, daily_stats, weekly_stats, monthly_stats, commits


def process_commits_and_generate_report(
    repo_owner: str,
    repo_name: str,
    commit_count: int,
    degree_of_parallelism: int,
    max_context_window: int,
    initial_analysis_prompt: str,
    final_summary_prompt: str,
    status_ui_update_callback,
) -> Tuple[str, str]:
    commit_text, daily, weekly, monthly, _ = fetch_commit_data(
        repo_owner,
        repo_name,
        commit_count,
        degree_of_parallelism,
        status_ui_update_callback,
    )
    if commit_text is None:
        return "", "Error: Could not fetch commits."
    if not commit_text and commit_count == 0:
        status_ui_update_callback(label="Commit count is 0. No analysis performed.", state="complete")
        return "", "Commit count is 0. No analysis performed."
    if not commit_text:
        status_ui_update_callback(label="Failed to extract commit information.", state="error")
        return "", "Error: Could not extract commit information."

    status_ui_update_callback(label="Analyzing commit information with LLM (initial pass)...")
    analyses = analyze_text_chunks_parallel(
        full_text=commit_text,
        degree_of_parallelism=degree_of_parallelism,
        max_context_window=max_context_window,
        system_prompt=initial_analysis_prompt,
    )
    if not analyses or all(a.startswith("Error:") for a in analyses):
        status_ui_update_callback(label="Failed during initial LLM analysis.", state="error")
        detail = analyses[0] if analyses else "No analysis results."
        return "", f"Error: Failed during initial analysis step. Details: {detail}"

    status_ui_update_callback(label="Generating final summary with LLM...")
    final_report = summarize_intermediate_results(
        analysis_strings=analyses,
        max_context_window=max_context_window,
        system_prompt=final_summary_prompt,
    )
    if not final_report or final_report.startswith("Error:"):
        status_ui_update_callback(label="Failed during final summary generation.", state="error")
        return "", final_report

    combined_stats = "\n\n".join([daily, weekly, monthly])
    return combined_stats, final_report
def process_commits_and_generate_stats(
    repo_owner: str,
    repo_name: str,
    commit_count: int,
    degree_of_parallelism: int,
    status_ui_update_callback,
):
    commit_text, daily, weekly, monthly, commits = fetch_commit_data(
        repo_owner,
        repo_name,
        commit_count,
        degree_of_parallelism,
        status_ui_update_callback,
    )
    if commit_text is None:
        return "", "", "", None, None, None

    daily_df = compute_daily_stats_df(commits, commit_count)
    weekly_df = compute_weekly_stats_df(commits, commit_count)
    monthly_df = compute_monthly_stats_df(commits, commit_count)

    return daily, weekly, monthly, daily_df, weekly_df, monthly_df
