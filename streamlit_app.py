import logging
import time
from typing import List

import streamlit as st

from github_utils import get_repo_candidates
from report_generator import (
    process_commits_and_generate_report,
    process_commits_and_generate_stats,
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


def run_line_count_analysis(repo_owner: str, repo_name: str, commit_count: int, parallelism: int):
    st.title(f"Line Count Report for {repo_owner}/{repo_name}")
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

    with st.spinner("Running line count analysis..."):
        daily_table, weekly_table, monthly_table, daily_df, weekly_df, monthly_df = process_commits_and_generate_stats(
            repo_owner=repo_owner,
            repo_name=repo_name,
            commit_count=commit_count,
            degree_of_parallelism=parallelism,
            status_ui_update_callback=update_status,
        )
        update_status("Line count analysis complete!", state="complete")

        tabs = st.tabs(["Graphs", "Tables"])

        with tabs[0]:
            if daily_df is not None:
                st.subheader("Daily")
                st.line_chart(
                    daily_df.set_index(daily_df.columns[0])[ ["Lines Added", "Lines Deleted"] ]
                )

            if weekly_df is not None:
                st.subheader("Weekly")
                st.line_chart(
                    weekly_df.set_index(weekly_df.columns[0])[ ["Lines Added", "Lines Deleted"] ]
                )

            if monthly_df is not None:
                st.subheader("Monthly")
                st.line_chart(
                    monthly_df.set_index(monthly_df.columns[0])[ ["Lines Added", "Lines Deleted"] ]
                )

        with tabs[1]:
            if daily_table:
                st.subheader("Daily")
                st.markdown(daily_table)

            if weekly_table:
                st.subheader("Weekly")
                st.markdown(weekly_table)

            if monthly_table:
                st.subheader("Monthly")
                st.markdown(monthly_table)
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
        run_line_count_analysis(repo_owner, repo_name, commit_count, parallelism)
    elif run_contrib:
        run_contributor_analysis(repo_owner, repo_name, commit_count, parallelism, max_context)


if __name__ == "__main__":
    main()
