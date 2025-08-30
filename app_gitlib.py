import streamlit as st
import os
import re

from code_analyzer import analyze_repo

def generate_report(repo_owner, repo_name):
    st.write(f"Generating report for {repo_owner}/{repo_name}...")
    report = analyze_repo(repo_owner, repo_name)
    return report

def get_report_filename(github_url):
    # Sanitize the URL to create a valid filename
    sanitized_url = re.sub(r'https?://', '', github_url)
    sanitized_url = re.sub(r'[^a-zA-Z0-9_-]', '_', sanitized_url)
    return f"{sanitized_url}.md"

def parse_github_url(url):
    match = re.search(r"github\.com/([^/]+)/([^/]+)", url)
    if match:
        return match.group(1), match.group(2)
    return None, None

def main():
    st.title("GitHub Repository Analyzer")

    github_url = st.text_input("Enter a GitHub Repository URL:")
    force_refresh = st.checkbox("Force Refresh")
    analyze_button = st.button("Analyze Repository")

    if analyze_button:
        if not github_url:
            st.error("Please enter a GitHub URL.")
            return

        repo_owner, repo_name = parse_github_url(github_url)
        if not repo_owner or not repo_name:
            st.error("Invalid GitHub URL.")
            return

        report_filename = get_report_filename(github_url)

        if not force_refresh and os.path.exists(report_filename):
            st.info("Loading report from cache...")
            with open(report_filename, "r") as f:
                report_content = f.read()
            st.markdown(report_content)
        else:
            st.info("Generating new report...")
            report_content = generate_report(repo_owner, repo_name)
            with open(report_filename, "w") as f:
                f.write(report_content)
            st.markdown(report_content)

if __name__ == "__main__":
    main()
