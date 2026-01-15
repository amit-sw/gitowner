# GitOwner

GitOwner is a Streamlit app that analyzes GitHub repositories to surface contributor insights and code change patterns. It can summarize commit activity with LLM-assisted analysis and generate line-count trends (daily/weekly/monthly) for a selected repo.

## Features
- Repository picker for an owner/org and repo
- Line count reports with graphs and tables
- Contributor analysis powered by OpenAI (chunked commit summaries + final synthesis)

## Requirements
- Python 3.9+
- GitHub API token with read access
- OpenAI API key

## Setup
1. Create and activate a virtual environment (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Configure Streamlit secrets (see below)
4. Run the app
   ```bash
   streamlit run streamlit_app.py
   ```

## Streamlit Secrets Configuration
This app reads secrets via `st.secrets`. Create a local file at `.streamlit/secrets.toml`:

```toml
GITHUB_API_KEY = "ghp_your_github_token"
OPENAI_API_KEY = "sk_your_openai_key"
```

Notes:
- GitHub token needs permissions to read the target repositories.
- For Streamlit Cloud, add the same keys under your app’s **Settings → Secrets**.
- Do not commit `.streamlit/secrets.toml` to source control.

## Usage
- Enter a GitHub owner/org in the sidebar.
- Pick a repository, set commit count, and choose analysis type.
- Use "Run Line Count Analysis" for stats or "Run Contributor Analysis" for LLM summaries.
