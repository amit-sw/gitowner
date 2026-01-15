# Repository Guidelines

## Project Structure & Module Organization
- Root-level Python modules: `streamlit_app.py` (UI entry point), `github_utils.py`, `app_gitlib.py`, `code_analyzer.py`, `report_generator.py`, and `llm_utils.py`.
- Configuration and dependencies: `requirements.txt`.
- Documentation: `README.md`.
- There is no dedicated `tests/` directory in the current repository.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs Python dependencies.
- `streamlit run streamlit_app.py` runs the Streamlit app locally.
- Optional: use a virtual environment (`python -m venv .venv` then `source .venv/bin/activate`) before installing dependencies.

## Coding Style & Naming Conventions
- Follow Python PEP 8 conventions: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes.
- Keep modules focused and avoid circular imports between utility files.
- No formatter or linter is configured; if you introduce one, document it in `README.md` and update this file.

## Testing Guidelines
- No automated test framework is currently set up.
- If you add tests, place them under `tests/` and name files `test_*.py`.
- Prefer small, deterministic unit tests for analysis utilities and keep Streamlit UI tests minimal.

## Commit & Pull Request Guidelines
- Commit history uses short, sentence-case summaries (e.g., "Updated Langchain library"). Keep messages concise and descriptive.
- For pull requests, include:
  - A brief summary of changes and motivation.
  - Steps to validate (commands run, expected output).
  - Screenshots/GIFs for UI changes in `streamlit_app.py`.

## Configuration & Secrets
- Keep secrets out of source control; use environment variables or a local `.env` (not committed).
- If you add configuration files, document defaults and required variables in `README.md`.
