import streamlit as st
from github_utils import get_repo_contents
from llm_utils import analyze_text_chunks_parallel, summarize_intermediate_results
import base64

def get_file_content(file):
    """Get the content of a file, decoding it from base64."""
    try:
        content_b64 = file.content
        content_bytes = base64.b64decode(content_b64)
        return content_bytes.decode('utf-8')
    except Exception as e:
        return f"Error decoding file content: {e}"

def analyze_repo(repo_owner, repo_name):
    st.write(f"Analyzing repository: {repo_owner}/{repo_name}")

    contents = get_repo_contents(repo_owner, repo_name)
    if not contents:
        return "Could not retrieve repository contents."

    report = f"# Analysis for {repo_owner}/{repo_name}\n\n"

    # First, find and summarize the README
    readme = None
    for content_file in contents:
        if content_file.name.lower() == "readme.md":
            readme = content_file
            break

    readme_summary = "No README file found."
    if readme:
        readme_content = get_file_content(readme)
        prompt = "Summarize the purpose of the repository based on this README file."
        readme_summary = summarize_intermediate_results([readme_content], 16000, prompt)

    report += f"## Repository Purpose\n\n{readme_summary}\n\n"

    # Then, analyze the rest of the files
    files_to_analyze = []
    for content_file in contents:
        if content_file.type == 'dir':
            # In a real implementation, we would recurse into directories
            report += f"### Directory: {content_file.path}\n\n"
        else:
            files_to_analyze.append(content_file)

    for file in files_to_analyze:
        report += f"### File: {file.path}\n"

        # Simple language detection based on extension
        language = "Unknown"
        if '.' in file.name:
            extension = file.name.split('.')[-1]
            if extension in ['py', 'python']:
                language = 'Python'
            elif extension in ['js', 'javascript']:
                language = 'JavaScript'
            elif extension in ['java']:
                language = 'Java'
            elif extension in ['go']:
                language = 'Go'
            elif extension in ['rb']:
                language = 'Ruby'
            elif extension in ['php']:
                language = 'PHP'
            elif extension in ['cs']:
                language = 'C#'
            elif extension in ['cpp', 'cxx', 'cc']:
                language = 'C++'
            elif extension in ['c']:
                language = 'C'
            elif extension in ['html']:
                language = 'HTML'
            elif extension in ['css']:
                language = 'CSS'

        report += f"- **Language:** {language}\n"

        # Get file content and generate pseudo-code
        file_content = get_file_content(file)
        if "Error" not in file_content:
            prompt = f"Generate a brief, high-level pseudo-code for the following {language} file. Focus on the main logic and functions."
            pseudo_code = summarize_intermediate_results([file_content], 16000, prompt)
            report += f"- **Pseudo-code:**\n```\n{pseudo_code}\n```\n\n"
        else:
            report += f"- **Content:** Could not decode file content.\n\n"

    return report
