import streamlit as st
from github import Github

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import json
import markdown
import concurrent.futures
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OPENAI_MODEL="gpt-4.1-mini"
DEFAULT_REPO_OWNER="deepchem"
DEFAULT_REPO_NAME="deepchem"
DEFAULT_COMMIT_COUNT=20

DEFAULT_PROMPT="""
You are a code analysis expoert.

Based on these check-ins, please identify the key code areas, as well as the best two experts on each.
"""

def get_commits(repo_owner=DEFAULT_REPO_OWNER, repo_name=DEFAULT_REPO_NAME):
  try:
    access_token = st.secrets['GITHUB_API_KEY']
    g = Github(access_token)
    logging.info(f"Fetching repo: {repo_owner}/{repo_name}")
    repo = g.get_repo(f"{repo_owner}/{repo_name}")
    logging.info(f"Fetching commits for {repo_owner}/{repo_name}")
    commits = repo.get_commits()
    return commits
  except Exception as e:
    logging.error(f"Error fetching commits for {repo_owner}/{repo_name}: {e}")
    st.error(f"Failed to fetch commits for {repo_owner}/{repo_name}. Please check the repository details and your GitHub API key. Error: {e}")
    return []

def _process_single_commit(commit):
  commit_info = f"\n\nCommit: {commit.sha}" + \
                f"\n  Comment: {commit.commit.message}" + \
                f"\n  Author: {commit.commit.author.name}" + \
                f"\n  Date: {commit.commit.author.date}"
  files = commit.files
  commit_info += "\n  Files:"
  for file in files:
      commit_info += f"\n    - {file.filename}"
  commit_info += "\n-------------------------\n"
  return commit_info

def extract_commit_info(commits, max_count, degree_of_parallelism):
  logging.info(f"Starting commit info extraction for max {max_count} commits with {degree_of_parallelism} workers.")
  # Convert PaginatedList to a list and slice it
  commits_to_process = []
  count = 0
  for commit in commits:
      if count >= max_count:
          break
      commits_to_process.append(commit)
      count += 1
  
  if not commits_to_process:
      logging.warning("No commits to process in extract_commit_info.")
      return ""

  logging.info(f"Processing {len(commits_to_process)} commits in parallel.")
  with concurrent.futures.ThreadPoolExecutor(max_workers=degree_of_parallelism) as executor:
      try:
          commit_strings = list(executor.map(_process_single_commit, commits_to_process))
      except Exception as e:
          logging.error(f"Error during parallel commit processing: {e}")
          st.error(f"An error occurred while processing commit data: {e}")
          return "" # Or handle more gracefully depending on desired UX
  
  logging.info("Finished parallel extraction of commit info.")
  return "".join(commit_strings)

def get_llm_response(input_string: str, system_prompt: str=DEFAULT_PROMPT, max_context_window: int=3000) -> str:
    estimated_tokens = len(input_string) / 4
    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=st.secrets['OPENAI_API_KEY'])

    if estimated_tokens <= max_context_window:
        logging.info("Input fits within context window. Processing as a single call.")
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=input_string)
            ]
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            logging.error(f"Error calling LLM for single chunk: {e}")
            st.error(f"An error occurred while communicating with the LLM: {e}")
            return "Error: Could not get response from LLM."

    logging.info("Input exceeds context window. Splitting into chunks.")
    llm_responses = []
    commit_separator = "\n-------------------------\n"
    # Split the input string into commits, keeping the separator
    # Add the separator back to each part, as split removes it.
    # Filter out empty strings that might result from splitting if input_string starts/ends with separator or has multiple separators together.
    commit_texts_parts = input_string.split(commit_separator)
    individual_commits = []
    for i, part in enumerate(commit_texts_parts):
        if not part.strip(): # Skip empty parts
            continue
        # Add separator back, unless it's the last part and the original string didn't end with a separator
        if i < len(commit_texts_parts) -1 or input_string.endswith(commit_separator):
             individual_commits.append(part + commit_separator)
        else:
            individual_commits.append(part)


    current_chunk = ""
    for commit_text in individual_commits:
        # Estimate token count for current_chunk + commit_text
        potential_chunk = current_chunk + commit_text
        if len(potential_chunk) / 4 > max_context_window:
            # If current_chunk is not empty, process it
            if current_chunk:
                logging.info(f"Processing chunk of size {len(current_chunk)/4} tokens.")
                try:
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=current_chunk)
                    ]
                    response = llm.invoke(messages)
                    llm_responses.append(response.content)
                except Exception as e:
                    logging.error(f"Error calling LLM for a chunk: {e}")
                    llm_responses.append(f"Error processing this chunk: {e}")
            # Start new chunk with the current commit_text
            current_chunk = commit_text
            # If the commit_text itself is too large, it will be processed as is in the next iteration or after loop
        else:
            current_chunk = potential_chunk

    # Process any remaining chunk
    if current_chunk:
        logging.info(f"Processing final chunk of size {len(current_chunk)/4} tokens.")
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=current_chunk)
            ]
            response = llm.invoke(messages)
            llm_responses.append(response.content)
        except Exception as e:
            logging.error(f"Error calling LLM for the final chunk: {e}")
            llm_responses.append(f"Error processing this chunk: {e}")

    logging.info("Finished processing all chunks.")
    return "\n\n---\n\n".join(llm_responses)

def main():


    st.sidebar.title("🎈Who dun it?")
    repo_owner=st.sidebar.text_input(label='acct',value=DEFAULT_REPO_OWNER,key='a')
    repo_name=st.sidebar.text_input(label='repo',value=DEFAULT_REPO_NAME,key='b')
    commit_count=st.sidebar.number_input(label='commits',value=DEFAULT_COMMIT_COUNT,key='c')
    parallelism = st.sidebar.number_input(label='Parallelism', value=4, key='parallelism')
    max_context = st.sidebar.number_input(label='Max LLM Context (Tokens)', value=3000, key='max_context')
    if st.sidebar.button("Run Analysis"):
        st.title(f"Running git analysis for {repo_owner}/{repo_name} ")
        try:
            with st.spinner("Working... Please wait.", show_time=True):
                logging.info(f"Starting analysis for {repo_owner}/{repo_name} with {commit_count} commits, parallelism {parallelism}, max_context {max_context}")
                
                commits = get_commits(repo_owner, repo_name)
                if not commits:
                    logging.warning("No commits returned, aborting analysis.")
                    # st.error is already called in get_commits if there's an issue
                    return # Stop further processing

                stri = extract_commit_info(commits, commit_count, parallelism)
                if not stri and commit_count > 0 : # Check if stri is empty but we expected commits
                    logging.warning("Commit info extraction returned empty, though commits were present.")
                    # extract_commit_info might show an st.error, or we can add one here
                    # st.warning("Could not extract information from commits, though commits were found.")
                    # Depending on desired behavior, we might want to return or continue
                # If stri is empty at this point, get_llm_response will handle it.

                response_text = get_llm_response(stri, max_context_window=max_context)
            
            logging.info("Analysis finished. Displaying response.")
            st.markdown(response_text)

        except Exception as e:
            logging.error(f"An unexpected error occurred in main analysis block: {e}", exc_info=True)
            st.error(f"An unexpected error occurred during the analysis: {e}")


if __name__ == "__main__":
    main()