import streamlit as st
from github import Github

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import json
import markdown
import concurrent.futures
import logging
import time

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
DEFAULT_INITIAL_ANALYSIS_PROMPT = DEFAULT_PROMPT

DEFAULT_FINAL_SUMMARY_PROMPT = """
You are a code summarization expert.
Based on the following individual analyses of commit chunks, please synthesize a single, coherent summary.
Identify key overall themes, common areas of change, and the most impacted experts.
Ensure the final output is a unified report.
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

def compute_daily_stats_table(commits, max_count):
    """Return a markdown table summarizing daily commit stats."""
    from collections import defaultdict

    daily = defaultdict(lambda: {"checkins": 0, "additions": 0, "deletions": 0})
    count = 0
    for commit in commits:
        if count >= max_count:
            break
        date = commit.commit.author.date.date()
        stats = commit.stats
        daily[date]["checkins"] += 1
        daily[date]["additions"] += stats.additions
        daily[date]["deletions"] += stats.deletions
        count += 1

    header = "| Date | Check-ins | Lines Added | Lines Deleted |"
    separator = "|------|-----------|-------------|---------------|"
    lines = [header, separator]

    total_checkins = total_additions = total_deletions = 0
    for day in sorted(daily):
        stats = daily[day]
        lines.append(
            f"| {day} | {stats['checkins']} | {stats['additions']} | {stats['deletions']} |")
        total_checkins += stats["checkins"]
        total_additions += stats["additions"]
        total_deletions += stats["deletions"]

    lines.append(
        f"| **Total** | **{total_checkins}** | **{total_additions}** | **{total_deletions}** |")
    return "\n".join(lines)

def compute_weekly_stats_table(commits, max_count):
    """Return a markdown table summarizing weekly commit stats."""
    from collections import defaultdict
    weekly = defaultdict(lambda: {"checkins": 0, "additions": 0, "deletions": 0})
    count = 0
    for commit in commits:
        if count >= max_count:
            break
        dt = commit.commit.author.date
        year, week, _ = dt.isocalendar()
        key = f"{year}-W{week:02d}"
        stats = commit.stats
        weekly[key]["checkins"] += 1
        weekly[key]["additions"] += stats.additions
        weekly[key]["deletions"] += stats.deletions
        count += 1

    header = "| Week | Check-ins | Lines Added | Lines Deleted |"
    separator = "|------|-----------|-------------|---------------|"
    lines = [header, separator]

    total_checkins = total_additions = total_deletions = 0
    for week in sorted(weekly):
        stats = weekly[week]
        lines.append(
            f"| {week} | {stats['checkins']} | {stats['additions']} | {stats['deletions']} |")
        total_checkins += stats["checkins"]
        total_additions += stats["additions"]
        total_deletions += stats["deletions"]

    lines.append(
        f"| **Total** | **{total_checkins}** | **{total_additions}** | **{total_deletions}** |")
    return "\n".join(lines)

def compute_monthly_stats_table(commits, max_count):
    """Return a markdown table summarizing monthly commit stats."""
    from collections import defaultdict
    monthly = defaultdict(lambda: {"checkins": 0, "additions": 0, "deletions": 0})
    count = 0
    for commit in commits:
        if count >= max_count:
            break
        dt = commit.commit.author.date
        key = dt.strftime("%Y-%m")
        stats = commit.stats
        monthly[key]["checkins"] += 1
        monthly[key]["additions"] += stats.additions
        monthly[key]["deletions"] += stats.deletions
        count += 1

    header = "| Month | Check-ins | Lines Added | Lines Deleted |"
    separator = "|------|-----------|-------------|---------------|"
    lines = [header, separator]

    total_checkins = total_additions = total_deletions = 0
    for month in sorted(monthly):
        stats = monthly[month]
        lines.append(
            f"| {month} | {stats['checkins']} | {stats['additions']} | {stats['deletions']} |")
        total_checkins += stats["checkins"]
        total_additions += stats["additions"]
        total_deletions += stats["deletions"]

    lines.append(
        f"| **Total** | **{total_checkins}** | **{total_additions}** | **{total_deletions}** |")
    return "\n".join(lines)

def _call_llm_for_chunk(chunk_text: str, system_prompt: str) -> str:
    """
    Calls the OpenAI LLM for a single chunk of text.
    """
    logging.info(f"Calling LLM for chunk starting with: '{chunk_text[:50]}...'")
    try:
        llm = ChatOpenAI(model=OPENAI_MODEL, api_key=st.secrets['OPENAI_API_KEY'])
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=chunk_text)
        ]
        response = llm.invoke(messages)
        logging.info(f"LLM call successful for chunk starting with: '{chunk_text[:50]}...'")
        return response.content
    except Exception as e:
        logging.error(f"Error processing chunk with LLM ('{chunk_text[:50]}...'): {e}", exc_info=True)
        # Display error to Streamlit UI as well, as this happens in a thread.
        # Consider if st.error is thread-safe or if errors should be collected and displayed in main thread.
        # For now, returning error message is safer.
        # st.error(f"LLM Error for a text chunk: {e}") 
        return f"Error processing chunk with LLM: {e}"

def analyze_text_chunks_parallel(full_text: str, degree_of_parallelism: int, max_context_window: int, system_prompt: str) -> list[str]:
    """
    Splits text into chunks and analyzes them in parallel using an LLM.
    """
    logging.info(f"Starting parallel analysis. Parallelism: {degree_of_parallelism}, Max Context: {max_context_window}.")
    
    if not full_text:
        logging.warning("analyze_text_chunks_parallel called with empty full_text.")
        return ["Error: No text provided for analysis."]

    # Chunking logic (adapted from get_llm_response)
    individual_chunks = []
    estimated_tokens = len(full_text) / 4

    if estimated_tokens <= max_context_window:
        logging.info("Input fits within context window. Processing as a single chunk.")
        individual_chunks.append(full_text)
    else:
        logging.info("Input exceeds context window. Splitting into chunks.")
        commit_separator = "\n-------------------------\n"
        commit_texts_parts = full_text.split(commit_separator)
        
        current_chunk_text = ""
        for i, part in enumerate(commit_texts_parts):
            # Determine if the part is a complete commit text or just a segment
            commit_text_entry = part
            if i < len(commit_texts_parts) -1 or full_text.endswith(commit_separator):
                 commit_text_entry += commit_separator
            
            if not commit_text_entry.strip(): # Skip effectively empty parts
                continue

            potential_chunk = current_chunk_text + commit_text_entry
            if len(potential_chunk) / 4 > max_context_window:
                if current_chunk_text: # Process the current_chunk_text accumulated so far
                    individual_chunks.append(current_chunk_text)
                current_chunk_text = commit_text_entry # Start new chunk
            else:
                current_chunk_text = potential_chunk
        
        if current_chunk_text: # Add any remaining part as the last chunk
            individual_chunks.append(current_chunk_text)

    if not individual_chunks: # Should not happen if full_text is not empty, but as a safeguard.
        logging.warning("No chunks were created from non-empty text. This is unexpected.")
        return ["Error: Failed to split text into manageable chunks."]

    logging.info(f"Created {len(individual_chunks)} chunks for parallel LLM analysis.")

    results = []
    if individual_chunks: # Ensure there's something to process
        with concurrent.futures.ThreadPoolExecutor(max_workers=degree_of_parallelism) as executor:
            # Submit tasks
            future_to_chunk_summary = {
                executor.submit(_call_llm_for_chunk, chunk, system_prompt): chunk 
                for chunk in individual_chunks
            }
            
            for future in concurrent.futures.as_completed(future_to_chunk_summary):
                try:
                    data = future.result()
                    results.append(data)
                except Exception as exc:
                    chunk_info = future_to_chunk_summary[future][:50] # Get first 50 chars of chunk for context
                    logging.error(f"Chunk (starting with '{chunk_info}...') generated an exception: {exc}", exc_info=True)
                    results.append(f"Error processing chunk '{chunk_info}...': {exc}")
    
    logging.info(f"Finished parallel analysis of {len(individual_chunks)} chunks.")
    return results

def summarize_intermediate_results(analysis_strings: list[str], max_context_window: int, system_prompt: str) -> str:
    """
    Summarizes a list of analysis strings, potentially chunking the combined text if it's too large.
    """
    logging.info(f"Starting final summarization with {len(analysis_strings)} analysis strings.")

    # Filter out error strings
    valid_analyses = [s for s in analysis_strings if not s.startswith("Error processing chunk with LLM:")]
    
    if not valid_analyses:
        logging.warning("No valid analysis strings to summarize after filtering errors.")
        return "Error: No valid analyses to summarize."

    # Combine valid analyses
    # Using a thematic break for clarity, as '\n\n' might already be in the LLM responses.
    combined_text = "\n\n---\n\n".join(valid_analyses)
    logging.info(f"Combined text for summarization has length: {len(combined_text)}")

    estimated_tokens = len(combined_text) / 4

    if estimated_tokens <= max_context_window:
        logging.info("Combined analysis text fits within context window. Making a single LLM call for final summary.")
        summary = _call_llm_for_chunk(combined_text, system_prompt)
        if summary.startswith("Error processing chunk with LLM:"):
            logging.error(f"LLM call failed during final summarization (single call): {summary}")
            return f"Error: LLM call failed during final summarization: {summary}"
        return summary
    else:
        logging.info("Combined analysis text exceeds context window. Chunking for final summarization.")
        summary_chunks_results = []
        # Split by paragraph, but a paragraph could be very long.
        # A more robust strategy might involve splitting by sentences or a fixed number of words/tokens.
        # For now, splitting by '\n\n' which is a common paragraph-like separator in LLM outputs.
        paragraphs = combined_text.split("\n\n")
        
        current_summary_chunk = ""
        for para in paragraphs:
            if not para.strip(): # Skip empty paragraphs
                continue

            # Estimate token count for current_summary_chunk + next paragraph
            potential_chunk = current_summary_chunk + "\n\n" + para if current_summary_chunk else para
            if len(potential_chunk) / 4 > max_context_window:
                if current_summary_chunk: # Process the current_summary_chunk accumulated so far
                    logging.info(f"Processing summary chunk of size {len(current_summary_chunk)/4} tokens.")
                    chunk_summary = _call_llm_for_chunk(current_summary_chunk, system_prompt)
                    if chunk_summary.startswith("Error processing chunk with LLM:"):
                        logging.warning(f"LLM call failed for a summary chunk: {chunk_summary}")
                        # Optionally, append the error or skip this chunk's summary
                        summary_chunks_results.append(f"Note: A sub-summary chunk resulted in an error: {chunk_summary}")
                    else:
                        summary_chunks_results.append(chunk_summary)
                current_summary_chunk = para # Start new chunk with the current paragraph
            else:
                current_summary_chunk = potential_chunk
        
        # Process any remaining chunk
        if current_summary_chunk:
            logging.info(f"Processing final summary chunk of size {len(current_summary_chunk)/4} tokens.")
            chunk_summary = _call_llm_for_chunk(current_summary_chunk, system_prompt)
            if chunk_summary.startswith("Error processing chunk with LLM:"):
                logging.warning(f"LLM call failed for the final summary chunk: {chunk_summary}")
                summary_chunks_results.append(f"Note: The final sub-summary chunk resulted in an error: {chunk_summary}")
            else:
                summary_chunks_results.append(chunk_summary)

        if not summary_chunks_results:
            logging.error("No summary chunks were processed successfully during chunked summarization.")
            return "Error: Failed to produce any summary from chunked analysis text."

        # Combine the summaries of chunks. For now, simple concatenation.
        # A more sophisticated approach might involve another LLM call if this combined text is also too large,
        # but that adds complexity and is not required by the current subtask.
        final_summary = "\n\n---\n\n".join(summary_chunks_results)
        logging.info("Successfully combined summaries from multiple chunks.")
        return final_summary

def process_commits_and_generate_report(
    repo_owner: str, 
    repo_name: str, 
    commit_count: int, 
    degree_of_parallelism: int, 
    max_context_window: int, 
    initial_analysis_prompt: str, 
    final_summary_prompt: str, 
    status_ui_update_callback
) -> str:
    """
    Orchestrates the entire process of fetching commits, extracting information,
    analyzing it, and generating a final summary.
    """
    logging.info(f"Starting report generation for {repo_owner}/{repo_name}.")

    # 1. Get Commits
    status_ui_update_callback(label="Fetching commits from repository...")
    actual_commits = get_commits(repo_owner, repo_name)
    if not actual_commits:
        logging.error("Failed to fetch commits.")
        status_ui_update_callback(label="Failed to fetch commits. Please check repository details and API key.", state="error")
        return "Error: Could not fetch commits. Please check repository details and connectivity."

    # 2. Extract Commit Info
    status_ui_update_callback(label="Extracting commit information (parallel)...")
    commit_text_data = extract_commit_info(actual_commits, commit_count, degree_of_parallelism)
    daily_stats_table = compute_daily_stats_table(actual_commits, commit_count)
    weekly_stats_table = compute_weekly_stats_table(actual_commits, commit_count)
    monthly_stats_table = compute_monthly_stats_table(actual_commits, commit_count)
    if not commit_text_data and commit_count > 0 : # If commit_count is 0, empty text data is expected.
        logging.error("Failed to extract commit information, or no information found.")
        status_ui_update_callback(label="Failed to extract commit information or no data found for the given commits.", state="error")
        return "Error: Could not extract commit information. Ensure commits exist and are accessible."
    elif not commit_text_data and commit_count == 0:
        logging.info("Commit count is 0. No commit information to extract or analyze.")
        status_ui_update_callback(label="Commit count set to 0. No analysis performed.", state="complete")
        return "Commit count is 0. No analysis performed."


    # 3. Analyze Text Chunks Parallel (Actual Implementation)
    status_ui_update_callback(label="Analyzing commit information with LLM (initial pass)...")
    intermediate_analyses = analyze_text_chunks_parallel(
        full_text=commit_text_data, 
        degree_of_parallelism=degree_of_parallelism, 
        max_context_window=max_context_window, 
        system_prompt=initial_analysis_prompt
    )
    # Check if all results are errors or if the list is empty
    if not intermediate_analyses or all(item.startswith("Error:") for item in intermediate_analyses):
        logging.error(f"Failed during initial analysis phase. All chunks resulted in errors or no analysis performed. Result: {intermediate_analyses}")
        status_ui_update_callback(label="Failed during initial LLM analysis. All chunks reported errors.", state="error")
        error_detail = intermediate_analyses[0] if intermediate_analyses else "No analysis results."
        return f"Error: Failed during initial analysis step. Details: {error_detail}"
    
    # Log if some chunks failed but not all
    if any(item.startswith("Error:") for item in intermediate_analyses):
        logging.warning(f"Some chunks failed during initial analysis. Results: {intermediate_analyses}")
        # Decide if to proceed or fail; for now, we proceed with successful chunks for summarization

    # 4. Summarize Intermediate Results (Actual Implementation)
    status_ui_update_callback(label="Generating final summary with LLM...")
    final_report_text = summarize_intermediate_results(
        analysis_strings=intermediate_analyses, 
        max_context_window=max_context_window, 
        system_prompt=final_summary_prompt
    )
    if not final_report_text or final_report_text.startswith("Error:"):
        logging.error(f"Failed during final summarization phase: {final_report_text}")
        status_ui_update_callback(label="Failed during final summary generation.", state="error")
        # final_report_text already contains the error message
        return final_report_text 
    
    logging.info("Successfully generated report.")
    combined_stats = "\n\n".join([daily_stats_table, weekly_stats_table, monthly_stats_table])
    return combined_stats + "\n\n" + final_report_text


def main():


    st.sidebar.title("🎈Who dun it?")
    repo_owner=st.sidebar.text_input(label='acct',value=DEFAULT_REPO_OWNER,key='a')
    repo_name=st.sidebar.text_input(label='repo',value=DEFAULT_REPO_NAME,key='b')
    commit_count=st.sidebar.number_input(label='commits',value=DEFAULT_COMMIT_COUNT,key='c')
    parallelism = st.sidebar.number_input(label='Parallelism', value=4, key='parallelism')
    max_context = st.sidebar.number_input(label='Max LLM Context (Tokens)', value=3000, key='max_context')
    if st.sidebar.button("Run Analysis"):
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

        with st.spinner("Running analysis..."):
            try:
                logging.info(f"Main: Kicking off analysis for {repo_owner}/{repo_name}.")

                final_report = process_commits_and_generate_report(
                    repo_owner=repo_owner,
                    repo_name=repo_name,
                    commit_count=commit_count,
                    degree_of_parallelism=parallelism,
                    max_context_window=max_context,
                    initial_analysis_prompt=DEFAULT_INITIAL_ANALYSIS_PROMPT,
                    final_summary_prompt=DEFAULT_FINAL_SUMMARY_PROMPT,
                    status_ui_update_callback=update_status
                )

                if "Error:" in final_report or not final_report:
                    update_status("Analysis encountered an error.", state="error")
                elif "No analysis performed." in final_report:
                    update_status("Analysis complete.", state="complete")
                else:
                    update_status("Analysis complete!", state="complete")

                logging.info("Main: Analysis process finished. Displaying report.")
                st.markdown(final_report)

            except Exception as e:
                logging.error(f"Main: An unexpected error occurred: {e}", exc_info=True)
                update_status(f"A critical unexpected error occurred: {e}", state="error")
                st.error(f"A critical unexpected error occurred: {e}")

        total_time = time.time() - start_time
        status_placeholder.success(f"Total time taken: {total_time:.2f}s")


if __name__ == "__main__":
    main()
