import concurrent.futures
import logging
from typing import List

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

OPENAI_MODEL = "gpt-4.1-mini"


def _call_llm_for_chunk(chunk_text: str, system_prompt: str) -> str:
    """Call the OpenAI LLM for a single chunk of text."""
    logging.info(f"Calling LLM for chunk starting with: '{chunk_text[:50]}...'")
    try:
        llm = ChatOpenAI(model=OPENAI_MODEL, api_key=st.secrets['OPENAI_API_KEY'])
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=chunk_text)]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logging.error(
            f"Error processing chunk with LLM ('{chunk_text[:50]}...'): {e}",
            exc_info=True,
        )
        return f"Error processing chunk with LLM: {e}"


def split_text_into_chunks(full_text: str, max_context_window: int) -> List[str]:
    if not full_text:
        return []

    estimated_tokens = len(full_text) / 4
    if estimated_tokens <= max_context_window:
        return [full_text]

    commit_separator = "\n-------------------------\n"
    parts = full_text.split(commit_separator)
    chunks = []
    current = ""
    for i, part in enumerate(parts):
        entry = part
        if i < len(parts) - 1 or full_text.endswith(commit_separator):
            entry += commit_separator
        if not entry.strip():
            continue
        potential = current + entry
        if len(potential) / 4 > max_context_window:
            if current:
                chunks.append(current)
            current = entry
        else:
            current = potential
    if current:
        chunks.append(current)
    return chunks


def parallel_llm_analysis(chunks: List[str], degree_of_parallelism: int, system_prompt: str) -> List[str]:
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=degree_of_parallelism) as ex:
        future_map = {ex.submit(_call_llm_for_chunk, c, system_prompt): c for c in chunks}
        for future in concurrent.futures.as_completed(future_map):
            try:
                results.append(future.result())
            except Exception as exc:
                chunk_info = future_map[future][:50]
                logging.error(
                    f"Chunk (starting with '{chunk_info}...') generated an exception: {exc}",
                    exc_info=True,
                )
                results.append(f"Error processing chunk '{chunk_info}...': {exc}")
    return results


def analyze_text_chunks_parallel(
    full_text: str,
    degree_of_parallelism: int,
    max_context_window: int,
    system_prompt: str,
) -> List[str]:
    logging.info(
        f"Starting parallel analysis. Parallelism: {degree_of_parallelism}, Max Context: {max_context_window}."
    )
    chunks = split_text_into_chunks(full_text, max_context_window)
    if not chunks:
        logging.warning("No chunks were created from the provided text.")
        return ["Error: No text provided for analysis."]
    logging.info(f"Created {len(chunks)} chunks for parallel LLM analysis.")
    return parallel_llm_analysis(chunks, degree_of_parallelism, system_prompt)


def chunk_text_by_paragraph(text: str, max_context_window: int) -> List[str]:
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        potential = current + ("\n\n" + para if current else para)
        if len(potential) / 4 > max_context_window:
            if current:
                chunks.append(current)
            current = para
        else:
            current = potential
    if current:
        chunks.append(current)
    return chunks


def summarize_chunks(chunks: List[str], system_prompt: str) -> List[str]:
    summaries = []
    for chunk in chunks:
        summary = _call_llm_for_chunk(chunk, system_prompt)
        summaries.append(summary)
    return summaries


def summarize_intermediate_results(
    analysis_strings: List[str],
    max_context_window: int,
    system_prompt: str,
) -> str:
    logging.info(
        f"Starting final summarization with {len(analysis_strings)} analysis strings."
    )
    valid = [s for s in analysis_strings if not s.startswith("Error processing chunk with LLM:")]
    if not valid:
        return "Error: No valid analyses to summarize."

    combined = "\n\n---\n\n".join(valid)
    if len(combined) / 4 <= max_context_window:
        summary = _call_llm_for_chunk(combined, system_prompt)
        return summary

    summary_chunks = chunk_text_by_paragraph(combined, max_context_window)
    results = summarize_chunks(summary_chunks, system_prompt)
    if not results:
        return "Error: Failed to produce any summary."
    return "\n\n---\n\n".join(results)
