import streamlit as st
from github import Github

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import json
import markdown

OPENAI_MODEL="gpt-4.1-mini"
DEFAULT_REPO_OWNER="deepchem"
DEFAULT_REPO_NAME="deepchem"
DEFAULT_COMMIT_COUNT=20

DEFAULT_PROMPT="""
You are a code analysis expoert.

Based on these check-ins, please identify the key code areas, as well as the best two experts on each.
"""

def get_commits(repo_owner=DEFAULT_REPO_OWNER, repo_name=DEFAULT_REPO_NAME):
  access_token = st.secrets['GITHUB_API_KEY']

  g = Github(access_token)
  repo = g.get_repo(f"{repo_owner}/{repo_name}")

  commits = repo.get_commits()
  return commits

def extract_commit_info(commits, max_count):
  commits_processed=0
  stri=""
  for commit in commits:
      stri += f"\n\nCommit: {commit.sha}" + \
      f"\n  Comment: {commit.commit.message}" + \
      f"\n  Author: {commit.commit.author.name}" + \
      f"\n  Date: {commit.commit.author.date}"

      files = commit.files
      stri += "\n  Files:"
      for file in files:
          stri += f"\n    - {file.filename}"
      stri += "\n-------------------------\n"
      commits_processed+=1
      #print(f"{commits_processed=}")
      if(commits_processed>max_count):
        break
  return stri


def get_llm_response(input_string: str, system_prompt: str=DEFAULT_PROMPT) -> str:
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=input_string)
    ]

    #print(f"Messages: {messages}\n\n")
    llm = ChatOpenAI(model=OPENAI_MODEL, api_key=st.secrets['OPENAI_API_KEY'])
    response = llm.invoke(messages)
    #print(f"Chat Response: {response}\n\n")
    resp = response.content
    #print(f"For input={input_string}, extracted response={resp}\n***********\n")
    return resp

def main():


    st.sidebar.title("🎈Who dun it?")
    repo_owner=st.sidebar.text_input(label='acct',value=DEFAULT_REPO_OWNER,key='a')
    repo_name=st.sidebar.text_input(label='repo',value=DEFAULT_REPO_NAME,key='b')
    commit_count=st.sidebar.number_input(label='commits',value=DEFAULT_COMMIT_COUNT,key='c')
    if st.sidebar.button("Run Analysis"):
        st.title(f"Running git analysis for {repo_owner}/{repo_name} ")
        with st.spinner("Working", show_time=True):
            commits=get_commits(repo_owner, repo_name)
            stri=extract_commit_info(commits, commit_count)
            response_text = get_llm_response(stri)
        st.markdown(response_text)


if __name__ == "__main__":
    main()