"""Code Searcher Agent - Finds existing implementations of papers."""

from typing import TypedDict
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from ..tools.code_search_tools import code_search_tools
from ..utils.llm_factory import create_llm
from ..utils.message_utils import normalize_message_content


class CodeSearcherState(TypedDict):
    """State for Code Searcher Agent."""
    paper_title: str
    paper_keywords: list
    github_repos: list
    papers_with_code_results: dict
    selected_repo: dict
    repo_cloned: bool


class CodeSearcherAgent:
    """Agent responsible for finding existing code implementations."""

    def __init__(self, llm=None):
        """Initialize the Code Searcher Agent."""
        self.llm = llm or create_llm(temperature=0.1)

        self.system_prompt = """You are an expert at finding code implementations for research papers.

Your responsibilities:
1. Search GitHub for repositories matching the paper
2. Search Papers with Code database
3. Evaluate repository quality (stars, documentation, recent updates)
4. Identify official vs. third-party implementations
5. Find the most promising implementation to use

When searching:
- Use multiple search strategies (paper title, algorithm name, author names)
- Prioritize official implementations
- Check for repositories with good documentation
- Look for repos with active maintenance
- Consider implementation quality indicators (tests, examples, README)

Use the available tools to search efficiently and find the best implementation."""

        self.agent = create_react_agent(
            self.llm,
            tools=code_search_tools,
            prompt=self.system_prompt
        )

    def search_implementations(self, paper_title: str, paper_keywords: list = None) -> dict:
        """
        Search for code implementations of a paper.

        Args:
            paper_title: Title of the paper
            paper_keywords: Additional keywords for search

        Returns:
            Dictionary with search results
        """
        keywords_str = ", ".join(paper_keywords) if paper_keywords else ""

        task = f"""Find code implementations for this paper:
Title: {paper_title}
Keywords: {keywords_str}

Search strategy:
1. Search Papers with Code database
2. Search GitHub repositories
3. Search GitHub code
4. Rank results by quality (stars, official status, documentation)
5. Recommend the top 3 implementations

Provide a comprehensive summary of findings."""

        messages = [HumanMessage(content=task)]
        result = self.agent.invoke({"messages": messages})

        return self._parse_search_result(result)

    def evaluate_repository(self, repo_full_name: str) -> dict:
        """
        Evaluate a specific repository's quality and usability.

        Args:
            repo_full_name: Repository full name (owner/repo)

        Returns:
            Repository evaluation
        """
        task = f"""Evaluate the repository: {repo_full_name}

Check:
1. README quality and completeness
2. Code structure and organization
3. Presence of requirements/dependencies file
4. Examples or demo scripts
5. Tests
6. Documentation
7. Recent activity

Provide a detailed evaluation and usability score."""

        messages = [HumanMessage(content=task)]
        result = self.agent.invoke({"messages": messages})

        return {
            "repo": repo_full_name,
            "evaluation": normalize_message_content(result.get("messages", [])[-1].content) if result.get("messages") else "",
        }

    def find_main_implementation_files(self, repo_full_name: str) -> list:
        """
        Find the main implementation files in a repository.

        Args:
            repo_full_name: Repository full name

        Returns:
            List of important files
        """
        task = f"""Explore the repository {repo_full_name} and identify:

1. Main model/algorithm implementation files
2. Training scripts
3. Evaluation/testing scripts
4. Configuration files
5. Requirements/dependencies

List the most important files for understanding and running the implementation."""

        messages = [HumanMessage(content=task)]
        result = self.agent.invoke({"messages": messages})

        return normalize_message_content(result.get("messages", [])[-1].content) if result.get("messages") else ""

    def _parse_search_result(self, result: dict) -> dict:
        """Parse search results into structured format."""
        messages = result.get("messages", [])

        search_results = {
            "repositories": [],
            "papers_with_code": {},
            "recommendations": [],
            "raw_output": "",
        }

        # Get final output
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content:
                search_results["raw_output"] = normalize_message_content(msg.content)
                break

        return search_results
