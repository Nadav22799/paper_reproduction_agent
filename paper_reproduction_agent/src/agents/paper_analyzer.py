"""Paper Analyzer Agent - Extracts and analyzes information from academic papers."""

from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from ..tools.paper_tools import paper_analysis_tools
from ..utils.llm_factory import create_llm
from ..utils.message_utils import normalize_message_content


class PaperAnalyzerState(TypedDict):
    """State for Paper Analyzer Agent."""
    paper_id: str
    paper_metadata: dict
    full_text: str
    algorithms: list
    experimental_setup: dict
    results: list
    code_references: list
    analysis_complete: bool


class PaperAnalyzerAgent:
    """Agent responsible for analyzing academic papers."""

    def __init__(self, llm=None):
        """Initialize the Paper Analyzer Agent."""
        self.llm = llm or create_llm(temperature=0.1)

        # System prompt for paper analysis
        self.system_prompt = """You are an expert research assistant specialized in analyzing academic papers.

Your responsibilities:
1. Extract paper metadata (title, authors, abstract, publication date)
2. Identify and extract algorithm descriptions and pseudocode
3. Extract experimental setup details (datasets, metrics, hyperparameters)
4. Extract reported results and performance numbers
5. Find any code repository references or links

When analyzing a paper:
- Be thorough and systematic
- Extract specific numerical results when available
- Identify key contributions and novel aspects
- Note any implementation details mentioned
- Look for reproducibility information

Use the available tools to extract information from papers efficiently."""

        # Create the agent with tools
        self.agent = create_react_agent(
            self.llm,
            tools=paper_analysis_tools,
            prompt=self.system_prompt
        )

    def analyze_paper(self, paper_input: str) -> dict:
        """
        Analyze a paper and extract all relevant information.

        Args:
            paper_input: arXiv ID, PDF path, or paper text

        Returns:
            Dictionary with extracted information
        """
        # Determine input type
        if paper_input.startswith("arxiv:") or len(paper_input.split()) == 1:
            arxiv_id = paper_input.replace("arxiv:", "")
            task = f"Fetch and analyze the arXiv paper {arxiv_id}. Extract all algorithms, experimental setup, results, and code references."
        elif paper_input.endswith(".pdf"):
            task = f"Analyze the PDF file at {paper_input}. Extract all algorithms, experimental setup, results, and code references."
        else:
            task = "Analyze the provided paper text. Extract all algorithms, experimental setup, results, and code references."

        messages = [HumanMessage(content=task)]
        result = self.agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 50}
        )

        return self._parse_analysis_result(result)

    def _parse_analysis_result(self, result: dict) -> dict:
        """Parse the agent's analysis result into structured format."""
        # Extract information from agent messages
        messages = result.get("messages", [])

        # Initialize response structure
        analysis = {
            "paper_metadata": {},
            "algorithms": [],
            "experimental_setup": {},
            "results": [],
            "code_references": [],
            "raw_output": "",
        }

        # Get the final AI message
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content:
                analysis["raw_output"] = normalize_message_content(msg.content)
                break

        return analysis

    def extract_key_contributions(self, paper_text: str) -> list:
        """
        Extract key contributions from paper.

        Args:
            paper_text: Full text of the paper

        Returns:
            List of key contributions
        """
        task = f"""Analyze this paper text and extract the key contributions:

{paper_text[:5000]}

List the main contributions in a structured format."""

        messages = [HumanMessage(content=task)]
        result = self.agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 50}
        )

        return normalize_message_content(result.get("messages", [])[-1].content) if result.get("messages") else ""

    def identify_reproducibility_info(self, paper_text: str) -> dict:
        """
        Identify information relevant to reproducing the paper's results.

        Args:
            paper_text: Full text of the paper

        Returns:
            Dictionary with reproducibility information
        """
        task = f"""Analyze this paper and identify all information needed to reproduce the results:

{paper_text[:5000]}

Focus on:
- Implementation details
- Hyperparameters and their values
- Training procedures
- Dataset preprocessing steps
- Evaluation protocols
- Any code or supplementary materials mentioned

Provide a structured summary."""

        messages = [HumanMessage(content=task)]
        result = self.agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 50}
        )

        return {
            "reproducibility_info": normalize_message_content(result.get("messages", [])[-1].content) if result.get("messages") else "",
        }
