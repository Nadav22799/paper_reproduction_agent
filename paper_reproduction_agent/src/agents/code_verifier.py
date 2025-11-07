"""Code Verifier Agent - Tests and verifies implementation results."""

from typing import TypedDict
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from ..tools.code_execution_tools import (
    search_file,
    read_file,
    list_directory,
    execute_shell_command,
    execute_python_script,
    install_dependencies,
)
from ..utils.llm_factory import create_llm
from ..utils.message_utils import normalize_message_content


class CodeVerifierState(TypedDict):
    """State for Code Verifier Agent."""
    code_path: str
    paper_results: dict
    execution_results: dict
    metrics_match: bool
    verification_report: str


class CodeVerifierAgent:
    """Agent responsible for verifying code implementations."""

    def __init__(self, llm=None):
        """Initialize the Code Verifier Agent."""
        self.llm = llm or create_llm(temperature=0.1)

        self.system_prompt = """You verify research code implementations.

Process:
1. Read README to understand setup
2. Check directory structure
3. Run installation/demo commands
4. Compare results with paper

Use tools systematically. Document findings."""

        # Use essential tools (enough for Groq but not too many for small models)
        essential_tools = [
            search_file,
            read_file,
            list_directory,
            execute_shell_command,
            execute_python_script,
            install_dependencies,
        ]

        self.agent = create_react_agent(
            self.llm,
            tools=essential_tools,
            prompt=self.system_prompt
        )

    def verify_implementation(self, code_path: str, paper_results: dict) -> dict:
        """
        Verify an implementation against paper results.

        Args:
            code_path: Path to code directory
            paper_results: Expected results from paper

        Returns:
            Verification results
        """
        import os

        # Check if path exists first
        if not os.path.exists(code_path):
            return {
                "setup_successful": False,
                "execution_successful": False,
                "results_match_paper": False,
                "discrepancies": ["Code path does not exist"],
                "report": f"Code path {code_path} not found.",
            }

        # Clean paper results - remove any <think> tags or XML that could confuse the model
        import re
        clean_results = "No specific metrics provided"
        if paper_results and isinstance(paper_results, dict) and 'summary' in paper_results:
            summary = paper_results['summary']
            # Remove <think>...</think> blocks
            summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL)
            # Remove <tool_call>...</tool_call> blocks if any
            summary = re.sub(r'<tool_call>.*?</tool_call>', '', summary, flags=re.DOTALL)
            # Clean up whitespace
            summary = re.sub(r'\n\s*\n', '\n', summary).strip()
            clean_results = summary if summary else "No specific metrics provided"

        # Simplified task - avoid confusing the model with example function calls
        task = f"""Verify this code repository: {code_path}

Expected paper results:
{clean_results}

Your task:
1. Read README.md to understand setup and usage
2. Install any dependencies you find
3. Download or prepare any datasets mentioned
4. Run the code (prefer demos or quick tests over full training)
5. Capture the execution output and any metrics
6. Compare results with expected paper results if provided

Work step by step. Use your available tools to explore files, install packages, and run scripts.

When done, provide a final report with what you did and what results you found."""

        messages = [HumanMessage(content=task)]
        result = self.agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 50}
        )

        return self._parse_verification_result(result, paper_results)

    def run_experiments(self, script_path: str, experiments: list) -> dict:
        """
        Run a series of experiments.

        Args:
            script_path: Path to script
            experiments: List of experiment configurations

        Returns:
            Experiment results
        """
        task = f"""Run these experiments using {script_path}:

{experiments}

For each experiment:
1. Set up configuration
2. Run the script
3. Capture metrics and outputs
4. Record execution time and resource usage

Compile all results into a summary."""

        messages = [HumanMessage(content=task)]
        result = self.agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 50}
        )

        return {
            "experiments": experiments,
            "results": normalize_message_content(result.get("messages", [])[-1].content) if result.get("messages") else "",
        }

    def compare_with_baseline(self, implementation_results: dict, baseline_results: dict) -> dict:
        """
        Compare implementation results with baseline/paper results.

        Args:
            implementation_results: Results from running implementation
            baseline_results: Expected results from paper

        Returns:
            Comparison report
        """
        task = f"""Compare these results:

Implementation results:
{implementation_results}

Baseline (paper) results:
{baseline_results}

Analysis:
1. Calculate differences for each metric
2. Assess if differences are within acceptable range
3. Identify metrics that match vs. don't match
4. Suggest possible reasons for discrepancies
5. Overall assessment: Does implementation reproduce paper results?

Provide detailed comparison report."""

        messages = [HumanMessage(content=task)]
        result = self.agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 50}
        )

        return {
            "comparison": normalize_message_content(result.get("messages", [])[-1].content) if result.get("messages") else "",
            "matches_paper": self._assess_match(result),
        }

    def validate_environment(self, requirements_path: str) -> dict:
        """
        Validate execution environment setup.

        Args:
            requirements_path: Path to requirements file

        Returns:
            Environment validation results
        """
        task = f"""Validate the environment setup:

1. Check Python version
2. Review requirements from {requirements_path}
3. Install dependencies
4. Verify installations
5. Check for potential conflicts

Report on environment readiness."""

        messages = [HumanMessage(content=task)]
        result = self.agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 50}
        )

        return {
            "environment": normalize_message_content(result.get("messages", [])[-1].content) if result.get("messages") else "",
        }

    def create_test_suite(self, implementation_path: str, paper_details: dict) -> str:
        """
        Create a test suite for the implementation.

        Args:
            implementation_path: Path to implementation
            paper_details: Details from paper for test cases

        Returns:
            Path to test file
        """
        task = f"""Create a pytest test suite for the implementation at {implementation_path}

Based on paper details:
{paper_details}

Tests should include:
1. Unit tests for core functions
2. Integration tests
3. Tests that verify against paper results
4. Edge case tests

Save to {implementation_path}/test_implementation.py"""

        messages = [HumanMessage(content=task)]
        result = self.agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 50}
        )

        return f"{implementation_path}/test_implementation.py"

    def _parse_verification_result(self, result: dict, paper_results: dict) -> dict:
        """Parse verification results."""
        messages = result.get("messages", [])

        verification = {
            "setup_successful": False,
            "execution_successful": False,
            "results_match_paper": False,
            "discrepancies": [],
            "report": "",
        }

        # Check if agent used any execution tools
        has_execution = False
        for msg in messages:
            if hasattr(msg, 'content') and msg.content:
                content_str = normalize_message_content(msg.content)
                content_lower = content_str.lower()
                # Check if execution tools were called
                if any(keyword in content_lower for keyword in ['execute_python_script', 'run_pytest', 'stdout', 'stderr', 'returncode']):
                    has_execution = True
                    verification["setup_successful"] = True

        # Get final output
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content:
                content_str = normalize_message_content(msg.content)
                verification["report"] = content_str
                content_lower = content_str.lower()

                # Check for successful execution
                if has_execution and "success" in content_lower:
                    verification["execution_successful"] = True

                # Check for results comparison
                if any(keyword in content_lower for keyword in ['match', 'same', 'reproduce', 'similar', 'comparable']):
                    verification["results_match_paper"] = True

                # Look for discrepancies
                if any(keyword in content_lower for keyword in ['differ', 'mismatch', 'discrepancy', 'not match', 'lower', 'higher']):
                    # Extract discrepancy info if mentioned
                    if "discrepancy" in content_lower or "differ" in content_lower:
                        verification["discrepancies"].append("Results differ from paper - see report for details")
                        verification["results_match_paper"] = False

                break

        return verification

    def _assess_match(self, result: dict) -> bool:
        """Assess if results match paper."""
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, 'content'):
                content = normalize_message_content(msg.content).lower()
                if "match" in content or "reproduce" in content:
                    return True
        return False
