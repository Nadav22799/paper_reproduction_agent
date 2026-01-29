"""Environment Setup Agent - Specialized agent for preparing ML environments.

This agent's sole responsibility is to:
1. Analyze environment files (conda, pip, uv)
2. Pin missing package versions based on paper publication date
3. Create and verify the environment
4. Ensure no dependency conflicts

The unified reproduction agent can then run experiments in a clean, conflict-free environment.
"""

from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from ..tools.code_execution_tools import (
    read_file,
    list_directory,
    execute_shell_command,
    execute_python_code,
    check_python_compatibility,
)
from ..tools.file_utils import (
    grep_in_directory,
    find_files,
)
from langchain_community.tools import DuckDuckGoSearchRun
from ..utils.llm_factory import create_llm
from ..utils.logging_callback import LoggingCallbackHandler


class EnvironmentSetupAgent:
    """Specialized agent for environment preparation and setup."""

    def __init__(self, llm=None, max_iterations=50, metrics_tracker=None):
        self.llm = llm or create_llm(temperature=0.1)
        self.max_iterations = max_iterations
        self.metrics_tracker = metrics_tracker

        from ..config.prompts import ENVIRONMENT_AGENT_PROMPT
        self.system_prompt = ENVIRONMENT_AGENT_PROMPT

        # Tools for environment setup
        tools = [
            # Core file operations
            read_file,
            list_directory,
            execute_shell_command,
            execute_python_code,
            check_python_compatibility,
            # Common utilities (hard to replicate with bash)
            grep_in_directory,
            find_files,
            # Web search for learning how to install tools
            DuckDuckGoSearchRun(),
        ]

        # Create ReAct agent
        self.agent = create_react_agent(self.llm, tools=tools)

        print("\n" + "=" * 60)
        print("🔧 Environment Setup Agent Initialized")
        print(f"   Max Iterations: {max_iterations}")
        print("=" * 60)

    def setup_environment(
        self,
        repo_path: str,
        readme_content: str,
        paper_date: Optional[str] = None,
        paper_title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Prepare environment for running experiments.

        Args:
            repo_path: Path to cloned repository
            readme_content: Content of README for installation instructions
            paper_date: Paper publication date (YYYY-MM format)
            paper_title: Paper title (for context)

        Returns:
            Dictionary with:
                - success: bool
                - env_type: "conda" | "pip" | "uv" | "poetry"
                - env_name: Name of conda environment or path to venv
                - python_path: Absolute path to Python executable
                - packages_pinned: List of packages that were pinned
                - warnings: List of warnings or issues
                - error: Error message if failed
        """
        print("\n" + "=" * 60)
        print("🚀 Starting Environment Setup")
        print("=" * 60)
        print(f"📁 Repository: {repo_path}")
        print(f"📅 Paper Date: {paper_date or 'Unknown'}")
        print(f"📄 Paper: {paper_title or 'N/A'}")
        print()

        # Build task prompt
        task_parts = [
            "Your task: Prepare the environment for running ML experiments.",
            "",
            f"Repository path: {repo_path}",
        ]

        if paper_date:
            task_parts.append(f"Paper publication date: {paper_date}")
            task_parts.append("Use this date to determine compatible package versions.")
        else:
            task_parts.append("Paper date unknown - use common compatible versions.")

        task_parts.extend(
            [
                "",
                "README excerpt (installation instructions):",
                "---",
                readme_content[:2000] if len(readme_content) > 2000 else readme_content,
                "---",
                "",
                "Follow the WORKFLOW phases:",
                "1. Analyze environment files",
                "2. Check for unpinned packages",
                "3. Determine compatible versions",
                "4. Edit files to pin versions",
                "5. Create environment",
                "6. Verify installation",
                "7. Report results",
                "",
                "Start by listing the repository directory to find environment files.",
            ]
        )

        task_prompt = "\n".join(task_parts)

        # Prepare messages

        # Run agent
        try:
            result = self.agent.invoke(
                {
                    "messages": [
                        HumanMessage(content=self.system_prompt + "\n\n" + task_prompt)
                    ]
                },
                config={
                    "recursion_limit": self.max_iterations * 3,
                    "callbacks": [
                        LoggingCallbackHandler(metrics_tracker=self.metrics_tracker)
                    ],
                },
            )

            # Parse results from agent conversation
            result_summary = self._parse_agent_results(result)

            print("\n" + "=" * 60)
            print("✅ Environment Setup Complete")
            print("=" * 60)
            print(f"Status: {result_summary['status']}")
            if result_summary.get("env_name"):
                print(f"Environment: {result_summary['env_name']}")
            if result_summary.get("python_path"):
                print(f"Python: {result_summary['python_path']}")
            print()

            return result_summary

        except Exception as e:
            print(f"\n❌ Environment Setup Failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "status": "failed",
                "last_message": f"Exception: {str(e)}",  # Error as reasoning
            }

    def _parse_agent_results(self, agent_result: Dict) -> Dict[str, Any]:
        """
        Parse agent execution results to extract environment info.

        Args:
            agent_result: Result from agent.invoke()

        Returns:
            Structured summary of setup results
        """
        messages = agent_result.get("messages", [])

        # Extract last_message for reasoning output
        last_message = ""
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                last_message = last_msg.content

        # Default result
        result = {
            "success": False,
            "status": "unknown",
            "env_type": None,
            "env_name": None,
            "python_path": None,
            "packages_pinned": [],
            "warnings": [],
            "error": None,
            "last_message": last_message,  # Agent reasoning for verbose output
        }

        # Search for key indicators in agent messages
        full_conversation = "\n".join(
            [str(m.content) if hasattr(m, "content") else str(m) for m in messages]
        )
        full_conversation_lower = full_conversation.lower()

        # Check for incomplete execution (agent hit iteration limit)
        incomplete_indicators = [
            "sorry",
            "need more steps",
            "cannot complete",
            "unable to finish",
            "i apologize",
            "unfortunately",
            "more iterations",
        ]
        is_incomplete = any(ind in full_conversation_lower for ind in incomplete_indicators)

        # Check for SMOKE TEST success (Critical - env is only ready if smoke test passed!)
        smoke_test_markers = [
            "smoke test passed",
            "smoke test successful",
            "script ran successfully",
            "experiment started successfully",
            "training started",
            "started training",
            "epoch 0",
            "epoch 1",
            "step 0",
            "step 1",
            "training loop",
        ]
        smoke_test_passed = any(marker in full_conversation_lower for marker in smoke_test_markers)

        # Check for basic success indicators (environment created, imports work)
        basic_success_markers = [
            "environment created successfully",
            "environment setup complete",
            "successfully prepared",
            "imports ok",
            "verification successful",
            "✅ imports ok",
        ]
        has_basic_success = any(marker in full_conversation_lower for marker in basic_success_markers)

        # Check for explicit failure indicators
        has_explicit_failure = (
            "modulenotfounderror" in full_conversation_lower
            or "importerror" in full_conversation_lower
            or "environment creation failed" in full_conversation_lower
            or "setup failed" in full_conversation_lower
            or "no such file" in full_conversation_lower
            or "filenotfounderror" in full_conversation_lower
        )

        # Determine success status based on all conditions
        # Determine success status based on all conditions
        if smoke_test_passed:
            # Full success: smoke test passed implies environment is working
            result["success"] = True
            result["status"] = "success"
        elif is_incomplete:
            result["success"] = False
            result["status"] = "incomplete"
            result["error"] = "Agent hit iteration limit without completing environment setup"
        elif has_explicit_failure:
            result["status"] = "failed"
            # Try to extract error message
            for msg in reversed(messages):
                content = str(msg.content) if hasattr(msg, "content") else str(msg)
                if "error" in content.lower():
                    result["error"] = content[:500]
                    break
        elif has_basic_success:
            # Partial success: basic setup done but no smoke test confirmation
            result["success"] = False
            result["status"] = "partial"
            result["error"] = "Basic environment setup done but smoke test not completed. Environment may not work for actual experiments."
        elif (
            "failed" in full_conversation_lower
            or "error" in full_conversation_lower
        ):
            result["status"] = "failed"
            # Try to extract error message
            for msg in reversed(messages):
                content = str(msg.content) if hasattr(msg, "content") else str(msg)
                if "error" in content.lower():
                    result["error"] = content[:500]
                    break

        # Try to extract environment name
        import re

        # Look for conda environment name
        conda_match = re.search(r"conda.*?-n\s+(\w+)", full_conversation)
        if conda_match:
            result["env_name"] = conda_match.group(1)
            result["env_type"] = "conda"

        # Look for venv path
        venv_match = re.search(r"(\.?/?\w*/venv)", full_conversation)
        if venv_match and not result["env_name"]:
            result["env_name"] = venv_match.group(1)
            result["env_type"] = "venv"

        # Try to extract python path
        python_match = re.search(r"(/[\w/]+/python\d?\.\d?)", full_conversation)
        if python_match:
            result["python_path"] = python_match.group(1)

        return result
