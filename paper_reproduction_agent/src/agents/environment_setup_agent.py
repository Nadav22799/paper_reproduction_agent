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
from ..utils.hierarchical_context import HierarchicalContextManager


class EnvironmentSetupAgent:
    """Specialized agent for environment preparation and setup."""

    def __init__(
        self,
        llm=None,
        max_iterations=50,
        metrics_tracker=None,
        callbacks=None,
        hierarchical_context: HierarchicalContextManager = None,
    ):
        self.llm = llm or create_llm(temperature=0.1)
        self.max_iterations = max_iterations
        self.metrics_tracker = metrics_tracker
        self.callbacks = callbacks or []
        self.hierarchical_context = hierarchical_context

        from ..config.prompts import ENVIRONMENT_AGENT_PROMPT
        self.system_prompt = ENVIRONMENT_AGENT_PROMPT

        # Tools for environment setup
        self.tools = [
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
        self.agent = create_react_agent(self.llm, tools=self.tools)

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

        # RETRIEVE relevant context from previous agents (especially for recovery)
        # IMPORTANT: Exclude own source to prevent self-referencing (seeing own "Smoke Test: PASSED")
        previous_context = ""
        if self.hierarchical_context:
            previous_context = self.hierarchical_context.compile_context(
                query="environment setup python conda pip micromamba error",
                max_tokens=1500,
                exclude_sources=["environment_setup"],
            )
            if previous_context:
                print(f"   📋 Retrieved {len(previous_context)} chars of previous context")

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

        # Add context from previous attempts if available (helps with recovery)
        if previous_context:
            task_parts.extend(
                [
                    "",
                    "=== CONTEXT FROM PREVIOUS ATTEMPTS ===",
                    previous_context,
                    "=======================================",
                ]
            )

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
            # Use provided callbacks (orchestrator already includes LoggingCallbackHandler with
            # verbose logging AND file logging). Only create our own as fallback if none provided.
            if self.callbacks:
                invoke_callbacks = list(self.callbacks)
            else:
                # Fallback: create callback if none provided (e.g., standalone usage)
                invoke_callbacks = [LoggingCallbackHandler(metrics_tracker=self.metrics_tracker)]

            result = self.agent.invoke(
                {
                    "messages": [
                        HumanMessage(content=self.system_prompt + "\n\n" + task_prompt)
                    ]
                },
                config={
                    "recursion_limit": self.max_iterations,
                    "callbacks": invoke_callbacks,
                },
            )

            # Parse results from agent conversation
            result_summary = self._parse_agent_results(result)

            # Store results in hierarchical context for next agent
            if self.hierarchical_context:
                from ..utils.context_utils import build_context_entry

                messages = result.get("messages", [])
                context_entry = build_context_entry(
                    agent_name="environment_setup",
                    result=result_summary,
                    messages=messages,
                    max_detail_tokens=5000,
                )
                self.hierarchical_context.add(
                    content=context_entry,
                    source="environment_setup",
                    entry_type="result" if result_summary.get("success") else "error",
                    importance=0.9,
                )

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
        # Handle structured content blocks (e.g., {'type': 'thinking', 'thinking': '...'})
        last_message = ""
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                if isinstance(last_msg.content, list):
                    # Handle list of content blocks (Gemini, Claude structured responses)
                    text_parts = []
                    for part in last_msg.content:
                        if isinstance(part, str):
                            text_parts.append(part)
                        elif isinstance(part, dict):
                            # Extract text from various block types
                            if "text" in part:
                                text_parts.append(part["text"])
                            elif "thinking" in part:
                                text_parts.append(part["thinking"])
                            elif "content" in part:
                                text_parts.append(part["content"])
                    last_message = "\n".join(text_parts)
                else:
                    last_message = str(last_msg.content)

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

        # Helper function to extract text from message content
        def extract_text_from_content(content) -> str:
            """Extract text from message content, handling structured blocks."""
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict):
                        # Extract text from various block types
                        if "text" in part:
                            text_parts.append(part["text"])
                        elif "thinking" in part:
                            text_parts.append(part["thinking"])
                        elif "content" in part:
                            text_parts.append(part["content"])
                return "\n".join(text_parts)
            else:
                return str(content)

        # Search for key indicators in agent messages
        # CRITICAL: Exclude HumanMessage from success detection!
        # HumanMessage contains the input prompt which includes previous context from
        # hierarchical storage. If that context has "smoke test passed" from a prior run,
        # it causes false positive success detection. Only check AI and Tool messages.
        full_conversation = "\n".join(
            [
                extract_text_from_content(m.content) if hasattr(m, "content") else str(m)
                for m in messages
                if not isinstance(m, HumanMessage)  # Exclude input prompts
            ]
        )
        full_conversation_lower = full_conversation.lower()

        # =====================================================================
        # USE STRUCTURED LLM OUTPUT FOR ROBUST SUCCESS DETECTION
        # This replaces brittle keyword matching with LLM judgment
        # Pattern from execution_agent.py:487-514
        # =====================================================================
        try:
            # Import pydantic with fallback for different versions
            try:
                from langchain_core.pydantic_v1 import BaseModel, Field
            except ImportError:
                try:
                    from pydantic.v1 import BaseModel, Field
                except ImportError:
                    from pydantic import BaseModel, Field

            from typing import Optional as Opt

            class EnvSetupAnalysis(BaseModel):
                smoke_test_passed: bool = Field(
                    description="True if the smoke test ran successfully (script started, printed output, no crash)"
                )
                env_name: Opt[str] = Field(
                    default=None,
                    description="The environment name that was created (e.g., 'gat_env_a1b2')"
                )
                env_type: Opt[str] = Field(
                    default=None,
                    description="Environment type: 'micromamba', 'conda', 'pip', 'venv', etc."
                )
                failure_reason: Opt[str] = Field(
                    default=None,
                    description="If smoke test failed, explain why. None if successful."
                )

            # Use LLM to analyze the conversation
            analyzer = self.llm.with_structured_output(EnvSetupAnalysis)
            analysis_prompt = f"""Analyze this environment setup conversation.

CONVERSATION (last 8000 chars):
{full_conversation[-8000:]}

Did the smoke test pass? A smoke test passes if:
- The script ran and printed output (hyperparameters, dataset info, training started)
- No Python exceptions (Traceback) crashed the script
- IGNORE warnings (DeprecationWarning, FutureWarning, distutils, etc.)
- IGNORE the word "error" if it's just in a variable name or log format string

Extract the environment name from commands like "micromamba create -n NAME" or "micromamba run -n NAME".
Determine the environment type (micromamba, conda, pip, venv) based on the commands used.
"""

            analysis = analyzer.invoke(analysis_prompt)
            print(f"DEBUG: LLM EnvSetup Analysis: {analysis}")

            # Apply LLM's judgment
            if analysis.smoke_test_passed:
                result["success"] = True
                result["status"] = "success"
            else:
                result["success"] = False
                result["status"] = "failed"
                if analysis.failure_reason:
                    result["error"] = analysis.failure_reason

            # Use LLM-extracted env info if available (will be overridden by regex if found)
            if analysis.env_name:
                result["env_name"] = analysis.env_name
            if analysis.env_type:
                result["env_type"] = analysis.env_type

        except Exception as e:
            print(f"⚠️ Structured LLM analysis failed: {e}. Falling back to keyword matching.")
            # FALLBACK: Original keyword-based logic
            incomplete_indicators = [
                "sorry", "need more steps", "cannot complete", "unable to finish",
                "i apologize", "unfortunately", "more iterations",
            ]
            is_incomplete = any(ind in full_conversation_lower for ind in incomplete_indicators)

            smoke_test_markers = [
                "smoke test passed", "smoke test successful", "script ran successfully",
                "experiment started successfully", "training started", "started training",
                "epoch 0", "epoch 1", "step 0", "step 1", "training loop",
            ]
            smoke_test_passed = any(marker in full_conversation_lower for marker in smoke_test_markers)

            if smoke_test_passed:
                result["success"] = True
                result["status"] = "success"
            elif is_incomplete:
                result["success"] = False
                result["status"] = "incomplete"
                result["error"] = "Agent hit iteration limit without completing environment setup"
            elif "modulenotfounderror" in full_conversation_lower or "importerror" in full_conversation_lower:
                result["status"] = "failed"
            elif "failed" in full_conversation_lower:
                result["status"] = "failed"

        # Try to extract environment name
        import re

        # PRIORITY 1: Look for "run" commands (the actual working command for smoke test)
        # This is what execution agent needs - the env that was successfully used
        micromamba_run = re.findall(
            r"micromamba\s+run\s+-n\s+(\w+)",
            full_conversation,
        )
        mamba_run = re.findall(
            r"(?<!micro)mamba\s+run\s+-n\s+(\w+)",
            full_conversation,
        )
        conda_run = re.findall(
            r"conda\s+run\s+-n\s+(\w+)",
            full_conversation,
        )

        if micromamba_run:
            result["env_name"] = micromamba_run[-1]  # Take LAST run command
            result["env_type"] = "micromamba"
        elif mamba_run:
            result["env_name"] = mamba_run[-1]
            result["env_type"] = "mamba"
        elif conda_run:
            result["env_name"] = conda_run[-1]
            result["env_type"] = "conda"
        # PRIORITY 2: Fall back to create/activate commands if no run found
        elif (
            micromamba_create := re.findall(
                r"micromamba\s+(?:create|activate|env\s+create).*?-n\s+(\w+)",
                full_conversation,
            )
        ):
            result["env_name"] = micromamba_create[-1]
            result["env_type"] = "micromamba"
        elif (
            mamba_create := re.findall(
                r"(?<!micro)mamba\s+(?:create|activate|env\s+create).*?-n\s+(\w+)",
                full_conversation,
            )
        ):
            result["env_name"] = mamba_create[-1]
            result["env_type"] = "mamba"
        elif (
            conda_create := re.findall(
                r"conda\s+(?:create|activate|env\s+create).*?-n\s+(\w+)",
                full_conversation,
            )
        ):
            result["env_name"] = conda_create[-1]
            result["env_type"] = "conda"
        # PRIORITY 3: Look for venv path
        elif venv_match := re.search(r"(\.?/?\w*/venv)", full_conversation):
            result["env_name"] = venv_match.group(1)
            result["env_type"] = "venv"

        # Try to extract python path
        python_match = re.search(r"(/[\w/]+/python\d?\.\d?)", full_conversation)
        if python_match:
            result["python_path"] = python_match.group(1)

        return result
