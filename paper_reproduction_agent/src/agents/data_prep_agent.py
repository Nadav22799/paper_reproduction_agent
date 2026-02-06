"""Data Preparation Agent - Downloads and prepares datasets for experiments.

Split from UnifiedReproductionAgent to handle data preparation as a separate concern.
This agent:
1. Reads the checklist to understand data requirements
2. Downloads datasets following README instructions
3. Verifies data integrity
4. Updates the checklist with data locations
"""

import os
from typing import Dict, Optional
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from ..tools.code_execution_tools import (
    read_file,
    list_directory,
    execute_shell_command,
    execute_python_code,
    search_error_solution,
)
from ..utils.llm_factory import create_llm
from ..utils.hierarchical_context import HierarchicalContextManager
from ..utils.logging_callback import LoggingCallbackHandler


class DataPrepAgent:
    """Handles dataset preparation for reproduction experiments."""

    def __init__(
        self,
        llm=None,
        max_iterations: int = 50,
        metrics_tracker=None,
        hierarchical_context: HierarchicalContextManager = None,
        callbacks=None,
    ):
        """Initialize the Data Preparation Agent.

        Args:
            llm: Language model to use
            max_iterations: Maximum iterations for the ReAct agent
            metrics_tracker: Optional metrics tracker for observability
            hierarchical_context: Shared context manager for cross-agent knowledge
        """
        self.llm = llm or create_llm(temperature=0.1)
        self.max_iterations = max_iterations
        self.metrics_tracker = metrics_tracker
        self.hierarchical_context = hierarchical_context
        self.callbacks = callbacks or []

        from ..config.prompts import DATA_PREP_AGENT_PROMPT
        self.system_prompt = DATA_PREP_AGENT_PROMPT

        self.tools = [
            read_file,
            list_directory,
            execute_shell_command,
            execute_python_code,
            search_error_solution,
        ]

        print("\n" + "=" * 60)
        print("Data Preparation Agent Initialized")
        print(f"   Max Iterations: {max_iterations}")
        print("=" * 60)

    def prepare_data(self, state: Dict) -> Dict:
        """Prepare datasets for experiments.

        Args:
            state: Current PaperReproductionState

        Returns:
            Dict with:
                - datasets_ready: bool
                - dataset_results: dict with details
                - failure_metadata: Optional dict if failed
        """
        code_path = state.get("implementation_path", "./cloned_repo")
        checklist_path = state.get("checklist_path", "")
        env_info = state.get("agent_contexts", {}).get("environment_setup", {})
        env_name = env_info.get("env_name", "")

        print("📦 Data Prep Agent: Preparing datasets...")

        # RETRIEVE relevant context from previous agents (exclude own to prevent self-referencing)
        previous_context = ""
        if self.hierarchical_context:
            previous_context = self.hierarchical_context.compile_context(
                query="data download dataset path environment setup smoke test",
                max_tokens=1500,
                exclude_sources=["data_prep"],
            )
            if previous_context:
                print(f"   📋 Retrieved {len(previous_context)} chars of previous context")

        # Read checklist to understand requirements
        checklist_content = ""
        if checklist_path and os.path.exists(checklist_path):
            try:
                with open(checklist_path, "r", encoding="utf-8") as f:
                    checklist_content = f.read()
            except Exception as e:
                print(f"⚠️  Could not read checklist: {e}")

        # Read README to check for data instructions (Critical for redundancy check)
        readme_content = ""
        readme_path = os.path.join(code_path, "README.md")
        if os.path.exists(readme_path):
            try:
                with open(readme_path, "r", encoding="utf-8") as f:
                    readme_content = f.read()
            except Exception as e:
                print(f"⚠️  Could not read README: {e}")

        # Build the data preparation prompt
        data_prompt = f"""Prepare datasets for this ML repository.

Repository Path: {code_path}
Environment Name: {env_name if env_name else "Check checklist for environment info"}

=== CONTEXT FROM PREVIOUS AGENTS ===
{previous_context if previous_context else "No previous context available"}
====================================

Current Checklist:
{checklist_content[:3000] if checklist_content else "No checklist found"}

README Content (excerpt):
{readme_content[:3000] if readme_content else "No README found"}

STEPS:
1. Analyze the Checklist and README for specific data preparation instructions.
2. Check if data already exists in common locations (./data/, ./datasets/) or subdirectories.
3. If specific instructions exist, follow them to download/prepare data.
4. CRITICAL: If NO specific data instructions are found in the README/Checklist, assume data is handled automatically by the scripts or is already present. In this case, verify no obvious data errors exist, then REPORT SUCCESS (Verification status: Passed).

Start by analyzing the provided context."""

        # Create and run the ReAct agent
        agent = create_react_agent(
            self.llm,
            self.tools,
            prompt=self.system_prompt,
        )


        print("\n" + "-" * 60)
        print(f"Data Prep Agent: Starting data preparation for {code_path}")
        print("-" * 60)
        
        # ------------------------------------------------------------------
        # PHASE 1: VERIFICATION (Agent-driven search)
        # ------------------------------------------------------------------
        # Instead of heuristics, we let the agent search first.
        # But we can also do a quick programmatic check for common patterns 
        # to save time/tokens if obvious. 
        # Actually, let's follow the user's request: CLEAN logic, no heuristics.
        # So we do NOT add heuristics here. We rely on the prompt to drive search.
        
        try:
            config = {"recursion_limit": self.max_iterations}
            if self.callbacks:
                config["callbacks"] = self.callbacks
            result = agent.invoke(
                {"messages": [HumanMessage(content=data_prompt)]},
                config,
            )

            # Analyze result to determine success
            success, details, last_message = self._analyze_result(result, code_path)

            # Store FULL messages in hierarchical context (including tool calls)
            if self.hierarchical_context:
                from ..utils.context_utils import build_context_entry

                messages = result.get("messages", [])
                data_result = {
                    "datasets_ready": success,
                    "dataset_results": details,
                }

                context_entry = build_context_entry(
                    agent_name="data_prep",
                    result=data_result,
                    messages=messages,
                    max_detail_tokens=4000,
                )

                self.hierarchical_context.add(
                    content=context_entry,
                    source="data_prep",
                    entry_type="result" if success else "error",
                    importance=0.8,
                    lazy=True,
                )

            if success:
                return {
                    "datasets_ready": True,
                    "dataset_results": details,
                    "failure_metadata": None,
                    "phase_status": {"data_prep": "completed"},
                    "last_message": last_message,  # Agent reasoning for verbose output
                }
            else:
                return {
                    "datasets_ready": False,
                    "dataset_results": details,
                    "failure_metadata": self._create_failure_metadata(details),
                    "phase_status": {"data_prep": "failed"},
                    "last_message": last_message,  # Agent reasoning for verbose output
                }

        except Exception as e:
            print(f"⚠️  Data Prep Agent error: {e}")
            return {
                "datasets_ready": False,
                "dataset_results": {"error": str(e)},
                "failure_metadata": {
                    "error_type": "data",
                    "error_message": str(e),
                    "error_source": "data_prep_agent",
                    "attempted_fixes": [],
                    "recovery_hints": ["Check README for manual download instructions"],
                    "retry_count": state.get("failure_metadata", {}).get("retry_count", 0) + 1,
                },
                "phase_status": {"data_prep": "failed"},
                "last_message": f"Exception: {str(e)}",  # Error as reasoning
            }

    def _analyze_result(self, result: Dict, code_path: str) -> tuple:
        """Analyze agent result to determine success using structured LLM output.

        Args:
            result: Agent execution result
            code_path: Repository path

        Returns:
            Tuple of (success: bool, details: dict, last_message: str)
        """
        import re
        from typing import List

        # Extract info from agent messages
        messages = result.get("messages", [])
        last_message = ""
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                if isinstance(last_msg.content, list):
                    # Handle list of content blocks
                    text_parts = []
                    for part in last_msg.content:
                        if isinstance(part, str):
                            text_parts.append(part)
                        elif isinstance(part, dict) and "text" in part:
                            text_parts.append(part["text"])
                    last_message = "\n".join(text_parts)
                else:
                    last_message = str(last_msg.content)

        # =====================================================================
        # USE STRUCTURED LLM OUTPUT FOR ROBUST SUCCESS DETECTION
        # This replaces brittle keyword matching with LLM judgment
        # Pattern from execution_agent.py:487-514
        # =====================================================================
        success = False
        data_locations = []

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

            class DataPrepAnalysis(BaseModel):
                datasets_ready: bool = Field(
                    description="True if datasets were successfully prepared, verified, or confirmed to already exist"
                )
                data_locations: List[str] = Field(
                    default_factory=list,
                    description="List of paths where data was found or prepared"
                )
                failure_reason: Opt[str] = Field(
                    default=None,
                    description="If data prep failed, explain why. None if successful."
                )

            # Use LLM to analyze the agent's output
            analyzer = self.llm.with_structured_output(DataPrepAnalysis)
            analysis_prompt = f"""Analyze this data preparation result.

AGENT OUTPUT:
{last_message[:4000]}

Did data preparation succeed? It succeeds if:
- Datasets were downloaded successfully
- Datasets already existed and were verified
- Agent confirmed data is ready (e.g., "prepared the datasets", "data verified", "datasets loaded")
- No missing data errors that block execution

Extract any data paths mentioned (e.g., "./data/cora", "/path/to/datasets").
"""

            analysis = analyzer.invoke(analysis_prompt)
            print(f"DEBUG: LLM DataPrep Analysis: {analysis}")

            success = analysis.datasets_ready
            data_locations = analysis.data_locations or []

            if not success and analysis.failure_reason:
                print(f"⚠️ Data prep failed: {analysis.failure_reason}")

        except Exception as e:
            print(f"⚠️ Structured LLM analysis failed: {e}. Falling back to keyword matching.")
            # FALLBACK: keyword-based logic
            success_indicators = [
                "datasets ready", "successfully loaded", "prepared the datasets",
                "data verified", "datasets exist", "data is ready", "verification passed",
                "data preparation complete", "datasets downloaded",
            ]
            success = any(ind in last_message.lower() for ind in success_indicators)

            # Try to extract paths from the message
            path_match = re.search(r"data_path:\s*(.+)", last_message, re.IGNORECASE)
            if path_match:
                path = path_match.group(1).strip()
                if path and path.lower() != "n/a":
                    data_locations.append(path)

        details = {
            "data_locations": data_locations,
            "found_existing": success,
            "agent_output": last_message[:500] if last_message else "",
        }

        return success, details, last_message

    def _create_failure_metadata(self, details: Dict) -> Dict:
        """Create failure metadata for routing.

        Args:
            details: Details from failed attempt

        Returns:
            FailureMetadata dict
        """
        error_message = details.get("agent_output", "Data preparation failed")
        if len(error_message) > 200:
            error_message = error_message[:200] + "..."

        hints = []
        error_lower = error_message.lower()

        if "permission" in error_lower:
            hints.append("Check file permissions")
        if "connection" in error_lower or "network" in error_lower:
            hints.append("Check network connection")
        if "not found" in error_lower:
            hints.append("Verify download URL is correct")
        if "disk" in error_lower or "space" in error_lower:
            hints.append("Check available disk space")

        return {
            "error_type": "data",
            "error_message": error_message,
            "error_source": "data_prep_agent",
            "attempted_fixes": [],
            "recovery_hints": hints or ["Check README for manual download instructions"],
            "retry_count": 0,
        }

    def verify_data(self, state: Dict) -> Dict:
        """Verify that data is ready for experiments.

        Args:
            state: Current state

        Returns:
            Verification result dict
        """
        code_path = state.get("implementation_path", "./cloned_repo")

        # Check common data directories
        data_dirs = ["data", "datasets", "DATA", "Datasets"]
        found_data = []

        for data_dir in data_dirs:
            full_path = os.path.join(code_path, data_dir)
            if os.path.exists(full_path):
                contents = os.listdir(full_path)
                if contents:
                    found_data.append({
                        "path": full_path,
                        "files": contents[:10],  # First 10 files
                        "total_files": len(contents),
                    })

        if found_data:
            return {
                "verified": True,
                "data_locations": found_data,
                "message": f"Found data in {len(found_data)} location(s)",
            }
        else:
            return {
                "verified": False,
                "data_locations": [],
                "message": "No data found in standard locations",
            }

    def request_planning_update(self, state: Dict, new_info: Dict) -> Dict:
        """Request a planning update when new data requirements are discovered.

        Args:
            state: Current state
            new_info: New information to add to plan

        Returns:
            State with planning_update_request set
        """
        state["planning_update_request"] = {
            "source": "data_prep_agent",
            "reason": new_info.get("reason", "New data requirements discovered"),
            "context": new_info,
        }
        return state
