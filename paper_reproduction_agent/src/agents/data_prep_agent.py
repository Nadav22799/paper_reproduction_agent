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


class DataPrepAgent:
    """Handles dataset preparation for reproduction experiments."""

    def __init__(
        self,
        llm=None,
        max_iterations: int = 50,
        metrics_tracker=None,
        hierarchical_context: HierarchicalContextManager = None,
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

        from ..config.prompts import DATA_PREP_AGENT_PROMPT
        self.system_prompt = DATA_PREP_AGENT_PROMPT

        self.tools = [
            read_file,
            list_directory,
            execute_shell_command,
            execute_python_code,
            search_error_solution,
        ]

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

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=data_prompt)]},
                {"recursion_limit": self.max_iterations},
            )

            # Analyze result to determine success
            success, details, last_message = self._analyze_result(result, code_path)

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
        """Analyze agent result to determine success.

        Args:
            result: Agent execution result
            code_path: Repository path

        Returns:
            Tuple of (success: bool, details: dict, last_message: str)
        """
        # Check common data directories (searching up to depth 2)
        data_dirs_names = ["data", "datasets", "DATA", "Datasets", "corpus", "input"]
        found_data = False
        data_locations = []

        # Walk directory up to depth 2 to find data folders
        for root, dirs, files in os.walk(code_path):
            # Calculate depth
            depth = root[len(code_path):].count(os.sep)
            if depth > 2:
                dirs[:] = [] # Stop recursing
                continue
                
            for d in dirs:
                if d in data_dirs_names:
                    full_path = os.path.join(root, d)
                    if os.listdir(full_path):
                        found_data = True
                        data_locations.append(full_path)

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

        # Look for success indicators in output
        success_indicators = [
            "download complete",
            "downloaded successfully",
            "data ready",
            "datasets prepared",
            "verification passed",
            "already exists",
            "found existing",
            "data was already present",
            "verification status: passed",
            "datasets downloaded:",
        ]

        # Indicators that definitely mean failure, unless overridden by strong success
        failure_indicators = [
            "download failed",
            "error downloading",
            "could not download",
            "permission denied",
        ]

        output_lower = last_message.lower()
        has_success = any(ind in output_lower for ind in success_indicators)
        has_failure = any(ind in output_lower for ind in failure_indicators)
        
        # Check for strong success signal that overrides incidental "not found" text
        strong_success = "verification status: passed" in output_lower or "any issues encountered: none" in output_lower

        # Determine success
        # Prioritize strong success signals or finding actual data
        success = found_data or strong_success or (has_success and not has_failure)

        details = {
            "data_locations": data_locations,
            "found_existing": found_data,
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
