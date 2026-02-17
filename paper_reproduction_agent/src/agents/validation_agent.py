"""Validation Agent - Verifies experiment results against paper claims.

Split from UnifiedReproductionAgent to handle result verification as a separate concern.
This agent uses a CODE-FIRST approach:
1. Writes Python code to extract metrics from result files
2. Compares extracted metrics with paper values
3. Reports success rate with 5% tolerance threshold
"""

import os
import re
from typing import Dict, List
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from ..tools.code_execution_tools import (
    read_file,
    list_directory,
    search_file,
    execute_python_code,
    write_file,
)
from ..utils.llm_factory import create_llm
from ..utils.hierarchical_context import HierarchicalContextManager


class ValidationAgent:
    """Verifies experiment results against paper claims using code-first approach."""

    def __init__(
        self,
        llm=None,
        max_iterations: int = 30,
        metrics_tracker=None,
        hierarchical_context: HierarchicalContextManager = None,
        callbacks=None,
    ):
        """Initialize the Validation Agent.

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

        from ..config.prompts import VALIDATION_AGENT_PROMPT
        self.system_prompt = VALIDATION_AGENT_PROMPT

        self.tools = [
            read_file,
            list_directory,
            search_file,
            execute_python_code,
            write_file,
        ]

        print("\n" + "=" * 60)
        print("Validation Agent Initialized")
        print(f"   Max Iterations: {max_iterations}")
        print("=" * 60)

    def verify_results(self, state: Dict) -> Dict:
        """Verify experiment results against paper claims.

        Args:
            state: Current PaperReproductionState

        Returns:
            Dict with:
                - results_match: bool
                - extracted_metrics: dict
                - metrics_comparison: dict
                - verification_results: dict
        """
        code_path = state.get("implementation_path", "./cloned_repo")
        checklist_path = state.get("checklist_path", "reproduction_checklist.md")
        
        # Ensure path is absolute if possible, but don't check existence - let Agent find it
        if not os.path.isabs(checklist_path) and os.path.basename(checklist_path) == checklist_path:
             checklist_path = os.path.join(code_path, checklist_path)
             
        print(f"📊 Debug: Checklist Path='{checklist_path}'")

        # Read experiment selection mode and plan
        experiment_mode = str(state.get("experiment_selection_mode", "all")).lower()
        reproduction_plan = state.get("reproduction_plan", {})
        selected_datasets = reproduction_plan.get("selected_datasets", [])

        print(f"📊 Debug: Mode='{experiment_mode}', Datasets={selected_datasets}")

        print("📊 Validation Agent: Verifying results...")

        # RETRIEVE relevant context from previous agents (exclude own to prevent self-referencing)
        previous_context = ""
        if self.hierarchical_context:
            previous_context = self.hierarchical_context.compile_context(
                query="experiment results metrics accuracy expected values execution output",
                max_tokens=2000,
                exclude_sources=["validation"],
            )
            if previous_context:
                print(f"   📋 Retrieved {len(previous_context)} chars of previous context")

        # Build verification prompt
        verification_prompt = f"""Verify experiment results.

Repository Path: {code_path}
Checklist Path Hint: {checklist_path}

=== CONTEXT FROM PREVIOUS AGENTS ===
{previous_context if previous_context else "No previous context available"}
====================================

Task:
1. Find and read the checklist to get Expected Metrics.
2. Find actual results (in checklist "Experiments" section OR in result files).
3. Compare and report.
"""

        # Create and run the ReAct agent
        agent = create_react_agent(
            self.llm,
            self.tools,
            prompt=self.system_prompt,
        )


        print("\n" + "-" * 60)
        print(f"Validation Agent: Verifying results for {code_path}")
        print("-" * 60)

        try:
            # Stream the agent's execution to provide visibility
            print("   💭 Validation verification plan:")

            config = {"recursion_limit": self.max_iterations}
            if self.callbacks:
                config["callbacks"] = self.callbacks

            all_messages = []  # Collect all messages for analysis
            for event in agent.stream(
                {"messages": [HumanMessage(content=verification_prompt)]},
                config
            ):
                # Collect messages from both agent and tools events
                if "agent" in event and "messages" in event["agent"]:
                    all_messages.extend(event["agent"]["messages"])
                if "tools" in event and "messages" in event["tools"]:
                    all_messages.extend(event["tools"]["messages"])

                # Handle streaming events to print progress
                if "agent" in event:
                    agent_data = event["agent"]
                    if "messages" in agent_data:
                        last_msg = agent_data["messages"][-1]
                        
                        # Print agent responses
                        has_tool_calls = hasattr(last_msg, "tool_calls") and last_msg.tool_calls
                        if hasattr(last_msg, "content") and last_msg.content:
                            content = last_msg.content
                            if isinstance(content, list):
                                content = " ".join(str(c) for c in content)
                            if isinstance(content, str) and content.strip():
                                if not has_tool_calls:
                                    preview = content[:500].replace('\n', ' ')
                                    print(f"   📝 Agent response: {preview}...")
                                else:
                                    lines = content.split('\n')
                                    for line in lines:
                                        if "thought" in line.lower() or "plan" in line.lower():
                                            print(f"   🤔 {line.strip()}")
                        
                        # Print Tool Calls (Inputs)
                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            for tool_call in last_msg.tool_calls:
                                tool_name = tool_call.get("name", "unknown")
                                print(f"   🛠️  Calling: {tool_name}")
                
                if "tools" in event:
                    tool_data = event["tools"]
                    if "messages" in tool_data:
                        for msg in tool_data["messages"]:
                            status = "✅" 
                            if "error" in str(msg.content).lower():
                                    status = "⚠️"
                            output_preview = str(msg.content)[:200].replace('\n', ' ')
                            print(f"      {status} Result: {output_preview}...")

            # Analyze result
            result = {"messages": all_messages}
            verification = self._analyze_verification_result(result, [], experiment_mode, selected_datasets)

            # Store FULL messages in hierarchical context (including tool calls)
            if self.hierarchical_context:
                from ..utils.context_utils import build_smart_context_entry

                validation_result = {
                    "results_match": verification["success"],
                    "match_ratio": verification.get("match_ratio", "N/A"),
                    "success_level": verification["success_level"],
                }

                context_entry = build_smart_context_entry(
                    agent_name="validation",
                    result=validation_result,
                    messages=all_messages,
                    max_detail_tokens=4000,
                )

                self.hierarchical_context.add(
                    content=context_entry,
                    source="validation",
                    entry_type="result" if verification["success"] else "error",
                    importance=0.9,
                    lazy=True,
                )

            return {
                "results_match": verification["success"],
                "extracted_metrics": verification["extracted_metrics"],
                "metrics_comparison": verification["comparison"],
                "verification_results": {
                    "report": verification["report"],
                    "success_level": verification["success_level"],
                    "results_match_paper": verification["success"],
                    "experiments_passed": verification["experiments_passed_count"],
                    "experiments_total": verification["experiments_total_count"],
                    "match_ratio": verification.get("match_ratio", "N/A"),
                },
                "phase_status": {"validation": "completed"},
            }

        except Exception as e:
            print(f"⚠️  Validation Agent error: {e}")
            return {
                "results_match": False,
                "extracted_metrics": {},
                "metrics_comparison": {"error": str(e)},
                "verification_results": {
                    "report": f"Verification failed: {e}",
                    "success_level": "failed",
                    "results_match_paper": False,
                },
                "phase_status": {"validation": "failed"},
            }

    def _analyze_verification_result(self, result: Dict, expected: List[Dict], mode: str = "all", selected_datasets: List[str] = None) -> Dict:
        """Analyze verification result from agent."""
        messages = result.get("messages", [])
        all_content = []
        for msg in messages:
            content = None
            if hasattr(msg, "content"):
                content = msg.content
            elif isinstance(msg, dict) and "content" in msg:
                content = msg["content"]
            if content:
                all_content.append(str(content))

        combined_output = "\n".join(all_content).lower()

        # Check for success indicators
        success = False
        success_level = "failed"
        matched_count = 0
        total_count = 0
        experiments_passed_count = 0
        experiments_total_count = 0

        match_ratio = "N/A"
    
        # Robust Success Detection
        if "overall status: ✅ passed" in combined_output or "primary metric matched: yes" in combined_output:
             success = True
             success_level = "full"
             experiments_passed_count = 1
             experiments_total_count = 1
        elif "metrics matched" in combined_output:
             # Legacy/Fallback parsing
             if "✅" in combined_output and "❌" not in combined_output:
                 success = True
                 success_level = "full"
                 experiments_passed_count = 1
                 experiments_total_count = 1

        # Extract Match Ratio if present (e.g., "Match Ratio: 1/1")
        ratio_match = re.search(r"match ratio:\s*(\d+)/(\d+)", combined_output)
        if ratio_match:
            match_ratio = f"{ratio_match.group(1)}/{ratio_match.group(2)}"
            try:
                experiments_passed_count = int(ratio_match.group(1))
                experiments_total_count = int(ratio_match.group(2))
            except ValueError:
                pass

        return {
            "success": success,
            "success_level": success_level,
            "match_ratio": match_ratio,
            "extracted_metrics": {
                "matched_count": matched_count,
                "total_count": total_count,
            },
            "comparison": {
                "matches": [],
                "mismatches": [],
            },
            "report": combined_output[-1000:] if combined_output else "No report generated",
            "experiments_passed_count": experiments_passed_count,
            "experiments_total_count": experiments_total_count,
        }
