"""Generalization Gap Agent - Tests if the paper's novelty generalizes to external data.

This agent runs ONLY after successful validation (match_ratio > 0) and:
1. Identifies the paper's novel contribution and baselines
2. Finds an external dataset not used in the paper for the same task
3. Adapts the code (via NEW files, not modifying originals) to run on external data
4. Runs the novel method and at least one baseline on external data
5. Compares: novel > baseline = SUCCESS
"""

import os
import re
from typing import Dict
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from ..tools.code_execution_tools import (
    read_file,
    list_directory,
    write_file,
    execute_shell_command,
    execute_python_code,
    start_background_process,
    wait_for_process,
    stop_process,
    search_error_solution,
)
from ..utils.llm_factory import create_llm
from ..utils.hierarchical_context import HierarchicalContextManager
from ..utils.tool_guard import guard_tool


class GeneralizationAgent:
    """Tests whether the paper's novelty generalizes beyond the paper's datasets."""

    def __init__(
        self,
        llm=None,
        max_iterations: int = 150,
        metrics_tracker=None,
        hierarchical_context: HierarchicalContextManager = None,
        callbacks=None,
        critic_mode: str = "auto",
    ):
        self.llm = llm or create_llm(temperature=0.1)
        self.max_iterations = max_iterations
        self.metrics_tracker = metrics_tracker
        self.hierarchical_context = hierarchical_context
        self.callbacks = callbacks or []

        from ..config.prompts import GENERALIZATION_AGENT_PROMPT, EFFICIENCY_RULES
        self.system_prompt = GENERALIZATION_AGENT_PROMPT.replace(
            "{efficiency_rules}", EFFICIENCY_RULES
        )

        # Wrap DuckDuckGo so network failures don't crash the agent
        _ddg = DuckDuckGoSearchRun()
        _ddg.handle_tool_error = True

        self.tools = [
            read_file,
            list_directory,
            write_file,
            guard_tool(execute_shell_command, mode=critic_mode),
            guard_tool(execute_python_code, mode=critic_mode),
            guard_tool(start_background_process, mode=critic_mode),
            wait_for_process,
            stop_process,
            search_error_solution,
            _ddg,
        ]

        from rich.console import Console
        from rich.panel import Panel
        Console().print(Panel(f"Max Iterations: {max_iterations}", title="Generalization Gap Agent Initialized", border_style="cyan", expand=False))

    def analyze_generalization(self, state: Dict) -> Dict:
        """Test the paper's novelty on external data.

        Args:
            state: Current PaperReproductionState

        Returns:
            Dict with generalization_success, external_dataset, novel_metrics,
            baseline_metrics, generalization_report, phase_status, and
            optionally user_input_required if blocked.
        """
        code_path = state.get("implementation_path", "./cloned_repo")
        checklist_path = state.get("checklist_path", "")
        paper_title = state.get("paper_title", "Unknown")
        verification_results = state.get("verification_results", {})
        print("🔬 Generalization Agent: Testing novelty on external data...")
        print(f"   📋 Paper: {paper_title}")
        print(f"   📊 Reproduction match ratio: {verification_results.get('match_ratio', 'N/A')}")

        # Retrieve context from previous agents
        previous_context = ""
        if self.hierarchical_context:
            previous_context = self.hierarchical_context.compile_context(
                query="experiment results metrics baselines novel method dataset validation",
                max_tokens=3000,
                exclude_sources=["generalization"],
            )
            if previous_context:
                print(f"   📋 Retrieved {len(previous_context)} chars of previous context")

        # Read checklist for env info
        checklist_content = ""
        tool_detected = ""
        env_name = ""

        # Try programmatic context first
        env_context = state.get("agent_contexts", {}).get("environment_setup", {})
        tool_detected = env_context.get("env_type", "")
        env_name = env_context.get("env_name", "")

        if not tool_detected or not env_name:
            env_results = state.get("env_setup_results", {})
            tool_detected = tool_detected or env_results.get("env_type", "")
            env_name = env_name or env_results.get("env_name", "")

        if checklist_path and os.path.exists(checklist_path):
            try:
                with open(checklist_path, "r", encoding="utf-8") as f:
                    checklist_content = f.read()
                if not tool_detected:
                    match = re.search(r"\*\*Tool Detected:\*\*\s*(\w+)", checklist_content)
                    if match:
                        tool_detected = match.group(1).lower()
                if not env_name:
                    match = re.search(r"\*\*Environment Name:\*\*\s*(\S+)", checklist_content)
                    if match:
                        env_name = match.group(1)
            except Exception as e:
                print(f"⚠️  Could not read checklist: {e}")

        # Build tool info string
        tool_info = ""
        if tool_detected and env_name:
            if tool_detected in ["conda", "mamba", "micromamba"]:
                tool_info = f"Tool: {tool_detected}, Env: {env_name}, Pattern: `{tool_detected} run -n {env_name} python <script>`"
            elif tool_detected in ["pip", "venv"]:
                tool_info = f"Tool: {tool_detected}, Env: {env_name}, Pattern: `./venv/bin/python <script>`"
            elif tool_detected == "poetry":
                tool_info = "Tool: poetry, Pattern: `poetry run python <script>`"
            elif tool_detected == "uv":
                tool_info = "Tool: uv, Pattern: `uv run python <script>`"

        # Build the generalization prompt
        prompt = f"""Test whether the paper's novelty generalizes to external data.

Paper: {paper_title}
Repository Path: {code_path}
Checklist Path: {checklist_path}

ENVIRONMENT: {tool_info if tool_info else "READ checklist for tool/env info"}

Reproduction Results:
- Match Ratio: {verification_results.get('match_ratio', 'N/A')}
- Success Level: {verification_results.get('success_level', 'N/A')}

=== CONTEXT FROM PREVIOUS AGENTS ===
{previous_context if previous_context else "No previous context available"}
====================================

Checklist (partial):
{checklist_content[:2000] if checklist_content else "Read reproduction_checklist.md"}

YOUR TASK:
1. Read the checklist to understand what was reproduced (novel method, baselines, datasets, metrics)
2. Search the web for an external dataset for the SAME task (NOT one used in the paper)
3. Download the external dataset
4. Create NEW files to adapt the novel method for external data (do NOT modify original repo files)
5. Run smoke test on external data
6. Run full novel method on external data
7. Run at least one baseline on the same external data
8. Compare results and report using the MANDATORY output format
"""

        agent = create_react_agent(
            self.llm,
            self.tools,
            prompt=self.system_prompt,
        )

        print("\n" + "-" * 60)
        print(f"Generalization Agent: Testing on external data for {code_path}")
        print("-" * 60)

        try:
            config = {"recursion_limit": self.max_iterations}
            if self.callbacks:
                config["callbacks"] = self.callbacks

            all_messages = []
            for event in agent.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config,
            ):
                if "agent" in event and "messages" in event["agent"]:
                    all_messages.extend(event["agent"]["messages"])
                if "tools" in event and "messages" in event["tools"]:
                    all_messages.extend(event["tools"]["messages"])

                # Print progress
                if "agent" in event:
                    agent_data = event["agent"]
                    if "messages" in agent_data:
                        last_msg = agent_data["messages"][-1]
                        has_tool_calls = hasattr(last_msg, "tool_calls") and last_msg.tool_calls
                        if hasattr(last_msg, "content") and last_msg.content:
                            content = last_msg.content
                            if isinstance(content, list):
                                content = " ".join(str(c) for c in content)
                            if isinstance(content, str) and content.strip():
                                if not has_tool_calls:
                                    preview = content[:500].replace('\n', ' ')
                                    print(f"   📝 Agent: {preview}...")
                                else:
                                    for line in content.split('\n'):
                                        if "thought" in line.lower() or "plan" in line.lower():
                                            print(f"   🤔 {line.strip()}")
                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                print(f"   🛠️  Calling: {tc.get('name', 'unknown')}")

                if "tools" in event:
                    tool_data = event["tools"]
                    if "messages" in tool_data:
                        for msg in tool_data["messages"]:
                            status = "✅" if "error" not in str(msg.content).lower() else "⚠️"
                            preview = str(msg.content)[:200].replace('\n', ' ')
                            print(f"      {status} Result: {preview}...")

            # Analyze the result
            result = self._analyze_result(all_messages)

            # Store in hierarchical context
            if self.hierarchical_context:
                from ..utils.context_utils import build_smart_context_entry

                context_entry = build_smart_context_entry(
                    agent_name="generalization",
                    result={
                        "generalization_success": result["generalization_success"],
                        "external_dataset": result.get("external_dataset", ""),
                    },
                    messages=all_messages,
                    max_detail_tokens=4000,
                )
                self.hierarchical_context.add(
                    content=context_entry,
                    source="generalization",
                    entry_type="result" if result["generalization_success"] else "error",
                    importance=0.9,
                    lazy=True,
                )

            return result

        except Exception as e:
            print(f"⚠️  Generalization Agent error: {e}")
            return {
                "generalization_success": False,
                "external_dataset": "",
                "novel_metrics": {},
                "baseline_metrics": {},
                "generalization_report": f"Generalization analysis failed: {e}",
                "phase_status": {"generalization": "failed"},
            }

    def _analyze_result(self, messages: list) -> Dict:
        """Parse agent output for generalization results."""
        all_content = []
        for msg in messages:
            content = None
            if hasattr(msg, "content"):
                content = msg.content
            elif isinstance(msg, dict) and "content" in msg:
                content = msg["content"]
            if content:
                all_content.append(str(content))

        combined = "\n".join(all_content).lower()

        # Detect success
        success = False
        if "generalization status: ✅ passed" in combined:
            success = True
        elif "generalization status:" in combined and "passed" in combined:
            success = True

        # Detect blocked (user input needed)
        blocked = "blocked" in combined and "user input" in combined

        # Extract external dataset name
        external_dataset = ""
        ds_match = re.search(r"external dataset:\s*(.+?)(?:\n|$)", combined)
        if ds_match:
            external_dataset = ds_match.group(1).strip()

        report = combined[-2000:] if combined else "No report generated"

        result = {
            "generalization_success": success,
            "external_dataset": external_dataset,
            "novel_metrics": {},
            "baseline_metrics": {},
            "generalization_report": report,
            "phase_status": {"generalization": "completed" if not blocked else "blocked"},
        }

        if blocked:
            result["user_input_required"] = {
                "description": "Generalization analysis requires user input (see checklist for details)",
                "items": [{"name": "Generalization adaptation", "type": "other",
                           "description": "Code adaptation too complex for auto-adaptation"}],
            }

        return result
