"""Validation Agent - Verifies experiment results against paper claims.

Split from UnifiedReproductionAgent to handle result verification as a separate concern.
This agent uses a CODE-FIRST approach:
1. Writes Python code to extract metrics from result files
2. Compares extracted metrics with paper values
3. Reports success rate with 5% tolerance threshold
"""

import os
import re
from typing import Dict, List, Tuple
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from ..tools.code_execution_tools import (
    read_file,
    list_directory,
    search_file,
    execute_python_code,
    write_file,
)
from ..utils.llm_factory import create_llm


class ValidationAgent:
    """Verifies experiment results against paper claims using code-first approach."""

    def __init__(self, llm=None, max_iterations: int = 30, metrics_tracker=None):
        """Initialize the Validation Agent.

        Args:
            llm: Language model to use
            max_iterations: Maximum iterations for the ReAct agent
            metrics_tracker: Optional metrics tracker for observability
        """
        self.llm = llm or create_llm(temperature=0.1)
        self.max_iterations = max_iterations
        self.metrics_tracker = metrics_tracker

        self.system_prompt = """You are a Verification Specialist for ML paper reproduction.

GOAL: VERIFY REPRODUCTION RESULTS
You need to confirm if the experiment results match the paper's claims.

═══════════════════════════════════════════════════════════════
YOUR WORKFLOW
═══════════════════════════════════════════════════════════════

1. **LOCATE CONTEXT**:
   - Find and read `reproduction_checklist.md`.
   - The user provided path might be wrong. Use `search_file` if needed.
   - Extract expected metrics and ANY ACTUAL results already recorded there (e.g. in "Experiments" section).

2. **LOCATE EVIDENCE**:
   - If results are NOT in the checklist, find the result files.
   - They could be `.log`, `.json`, `.csv` or `.txt`.
   - Use `list_directory` and `search_file` to find them. Do not guess paths.

3. **VERIFY & REPORT**:
   - Use `execute_python_code` to parse files and compare values.
   - Calculate relative error (Target < 5%).
   - **CRITICAL SUCCESS LOGIC**:
     - **Primary Metrics** (Accuracy, F1, Score, Loss): MUST MATCH.
     - **Secondary Metrics** (Time, Memory, Epochs): Informational only. DO NOT include them in the Markdown table.
     - If Primary Metric matches, count the Experiment as PASSED.
     - Do NOT mark the experiment as failed just because Training Time didn't match.

   - **SUCCESS RATIO CALCULATION (n/N)**:
     - `N` = Total count of expected **Primary Metrics** across all experiments.
       - If Single mode: N = count of primary metrics in the one experiment (e.g., 1 for Accuracy, or 2 if F1 & Accuracy).
       - If Full mode: N = sum of primary metrics across all experiments.
     - `n` = Total count of matching **Primary Metrics**.
     - **REPORT OUTPUT**:
       - You MUST include the line: `Match Ratio: n/N` (e.g., "Match Ratio: 1/1" or "Match Ratio: 2/3").
       
═══════════════════════════════════════════════════════════════
TOOLS
═══════════════════════════════════════════════════════════════
- `read_file(path)`: Read file content.
- `search_file(pattern, path)`: Find files matching a pattern.
- `execute_python_code(code)`: Run Python to parse/calculate.
- `list_directory(path)`: See folder contents.

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════

Report format (YOU MUST OUTPUT THIS):
```
VERIFICATION RESULTS
====================
Primary Metric Matched: YES/NO
Secondary Metrics Matched: X/Y (Informational) 
Overall Status: ✅ PASSED / ❌ FAILED

| Metric      | Expected | Actual | Error   | Status |
|-------------|----------|--------|---------|--------|
| Accuracy    | 95.5     | 94.8   | 0.73%   | ✅     |
| F1 Score    | 0.92     | 0.85   | 7.6%    | ❌     |
```

Then update the `reproduction_checklist.md` verification section.
"""

        self.tools = [
            read_file,
            list_directory,
            search_file,
            execute_python_code,
            write_file,
        ]

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

        paper_results = state.get("paper_results", {})
        experiment_results = state.get("experiment_results", {})

        # Read experiment selection mode and plan
        experiment_mode = str(state.get("experiment_selection_mode", "all")).lower()
        reproduction_plan = state.get("reproduction_plan", {})
        selected_experiments = reproduction_plan.get("selected_experiments", [])
        selected_datasets = reproduction_plan.get("selected_datasets", [])

        print(f"📊 Debug: Mode='{experiment_mode}', Datasets={selected_datasets}")

        print("📊 Validation Agent: Verifying results...")

        # Build verification prompt
        verification_prompt = f"""Verify experiment results.

Repository Path: {code_path}
Checklist Path Hint: {checklist_path}

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

        try:
            # Stream the agent's execution to provide visibility
            print("   💭 Validation verification plan:")
            
            all_messages = []  # Collect all messages for analysis
            for event in agent.stream(
                {"messages": [HumanMessage(content=verification_prompt)]},
                {"recursion_limit": self.max_iterations}
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
        import re
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
