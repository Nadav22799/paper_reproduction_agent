"""Experiment Runner Agent - Executes experiments and captures outputs."""

from typing import Dict
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from ..tools.code_execution_tools import (
    read_file,
    search_file,
    list_directory,
    execute_shell_command,
    execute_python_script,
)
from ..utils.llm_factory import create_llm
from ..utils.logging_callback import LoggingCallbackHandler
from ..utils.message_utils import normalize_message_content


class ExperimentRunnerAgent:
    """Agent for running experiments and capturing results."""

    def __init__(self, llm=None):
        self.llm = llm or create_llm(temperature=0.1)

        self.system_prompt = """You run ML experiments and capture results.

Priority:
1. Find demo/test scripts (demo.py, test.py, eval.py, inference.py)
2. Look for main training entry points (train.py, main.py, run.py)
3. Check README for usage examples
4. Find smallest/quickest experiment config

Execution strategy:
- Prefer demos over full training (faster, less resources)
- Use --help to discover arguments
- Look for pretrained models to skip training
- Use minimal configs (small batch size, few epochs)
- Capture all stdout/stderr

Report: what you ran, execution time, output summary."""

        tools = [read_file, search_file, list_directory, execute_shell_command, execute_python_script]
        # Don't pass custom prompt - let ReAct use its default optimized for tool calling
        self.agent = create_react_agent(self.llm, tools=tools)

    def run_experiments(self, code_path: str, paper_experiments: Dict = None) -> Dict:
        """
        Run experiments from the repository.

        Args:
            code_path: Path to repository
            paper_experiments: Expected experiments from paper (optional)

        Returns:
            Execution results with outputs and metrics
        """
        experiment_info = f"Expected experiments: {paper_experiments}" if paper_experiments else "Find available experiments"

        # Prepend system context to task (since we can't override ReAct's prompt)
        task = f"""{self.system_prompt}

Task: Run quick experiment from repository at: {code_path}

{experiment_info}

Use these EXACT tool calls in order:

1. read_file(file_path="{code_path}/README.md", max_lines=300)
   Look for: Quick Start section, usage examples, command-line examples

2. list_directory(dir_path="{code_path}", recursive=False)
   Find: demo.py, test.py, eval.py, train.py, main.py, or command-line tools

3. Identify the QUICKEST experiment from README:
   - Prefer: demo scripts, pre-trained model inference, minimal training examples
   - Avoid: full training runs (too slow)
   - Look for commands starting with: t2t-trainer, python, bash scripts

4. Execute the quickest command you found:
   execute_shell_command(command="<command from README>", cwd="{code_path}", timeout=300)

   Example: If README shows "t2t-trainer --generate_data --problem=image_mnist --train_steps=100"
   Run: execute_shell_command(command="t2t-trainer --generate_data --problem=image_mnist --train_steps=100", cwd="{code_path}", timeout=300)

IMPORTANT:
- Always use cwd="{code_path}" for shell commands
- Use full path {code_path}/filename for file operations
- Prioritize examples from README Quick Start section
- If training needed: use minimal steps (--train_steps=100 or similar)

Report: Command executed, arguments used, stdout/stderr output, and any metrics found."""

        messages = [HumanMessage(content=task)]
        callback = LoggingCallbackHandler(verbose=True)

        try:
            result = self.agent.invoke(
                {"messages": messages},
                config={"recursion_limit": 40, "callbacks": [callback]}
            )
        except Exception as e:
            print(f"\n❌ Experiment runner failed: {e}")
            result = {"messages": [], "error": str(e)}

        return self._parse_experiment_result(result)

    def _parse_experiment_result(self, result: Dict) -> Dict:
        """Extract experiment results from agent output."""
        messages = result.get("messages", [])

        experiment_info = {
            "scripts_found": False,
            "execution_attempted": False,
            "execution_successful": False,
            "output": "",
            "metrics": {},
            "errors": [],
            "report": ""
        }

        full_output = []

        for msg in messages:
            if hasattr(msg, 'content') and msg.content:
                content = normalize_message_content(msg.content)
                content_lower = content.lower()

                # Check for script discovery
                if any(kw in content_lower for kw in [".py", "train", "demo", "test", "eval", "main"]):
                    experiment_info["scripts_found"] = True

                # Check for execution
                if any(kw in content_lower for kw in ["execute", "running", "python", "stdout", "stderr"]):
                    experiment_info["execution_attempted"] = True
                    full_output.append(content)

                # Check for successful completion
                if "returncode" in content_lower or "success" in content_lower:
                    if "returncode: 0" in content_lower or "success: true" in content_lower:
                        experiment_info["execution_successful"] = True

                # Check for errors
                if "error" in content_lower or "failed" in content_lower or "exception" in content_lower:
                    experiment_info["errors"].append(content[:300])

        # Combine output
        experiment_info["output"] = "\n".join(full_output)

        # Get final report
        if messages:
            experiment_info["report"] = normalize_message_content(messages[-1].content) if hasattr(messages[-1], 'content') else ""

        return experiment_info
