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
from ..utils.resource_detector import (
    detect_system_resources,
    get_resource_summary,
    get_experiment_strategy
)


class ExperimentRunnerAgent:
    """Agent for running experiments and capturing results."""

    def __init__(self, llm=None):
        self.llm = llm or create_llm(temperature=0.1)

        # Detect system resources
        self.resources = detect_system_resources()
        self.experiment_strategy = get_experiment_strategy(self.resources)

        print("\n" + "="*60)
        print(get_resource_summary(self.resources))
        print(f"   Experiment Strategy: {self.experiment_strategy.upper()}")
        print("="*60)

        self.system_prompt = """You run ML experiments by following README instructions.

ALWAYS START BY READING README.md - it contains usage examples!

Your job:
1. Extract experiment commands from README
2. Follow references to nested READMEs if mentioned
3. Try to execute commands
4. Report what happened

README sections to look for:
- "Quick Start" / "Quickstart" / "Usage" / "Examples"
- "Getting Started" / "How to Run"
- "Reproduce Results" / "Reproduction"
- Code examples with commands to run

IMPORTANT - Nested READMEs:
If README says "see instructions in X/" or "refer to X/README.md":
→ Read that nested README too! Use read_file(file_path="<repo>/X/README.md")
→ The nested README often has the actual commands to run

Example:
Root README: "See instructions in examples/NLG/ to reproduce results"
→ You should: read_file(file_path="<repo>/examples/NLG/README.md")
→ Then use commands from that nested README

What to extract from README:
✅ Exact commands to run (e.g., "python train.py --config ...")
✅ Demo/test examples (simpler = better)
✅ Paper reproduction commands
✅ Required arguments/configs
✅ References to other READMEs or instruction files

Execution strategy:
- Trust README commands over blind exploration
- If README points to subdirectory → read that subdirectory's README
- Prefer simple demos for sanity checks
- Use paper-specific commands for main experiments
- Capture all stdout/stderr

Report: what commands README mentioned, what you executed, results."""

        tools = [read_file, search_file, list_directory, execute_shell_command, execute_python_script]
        # Don't pass custom prompt - let ReAct use its default optimized for tool calling
        self.agent = create_react_agent(self.llm, tools=tools)

    def run_experiments(self, code_path: str, paper_experiments: Dict = None,
                        agent_context: str = "") -> Dict:
        """
        Run experiments from the repository using two-phase strategy.

        Phase 1: Sanity check (demo/test scripts)
        Phase 2: Main experiment (paper reproduction)

        Args:
            code_path: Path to repository
            paper_experiments: Expected experiments from paper (optional)
            agent_context: Context from previous agents (NEW!)

        Returns:
            Execution results with outputs and metrics from both phases
        """
        # Store context for use in experiment phases
        self.agent_context = agent_context
        print("\n" + "="*60)
        print("🔬 TWO-PHASE EXPERIMENT STRATEGY")
        print("="*60)

        # Phase 1: Sanity Check
        print("\n📋 PHASE 1: Sanity Check")
        print("-" * 60)
        sanity_result = self._run_sanity_check(code_path)

        if not sanity_result.get("execution_successful"):
            print("\n⚠️  Sanity check failed - skipping main experiment")
            return {
                **sanity_result,
                "sanity_check_passed": False,
                "main_experiment_run": False,
                "report": f"Sanity check failed:\n{sanity_result.get('report', '')}"
            }

        print("\n✅ Sanity check passed!")

        # Phase 2: Main Experiment
        print("\n" + "="*60)
        print("🚀 PHASE 2: Main Experiment (Paper Reproduction)")
        print("="*60)
        main_result = self._run_main_experiment(code_path, paper_experiments)

        # Combine results
        return {
            **main_result,
            "sanity_check_passed": True,
            "sanity_check_output": sanity_result.get("output", ""),
            "main_experiment_run": True,
            "report": f"SANITY CHECK:\n{sanity_result.get('report', '')}\n\n" +
                     f"MAIN EXPERIMENT:\n{main_result.get('report', '')}"
        }

    def _run_sanity_check(self, code_path: str) -> Dict:
        """Phase 1: Run quick sanity check (demo/test)."""
        task = f"""{self.system_prompt}

===== PHASE 1: SANITY CHECK =====

Task: Run a simple demo/test to verify the code works

Repository: {code_path}

===== YOUR WORKFLOW =====

STEP 1: Read root README.md
   read_file(file_path="{code_path}/README.md")

   Look for sections (PRIORITY ORDER):
   1. "Quickstart" / "Quick Start" (HIGHEST PRIORITY - simplest examples)
   2. "Getting Started" / "Usage" / "Examples"
   3. "Demo" / "Example" / "Test"

   Extract: Simple demo/test commands

STEP 2: Check if README mentions nested instructions
   If README says things like:
   - "See instructions in examples/X/"
   - "Refer to X/README.md"
   - "Follow examples in X/"

   → Read that nested README: read_file(file_path="{code_path}/X/README.md")
   → Look for Quickstart or simple demo commands there

STEP 3: Execute the simplest command you found
   Prefer: Quickstart > Demo > Test > Example

   Example: README says "python examples/demo.py"
   → execute_shell_command(command="python examples/demo.py", cwd="{code_path}", timeout=180)

   If command needs to run from subdirectory (common with nested READMEs):
   → execute_shell_command(command="python demo.py", cwd="{code_path}/examples/X", timeout=180)

STEP 4: Fallback if no README commands
   Only if README has NO commands at all:
   1. Look for demo.py, test.py, example.py (root level)
   2. Look in examples/, tests/ directories
   3. Report "No simple demo found" if nothing exists

STEP 5: Report results
   - What README(s) you read (root + nested if any)
   - What command you found/ran
   - Execution output
   - Success or failure

===== CRITICAL RULES =====
- PRIORITIZE "Quickstart" sections - they're the simplest!
- If README mentions subdirectory with instructions → READ THAT README TOO
- Keep it simple: prefer single python script
- NO distributed training (no mpirun, torch.distributed)
- Timeout: 3 minutes max

Report: READMEs consulted + command executed + result."""

        messages = [HumanMessage(content=task)]
        callback = LoggingCallbackHandler(verbose=True)

        try:
            result = self.agent.invoke(
                {"messages": messages},
                config={"recursion_limit": 30, "callbacks": [callback]}
            )
        except Exception as e:
            print(f"\n❌ Sanity check failed: {e}")
            result = {"messages": [], "error": str(e)}

        return self._parse_experiment_result(result)

    def _run_main_experiment(self, code_path: str, paper_experiments: Dict = None) -> Dict:
        """Phase 2: Run main experiment for paper reproduction."""
        experiment_info = f"Paper experiments: {paper_experiments}" if paper_experiments else "Find experiments that match the paper's claims"

        # Add context from previous agents
        context_note = ""
        if hasattr(self, 'agent_context') and self.agent_context:
            context_note = f"\n\nContext from paper analysis:\n{self.agent_context}\n"

        # Build resource-aware instructions
        resource_instructions = self._get_resource_aware_instructions()

        task = f"""{self.system_prompt}

===== PHASE 2: MAIN EXPERIMENT - Reproduce Paper Results =====

Repository: {code_path}

Paper Information:
{experiment_info}{context_note}

SYSTEM RESOURCES:
{get_resource_summary(self.resources)}

EXECUTION STRATEGY: {self.experiment_strategy.upper()}
{resource_instructions}

===== YOUR WORKFLOW =====

STEP 1: Read root README.md for experiment commands
   read_file(file_path="{code_path}/README.md")

   Look for sections that match the paper:
   - "Reproduce Results" / "Reproduction" / "Paper Results"
   - "Training" / "Evaluation" / "Experiments"
   - Commands that mention the paper's dataset/model
   - Examples that match paper context above

   Extract: Commands to reproduce paper results

STEP 1b: Follow nested README references if mentioned
   If README mentions subdirectories with instructions:
   - "See instructions in examples/X/" → read examples/X/README.md
   - "Refer to X/ for reproduction" → read X/README.md

   Read nested READMEs to find actual experiment commands
   Nested READMEs often have the detailed reproduction steps!

STEP 2: Match README commands to paper
   Compare:
   - Dataset mentioned in README vs paper (e.g., "SQuAD", "CIFAR-10")
   - Model/architecture (e.g., "GPT-2", "RoBERTa")
   - Experiment type (training, evaluation, fine-tuning)

   Find the command that best matches the paper's experiments

STEP 3: Execute the matching command
   Adapt based on resource strategy:
   {resource_instructions}

   execute_shell_command(command="<readme_command>", cwd="{code_path}", timeout=600)

STEP 4: If README doesn't specify → Search for training scripts
   Fallback:
   - Look for train.py, main.py, run_experiments.py
   - Check --help for arguments
   - Use minimal configs

STEP 5: Report results
   - README command you found (if any)
   - What you executed
   - Dataset/model used
   - Results/metrics obtained

===== CRITICAL =====
- Read README FIRST - it often has exact reproduction commands!
- Match README commands to paper's dataset/model
- Follow resource strategy to avoid expensive experiments
- Report the exact command run + all output

Report: README command + execution details + results."""

        messages = [HumanMessage(content=task)]
        callback = LoggingCallbackHandler(verbose=True)

        try:
            result = self.agent.invoke(
                {"messages": messages},
                config={"recursion_limit": 40, "callbacks": [callback]}
            )
        except Exception as e:
            print(f"\n❌ Main experiment failed: {e}")
            result = {"messages": [], "error": str(e)}

        return self._parse_experiment_result(result)

    def _get_resource_aware_instructions(self) -> str:
        """Generate resource-aware experiment instructions."""
        strategy = self.experiment_strategy

        if strategy == "all_experiments":
            return """
🚀 HIGH RESOURCES - Run ALL experiments from paper:
- Execute ALL experiment configurations mentioned in the paper
- Use full datasets (ImageNet, WMT, etc. are OK)
- Run with full epochs/steps as specified in paper
- Use distributed training if paper used it
- Multi-GPU training is allowed
- Goal: Reproduce ALL results from paper

Execution order:
1. Start with main/flagship experiment
2. Then run secondary experiments
3. Then run ablation studies if mentioned
"""

        elif strategy == "main_experiment":
            return """
⚙️  MEDIUM RESOURCES - Run MAIN experiment with resource limits:
- Execute the PRIMARY/MAIN experiment from paper
- Use smaller dataset variants when available (CIFAR-10 instead of ImageNet)
- Limit epochs: use --epochs 3-5 (or --max_steps to limit)
- Single GPU training preferred
- Skip very expensive experiments
- Goal: Reproduce CORE claim of paper

If full experiment is too expensive:
- Use abbreviated versions
- Use subset of data (--max_samples)
- Reduce model size if configs available
"""

        else:  # minimal_experiment
            return """
⚠️  LOW RESOURCES - Run MINIMAL experiment for verification:
- Find the SIMPLEST experiment that validates the core idea
- Use smallest dataset (MNIST, toy datasets)
- Minimal epochs (--epochs 1 or --max_steps 100)
- Smallest model variant
- NO distributed training
- NO large datasets (skip ImageNet, WMT, etc.)
- Goal: Verify setup works and approach is sound

Priority:
1. Look for "quick" or "demo" experiments
2. Use smallest dataset mentioned
3. If no small variant, SKIP expensive experiments
4. Report: "Skipped due to resource constraints"
"""

    def _parse_experiment_result(self, result: Dict) -> Dict:
        """Extract experiment results from agent output."""
        messages = result.get("messages", [])

        experiment_info = {
            "readme_commands_found": [],  # NEW: Commands extracted from README
            "executed_command": "",  # NEW: Actual command that was run
            "used_readme": False,  # NEW: Did agent use README command?
            "scripts_found": False,
            "execution_attempted": False,
            "execution_successful": False,
            "output": "",
            "metrics": {},
            "errors": [],
            "report": ""
        }

        # Handle agent errors
        if "error" in result:
            experiment_info["errors"].append(result["error"])
            experiment_info["report"] = f"Experiment execution failed: {result['error']}"
            return experiment_info

        all_messages = []
        full_output = []

        for msg in messages:
            if hasattr(msg, 'content') and msg.content:
                content = normalize_message_content(msg.content)
                all_messages.append(content)
                content_lower = content.lower()

                # Extract README commands mentioned
                if "readme" in content_lower and ("command" in content_lower or "run" in content_lower or "python" in content_lower):
                    # Try to extract command snippets
                    import re
                    commands = re.findall(r'python\s+[\w/\.\-]+(?:\s+--[\w\-]+\s+[\w\.\-]+)*', content, re.IGNORECASE)
                    experiment_info["readme_commands_found"].extend(commands[:3])  # Limit to 3
                    if commands:
                        experiment_info["used_readme"] = True

                # Check for script discovery
                if any(kw in content_lower for kw in [".py", "train", "demo", "test", "eval", "main"]):
                    experiment_info["scripts_found"] = True

                # Check for execution and extract command
                if any(kw in content_lower for kw in ["execute", "running", "python", "stdout", "stderr"]):
                    experiment_info["execution_attempted"] = True
                    full_output.append(content)

                    # Try to extract the executed command
                    if "command:" in content_lower or "executing" in content_lower:
                        import re
                        cmd_match = re.search(r'(?:command|executing):\s*([^\n]+)', content, re.IGNORECASE)
                        if cmd_match:
                            experiment_info["executed_command"] = cmd_match.group(1).strip()[:200]

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
        if all_messages:
            experiment_info["report"] = all_messages[-1] if all_messages else ""

        return experiment_info
