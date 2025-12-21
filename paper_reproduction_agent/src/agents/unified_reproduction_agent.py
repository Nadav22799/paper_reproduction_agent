"""Unified Reproduction Agent - Single agent that follows README instructions to reproduce paper results.

This agent replaces three separate agents (environment_setup, dataset_manager, experiment_runner)
with a unified approach that:
1. Reads README files (root and nested)
2. Follows instructions step-by-step starting with quickstart
3. Handles setup, data preparation, and experiments in a coherent flow
4. Maintains context across all phases
"""

import os
from typing import Dict, List
from pathlib import Path
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from ..tools.code_execution_tools import (
    read_file,
    search_file,
    list_directory,
    execute_shell_command,
    execute_python_script,
    check_python_compatibility,
    smart_install_dependencies,
    search_log_errors,
    search_error_solution,
    extract_experiment_metrics,
    compare_with_paper_results,
    # New result discovery and verification tools
    discover_result_files,
    read_result_files,
    read_log_tail,
    verify_experiment_results,
    get_experiment_checkpoint_status,
    generate_comparison_report,
    # Smart extraction tools for custom formats
    smart_extract_results,
    align_and_compare_results,
)
from ..utils.llm_factory import create_llm
from ..utils.logging_callback import LoggingCallbackHandler
from ..utils.message_utils import normalize_message_content
from ..utils.resource_detector import (
    detect_system_resources,
    get_resource_summary,
    get_experiment_strategy
)
from ..utils.context_manager import ContextManager


class UnifiedReproductionAgent:
    """Agent that follows README instructions to reproduce paper results."""

    def __init__(self, llm=None, max_iterations=50):
        self.llm = llm or create_llm(temperature=0.1)
        self.max_iterations = max_iterations

        # Initialize context manager to prevent context explosion
        self.context_manager = ContextManager(
            max_context_chars=200000,  # 200K char limit (~50K tokens)
            sliding_window_size=3      # Keep last 3 tool interactions in detail
        )

        # Detect system resources
        self.resources = detect_system_resources()
        self.experiment_strategy = get_experiment_strategy(self.resources)

        print("\n" + "="*60)
        print(get_resource_summary(self.resources))
        print(f"   Experiment Strategy: {self.experiment_strategy.upper()}")
        print("="*60)

        self.system_prompt = """You are a README-following agent that reproduces paper results by following repository instructions.

═══════════════════════════════════════════════════════════════
CORE PRINCIPLES
═══════════════════════════════════════════════════════════════

1. README IS YOUR GUIDE - Read it first, follow it literally
2. EXECUTE, DON'T IMPROVISE - If README says "bash script.sh", run that exact command
3. FOLLOW NESTED READMES - When README references subdirectories, read those READMEs too
4. USE CORRECT WORKING DIRECTORY - Run commands from the directory the README expects
5. VERIFY INSTALLATIONS - After installing, confirm with `pip list | grep package_name`
6. USE DIAGNOSTIC TOOLS - When errors occur, use search_log_errors and search_error_solution

═══════════════════════════════════════════════════════════════
⚠️  CRITICAL: YOU ARE AN AUTOMATED BOT, NOT A HUMAN
═══════════════════════════════════════════════════════════════

You execute commands in isolated shell sessions - each command runs in a NEW shell.
This means `conda activate myenv` followed by `python script.py` will NOT work!

ENVIRONMENT ACTIVATION RULES:
1. NEVER run `conda activate` or `source activate` as a standalone command
   - It only affects that shell session, which ends immediately

2. ALWAYS combine activation with the command in ONE LINE:
   - `conda run -n myenv python script.py`  (PREFERRED - cleanest approach)
   - `source /path/to/conda/etc/profile.d/conda.sh && conda activate myenv && python script.py`
   - `bash -c "source activate myenv && python script.py"`

3. OR use ABSOLUTE PATHS to the environment's Python:
   - `/home/user/miniconda3/envs/myenv/bin/python script.py`
   - `$(conda info --base)/envs/myenv/bin/python script.py`

4. For pip in conda env:
   - `conda run -n myenv pip install package`
   - `/path/to/envs/myenv/bin/pip install package`

5. For bash scripts that need the env:
   - Modify the script: `sed -i '1a source activate myenv' script.sh`
   - Or run: `conda run -n myenv bash script.sh`

REMEMBER: Every execute_shell_command() starts fresh - no environment persists!

═══════════════════════════════════════════════════════════════
ERROR HANDLING STRATEGY
═══════════════════════════════════════════════════════════════

When something fails:
1. First failure → Read the error, try obvious fix
2. Second failure → Use search_error_solution("error message")
3. Third failure → Try fundamentally different approach (not minor variations)
4. After 3+ different approaches fail → Document blocker, move to next phase

CRITICAL: ERROR LOCATION ≠ FIX LOCATION
└─ Traceback shows WHERE code crashed, not WHERE to fix
└─ If error is in installed package → fix in YOUR project code, not the package

WHEN ERROR IS IN AN INSTALLED PACKAGE (site-packages/, conda env libs):
└─ NEVER modify installed package files directly
   - Not version controlled, lost on reinstall, breaks reproducibility
└─ Find workaround at PROJECT level instead:
   1. Environment variable to change behavior?
   2. Config option in the library?
   3. Monkey patch in main.py to override the problematic function?
   4. Pin to compatible version in requirements.txt?
└─ Think: "How can MY code prevent this crash before it reaches the library?"

Signs you're stuck (STOP and change approach):
- Retrying same command with minor flag changes
- Same error appearing repeatedly
- Going in circles without progress

═══════════════════════════════════════════════════════════════
WORKFLOW PHASES
═══════════════════════════════════════════════════════════════

PHASE 0: STRUCTURE CHECK
└─ Quick scan: find setup.py/pyproject.toml locations, list top-level directories

PHASE 1: UNDERSTAND REPOSITORY
└─ Read root README.md
└─ Identify: Installation, Data, Usage/Examples, Training/Evaluation sections
└─ Note any nested README references

PHASE 2: ENVIRONMENT SETUP
└─ Execute installation commands from README (pip install, conda create, etc.)
└─ NON-INTERACTIVE EXECUTION (critical!):
   - Add -y to conda commands: `conda create -n env python=3.9 -y`
   - Add -y to apt/yum: `apt-get install -y package`
   - Timeouts: conda/pip installs can take 10-30 min, use timeout=1800
└─ REMEMBER: You are a BOT - see "YOU ARE AN AUTOMATED BOT" section above!
   - Never run `conda activate` alone - combine with command or use absolute paths
   - Use `conda run -n envname command` for all commands in conda envs
   - Or use absolute path: `/path/to/envs/myenv/bin/python script.py`
└─ Verify installation: `conda run -n envname pip list | grep package`
└─ Fallback: use smart_install_dependencies() only if no explicit commands

PHASE 3: DATA PREPARATION
└─ Check README for data sections
└─ Common patterns:
   - Auto-download (HuggingFace, torchvision) → proceed to experiments
   - Download scripts → execute them
   - Manual download required → document and proceed

PHASE 4: RUN EXPERIMENTS
└─ **PRE-CHECK: Look for existing results/checkpoints FIRST!**
   - `get_experiment_checkpoint_status(repo_path)` - Check if previous run exists
   - `discover_result_files(repo_path)` - Check if results already saved
   - If results exist → SKIP to PHASE 5 (verification)
   - If checkpoints exist → Resume from checkpoint, don't restart
└─ Pre-flight: Verify imports work before running experiments
   - Use: `conda run -n envname python -c "import torch; print(torch.__version__)"`
└─ Quick test: Run with 60s timeout first to catch setup errors
   `timeout 60 conda run -n envname bash script.sh 2>&1 | tee quick_test.log || true`
└─ GPU adaptation: Check available GPUs, adapt scripts accordingly
   - For accelerate/torchrun: use CUDA_VISIBLE_DEVICES
   - For scripts with num_gpus variable: use sed to modify
└─ Run in FOREGROUND with env (critical!):
   - `conda run -n envname bash script.sh 2>&1 | tee output.log`
   - Or: `source /path/to/conda.sh && conda activate env && bash script.sh 2>&1 | tee output.log`
   - Never use & (background) - agent must wait for results
   - Set timeout for long experiments: timeout=7200 (2h), timeout=14400 (4h)
└─ After experiment completes → Proceed to PHASE 5

PHASE 5: VERIFY RESULTS (NEW - CRITICAL!)
└─ **Step 1: Discover result files FIRST (NOT the log!)**
   - `discover_result_files(repo_path)` → Find results.json, eval_results.json, etc.
└─ **Step 2: Extract metrics from result files**
   - `read_result_files(file_paths)` → Get accuracy, F1, loss values
└─ **Step 3: Check log tail ONLY for errors (last 20-30 lines)**
   - `read_log_tail(log_path, num_lines=30)` → Check completion status
   - Do NOT read entire log file!
└─ **Step 4: Compare with paper results**
   - `compare_with_paper_results(extracted_metrics, expected_results_str)`
└─ OR use all-in-one: `verify_experiment_results(repo_path, paper_expected_results, log_path)`

═══════════════════════════════════════════════════════════════
GPU CONTROL (Read script first, then adapt)
═══════════════════════════════════════════════════════════════

1. Check GPUs: nvidia-smi --query-gpu=index --format=csv,noheader | wc -l
2. Read script to identify control method (accelerate, torchrun, num_gpus variable)
3. Use appropriate method:
   - accelerate/torchrun → export CUDA_VISIBLE_DEVICES=0,1,2,3
   - Script variables → sed -i "s/num_gpus=[0-9]*/num_gpus=$count/g" script.sh

═══════════════════════════════════════════════════════════════
COMMON ISSUES & QUICK FIXES
═══════════════════════════════════════════════════════════════

| Issue | Quick Fix |
|-------|-----------|
| CUDA version mismatch | Adapt pytorch-cuda to match nvidia-smi output |
| Package version conflict | Try without version pin: `pip install package` |
| Build fails (gcc error) | Try: `pip install package --only-binary :all:` |
| Module not found after install | Wrong Python - use `conda run -n env` or absolute path |
| Conda activate doesn't work | You're a BOT! Use `conda run -n env cmd` instead |
| Env not persisting between cmds | Each cmd is new shell - use `conda run -n env cmd` |
| Out of memory (OOM) | Reduce batch_size, use gradient accumulation |
| Port already in use | `export MASTER_PORT=$((29500 + RANDOM % 1000))` |
| Download hangs/corrupts | Delete partial files, retry with longer timeout |
| Error in installed package | Monkey patch in file.py, DON'T edit the package |

═══════════════════════════════════════════════════════════════
TOOLS REFERENCE
═══════════════════════════════════════════════════════════════

| Tool                        | When to Use                              |
|-----------------------------|------------------------------------------|
| read_file                   | Read README, scripts, config files       |
| list_directory              | Explore repo structure                   |
| execute_shell_command       | Run installation/training commands       |
| search_log_errors           | Analyze log files for errors             |
| search_error_solution       | Find solutions for specific errors       |
| extract_experiment_metrics  | Parse results from output                |
| compare_with_paper_results  | Validate reproduction accuracy           |
|                             |                                          |
| **NEW RESULT VERIFICATION TOOLS:**                           |
| discover_result_files       | Find result files (JSON, CSV, TXT)       |
| read_result_files           | Extract metrics from result files        |
| read_log_tail               | Read last N lines of log (error check)   |
| verify_experiment_results   | Complete verification workflow           |
| get_experiment_checkpoint_status | Check if experiments can resume     |
| generate_comparison_report  | Create detailed report comparing results |

═══════════════════════════════════════════════════════════════
⚠️  CRITICAL: RESULT VERIFICATION WORKFLOW (NEW!)
═══════════════════════════════════════════════════════════════

AFTER running experiments, ALWAYS follow this verification order:

1. **FIRST: Search for result files (NOT the log!)**
   ```
   discover_result_files(repo_path="/path/to/repo")
   ```
   Look for: results.json, eval_results.json, metrics.csv, scores.txt

2. **SECOND: Extract metrics from result files**
   ```
   read_result_files(file_paths="/path/to/results.json,/path/to/eval.json")
   ```
   This extracts accuracy, F1, BLEU, loss, etc. from structured files.

3. **THIRD: Only check log TAIL for errors (last 20-30 lines)**
   ```
   read_log_tail(log_path="/path/to/training.log", num_lines=30)
   ```
   Do NOT read the entire log file - it wastes context!

4. **FOURTH: Compare with paper results**
   ```
   comparison = compare_with_paper_results(extracted_metrics=..., expected_results_str="accuracy: 94.5%")
   ```

5. **FIFTH: Generate comparison report** (saves to file)
   ```
   generate_comparison_report(repo_path, extracted_metrics, paper_results, comparison)
   ```
   Creates `reproduction_report.md` with detailed comparison table.

OR use the all-in-one tool:
   ```
   verify_experiment_results(repo_path, paper_expected_results, experiment_log)
   ```

COMPARISON FEATURES:
- **Fuzzy matching**: test_accuracy matches "accuracy", F1-score matches "f1"
- **Value normalization**: 94.5% and 0.945 are treated as equal
- **5% tolerance**: Values within 5% relative error are considered matching
- **Partial success**: Reports "3/4 metrics matched" even if not all match

⚠️  COMMON MISTAKES TO AVOID:
   - ❌ Reading the entire log file (wastes context, misses result files)
   - ❌ Ignoring result files saved by the experiment
   - ❌ Re-running experiments that already completed successfully
   - ❌ Not checking for checkpoints before starting experiments

✅ CORRECT WORKFLOW:
   - ✅ FIRST check for existing result files
   - ✅ ONLY read log tail (last 30 lines) for error checking
   - ✅ Use get_experiment_checkpoint_status to check for resume points
   - ✅ Resume from last checkpoint instead of starting over

═══════════════════════════════════════════════════════════════
RESUME FROM CHECKPOINT (NEW!)
═══════════════════════════════════════════════════════════════

Before running any experiment, check if it can be resumed:

1. Check checkpoint status:
   ```
   get_experiment_checkpoint_status(repo_path="/path/to/repo")
   ```

2. If checkpoints exist, modify the training command to resume:
   - HuggingFace: `--resume_from_checkpoint /path/to/checkpoint`
   - PyTorch: `--resume checkpoint.pt`
   - Custom: Check README for resume instructions

3. If no checkpoints but results exist → skip to verification

═══════════════════════════════════════════════════════════════
RESOURCE-AWARE EXECUTION
═══════════════════════════════════════════════════════════════

{resource_instructions}

═══════════════════════════════════════════════════════════════
PROGRESS CHECKPOINTS (Every 15 iterations)
═══════════════════════════════════════════════════════════════

Ask yourself:
1. What phase am I in? Am I making progress?
2. Am I stuck in a retry loop? (Same error 3+ times = stuck)
3. Should I move forward with partial success?

Decision: If stuck for 10+ iterations → document blocker → move to next phase

═══════════════════════════════════════════════════════════════
FINAL REPORT SHOULD INCLUDE
═══════════════════════════════════════════════════════════════

- READMEs consulted (root + nested)
- Setup result (success/failure + what was installed)
- Data preparation result (location or manual steps needed)
- Experiments run (commands + metrics obtained)
- **RESULT FILES FOUND** (where results were saved)
- **EXTRACTED METRICS** (from result files, not just logs)
- Comparison with paper results (% match)
- Any blockers encountered
- **CHECKPOINT STATUS** (for resuming if needed)

Maximum {max_iterations} tool calls - be efficient and strategic!
"""

        tools = [
            read_file,
            search_file,
            list_directory,
            execute_shell_command,
            execute_python_script,
            check_python_compatibility,
            smart_install_dependencies,
            search_log_errors,
            search_error_solution,
            extract_experiment_metrics,
            compare_with_paper_results,
            # New result discovery and verification tools
            discover_result_files,
            read_result_files,
            read_log_tail,
            verify_experiment_results,
            get_experiment_checkpoint_status,
            generate_comparison_report,
            # Smart extraction tools for custom formats (e.g., poly 0.3 0.001 92.32 $\pm$ nan)
            smart_extract_results,
            align_and_compare_results,
        ]

        # Use ReAct agent with native tool calling
        self.agent = create_react_agent(self.llm, tools=tools)

    def _get_resource_aware_instructions(self) -> str:
        """Generate resource-aware experiment instructions."""
        strategy = self.experiment_strategy

        # Common instructions for all strategies
        common_instructions = """
🔧 AUTOMATIC SCRIPT ADAPTATION (MANDATORY):
Before running ANY experiment script (.sh or .py), you MUST:

1. Check available resources:
   ```bash
   nvidia-smi --query-gpu=name --format=csv,noheader | wc -l  # Count GPUs
   ```
   Or check for CPU-only: `which nvidia-smi` (if not found → CPU only)

2. Read the script FIRST to check resource requirements:
   ```bash
   cat script_name.sh  # Check for: num_gpus, nproc_per_node, batch_size
   ```

3. Adapt script to YOUR available resources:
   **IMPORTANT: Follow the GPU CONTROL guide in STEP 4B above!**
   - First READ the script to understand GPU control method
   - Use CUDA_VISIBLE_DEVICES for accelerate/torchrun
   - Use sed ONLY for scripts with explicit variables
   - Reduce batch_size if needed to fit in available VRAM

4. GPU adaptation (see STEP 4B for full details):
   ```bash
   # Check available GPUs
   gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l || echo "0")

   # Read script first to understand control method
   cat script.sh | grep -E "accelerate|torchrun|num_gpus"

   # Then use appropriate method (CUDA_VISIBLE_DEVICES or sed)
   # See STEP 4B for decision tree
   ```

❌ NEVER run scripts requiring more GPUs than available
❌ NEVER modify scripts blindly with sed
❌ NEVER assume scripts auto-detect GPU count
✅ ALWAYS read script first to understand GPU control
✅ ALWAYS use correct method for the framework
"""

        if strategy == "all_experiments":
            return common_instructions + """
🚀 HIGH RESOURCES - Run full experiments:
- Execute full training/evaluation as specified in paper
- Use full datasets (ImageNet, WMT, etc. are OK)
- Run with full epochs/steps from paper
- Multi-GPU / distributed training allowed (after adapting num_gpus!)
- Goal: Complete reproduction of all results
"""

        elif strategy == "main_experiment":
            return common_instructions + """
⚙️  MEDIUM RESOURCES - Run main experiment with limits:
- Focus on PRIMARY experiment from paper
- Use smaller dataset variants when available (CIFAR-10 instead of ImageNet)
- Limit epochs: use --epochs 3-5 or --max_steps to reduce runtime
- Single GPU preferred (edit scripts to num_gpus=1)
- Skip very expensive experiments
- Goal: Reproduce core claim of paper
"""

        else:  # minimal_experiment
            return common_instructions + """
⚠️  LOW RESOURCES - Run minimal validation:
- Find SIMPLEST experiment that validates the idea
- Use smallest dataset (MNIST, toy data)
- Minimal epochs (--epochs 1 or --max_steps 100)
- Smallest model variant
- NO distributed training (edit scripts to remove torch.distributed.launch)
- NO large datasets (skip ImageNet, WMT, etc.)
- Goal: Verify setup works and approach is sound
"""

    def _extract_experiment_names_from_readme(self, code_path: str) -> List[str]:
        """Extract experiment names/directories from README using LLM."""
        try:
            readme_path = os.path.join(code_path, "README.md")
            if not os.path.exists(readme_path):
                return []

            with open(readme_path, 'r', encoding='utf-8') as f:
                readme_content = f.read()[:10000]  # First 10k chars

            prompt = f"""Analyze this README and extract ALL experiment names, benchmark names, or task names mentioned.
Look for:
- Directory names containing experiments (e.g., "examples/NLU", "experiments/text8", "scripts/lambada")
- Benchmark/task names (e.g., "GLUE", "SQuAD", "text8", "lambada", "ImageNet")
- Evaluation/training script names

Return ONLY a comma-separated list of experiment/task names (no explanations).
If multiple are found, list all. If none found, return "NONE".

README excerpt:
{readme_content}

Experiment names (comma-separated):"""

            response = self.llm.invoke(prompt)
            result = response.content.strip()

            if result == "NONE" or not result:
                return []

            # Parse comma-separated list
            experiments = [exp.strip() for exp in result.split(',')]
            # Clean up - remove "NONE", empty strings, and generic terms
            experiments = [exp for exp in experiments if exp and exp.upper() != "NONE"
                          and len(exp) > 1 and not exp.lower() in ['experiments', 'examples', 'tasks']]

            print(f"📋 Extracted experiments from README: {experiments}")
            return experiments

        except Exception as e:
            print(f"⚠️  Could not extract experiments from README: {e}")
            return []

    def reproduce(self, code_path: str, paper_context: str = "") -> Dict:
        """
        Follow README instructions to reproduce paper results.

        Args:
            code_path: Path to repository
            paper_context: Context from paper analysis (datasets, results to reproduce, etc.)

        Returns:
            Reproduction results with setup status, data status, and experiment results
        """
        resource_instructions = self._get_resource_aware_instructions()

        # Add context from paper if available
        paper_info = ""
        if paper_context:
            paper_info = f"""

═══════════════════════════════════════════════════════════════
PAPER CONTEXT (from analysis)
═══════════════════════════════════════════════════════════════
{paper_context}

Use this context to:
- Identify which experiments to run
- Match README commands to paper's datasets/models
- Prioritize experiments that reproduce paper's key claims
═══════════════════════════════════════════════════════════════
"""

        task = f"""{self.system_prompt.format(max_iterations=self.max_iterations, resource_instructions=resource_instructions)}

═══════════════════════════════════════════════════════════════
YOUR TASK: Reproduce paper results from repository
═══════════════════════════════════════════════════════════════
Repository path: {code_path}
{paper_info}

SYSTEM RESOURCES:
{get_resource_summary(self.resources)}
Experiment Strategy: {self.experiment_strategy.upper()}

═══════════════════════════════════════════════════════════════
BEGIN WORKFLOW
═══════════════════════════════════════════════════════════════

**IMPORTANT: Check for existing results FIRST!**
→ get_experiment_checkpoint_status(repo_path="{code_path}")
→ discover_result_files(repo_path="{code_path}")

If results already exist → Skip to verification (Phase 5)
If checkpoints exist → Resume from checkpoint

Otherwise, start by reading the root README.md:
→ read_file(file_path="{code_path}/README.md")

Then identify and execute the workflow:
1. Environment Setup
2. Dataset Preparation
3. Run Experiments (Sanity Check → Main Experiment)
4. **VERIFY RESULTS** (discover_result_files → read_result_files → compare_with_paper_results)

Remember:
- Search for nested READMEs if mentioned
- Follow README instructions explicitly
- Report progress after each phase
- **After experiments: Search for RESULT FILES first, not log files!**
- **Only read log TAIL (last 20-30 lines) for error checking**
- Stop if critical phase fails
"""

        messages = [HumanMessage(content=task)]
        callback = LoggingCallbackHandler(verbose=True)

        try:
            print("\n" + "="*60)
            print("🚀 STARTING UNIFIED REPRODUCTION WORKFLOW")
            print("="*60)

            # Use custom agent loop with context management
            result = self._run_agent_with_context_management(
                messages=messages,
                callback=callback
            )

        except Exception as e:
            print(f"\n❌ Unified reproduction failed: {e}")
            import traceback
            traceback.print_exc()
            result = {"messages": [], "error": str(e)}

        # Extract experiment names from README for dynamic detection
        experiment_names = self._extract_experiment_names_from_readme(code_path)

        return self._parse_reproduction_result(result, code_path, experiment_names)

    def _run_agent_with_context_management(self, messages: List, callback) -> Dict:
        """
        Run agent with context pruning to prevent explosion.

        This custom loop:
        1. Runs agent for small batches of iterations (5 at a time)
        2. Prunes context after each batch
        3. Monitors context size and shows statistics
        """
        batch_size = 15  # Run 15 iterations then prune (increased from 10)
        max_batches = self.max_iterations // batch_size

        current_messages = messages

        try:
            for batch in range(max_batches):
                print(f"\n📦 Batch {batch + 1}/{max_batches} (iterations {batch*batch_size + 1}-{(batch+1)*batch_size})")

                # Run agent for this batch
                result = self.agent.invoke(
                    {"messages": current_messages},
                    config={
                        "recursion_limit": batch_size * 4,  # Increased to 4x (now 60 instead of 30)
                        "callbacks": [callback]
                    }
                )

                # Check if agent finished
                result_messages = result.get("messages", [])
                if not result_messages:
                    print("   Agent returned no messages - stopping")
                    break

                # Check if agent is done (no more tool calls)
                last_msg = result_messages[-1] if result_messages else None
                has_tool_calls = (
                    hasattr(last_msg, 'tool_calls') and
                    last_msg.tool_calls and
                    len(last_msg.tool_calls) > 0
                )

                if not has_tool_calls:
                    # Check if LLM described a plan but didn't execute it
                    last_content = str(getattr(last_msg, 'content', ''))
                    planning_indicators = ['i will', 'let me', 'i\'ll', 'next step', 'going to', 'proceed with']
                    has_plan = any(indicator in last_content.lower() for indicator in planning_indicators)

                    if has_plan and len(last_content) > 100:
                        # LLM made a plan but didn't execute - prompt it to continue
                        print("   ⚠️ LLM described plan but didn't execute - prompting to continue...")
                        from langchain_core.messages import HumanMessage
                        continue_msg = HumanMessage(content="Please execute the plan you described. Use the appropriate tools to carry out the actions.")
                        result_messages.append(continue_msg)
                        current_messages = result_messages
                        continue

                    print("   ✅ Agent completed (no more tool calls)")
                    return result

                # Prune context after this batch
                if len(result_messages) > 10:
                    original_count = len(result_messages)
                    original_size = sum(len(str(getattr(m, 'content', ''))) for m in result_messages)

                    # Prune messages
                    pruned_messages = self.context_manager.prune_messages(result_messages)

                    pruned_count = len(pruned_messages)
                    pruned_size = sum(len(str(getattr(m, 'content', ''))) for m in pruned_messages)

                    print(f"\n   📊 Context Pruning:")
                    print(f"      Messages: {original_count} → {pruned_count} ({pruned_count/original_count*100:.1f}% kept)")
                    print(f"      Size: {original_size:,} → {pruned_size:,} chars ({pruned_size/original_size*100:.1f}% kept)")

                    # Warn if still over limit
                    if pruned_size > self.context_manager.max_context_chars:
                        print(f"      ⚠️  WARNING: Still over {self.context_manager.max_context_chars:,} char limit!")
                    else:
                        print(f"      ✅ Under {self.context_manager.max_context_chars:,} char limit")

                    current_messages = pruned_messages
                else:
                    current_messages = result_messages

            # Return final result
            return {"messages": current_messages}

        except Exception as e:
            print(f"\n❌ Agent execution error: {e}")
            raise

    def _parse_reproduction_result(self, result: Dict, code_path: str, experiment_names: List[str] = None) -> Dict:
        """Extract reproduction status from agent result."""
        messages = result.get("messages", [])

        reproduction_info = {
            # Environment Setup
            "setup_attempted": False,
            "setup_successful": False,
            "dependencies_installed": False,
            "python_compatible": True,

            # Dataset Preparation
            "data_attempted": False,
            "data_successful": False,
            "data_location": "",
            "data_manual_steps": "",

            # Experiments (NEW: Track multiple experiments)
            "sanity_check_attempted": False,
            "sanity_check_passed": False,
            "main_experiment_attempted": False,
            "main_experiment_successful": False,
            "executed_commands": [],
            "experiments_tried": [],  # NEW: List of all experiments attempted
            "experiments_succeeded": [],  # NEW: List of successful experiments
            "partial_success": False,  # NEW: True if at least one experiment worked

            # READMEs
            "readmes_consulted": [],
            "nested_readmes_found": [],

            # NEW: Result Verification
            "result_files_found": [],  # NEW: Result files discovered
            "result_files_used": [],   # NEW: Files used for verification
            "extracted_metrics": {},   # NEW: Metrics extracted from result files
            "verification_status": "", # NEW: "verified", "partial", "failed", "not_run"
            "checkpoints_found": [],   # NEW: Checkpoints for resume
            "can_resume": False,       # NEW: Whether experiment can be resumed

            # Output
            "experiment_output": "",
            "errors": [],
            "report": ""
        }

        # Handle agent errors
        if "error" in result:
            reproduction_info["errors"].append(result["error"])
            reproduction_info["report"] = f"Reproduction failed: {result['error']}"
            return reproduction_info

        all_messages = []
        for msg in messages:
            if hasattr(msg, 'content') and msg.content:
                content_str = normalize_message_content(msg.content)
                all_messages.append(content_str)

        full_output = "\n".join(all_messages).lower()

        # Track README usage
        for msg in all_messages:
            msg_lower = msg.lower()
            if "readme" in msg_lower:
                # Extract README paths mentioned
                if "root readme" in msg_lower or "reading readme.md" in msg_lower:
                    if "root README" not in reproduction_info["readmes_consulted"]:
                        reproduction_info["readmes_consulted"].append("root README")

                # Look for nested README mentions
                import re
                nested_patterns = re.findall(r'(\w+/[\w/]*readme\.md)', msg, re.IGNORECASE)
                for pattern in nested_patterns:
                    if pattern not in reproduction_info["nested_readmes_found"]:
                        reproduction_info["nested_readmes_found"].append(pattern)
                        reproduction_info["readmes_consulted"].append(pattern)

        # Check environment setup
        if any(kw in full_output for kw in ["check_python_compatibility", "smart_install_dependencies", "installing dependencies", "setup", "conda env", "virtualenv", "pip install"]):
            reproduction_info["setup_attempted"] = True

            # STRICT: Only mark successful if we see VERIFICATION evidence
            # Don't trust "returncode 0" alone - need to see actual package presence
            has_verification = any(kw in full_output for kw in [
                "import loralib",  # Import test passed
                "import transformers",  # Import test passed
                "loralib" and "pip list",  # Package appears in pip list
                "transformers" and "pip list",  # Package appears in pip list
                "verification successful",  # Explicit verification message
                "all packages installed and verified"  # Explicit verification message
            ])

            has_success_markers = any(kw in full_output for kw in [
                "successfully installed", "installation successful", "dependencies installed", "setup successful",
                "already exists", "already installed", "requirement already satisfied",
                "environment is ready", "activated"
            ])

            # Only mark as successful if we have BOTH success markers AND verification
            # OR if we explicitly see the packages in pip list
            if has_verification or (has_success_markers and "modulenotfounderror" not in full_output):
                reproduction_info["setup_successful"] = True
                reproduction_info["dependencies_installed"] = True

            # Even if success markers present, mark as failed if we see module not found errors
            if "modulenotfounderror" in full_output or "no module named" in full_output:
                reproduction_info["setup_successful"] = False
                reproduction_info["dependencies_installed"] = False

            if "incompatible" in full_output or "not compatible" in full_output:
                reproduction_info["python_compatible"] = False

        # Check dataset preparation
        if any(kw in full_output for kw in ["data", "dataset", "download"]):
            reproduction_info["data_attempted"] = True

            # Check for successful data preparation
            if any(kw in full_output for kw in ["data is in", "dataset found", "download successful", "data downloaded"]):
                reproduction_info["data_successful"] = True

                # Extract data location
                for msg in all_messages:
                    if any(kw in msg.lower() for kw in ["data is in", "data/", "datasets/", "examples/"]):
                        import re
                        paths = re.findall(r'[\w/\-\.]+/(?:data|examples|datasets)[\w/\-\.]*', msg, re.IGNORECASE)
                        if paths:
                            reproduction_info["data_location"] = paths[0]
                            break

            # Check for manual steps needed
            if any(kw in full_output for kw in ["manual", "you need to", "please download", "register"]):
                # Extract manual step instructions
                for msg in all_messages:
                    if any(kw in msg.lower() for kw in ["manual", "you need to", "please download"]):
                        reproduction_info["data_manual_steps"] = msg[:300]
                        break

        # Check experiments (NEW: Track multiple experiments)
        # STRICT: Only mark as succeeded if ACTUALLY EXECUTED with results/metrics
        # Use dynamically extracted experiment names from README
        if not experiment_names:
            experiment_names = []

        for msg in all_messages:
            msg_lower = msg.lower()

            # Check for each extracted experiment name
            for exp_name in experiment_names:
                exp_name_lower = exp_name.lower()
                if exp_name_lower in msg_lower:
                    # Verify it's in an execution context
                    context_window = msg_lower[max(0, msg_lower.find(exp_name_lower)-100):
                                               min(len(msg_lower), msg_lower.find(exp_name_lower)+100)]
                    if any(kw in context_window for kw in ["run", "execut", "train", "evaluat", "experiment", "test", "script"]):
                        if exp_name not in reproduction_info["experiments_tried"]:
                            reproduction_info["experiments_tried"].append(exp_name)

                    # STRICT: Only mark as succeeded if we see EXECUTION + METRICS/RESULTS
                    # Must have BOTH experiment execution indicators AND success/results
                    has_execution = any(kw in msg_lower for kw in [
                        "python -m torch", "bash ", "running experiment", "executing",
                        "training", "evaluation", "testing", "inference"
                    ])
                    has_results = any(kw in msg_lower for kw in [
                        "accuracy", "loss", "f1", "bleu", "rouge", "perplexity",
                        "metric", "score", "result", "performance"
                    ])
                    has_success = any(kw in msg_lower for kw in [
                        "completed successfully", "experiment.*success", "returncode: 0"
                    ])

                    # Only mark succeeded if BOTH execution happened AND got results/success
                    # NOT just environment setup success!
                    if (has_execution and (has_results or has_success)) and "setup" not in msg_lower:
                        if exp_name not in reproduction_info["experiments_succeeded"]:
                            reproduction_info["experiments_succeeded"].append(exp_name)

        if any(kw in full_output for kw in ["sanity check", "quickstart", "demo"]):
            reproduction_info["sanity_check_attempted"] = True

            if any(kw in full_output for kw in ["sanity check passed", "sanity.*success", "demo.*success"]):
                reproduction_info["sanity_check_passed"] = True

        # STRICT: Only detect main experiment if we see ACTUAL execution in command outputs
        # Look for actual command execution with results
        has_training_cmd = any(kw in full_output for kw in [
            '"returncode": 0',  # From execute_shell_command success
            'torch.distributed.launch',  # PyTorch distributed training
            'torchrun',  # PyTorch distributed training
            'python -m torch'  # PyTorch training
        ])

        if has_training_cmd:
            reproduction_info["main_experiment_attempted"] = True

            # Look for successful execution WITH actual completion
            # Must see BOTH return code 0 AND not just from failed attempts
            has_success_returncode = '"returncode": 0' in full_output and '"success": true' in full_output
            has_error = 'modulenotfounderror' in full_output or 'error:' in full_output.lower() or 'no module named' in full_output
            has_import_failure = 'import error' in full_output or 'importerror' in full_output

            # Only mark as successful if we see returncode 0 WITHOUT any import/module errors
            if has_success_returncode and not has_error and not has_import_failure:
                reproduction_info["main_experiment_successful"] = True

            # If we see import errors, mark setup as failed (environment broken)
            if has_import_failure or (has_error and "modulenotfounderror" in full_output):
                reproduction_info["setup_successful"] = False
                reproduction_info["dependencies_installed"] = False

        # NEW: Detect partial success (STRICT: only if experiments actually succeeded)
        if reproduction_info["experiments_succeeded"] or reproduction_info["main_experiment_successful"]:
            reproduction_info["partial_success"] = True

        # Extract executed commands
        for msg in all_messages:
            if any(kw in msg.lower() for kw in ["executing", "running", "command:"]):
                import re
                # Look for python commands
                commands = re.findall(r'python\s+[\w/\.\-]+(?:\s+--[\w\-]+\s+[\w\.\-]+)*', msg, re.IGNORECASE)
                reproduction_info["executed_commands"].extend(commands[:3])  # Limit to 3

        # Collect experiment output
        output_msgs = []
        for msg in all_messages:
            if any(kw in msg.lower() for kw in ["stdout", "output", "result"]):
                output_msgs.append(msg)
        reproduction_info["experiment_output"] = "\n".join(output_msgs[-3:])  # Last 3 output messages

        # Collect errors
        for msg in all_messages:
            if any(kw in msg.lower() for kw in ["error", "failed", "exception"]) and "no error" not in msg.lower():
                reproduction_info["errors"].append(msg[:200])

        # NEW: Detect result verification from messages
        for msg in all_messages:
            msg_lower = msg.lower()

            # Detect result files found
            if "discover_result_files" in msg_lower or "result files" in msg_lower:
                # Look for file paths
                file_matches = re.findall(r'[\w/\-\.]+\.(?:json|csv|txt)', msg, re.IGNORECASE)
                for f in file_matches[:5]:
                    if f not in reproduction_info["result_files_found"]:
                        reproduction_info["result_files_found"].append(f)

            # Detect metrics extraction
            if "extracted metrics" in msg_lower or "read_result_files" in msg_lower:
                # Look for metric patterns like "accuracy: 0.95"
                metric_matches = re.findall(r'(\w+)\s*[:=]\s*(\d+\.?\d*)', msg, re.IGNORECASE)
                for name, value in metric_matches[:10]:
                    try:
                        reproduction_info["extracted_metrics"][name.lower()] = float(value)
                    except ValueError:
                        pass

            # Detect checkpoint status
            if "checkpoint" in msg_lower:
                checkpoint_matches = re.findall(r'(checkpoint[_\-]?\d+|epoch[_\-]?\d+)', msg, re.IGNORECASE)
                for ckpt in checkpoint_matches[:5]:
                    if ckpt not in reproduction_info["checkpoints_found"]:
                        reproduction_info["checkpoints_found"].append(ckpt)
                if "can_resume" in msg_lower or "resume from" in msg_lower:
                    reproduction_info["can_resume"] = True

            # Detect verification status
            if "verify_experiment_results" in msg_lower or "compare_with_paper" in msg_lower:
                if "verified" in msg_lower and "success" in msg_lower:
                    reproduction_info["verification_status"] = "verified"
                elif "partial" in msg_lower:
                    reproduction_info["verification_status"] = "partial"
                elif "failed" in msg_lower or "mismatch" in msg_lower:
                    reproduction_info["verification_status"] = "failed"

        # Set verification status if not already set
        if not reproduction_info["verification_status"]:
            if reproduction_info["extracted_metrics"]:
                reproduction_info["verification_status"] = "partial"
            else:
                reproduction_info["verification_status"] = "not_run"

        # Generate final report
        if all_messages:
            # Use last message as summary, or create one
            last_msg = all_messages[-1]

            report_parts = []
            report_parts.append("=== Unified Reproduction Report ===\n")

            report_parts.append(f"READMEs Consulted: {', '.join(reproduction_info['readmes_consulted']) if reproduction_info['readmes_consulted'] else 'None'}")

            report_parts.append(f"\nEnvironment Setup: {'✅ Success' if reproduction_info['setup_successful'] else '❌ Failed' if reproduction_info['setup_attempted'] else '⚠️  Not attempted'}")

            report_parts.append(f"Dataset Preparation: {'✅ Success' if reproduction_info['data_successful'] else '⚠️  Manual steps needed' if reproduction_info['data_manual_steps'] else '❌ Failed' if reproduction_info['data_attempted'] else '⚠️  Not attempted'}")
            if reproduction_info["data_location"]:
                report_parts.append(f"  Data Location: {reproduction_info['data_location']}")

            report_parts.append(f"\nExperiments:")

            # NEW: Show multiple experiments if attempted
            if reproduction_info["experiments_tried"]:
                report_parts.append(f"  Experiments Tried: {', '.join(reproduction_info['experiments_tried'])}")
                if reproduction_info["experiments_succeeded"]:
                    report_parts.append(f"  ✅ Succeeded: {', '.join(reproduction_info['experiments_succeeded'])}")
                    report_parts.append(f"  🎯 PARTIAL SUCCESS - Some experiments reproduced!")

            report_parts.append(f"  Sanity Check: {'✅ Passed' if reproduction_info['sanity_check_passed'] else '❌ Failed' if reproduction_info['sanity_check_attempted'] else '⚠️  Not run'}")
            report_parts.append(f"  Main Experiment: {'✅ Success' if reproduction_info['main_experiment_successful'] else '❌ Failed' if reproduction_info['main_experiment_attempted'] else '⚠️  Not run'}")

            # NEW: Result Verification Section
            report_parts.append(f"\nResult Verification:")
            if reproduction_info["result_files_found"]:
                report_parts.append(f"  Result Files Found: {len(reproduction_info['result_files_found'])}")
                for f in reproduction_info["result_files_found"][:3]:
                    report_parts.append(f"    → {f}")

            if reproduction_info["extracted_metrics"]:
                report_parts.append(f"  Extracted Metrics: {len(reproduction_info['extracted_metrics'])}")
                for key, val in list(reproduction_info["extracted_metrics"].items())[:5]:
                    report_parts.append(f"    {key}: {val}")

            verification_status_display = {
                "verified": "✅ VERIFIED - Results match paper!",
                "partial": "⚠️  PARTIAL - Some metrics matched",
                "failed": "❌ FAILED - Results don't match paper",
                "not_run": "⚠️  NOT RUN - Verification not completed"
            }
            report_parts.append(f"  Verification: {verification_status_display.get(reproduction_info['verification_status'], '⚠️  Unknown')}")

            if reproduction_info["checkpoints_found"]:
                report_parts.append(f"\nCheckpoints Found: {len(reproduction_info['checkpoints_found'])}")
                report_parts.append(f"  Can Resume: {'Yes' if reproduction_info['can_resume'] else 'No'}")
                for ckpt in reproduction_info["checkpoints_found"][:3]:
                    report_parts.append(f"    → {ckpt}")

            if reproduction_info["executed_commands"]:
                report_parts.append(f"\nCommands Executed:")
                for cmd in reproduction_info["executed_commands"][:3]:
                    report_parts.append(f"  - {cmd}")

            if reproduction_info["errors"]:
                report_parts.append(f"\nErrors: {len(reproduction_info['errors'])} error(s) encountered")

            report_parts.append(f"\n\nAgent Summary:\n{last_msg[:500]}")

            reproduction_info["report"] = "\n".join(report_parts)

        return reproduction_info
