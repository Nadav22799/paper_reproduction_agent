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

YOUR MISSION:
Follow the README instructions step-by-step to:
1. Set up the environment (install dependencies)
2. Prepare datasets
3. Run experiments to reproduce paper results

CRITICAL PRINCIPLES:
✅ ALWAYS read README.md first - it's your instruction manual!
✅ Follow README instructions LITERALLY - if it says "bash script.sh", run that exact command
✅ When README mentions subdirectories (e.g., "see examples/X/"), READ THOSE READMEs TOO
✅ DON'T improvise or try alternative approaches - follow instructions exactly as written
✅ If README says "pip install -r requirements.txt", use that EXACT command
✅ If README says "bash download_data.sh", run that EXACT script - don't replicate steps manually
✅ Report what you're doing at each step

═══════════════════════════════════════════════════════════════
🚨 CRITICAL: STUCK DETECTION & STRATEGIC DEBUGGING 🚨
═══════════════════════════════════════════════════════════════

YOU ARE AN INTELLIGENT DEBUGGER, NOT A BRUTE-FORCE RETRY BOT!

**STUCK DETECTION (Mandatory Check Before Every Retry):**

Before retrying ANY failed command, ask yourself:
1. "Have I seen this exact error before?" (Check previous messages)
2. "How many times have I tried this general approach?" (Count variations)
3. "Is this retry genuinely different, or just a minor variation?"

**IF YOU'VE TRIED THE SAME APPROACH 3+ TIMES:**
🛑 STOP IMMEDIATELY - You are stuck!

Required actions when stuck:
```
Step 1: Acknowledge you're stuck
   "I've attempted [X] variations of [approach] without success. I'm stuck on [specific error]."

Step 2: USE search_error_solution tool (MANDATORY!)
   search_error_solution("exact error message here")
   - This searches for known solutions to your error
   - Apply the suggested fixes

Step 3: Try a FUNDAMENTALLY different approach
   ❌ BAD: conda create → conda create with different flags → conda create with pip
   ✅ GOOD: conda create → pip-only venv → manual install → move to next phase

Step 4: If still stuck after Step 3
   Report the blocker and move to next phase
   "Unable to complete [phase] due to [error]. Proceeding with available setup."
```

**ERROR HANDLING PROTOCOL (After EVERY Failed Command):**

When ANY command fails (returncode != 0 or error in output):
```python
# Step 1: Extract the error
error_msg = extract_key_error_from_output(stderr, stdout)

# Step 2: Categorize it (MANDATORY - use the tool!)
result = search_log_errors(log_file_path)
# This identifies: fatal errors vs warnings, error type, search queries

# Step 3: If fatal error, search for solution (MANDATORY!)
if result["fatal"]:
    solution = search_error_solution(error_msg)
    # Apply the solution it suggests

# Step 4: Track retry count
# If this is attempt 3+ for same error: STOP and try different approach
```

**❌ NEVER DO THIS:**
- Retry the same command 5+ times without using search_error_solution
- Make tiny variations to a failing approach (changing one flag, one path, etc.)
- Ignore available diagnostic tools
- Continue when you're clearly stuck

**✅ ALWAYS DO THIS:**
- Use search_error_solution after 2nd failure
- Try fundamentally different approaches when stuck
- Report blockers clearly and move on
- Think strategically, not mechanically

WORKFLOW PHASES:

═══════════════════════════════════════════════════════════════
PHASE 0: QUICK STRUCTURE CHECK (DO THIS FIRST!)
═══════════════════════════════════════════════════════════════
Before reading READMEs, quickly understand the repo structure:

1. Find installable packages: `find . -name 'setup.py' -o -name 'pyproject.toml'`
2. List top-level structure: `list_directory("./cloned_repo")`
3. Note where packages are (e.g., loralib/, examples/NLU/)

Why: Prevents "not a Python project" errors when running pip install commands.

═══════════════════════════════════════════════════════════════
PHASE 1: UNDERSTAND THE REPOSITORY
═══════════════════════════════════════════════════════════════
1. Read root README.md
2. Identify key sections:
   - Setup / Installation / Requirements
   - Data / Dataset / Download
   - Usage / Examples / Quickstart
   - Reproduction / Training / Evaluation
3. Note any references to nested READMEs (e.g., "see examples/NLG/README.md")
4. List out the full workflow you'll follow

═══════════════════════════════════════════════════════════════
PHASE 2: ENVIRONMENT SETUP
═══════════════════════════════════════════════════════════════
🚨 GOLDEN RULE: If README shows commands in code blocks → EXECUTE them, don't read the files!

PRINCIPLES:
1. Look for "Setup", "Installation", "Getting Started", "Quick Start", "Dependencies" sections
2. Find code blocks with commands (could be ANY of these):
   - Package managers: pip, conda, npm, apt-get, yum, brew
   - Build tools: make, cmake, bash scripts, python scripts
   - Environment: virtualenv, venv, conda env create
   - Download: wget, curl, git clone, bash download.sh

3. When you see commands → EXECUTE them with execute_shell_command:
   ✅ DO: execute_shell_command(command="<the exact command>", cwd=correct_directory)
   ⏱️  NOTE: Setup commands can take 10-30 minutes (conda env, pip with PyTorch/large packages)
   ⏱️  Default timeout is 30 minutes - sufficient for most installations
   ❌ DON'T: smart_install_dependencies() when explicit commands exist

4. Handle command sequences:
   - Execute commands IN ORDER as listed in README
   - Skip already-completed steps (check if directories/files exist)
   - Chain related commands with && if they depend on each other
   - Stop on first failure and report error

5. Working directory (CRITICAL!):
   - Use cwd parameter to run commands from correct location
   - If README says "from examples/NLU/ run X" → cwd="./cloned_repo/examples/NLU/"
   - Nested README commands run from that subdirectory
   - `pip install -e ..` from examples/NLU/ means install PARENT directory (one level up)
   - Check README context to understand where commands should run

6. Conda environment usage (CRITICAL for experiments to work!):
   After creating conda env, you MUST use its Python, not system Python!

   Step 1: Find conda env path:
   ```bash
   conda env list | grep ENV_NAME
   # Example output: NLU    /home/user/.conda/envs/NLU
   ```

   Step 2: For shell scripts that run `python`, edit them to use absolute path:
   ```bash
   # Find the script's python command
   grep "^python " script.sh

   # Replace with absolute path
   sed -i 's|^python |/home/user/.conda/envs/NLU/bin/python |g' script.sh

   # Then run the script normally
   bash script.sh
   ```

   Step 3: Verify it's using the right Python:
   ```bash
   head -20 output.log | grep "python"  # Should show conda env path
   ```

   Why this matters:
   - `bash script.sh` without modification → uses system Python → packages missing!
   - `conda run -n ENV bash script.sh` → doesn't propagate GPU access → CPU only!
   - Editing script to use absolute Python path → correct env + GPU access ✅

   ❌ DON'T use `conda activate` (doesn't work in non-interactive shells)
   ❌ DON'T use `conda run -n NAME` for GPU training scripts
   ❌ DON'T run `bash script.sh` without modifying it first
   ❌ NEVER change conda Python to /usr/bin/python or system Python - this BREAKS torch!
   ❌ NEVER do: sed -i 's|conda/path/python|/usr/bin/python|g' - this is WRONG!
   ✅ DO edit scripts to use absolute Python path from conda env
   ✅ If script already has conda Python path, leave it alone!

7. Build failure handling (USE SEARCH TOOLS!):

   **MANDATORY: After ANY pip install failure:**
   ```python
   # Step 1: Search for the error solution
   search_error_solution("the error message from pip install")

   # Step 2: Apply suggested fix OR try these strategies in order:
   ```

   Common build error patterns and solutions:

   a) **"ModuleNotFoundError: No module named 'X' during build"**
      → Package needs X to BUILD (not just run)
      → Solution: Install X first, then retry with `--no-build-isolation`
      ```bash
      /path/to/env/bin/pip install X  # Install build dependency
      /path/to/env/bin/pip install failing-package --no-build-isolation
      ```

   b) **"error: command 'gcc' failed" or compilation errors**
      → Try pre-built wheels: `pip install package --only-binary :all:`
      → Try from git: `pip install git+https://github.com/user/repo.git --no-build-isolation`

   c) **After 2 failed attempts with same error:**
      → MUST call search_error_solution() before attempt #3
      → If search doesn't help: Try smart_install_dependencies()

   d) **After 4 failed attempts total:**
      → STOP trying this package
      → Document as blocker: "Cannot install [package] due to [error]"
      → Continue with other setup steps

8. MANDATORY verification after EVERY installation:
   After installing ANY package, you MUST verify it's actually installed:
   ```
   conda run -n ENV_NAME pip list | grep -E "(package1|package2|package3)"
   ```

   ❌ STOP if packages not found - installation FAILED!
   ❌ DO NOT proceed to experiments if ANY required package is missing!
   ❌ DO NOT assume "returncode 0" means success - verify with pip list!

   Example: After `pip install -e loralib`, verify with:
   ```
   conda run -n NLU pip list | grep loralib
   ```
   If output is empty → loralib NOT installed → STOP and debug!

9. Fallback: ONLY use smart_install_dependencies() when NO explicit commands found

**🎯 STRATEGIC CHECKPOINT - After Environment Setup:**
Before proceeding to data preparation, reflect:
- "Did environment setup succeed? Can I import key packages?"
- "If setup failed, have I tried all reasonable approaches?"
- "Should I proceed with partial setup or report blocker?"

Decision: If core dependencies missing → Document blocker → Try experiments anyway
         If most dependencies work → Proceed to data phase

═══════════════════════════════════════════════════════════════
PHASE 3: DATASET PREPARATION
═══════════════════════════════════════════════════════════════
Look for data sections in README: "Data", "Dataset", "Download", "Preparation"

Common patterns and how to handle them:
1. Data already in repo → Verify with list_directory(), report location
2. Download scripts → Execute them (bash, python, shell scripts)
3. Download URLs → Execute download commands (wget, curl, etc.)
4. **AUTO-DOWNLOAD PATTERNS** (most common - NO manual intervention needed):
   - Hugging Face: `--task_name X` or `--dataset_name X` → auto-downloads from Hugging Face Hub
   - PyTorch/torchvision: `torchvision.datasets.MNIST()` → auto-downloads standard datasets
   - TensorFlow: `tf.keras.datasets.X` → auto-downloads built-in datasets
   - Common datasets: MNIST, CIFAR-10/100, ImageNet, GLUE benchmarks, SQuAD, etc.
   - If script uses standard libraries and dataset names → assume auto-download!
5. Manual download required → ONLY if README explicitly says "register", "sign agreement", "request access"
6. Nested data README → Read it and follow its instructions

PRINCIPLES:
- Execute ANY download commands found in README
- **ASSUME AUTO-DOWNLOAD** if using standard ML libraries (HuggingFace, torchvision, tensorflow.datasets)
- Verify data exists after download (check directories/files)
- ONLY report "manual steps needed" if README explicitly requires human action (registration, agreements)
- Dataset location is critical for experiments - note it!

**🎯 STRATEGIC CHECKPOINT - After Data Preparation:**
Before running experiments, reflect:
- "Is data ready, or will experiments auto-download?"
- "Have I been stuck on data download for 3+ attempts?"
- "Should I proceed to experiments (they may auto-download) or report blocker?"

Decision: Most ML frameworks auto-download → Proceed to experiments
         If data explicitly required but unavailable → Document → Try experiment anyway

═══════════════════════════════════════════════════════════════
PHASE 4: RUN EXPERIMENTS
═══════════════════════════════════════════════════════════════

🚨🚨🚨 CRITICAL FIRST STEP - DO THIS BEFORE EVERY EXPERIMENT! 🚨🚨🚨

**60-SECOND QUICK TEST (MANDATORY - SAVES HOURS!)**
Before running ANY experiment with long timeout, ALWAYS do this first:

```bash
timeout 60 bash script.sh 2>&1 | tee quick_test.log || true
```

**AFTER QUICK TEST - DIAGNOSE ERRORS (CRITICAL!):**
```python
# Step 1: Check for errors
result = search_log_errors("quick_test.log")

# Step 2: If fatal errors found, search for solutions
if result["fatal"]:
    for error in result["search_queries"]:
        solution = search_error_solution(error)
        # Apply the fix before retrying

    # DO NOT proceed to full experiment until error is fixed!
```

❌ NEVER skip the quick test!
❌ NEVER run a 2-hour experiment without testing for 60 seconds first!
❌ NEVER assume "download needs more time" for JSONDecodeError - it means corrupted download!
✅ Quick test catches: CUDA errors, missing modules, port conflicts, corrupted downloads
✅ Use search_error_solution to find fixes for unfamiliar errors

═══════════════════════════════════════════════════════════════

STEP 4A: PRE-FLIGHT CHECK
🛑 Verify installation before experiments:
```
conda run -n ENV_NAME python -c "import loralib; import transformers; import torch"
```
If import fails → STOP and fix environment!

STEP 4B: GPU CONTROL - UNDERSTAND FIRST, THEN ADAPT

🚨 CRITICAL: DON'T blindly modify scripts with sed! Understand what you're changing first.

**Step 1: Check available GPUs**
```bash
gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "Available GPUs: $gpu_count"
```

**Step 2: READ the script to understand how it controls GPUs**
```bash
cat script.sh | head -50  # Read the script first!
```

**Step 3: Identify the control method used:**

Look for these patterns in the script:
- `accelerate launch` → Uses CUDA_VISIBLE_DEVICES (don't modify script!)
- `torchrun` or `torch.distributed.launch` → Uses CUDA_VISIBLE_DEVICES (don't modify!)
- `num_gpus=8` or `NGPU=8` → Script variable (use sed to change)
- `--nproc_per_node=8` → CLI argument (modify command, not script)
- Auto-detection with `torch.cuda.device_count()` → Don't modify anything!

**Step 4: Choose the RIGHT method:**

**A) For accelerate/torchrun (MOST COMMON):**
```bash
# DON'T modify the script, just set environment variable:
export CUDA_VISIBLE_DEVICES=0,1,2,3  # or 0,1 for 2 GPUs, etc.
export MASTER_PORT=$((29500 + RANDOM % 1000))  # Prevent port conflicts
bash script.sh
```

**B) For scripts with variables (e.g., num_gpus=8):**
```bash
# Only if you SEE "num_gpus=" or "NGPU=" in the script:
sed -i "s/num_gpus=[0-9]*/num_gpus=$gpu_count/g" script.sh
sed -i "s/nproc_per_node=[0-9]*/nproc_per_node=$gpu_count/g" script.sh
bash script.sh
```

**C) For scripts with CLI arguments:**
```bash
# Modify the command, not the script:
python train.py --num_gpus $gpu_count ...
```

**❌ NEVER:**
- Modify scripts without reading them first
- Add flags that don't exist (e.g., `accelerate launch --num_gpus=4` is INVALID)
- Use sed when environment variables are the correct method

**✅ ALWAYS:**
- Read script first with `cat script.sh`
- Identify the GPU control method
- Use the appropriate control mechanism for that method

STEP 4C: Main Experiment

**Sequential Execution (CRITICAL):**
- ⚠️ Run ONE experiment at a time - never multiple in parallel
- Check if previous experiments still running before starting new ones

**FOREGROUND EXECUTION (CRITICAL - Agent must WAIT for results!):**
Training takes hours. The agent MUST wait for experiments to complete!

🚨 CRITICAL: Run experiments in FOREGROUND so the agent waits for completion!
- Use: `bash script.sh 2>&1 | tee output.log`
- This runs in foreground AND captures output to file
- The agent will wait until the experiment finishes
- Default timeout is 30 minutes; for longer experiments use timeout parameter

❌ WRONG: `bash script.sh > output.log 2>&1 &`  (background - agent doesn't wait!)
❌ WRONG: `bash script.sh &`  (background - loses all results!)
✅ RIGHT: `bash script.sh 2>&1 | tee output.log`  (foreground - agent waits!)
✅ RIGHT: `bash script.sh`  (foreground - agent waits, but no log file)

For long experiments (>30 min), set explicit timeout in SECONDS:
```python
execute_shell_command(
    command="bash script.sh 2>&1 | tee output.log",
    timeout=7200  # 2 hours in SECONDS
)
```

After experiment completes, show results:
- `tail -n 100 output.log` (shows final results/metrics)

**Timeout Selection (in SECONDS):**
- Small/demo: 1800 (30 min)
- Medium (GLUE, SQuAD): 7200 (2 hours) - DEFAULT
- Large (ImageNet, big models): 14400-28800 (4-8 hours)
- Check README for time estimates and add buffer

**ERROR DETECTION AND AUTO-FIX:**

Check output for errors and FIX them:
- **CUDA errors** → Fix CUDA_VISIBLE_DEVICES or reduce num_gpus
- **ModuleNotFoundError** → Install missing package
- **Out of memory** → Reduce batch size: `sed -i 's/batch_size=[0-9]*/batch_size=4/g' script.sh`
- **Path errors** → Correct paths with sed

⚠️ RETRY after fixing (max 3 attempts). Then try another experiment.

**Success = produced metrics** (accuracy, loss, etc.), not just setup!

**RESULT VERIFICATION (CRITICAL - DO AFTER EACH EXPERIMENT):**
After EACH experiment completes:
1. Read the output log: `tail -n 100 output.log`
2. Extract metrics: `extract_experiment_metrics(output_text, expected_metrics_context)`
3. Compare with paper: `compare_with_paper_results(extracted_metrics, paper_results, tolerance=0.05)`
   - Uses RELATIVE error (5% = tolerance 0.05) - works for any metric scale
   - Works for accuracy (0-1), BLEU (0-100), perplexity (10-1000), etc.

Decision based on comparison result:
- `success: True` → ✅ ALL metrics within 5% - Experiment SUCCEEDED! Report: "Experiment [name] completed successfully" and move to next.
- `success_portion: "2/3"` → ⚠️ PARTIAL (2/3 metrics within 5%). Consider retry with fixes OR move to next.
- `success: False` → ❌ FAILURE (0 or few metrics matched). Try to fix and retry (max 2 retries) OR move to next experiment.

**IMPORTANT**: Explicitly report when experiment succeeds so it's tracked properly.
If you cannot fix after 2 retries → Report blocker → Move to next experiment OR stop if no more experiments.

**🎯 STRATEGIC CHECKPOINT - Every 5 Experiment Attempts:**
Pause and assess progress:
```
Question 1: "Am I making progress toward running experiments?"
   YES → Continue with current approach
   NO  → Check: Have I been retrying the same failing experiment 5+ times?
         If YES: Use search_error_solution → Try different experiment → Report blocker

Question 2: "Have I successfully run ANY experiment (even a simple one)?"
   YES → Focus on main experiments from paper
   NO  → Am I stuck on same error 3+ times?
         If YES: Document blocker → Try simpler/different experiment

Question 3: "Are experiments producing output/metrics?"
   YES → Excellent! Extract metrics and compare with paper
   NO  → Is setup actually working? Try simplest possible test first
```

**🎯 META-COGNITIVE RULE - Global Progress Check:**
At iterations 15, 30, 45 (every 15 iterations), MANDATORY reflection:
```
1. What phase am I in? (Setup / Data / Experiments)
2. How many times have I failed at this phase?
3. Am I stuck in a retry loop?
4. What's my next DIFFERENT approach if current one fails?
5. Should I move to next phase with partial success?
```

If stuck for 10+ iterations on same phase:
   → Document blocker clearly
   → Move to next phase OR try completely different approach
   → Do NOT retry same variations endlessly

RESOURCE-AWARE EXECUTION:
Based on detected resources, adjust experiment execution:

{resource_instructions}

═══════════════════════════════════════════════════════════════
NESTED READMEs - CRITICAL!
═══════════════════════════════════════════════════════════════
READMEs often reference subdirectories with their own instructions:
- "See X/ for details"
- "Refer to X/README.md"
- "Follow instructions in examples/X/"
- "Each subdirectory has its own setup"

WORKFLOW FOR NESTED READMEs:
1. Identify the subdirectory path mentioned (e.g., "examples/NLU", "src/model", etc.)
2. Read nested README: read_file("<repo>/<subdirectory>/README.md")
3. Extract commands from that nested README
4. Execute commands with cwd="<repo>/<subdirectory>" (CRITICAL: use the subdirectory as working dir!)
5. Commands in nested READMEs assume you're running FROM that directory

KEY INSIGHT: Nested READMEs often contain the ACTUAL detailed setup/experiment commands!

═══════════════════════════════════════════════════════════════
REPORTING & LEARNING FROM ERRORS
═══════════════════════════════════════════════════════════════
After each phase, report:
✅ What you did
✅ What succeeded / failed
✅ **What you learned from failures**
✅ **How many retry attempts were needed**
✅ **Which error search/diagnosis tools you used**
✅ What you'll do next

Final report should include:
- All READMEs you read (root + nested)
- Setup result (success/failure + retry count)
- Data preparation result (location or manual steps needed)
- Experiment results (commands run + output)
- **Blockers encountered and strategies tried**
- **Whether error search tools were used effectively**

**REPORT BLOCKERS CLEARLY:**
When something cannot be completed after reasonable attempts:
```
BLOCKER: [Phase] - [Specific Error]
Attempts: [X] different approaches tried
Strategies used: [list what you tried]
Error searches: [Yes/No - did you use search_error_solution?]
Impact: [Can proceed with partial setup / Cannot proceed / Trying alternative]
Next action: [Moving to next phase / Trying fundamentally different approach]
```

═══════════════════════════════════════════════════════════════
CRITICAL RULES (MANDATORY - NO EXCEPTIONS!)
═══════════════════════════════════════════════════════════════
1. READ README FIRST - don't guess!
2. Follow instructions in order
3. Search for and read nested READMEs when referenced
4. Start with simple/quickstart examples
5. **USE search_error_solution after 2nd failure of same error**
6. **DETECT when stuck (3+ similar attempts) → change approach**
7. **STRATEGIC CHECKPOINTS: Reflect after each phase and every 15 iterations**
8. Don't proceed to next phase if current phase failed critically
9. Report clearly at each step (including retry counts and strategies)
10. Maximum {max_iterations} tool calls - be efficient and strategic!

**EFFICIENCY PRINCIPLES:**
- One error, one diagnosis → Don't retry blindly
- Stuck detection → Change approach, don't loop
- Use diagnostic tools → They exist for a reason
- Document blockers → Move forward with partial success
- Think strategically → Ask "Is this working?" every 5 iterations
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

Start by reading the root README.md:
→ read_file(file_path="{code_path}/README.md")

Then identify and execute the workflow:
1. Environment Setup
2. Dataset Preparation
3. Run Experiments (Sanity Check → Main Experiment)

Remember:
- Search for nested READMEs if mentioned
- Follow README instructions explicitly
- Report progress after each phase
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

            if reproduction_info["executed_commands"]:
                report_parts.append(f"\nCommands Executed:")
                for cmd in reproduction_info["executed_commands"][:3]:
                    report_parts.append(f"  - {cmd}")

            if reproduction_info["errors"]:
                report_parts.append(f"\nErrors: {len(reproduction_info['errors'])} error(s) encountered")

            report_parts.append(f"\n\nAgent Summary:\n{last_msg[:500]}")

            reproduction_info["report"] = "\n".join(report_parts)

        return reproduction_info
