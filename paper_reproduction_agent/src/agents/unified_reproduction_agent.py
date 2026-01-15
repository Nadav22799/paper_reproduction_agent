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
    execute_python_code,  # NEW: Execute inline Python code
    create_python_file,   # Create Python scripts
    check_python_compatibility,
    # smart_install_dependencies,  # REMOVED: Conflicts with README-first approach
    search_error_solution,
    start_background_process,  # NEW: For long-running experiments
    wait_for_process,          # NEW: Smart blocking wait
    stop_process,              # NEW: Cleanup
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
from ..utils.hierarchical_context import HierarchicalContextManager


class UnifiedReproductionAgent:
    """Agent that follows README instructions to reproduce paper results."""

    def __init__(self, llm=None, max_iterations=50, hierarchical_context: HierarchicalContextManager = None, metrics_tracker=None):
        self.llm = llm or create_llm(temperature=0.1)
        self.max_iterations = max_iterations
        self.metrics_tracker = metrics_tracker

        # Initialize token-based context manager to prevent context explosion
        self.context_manager = ContextManager(
            max_tokens=50000,          # 50K token limit (accurate token counting)
            sliding_window_size=3      # Keep last 3 tool interactions in detail
        )

        # Hierarchical context for semantic retrieval (shared or create new)
        self.hierarchical_context = hierarchical_context or HierarchicalContextManager(
            model_name="gpt-4",
            hot_capacity=30,
            max_tokens=50000
        )

        # Detect system resources
        self.resources = detect_system_resources()
        self.experiment_strategy = get_experiment_strategy(self.resources)

        print("\n" + "="*60)
        print(get_resource_summary(self.resources))
        print(f"   Experiment Strategy: {self.experiment_strategy.upper()}")
        print("="*60)

        self.system_prompt = """You are an Expert AI Engineer specializing in reproducing machine learning research papers. Your task is to follow repository instructions precisely to validate published results.

═══════════════════════════════════════════════════════════════
CORE PRINCIPLES
═══════════════════════════════════════════════════════════════

1. README IS YOUR GUIDE - Read it first, follow it literally.
   (EXCEPTION: Skip environment creation steps - the setup is ALREADY DONE)
2. EXECUTE, DON'T IMPROVISE - If README says "bash script.sh", run that exact command.
3. FOLLOW NESTED READMES - Read sub-READMEs if referenced.
4. VERIFY INSTALLATIONS - Confirm with `pip list` or imports.
5. WRITE CODE TO SOLVE PROBLEMS - Use execute_python_code() for extraction/comparison.
6. RESOURCE AWARENESS - Check GPUs/CPU before running scripts and adapt commands.

═══════════════════════════════════════════════════════════════
⚠️  CRITICAL: ENVIRONMENT & SHELL RULES
═══════════════════════════════════════════════════════════════

You execute commands in ISOLATED shell sessions. `conda activate` ALONE DOES NOTHING.
You MUST use these ONE-LINE patterns for EVERY command:

1. **CONDA/MAMBA/MICROMAMBA**:
   - `[tool] run -n [env] python script.py` (PREFERRED)
   - `source /path/to/conda.sh && conda activate [env] && python script.py`

2. **VENV/UV**:
   - `./venv/bin/python script.py` (PREFERRED)
   - `source venv/bin/activate && python script.py`

3. **POETRY**:
   - `poetry run python script.py`

═══════════════════════════════════════════════════════════════
ERROR HANDLING & COMMON FIXES
═══════════════════════════════════════════════════════════════

1. **Unpinned Dependencies**: Pin versions in requirements/yaml to match paper date.
2. **Module Not Found**: You likely used the wrong python. USE THE ONE-LINE PATTERNS ABOVE.
3. **OOM**: Reduce batch_size or use gradient accumulation.
4. **Download Issues**: Delete partial files and retry.
5. **Package Errors**: Monkey patch in your script/main.py. DO NOT edit installed packages.

STRATEGY:
1. Read error -> Try obvious fix.
2. Search error -> `search_error_solution("error")`.
3. Stuck (3+ retries) -> Document blocker and move to next phase.

═══════════════════════════════════════════════════════════════
WORKFLOW PHASES
═══════════════════════════════════════════════════════════════

PHASE 0: STRUCTURE CHECK
└─ Locate setup.py, pyproject.toml, and top-level dirs.

PHASE 1: UNDERSTAND REPOSITORY & EXTEND CHECKLIST
└─ Read root README and nested READMEs.
└─ **EXTEND THE CHECKLIST (Critical Step)**:
   - Read `./cloned_repo/reproduction_checklist.md`.
   - Append `## Reproduction Workflow` with tasks derived from README:
     1. **Data Prep**: ONLY if explicit setup needed (e.g. `[ ] Download Data`).
     2. **Experiments**: ONE item per experiment found (e.g. `[ ] Run CIFAR-10`).
     3. **Verification**: MANDATORY `[ ] Compare generated results vs paper (Code-based)`.

PHASE 2: VERIFY ENVIRONMENT
└─ **STOP!** Do not re-create the environment.
└─ **Step 1**: Read `./cloned_repo/reproduction_checklist.md` to find `**Tool Used:**`.
└─ **Step 2**: If Environment Name is found, use it! (e.g. `[tool] run -n [env_name] ...`).
└─ **Step 3**: Verify imports using the ONE-LINE PATTERNS above.
└─ **Step 4**: Update checklist (mark setup/verify as done).

PHASE 3: DATA PREPARATION
└─ executes data download/prep scripts if required.

PHASE 4: RUN EXPERIMENTS
└─ **CHECK FIRST**: directory for existing results/checkpoints.
   - If results exist -> Skip to Phase 5.
   - If checkpoints exist -> Resume.
└─ **EXECUTE**: Run experiments using ONE-LINE patterns.
   - Run in FOREGROUND (no `&`).
   - Use `timeout` to prevent hangs.
   - Log output: `cmd 2>&1 | tee output.log`.

   ⚠️ EXPERIMENT EXECUTION RULE:
   - For ANY training, evaluation, or long script:
     1. **start_background_process**(cmd, log_file, cwd="path/to/repo")  <-- DON'T FORGET cwd!
     2. **IMMEDIATELY CALL**: **wait_for_process**(pid, log_file, timeout=604800)
     3. **DO NOT STOP** until you have called wait_for_process.
   - **NEVER** use `execute_shell_command` for training scripts - it will execute timeout!
   - **ALWAYS** use the background+wait pattern.

PHASE 5: VERIFY RESULTS (CODE-FIRST APPROACH)
└─ **DO NOT** just read logs. WRITE PYTHON CODE to:
   1. **Find** result files (JSON/CSV/Logs).
   2. **Parse** them to extract metrics.
   3. **Compare** against paper's expected values.
└─ **REPORT**: "X/Y metrics matched (Z% success)".
   - Success = Relative error < 5%.

═══════════════════════════════════════════════════════════════
RESOURCE-AWARE EXECUTION
═══════════════════════════════════════════════════════════════

{resource_instructions}

═══════════════════════════════════════════════════════════════
FINAL REPORT Requirements
═══════════════════════════════════════════════════════════════
- READMEs consulted.
- Setup Status (Tool used).
- Data Status.
- Experiments Run (Commands).
- **EXTRACTED METRICS** & Comparison (% match).
- Checkpoint Status.

Maximum {max_iterations} tool calls. Be efficient.
"""

        # Minimal tool set - LLM writes code for extraction/comparison
        tools = [
            # File operations
            read_file,
            search_file,
            list_directory,
            # Execution
            execute_shell_command,
            execute_python_code,      # PRIMARY: Write and run inline Python
            create_python_file,       # Save reusable scripts
            execute_python_script,    # Run saved scripts
            # Setup utilities
            check_python_compatibility,
            # smart_install_dependencies,  # REMOVED: Use README instructions instead
            search_error_solution,    # Gemini-powered error fixing
            start_background_process, # Async training
            wait_for_process,         # Smart blocking wait
            stop_process,             # Process control
        ]

        # Use ReAct agent with native tool calling
        self.agent = create_react_agent(self.llm, tools=tools)

    def _get_resource_aware_instructions(self, experiment_mode: str = "single", custom_experiments: List[str] = None) -> str:
        """Generate resource-aware experiment instructions."""
        strategy = self.experiment_strategy
        
        # Override based on user selection
        if experiment_mode == "all":
            strategy = "all_experiments"
        elif experiment_mode == "custom":
            strategy = "custom"
        elif experiment_mode == "single":
            # If user wants 1 experiment but we detected high resources, 
            # we should still restrict to main_experiment (single)
            strategy = "main_experiment"

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

        if strategy == "custom":
            exps = ", ".join(custom_experiments or ["specified by user"])
            return common_instructions + f"""
🎯 CUSTOM SELECTION - Run specific experiments:
- You MUST run ONLY the following experiments: {exps}
- Ignore other experiments mentioned in README unless required for your selection
- Goal: Reproduce specific selected results
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
            # Handle response content which might be a list or object
            content = response.content
            if isinstance(content, list):
                # If it's a list (e.g. from Anthropic/Gemini returning blocks), join them
                content = " ".join([str(c) for c in content])
            elif not isinstance(content, str):
                content = str(content)
                
            result = content.strip()

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

    def reproduce(self, code_path: str, paper_context: str = "", experiment_mode: str = "single", custom_experiments: List[str] = None) -> Dict:
        """
        Follow README instructions to reproduce paper results.

        Args:
            code_path: Path to repository
            paper_context: Context from paper analysis (datasets, results to reproduce, etc.)
            experiment_mode: 'single', 'all', or 'custom'
            custom_experiments: List of experiment names if mode is 'custom'

        Returns:
            Reproduction results with setup status, data status, and experiment results
        """
        resource_instructions = self._get_resource_aware_instructions(experiment_mode, custom_experiments)

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

        # Use replace() instead of format() to avoid conflicts with code example curly braces
        formatted_prompt = self.system_prompt.replace("{max_iterations}", str(self.max_iterations)).replace("{resource_instructions}", resource_instructions)
        task = f"""{formatted_prompt}

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

**IMPORTANT: Check for existing results/checkpoints FIRST using execute_python_code!**

```python
execute_python_code(code='''
from pathlib import Path
repo = Path("{code_path}")
results = list(repo.glob("**/*result*.json")) + list(repo.glob("**/*.csv"))
checkpoints = list(repo.glob("**/*.pt")) + list(repo.glob("**/*.pth"))
print(f"Results: {{len(results)}}, Checkpoints: {{len(checkpoints)}}")
for f in results[:3]: print(f"  → {{f}}")
''')
```

**SMART RESUME LOGIC:**
1. Check existing results vs paper's expected datasets (use execute_python_code to compare)
2. If ALL expected results exist → Skip to VERIFICATION (Phase 5)
3. If SOME results missing:
   - If checkpoints exist → Try to RESUME from checkpoint to complete missing experiments
   - Or write script to run ONLY missing experiments (e.g., if paper expects 13 datasets but only 9 found, run the 4 missing)
4. After completion (or if resume fails) → VERIFY all results with detailed success rate

Otherwise, start by reading the root README.md:
→ read_file(file_path="{code_path}/README.md")

Then identify and execute the workflow:
1. Environment Setup
2. Dataset Preparation
3. Run Experiments (Sanity Check → Main Experiment)
4. **VERIFY RESULTS** - Write Python code to extract and compare!

Remember:
- Search for nested READMEs if mentioned
- Follow README instructions explicitly
- Report progress after each phase
- **Write custom Python code to extract metrics from result files**
- **Write comparison code to check against paper results**
- Stop if critical phase fails
"""

        messages = [HumanMessage(content=task)]
        callback = LoggingCallbackHandler(verbose=True, metrics_tracker=self.metrics_tracker)

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
                
                # Handling empty/malformed responses from LLM which cause premature exit
                if not has_tool_calls and last_msg:
                    last_content = str(getattr(last_msg, 'content', '')).strip()
                    
                    # If content is empty or extremely short without tool calls, it's likely an error/confusion
                    if not last_content or len(last_content) < 5:
                        print("   ⚠️ LLM returned empty/short response without tool calls - prompting to continue...")
                        from langchain_core.messages import HumanMessage
                        
                        # Inspect recent history to provide better guidance
                        prev_tool_output = ""
                        if len(result_messages) >= 2:
                            prev_msg = result_messages[-2]
                            if hasattr(prev_msg, 'content'):
                                prev_tool_output = str(prev_msg.content)
                        
                        hint = "You returned an empty message. "
                        if "PID" in prev_tool_output and "success" in prev_tool_output:
                            hint += "You just started a background process. You MUST now call wait_for_process(pid=...) to wait for it to finish."
                        else:
                            hint += "Please continue with the reproduction task using appropriate tools."
                            
                        continue_msg = HumanMessage(content=hint)
                        result_messages.append(continue_msg)
                        current_messages = result_messages
                        continue

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
                    original_tokens = sum(
                        self.context_manager.count_tokens(str(getattr(m, 'content', '')))
                        for m in result_messages
                    )

                    # Prune messages
                    pruned_messages = self.context_manager.prune_messages(result_messages)

                    pruned_count = len(pruned_messages)
                    pruned_tokens = sum(
                        self.context_manager.count_tokens(str(getattr(m, 'content', '')))
                        for m in pruned_messages
                    )

                    print(f"\n   📊 Context Pruning:")
                    print(f"      Messages: {original_count} → {pruned_count} ({pruned_count/original_count*100:.1f}% kept)")
                    print(f"      Tokens: {original_tokens:,} → {pruned_tokens:,} ({pruned_tokens/max(1, original_tokens)*100:.1f}% kept)")

                    # Warn if still over limit
                    if pruned_tokens > self.context_manager.max_tokens:
                        print(f"      ⚠️  WARNING: Still over {self.context_manager.max_tokens:,} token limit!")
                    else:
                        print(f"      ✅ Under {self.context_manager.max_tokens:,} token limit")

                    # Store important context in hierarchical storage
                    self._store_batch_summary(result_messages, batch)

                    current_messages = pruned_messages
                else:
                    current_messages = result_messages

            # Return final result
            return {"messages": current_messages}

        except Exception as e:
            print(f"\n❌ Agent execution error: {e}")
            raise

    def _store_batch_summary(self, messages: List, batch_num: int):
        """
        Store important information from batch in hierarchical context.

        Extracts key results, errors, and decisions for semantic retrieval.
        """
        try:
            for msg in messages:
                content = str(getattr(msg, 'content', ''))
                content_lower = content.lower()

                # Skip empty or very short content
                if len(content) < 50:
                    continue

                # Store errors for future reference
                if any(kw in content_lower for kw in ['error', 'failed', 'exception', 'traceback']):
                    # Extract concise error summary
                    error_summary = content[:500] if len(content) > 500 else content
                    self.hierarchical_context.add(
                        content=f"[Batch {batch_num}] Error: {error_summary}",
                        source="reproduction",
                        entry_type="error",
                        importance=0.9
                    )

                # Store successful results
                elif any(kw in content_lower for kw in ['success', 'completed', 'accuracy', 'loss', 'metric']):
                    result_summary = content[:500] if len(content) > 500 else content
                    self.hierarchical_context.add(
                        content=f"[Batch {batch_num}] Result: {result_summary}",
                        source="reproduction",
                        entry_type="result",
                        importance=1.0
                    )

                # Store important decisions/observations
                elif any(kw in content_lower for kw in ['found', 'discovered', 'using', 'running']):
                    observation = content[:300] if len(content) > 300 else content
                    self.hierarchical_context.add(
                        content=f"[Batch {batch_num}] {observation}",
                        source="reproduction",
                        entry_type="observation",
                        importance=0.6
                    )

        except Exception as e:
            # Don't fail the main workflow if context storage fails
            print(f"   ⚠️  Warning: Failed to store batch context: {e}")

    def _get_relevant_context(self, query: str) -> str:
        """
        Get relevant historical context for current task.

        Uses hierarchical context manager for semantic retrieval.
        """
        try:
            relevant = self.hierarchical_context.retrieve(
                query=query,
                max_tokens=10000  # Budget for historical context
            )

            if not relevant:
                return ""

            sections = []
            for r in relevant[:5]:
                source = r.get('source', 'context')
                content = r.get('content', '')
                sections.append(f"[{source}] {content}")

            return "\n\n".join(sections)

        except Exception as e:
            print(f"   ⚠️  Warning: Failed to retrieve context: {e}")
            return ""

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
