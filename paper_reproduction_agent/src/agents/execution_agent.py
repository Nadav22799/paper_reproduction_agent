"""Execution Agent - Runs experiments using background process pattern.

Split from UnifiedReproductionAgent to handle experiment execution as a separate concern.
This agent:
1. Reads the checklist to understand experiment requirements AND environment info
2. Runs experiments using the background process pattern with correct env tool
3. Classifies errors for recovery routing (OOM, module errors, etc.)
4. Updates the checklist with experiment status
"""

import os
import re
from typing import Dict, Optional, Tuple
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
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
from ..utils.resource_detector import detect_system_resources, get_experiment_strategy
from ..utils.hierarchical_context import HierarchicalContextManager
from ..utils.logging_callback import LoggingCallbackHandler


class ExecutionAgent:
    """Handles experiment execution for reproduction."""

    def __init__(
        self,
        llm=None,
        max_iterations: int = 50,
        metrics_tracker=None,
        hierarchical_context: HierarchicalContextManager = None,
        callbacks=None,
    ):
        """Initialize the Execution Agent.

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

        # Detect system resources
        self.resources = detect_system_resources()
        self.experiment_strategy = get_experiment_strategy(self.resources)

        from ..config.prompts import EXECUTION_AGENT_PROMPT
        self.system_prompt = EXECUTION_AGENT_PROMPT

        self.tools = [
            read_file,
            list_directory,
            write_file,
            execute_shell_command,
            execute_python_code,
            start_background_process,
            wait_for_process,
            stop_process,
            search_error_solution,
        ]

        print("\n" + "=" * 60)
        print("Execution Agent Initialized")
        print(f"   Max Iterations: {max_iterations}")
        print(f"   Strategy: {self.experiment_strategy}")
        print("=" * 60)

    def run_experiments(self, state: Dict) -> Dict:
        """Run experiments from the checklist.

        Args:
            state: Current PaperReproductionState

        Returns:
            Dict with:
                - experiments_completed: bool
                - experiment_results: dict with details
                - failure_metadata: Optional dict if failed
        """
        code_path = state.get("implementation_path", "./cloned_repo")
        checklist_path = state.get("checklist_path", "")
        recovery_context = state.get("failure_metadata", {})

        # Read experiment selection mode and plan
        experiment_mode = state.get("experiment_selection_mode", "all")
        reproduction_plan = state.get("reproduction_plan", {})

        # Get selected datasets from plan (set by planning agent from paper analysis)
        selected_experiments = reproduction_plan.get("selected_experiments", [])
        selected_datasets = reproduction_plan.get("selected_datasets", [])

        print("🚀 Execution Agent: Running experiments...")
        print(f"   📋 Experiment mode: {experiment_mode}")
        if selected_datasets:
            print(f"   📋 Selected datasets to run: {selected_datasets}")

        # Check for recovery context (OOM retry, etc.)
        is_retry = recovery_context and recovery_context.get("retry_count", 0) > 0
        reduce_batch = recovery_context.get("reduce_batch", False) if is_retry else False

        # Read checklist to get tool and environment info
        checklist_content = ""
        tool_detected = ""
        env_name = ""

        if checklist_path and os.path.exists(checklist_path):
            try:
                with open(checklist_path, "r", encoding="utf-8") as f:
                    checklist_content = f.read()

                # Extract tool and environment from checklist
                tool_match = re.search(r"\*\*Tool Detected:\*\*\s*(\w+)", checklist_content)
                if tool_match:
                    tool_detected = tool_match.group(1).lower()

                env_match = re.search(r"\*\*Environment Name:\*\*\s*(\S+)", checklist_content)
                if env_match:
                    env_name = env_match.group(1)

                print(f"   📋 From checklist - Tool: {tool_detected or 'not found'}, Env: {env_name or 'not found'}")

            except Exception as e:
                print(f"⚠️  Could not read checklist: {e}")

        # Build resource-aware instructions
        resource_instructions = self._get_resource_instructions()

        # Build the execution prompt with explicit tool/env info
        tool_info = ""
        if tool_detected and env_name:
            if tool_detected in ["conda", "mamba", "micromamba"]:
                tool_info = f"""
ENVIRONMENT INFO (from checklist):
- Tool: {tool_detected}
- Environment: {env_name}
- Command pattern: `{tool_detected} run -n {env_name} python <script.py>`

Example: `{tool_detected} run -n {env_name} python train.py`
"""
            elif tool_detected in ["pip", "venv"]:
                tool_info = f"""
ENVIRONMENT INFO (from checklist):
- Tool: {tool_detected}
- Environment: {env_name}
- Command pattern: `./venv/bin/python <script.py>` or `source venv/bin/activate && python <script.py>`
"""
            elif tool_detected == "poetry":
                tool_info = f"""
ENVIRONMENT INFO (from checklist):
- Tool: poetry
- Command pattern: `poetry run python <script.py>`
"""
            elif tool_detected == "uv":
                tool_info = f"""
ENVIRONMENT INFO (from checklist):
- Tool: uv
- Command pattern: `uv run python <script.py>`
"""

        # Build mode-specific execution instructions using selected datasets from plan
        if experiment_mode == "single" and selected_datasets:
            dataset_to_run = selected_datasets[0]
            mode_exec_instructions = f"""
═══════════════════════════════════════════════════════════════
EXPERIMENT MODE: SINGLE (One Dataset Only)
═══════════════════════════════════════════════════════════════
Run experiments ONLY for dataset: {dataset_to_run}

This dataset was selected from the paper analysis as the primary/simplest target.

YOU MUST:
- Find the command that runs experiments on {dataset_to_run}
- Run ONLY that experiment
- Skip ALL other datasets/experiments
"""
        elif experiment_mode == "custom" and selected_datasets:
            mode_exec_instructions = f"""
═══════════════════════════════════════════════════════════════
EXPERIMENT MODE: CUSTOM (User-Selected Datasets)
═══════════════════════════════════════════════════════════════
Run experiments ONLY for these datasets: {', '.join(selected_datasets)}

YOU MUST:
- Find commands for EACH selected dataset
- Run ONLY those experiments
- Skip ALL other datasets/experiments
"""
        else:  # "all" or no selection
            if selected_datasets:
                datasets_list = ", ".join(selected_datasets)
                mode_exec_instructions = f"""
═══════════════════════════════════════════════════════════════
EXPERIMENT MODE: ALL (Full Reproduction)
═══════════════════════════════════════════════════════════════
Run experiments for ALL these datasets from the paper: {datasets_list}

Run each experiment in order from the checklist.
"""
            else:
                mode_exec_instructions = """
═══════════════════════════════════════════════════════════════
EXPERIMENT MODE: ALL (Full Reproduction)
═══════════════════════════════════════════════════════════════
Run ALL experiments from the checklist.
"""

        execution_prompt = f"""Run experiments for this ML repository.

{mode_exec_instructions}

Repository Path: {code_path}
Experiment Strategy: {self.experiment_strategy}

{tool_info if tool_info else "⚠️ No tool/env info found - READ THE CHECKLIST FIRST!"}

{"⚠️ RETRY MODE: Previous attempt failed. " + ("Reduce batch size!" if reduce_batch else "") if is_retry else ""}
{"Previous error: " + recovery_context.get("error_message", "")[:200] if is_retry else ""}

STEPS:
1. FIRST: Read reproduction_checklist.md to find:
   - **Tool Detected** (conda/micromamba/pip/poetry/uv)
   - **Environment Name**
   - Experiment commands in the "Experiments to Run" section
   - **Strategy** field to confirm which experiments to run
2. Check if results already exist (skip those experiments)
3. For EACH experiment listed in "Experiments to Run":
   a. Build command using the CORRECT tool pattern
   b. Use start_background_process with log file. MANDATORY: Store logs in 'cloned_repo/logs/' (or 'logs/' if CWD is already repo). NEVER use root.
      Example: `start_background_process("python <script.py>", log_file="cloned_repo/logs/<script_name>.log")`
   c. IMMEDIATELY call wait_for_process
   d. Check log file for errors
4. Report status of each experiment

Current Checklist (partial):
{checklist_content[:3000] if checklist_content else "⚠️ No checklist found - read reproduction_checklist.md"}

REMEMBER:
- ONLY run experiments listed in "Experiments to Run" section!
- Use the TOOL from the checklist (don't assume conda!)
- Use background process pattern for ALL training/evaluation scripts!

Start by reading the checklist to confirm tool, environment, and experiment list."""

        # Update system prompt with resource instructions
        full_prompt = self.system_prompt.replace("{resource_instructions}", resource_instructions)

        # Create and run the ReAct agent
        agent = create_react_agent(
            self.llm,
            self.tools,
            prompt=full_prompt,
        )


        print("\n" + "-" * 60)
        print(f"Execution Agent: Starting experiment execution for {code_path}")
        print("-" * 60)

        try:
            config = {"recursion_limit": self.max_iterations * 3}
            if self.callbacks:
                config["callbacks"] = self.callbacks
            result = agent.invoke(
                {"messages": [HumanMessage(content=execution_prompt)]},
                config,
            )

            # Analyze result to determine success and extract errors
            success, details, error_info, last_message = self._analyze_result(result, code_path)
            if success:
                # Store success in hierarchical context
                if self.hierarchical_context:
                    result_summary = str(details)[:500] if details else "Experiments completed"
                    self.hierarchical_context.add(
                        content=f"[Execution Success] {result_summary}",
                        source="execution",
                        entry_type="result",
                        importance=1.0,
                        lazy=True,  # Defer embedding to avoid loading SentenceTransformer
                    )
                return {
                    "experiments_completed": True,
                    "experiment_results": details,
                    "failure_metadata": None,
                    "phase_status": {"execution": "completed"},
                    "last_message": last_message,  # Agent reasoning for critic
                }
            else:
                # Classify the error for routing
                error_type = self._classify_error(error_info.get("error_message", ""))
                failure_meta = self._create_failure_metadata(error_info, error_type, state)

                # Store error in hierarchical context for recovery
                if self.hierarchical_context:
                    error_msg = error_info.get("error_message", "Unknown error")[:500]
                    self.hierarchical_context.add(
                        content=f"[Execution Error] Type: {error_type}\nMessage: {error_msg}",
                        source="execution",
                        entry_type="error",
                        importance=0.9,
                        lazy=True,  # Defer embedding to avoid loading SentenceTransformer
                    )

                return {
                    "experiments_completed": False,
                    "experiment_results": details,
                    "failure_metadata": failure_meta,
                    "phase_status": {"execution": "failed"},
                    "last_message": last_message,  # Agent reasoning for critic
                }

        except Exception as e:
            print(f"⚠️  Execution Agent error: {e}")
            error_type = self._classify_error(str(e))

            return {
                "experiments_completed": False,
                "experiment_results": {"error": str(e)},
                "failure_metadata": {
                    "error_type": error_type,
                    "error_message": str(e),
                    "error_source": "execution_agent",
                    "attempted_fixes": [],
                    "recovery_hints": self._get_recovery_hints(error_type, str(e)),
                    "retry_count": state.get("failure_metadata", {}).get("retry_count", 0) + 1,
                },
                "phase_status": {"execution": "failed"},
                "last_message": f"Exception: {str(e)}",  # Error as reasoning
            }

    def _get_resource_instructions(self) -> str:
        """Get resource-aware execution instructions.

        Returns:
            Instructions string based on system resources
        """
        if self.experiment_strategy == "minimal":
            return """MINIMAL RESOURCES DETECTED:
- Run only sanity check / smallest dataset
- Use batch_size=1 or 2
- Limit to 1 epoch or 100 steps
- Skip heavy experiments"""
        elif self.experiment_strategy == "single":
            return """MEDIUM RESOURCES DETECTED:
- Run main experiment only
- Use reduced batch_size if needed
- Limit to 3-5 epochs
- Skip auxiliary experiments"""
        else:
            return """FULL RESOURCES DETECTED:
- Run all experiments from checklist
- Use recommended batch sizes
- Full training as specified in README"""

    def _analyze_result(self, result: Dict, code_path: str) -> Tuple[bool, Dict, Dict, str]:
        """Analyze agent result using LLM to determine success and valid errors.

        Args:
            result: Agent execution result
            code_path: Repository path

        Returns:
            Tuple of (success: bool, details: dict, error_info: dict, last_message: str)
        """
        messages = result.get("messages", [])
        last_message = ""
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                if isinstance(last_msg.content, list):
                    text_parts = []
                    for part in last_msg.content:
                        if isinstance(part, str):
                            text_parts.append(part)
                        elif isinstance(part, dict) and "text" in part:
                            text_parts.append(part["text"])
                    last_message = "\n".join(text_parts)
                else:
                    last_message = str(last_msg.content)

        # 1. Look for result files
        result_files = []
        has_actual_results = False
        result_patterns = ["results", "output", "logs", "checkpoints"]
        
        log_content = ""

        # Scan specifically for logs to provide context to LLM
        for pattern in result_patterns:
            pattern_path = os.path.join(code_path, pattern)
            if os.path.exists(pattern_path):
                result_files.append(pattern_path)
                if os.path.isdir(pattern_path):
                    try:
                        for item in os.listdir(pattern_path):
                            item_path = os.path.join(pattern_path, item)
                            if os.path.isfile(item_path):
                                if os.path.getsize(item_path) > 0:
                                    has_actual_results = True
                                # Read log files for context (up to 2KB)
                                if item.endswith(".log") or item.endswith(".txt"):
                                    try:
                                        with open(item_path, "r", encoding="utf-8", errors="ignore") as f:
                                            content = f.read()
                                            log_content += f"\n--- File: {item} ---\n{content[-2000:]}"
                                    except Exception:
                                        pass
                    except OSError:
                        pass
                elif os.path.isfile(pattern_path) and os.path.getsize(pattern_path) > 0:
                    has_actual_results = True
                    # Read single log file
                    if pattern.endswith(".log") or pattern.endswith(".txt"):
                        try:
                            with open(pattern_path, "r", encoding="utf-8", errors="ignore") as f:
                                content = f.read()
                                log_content += f"\n--- File: {pattern} ---\n{content[-2000:]}"
                        except Exception:
                            pass

        # 2. Use LLM to judge success/failure
        # Define structured output
        try:
            from langchain_core.pydantic_v1 import BaseModel, Field
        except ImportError:
            try:
                from pydantic.v1 import BaseModel, Field
            except ImportError:
                from pydantic import BaseModel, Field

        class ExperimentAnalysis(BaseModel):
            success: bool = Field(description="True if the experiment completed its main task, even with warnings.")
            failure_reason: Optional[str] = Field(description="If failed, the specific error message. None if successful.")
            is_critical_error: bool = Field(description="True if the error is fatal (OOM, killed, exception). False for warnings.")
            error_type: str = Field(description="Classification: environment, data, execution, validation, or none")

        # Create analysis chain
        analyzer = self.llm.with_structured_output(ExperimentAnalysis)
        
        analysis_prompt = f"""Analyze this experiment execution.
        
        AGENT OUTPUT:
        {last_message[:1000]}
        
        LOG FILES CONTENT:
        {log_content[:4000]}
        
        TASK:
        Did the experiment run to completion?
        - IGNORE warnings (like FutureWarnings, UserWarnings, distutils errors).
        - IGNORE "No module named" usage in non-critical paths if the main loop finished.
        - "Optimization Finished" or "Training complete" usually means SUCCESS.
        - Only mark as FAILURE if the process crashed, timed out, or produced no results.
        """
        
        try:
            analysis = analyzer.invoke(analysis_prompt)
            print(f"DEBUG: LLM Analysis: {analysis}")
            
            success = analysis.success
            
            # Fallback: if LLM says success but no files found, force fail
            if success and not has_actual_results:
                success = False
                error_info = {
                    "error_message": "LLM reported success but no result files found on disk",
                    "error_type": "no_results",
                    "full_output": last_message
                }
            elif not success:
                error_info = {
                    "error_message": analysis.failure_reason or "Unknown failure",
                    "error_type": analysis.error_type,
                    "full_output": last_message
                }
            else:
                error_info = {}
                
        except Exception as e:
            print(f"⚠️ LLM Analysis failed: {e}. Falling back to basic check.")
            # Fallback to basic file check
            success = has_actual_results
            error_info = {}

        details = {
            "result_files": result_files,
            "has_actual_results": has_actual_results,
            "agent_output": last_message[:1000] if last_message else "",
        }

        return success, details, error_info, last_message

    def _extract_error_message(self, output: str) -> str:
        """Extract the main error message from output.

        Args:
            output: Full output text

        Returns:
            Extracted error message
        """
        # Look for common error patterns
        patterns = [
            r"(ModuleNotFoundError:.*?)(?:\n|$)",
            r"(ImportError:.*?)(?:\n|$)",
            r"(FileNotFoundError:.*?)(?:\n|$)",
            r"(CUDA out of memory.*?)(?:\n|$)",
            r"(RuntimeError:.*?)(?:\n|$)",
            r"(Error:.*?)(?:\n|$)",
            r"(Exception:.*?)(?:\n|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Return last 200 chars if no specific error found
        return output[-200:] if len(output) > 200 else output

    def _classify_error(self, error_message: str) -> str:
        """Classify error type for routing.

        Args:
            error_message: The error message

        Returns:
            Error type string
        """
        error_lower = error_message.lower()

        if "modulenotfounderror" in error_lower or "no module named" in error_lower:
            return "environment"
        elif "importerror" in error_lower:
            return "environment"
        elif "cuda out of memory" in error_lower or "out of memory" in error_lower:
            return "execution"  # OOM - retry with reduced batch
        elif "filenotfounderror" in error_lower and ("data" in error_lower or "dataset" in error_lower):
            return "data"
        elif "timeout" in error_lower:
            return "execution"
        else:
            return "execution"

    def _create_failure_metadata(self, error_info: Dict, error_type: str, state: Dict) -> Dict:
        """Create failure metadata for routing.

        Args:
            error_info: Error information
            error_type: Classified error type
            state: Current state

        Returns:
            FailureMetadata dict
        """
        error_message = error_info.get("error_message", "Experiment execution failed")
        existing_metadata = state.get("failure_metadata", {})
        attempted_fixes = existing_metadata.get("attempted_fixes", [])

        return {
            "error_type": error_type,
            "error_message": error_message,
            "error_source": "execution_agent",
            "attempted_fixes": attempted_fixes,
            "recovery_hints": self._get_recovery_hints(error_type, error_message),
            "retry_count": existing_metadata.get("retry_count", 0) + 1,
            "reduce_batch": "out of memory" in error_message.lower(),
        }

    def _get_recovery_hints(self, error_type: str, error_message: str) -> list:
        """Get recovery hints based on error type.

        Args:
            error_type: Classified error type
            error_message: The error message

        Returns:
            List of recovery hints
        """
        hints = []
        error_lower = error_message.lower()

        if error_type == "environment":
            if "no module named" in error_lower:
                module_match = re.search(r"no module named ['\"]?(\w+)", error_lower)
                if module_match:
                    hints.append(f"pip install {module_match.group(1)}")
            hints.append("Check environment is activated")
            hints.append("Verify all dependencies are installed")

        elif error_type == "execution":
            if "out of memory" in error_lower:
                hints.append("Reduce batch_size by 50%")
                hints.append("Enable gradient checkpointing")
                hints.append("Use mixed precision (fp16)")
            elif "timeout" in error_lower:
                hints.append("Increase timeout parameter")
                hints.append("Run smaller experiment first")

        elif error_type == "data":
            hints.append("Run data preparation again")
            hints.append("Check data paths in config")

        return hints or ["Check experiment logs for details"]

    def check_existing_results(self, code_path: str) -> Dict:
        """Check if results already exist in the repository.

        Args:
            code_path: Repository path

        Returns:
            Dict with existing results info
        """
        result_patterns = [
            ("results", [".json", ".csv", ".txt"]),
            ("output", [".json", ".csv", ".txt"]),
            ("logs", [".log", ".txt"]),
            ("checkpoints", [".pt", ".pth", ".ckpt"]),
        ]

        found = {
            "has_results": False,
            "result_files": [],
            "checkpoints": [],
            "log_files": [],
        }

        for directory, extensions in result_patterns:
            dir_path = os.path.join(code_path, directory)
            if os.path.exists(dir_path):
                for root, _, files in os.walk(dir_path):
                    for file in files:
                        if any(file.endswith(ext) for ext in extensions):
                            full_path = os.path.join(root, file)
                            if ".pt" in file or ".pth" in file or ".ckpt" in file:
                                found["checkpoints"].append(full_path)
                            elif ".log" in file:
                                found["log_files"].append(full_path)
                            else:
                                found["result_files"].append(full_path)
                            found["has_results"] = True

        return found
