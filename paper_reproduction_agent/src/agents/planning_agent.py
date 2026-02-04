"""Planning Agent - Creates comprehensive reproduction checklist from README analysis.

This agent implements a Hybrid approach:
1. Initial Deep Analysis: Reads README thoroughly, extracts ALL setup/experiment steps
2. Creates Comprehensive Checklist: reproduction_checklist.md with expandable sections
3. On-Demand Updates: Sub-agents can request planning updates when they encounter
   unexpected requirements (e.g., nested README discovered, undocumented dataset)
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from ..tools.code_execution_tools import (
    read_file,
    list_directory,
    execute_shell_command,
)
from ..tools.file_utils import (
    grep_in_directory,
    find_files,
)
from langchain_community.tools import DuckDuckGoSearchRun
from ..utils.llm_factory import create_llm
from ..utils.hierarchical_context import HierarchicalContextManager
from ..utils.logging_callback import LoggingCallbackHandler


class PlanningAgent:
    """Creates upfront reproduction plan based on README analysis."""

    def __init__(
        self,
        llm=None,
        max_iterations: int = 30,
        metrics_tracker=None,
        hierarchical_context: HierarchicalContextManager = None,
        callbacks: List = None,
    ):
        """Initialize the Planning Agent.

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

        self.system_prompt = """You are a Planning Specialist for ML paper reproduction.

GOAL: REPRODUCE PAPER RESULTS
You are part of an automated system designed to reproduce the results of a scientific paper. Your individual tasks must always serve this ultimate goal.

Your job is to create a comprehensive reproduction_checklist.md that guides all subsequent agents.

═══════════════════════════════════════════════════════════════
PHASE 1: READ AND ANALYZE
═══════════════════════════════════════════════════════════════


1. Read the root README.md thoroughly
2. Look for nested READMEs in subdirectories (examples/, docs/, data/, scripts/)
3. Identify environment files (requirements.txt, environment.yaml, pyproject.toml, setup.py)
4. Note ALL datasets mentioned and their download instructions
5. Find ALL experiment commands (training scripts, evaluation scripts)
6. Identify the main entry point script

═══════════════════════════════════════════════════════════════
PHASE 2: CREATE CHECKLIST
═══════════════════════════════════════════════════════════════

Create reproduction_checklist.md in the repository root with this structure:

```markdown
# Reproduction Checklist

**Paper:** {paper_title}
**Repository:** {repo_url}
**Created:** {timestamp}
**Status:** In Progress

---

## Environment Setup
**Tool Detected:** [PENDING - detect from README: conda/pip/poetry/uv/micromamba]
**Environment Name:** [PENDING - will be filled by environment agent]
**Python Version:** [detected from files]

- [ ] Read README and detect recommended tool (micromamba/conda/pip/etc)
- [ ] **MISSING PYTHON VERSION?** If not specified in README, USE THE SEARCH TOOL to find the correct version based on project name or key packages.
- [ ] Check if recommended tool is installed
- [ ] Install tool if missing
- [ ] Analyze environment.yaml/requirements.txt (check for unpinned packages)
- [ ] Pin versions if necessary (e.g. numpy<2 for torch compatibility)
- [ ] Create isolated environment (NOT base!)
- [ ] Install dependencies
- [ ] Verify installation (list packages)
- [ ] Test imports work (torch, transformers, etc.)
- [ ] **SMOKE TEST**: Run main script for 60s to verify setup

<!-- environment_agent will update this section with results -->

---

## Data Preparation
**Datasets Required:** [list from README]
**Total Size:** [if mentioned]
**Download Method:** [script/manual/automatic]
**Skip Data Prep:** [YES/NO - YES if scripts auto-download data]

- [ ] Identify all required datasets
- [ ] Download datasets (if needed)
- [ ] Verify data integrity

<!-- NOTE: Set "Skip Data Prep: YES" if README shows data is auto-downloaded by training scripts -->
<!-- EXPAND: data_prep_agent will add download commands here -->

---

## Experiments
**Total Experiments:** [count from README]
**Main Experiment:** [primary script/command]
**Strategy:** [single/all/custom]

### Experiment List:
[List each experiment found with command]

<!-- EXPAND: execution_agent will add run status here -->

---

## Verification
**Expected Metrics:** [from paper context if available]
**Success Threshold:** 5% relative error

- [ ] Extract metrics from result files
- [ ] Compare with paper values
- [ ] Report success rate

<!-- EXPAND: validation_agent will add comparison results here -->

---

## Notes
[Any special instructions, known issues, or requirements found in README]
```

═══════════════════════════════════════════════════════════════
CRITICAL RULES
═══════════════════════════════════════════════════════════════

1. Be THOROUGH - read nested READMEs, don't miss any setup steps
2. Extract EXACT commands from README - don't paraphrase
3. Note any dataset download scripts or URLs
4. Identify GPU requirements if mentioned
5. Flag any manual steps that can't be automated
6. Look for "quickstart" or "getting started" sections first

═══════════════════════════════════════════════════════════════
DATA PREPARATION ANALYSIS (Important!)
═══════════════════════════════════════════════════════════════

Check if data preparation can be SKIPPED:
- If training scripts AUTO-DOWNLOAD data (e.g., torchvision, HuggingFace datasets)
- If README says "data will be downloaded automatically"
- If there's a `--download` flag that's enabled by default

Set "**Skip Data Prep:** YES" in the checklist if:
- Scripts handle data download automatically
- Data is fetched on first run
- Using standard datasets (CIFAR, MNIST, ImageNet from torchvision)

Set "**Skip Data Prep:** NO" if:
- Manual download is required
- Data must be downloaded from external URLs
- Preprocessing scripts must be run first

After creating the checklist, respond with a summary of what you found.
"""

        # Minimal tools for planning - mostly reading
        self.tools = [
            read_file,
            list_directory,
            DuckDuckGoSearchRun(),
        ]

        print("\n" + "=" * 60)
        print("Planning Agent Initialized")
        print(f"   Max Iterations: {max_iterations}")
        print("=" * 60)

    def create_plan(self, state: Dict) -> Dict:
        """Create initial reproduction plan from README analysis.

        Args:
            state: Current PaperReproductionState

        Returns:
            Dict with:
                - reproduction_plan: Structured plan dict
                - checklist_path: Path to created checklist file
                - phase_status: Updated phase status
        """
        code_path = state.get("implementation_path", "./cloned_repo")
        paper_title = state.get("paper_title", "Unknown Paper")
        repo_url = state.get("selected_repo", {}).get("url", "Unknown")
        paper_context = state.get("agent_contexts", {}).get("paper_analyzer", "")

        # Read experiment selection mode from user's CLI choice
        experiment_mode = state.get("experiment_selection_mode", "all")
        custom_experiments = state.get("custom_experiment_list", [])

        # Get datasets/experiments from paper analyzer results
        paper_results = state.get("paper_results", {})
        paper_datasets = state.get("experimental_setup", {}).get("datasets", [])
        paper_metrics = paper_results.get("metrics", []) if isinstance(paper_results, dict) else []

        print("📋 Planning Agent: Creating reproduction checklist...")
        print(f"   📋 Experiment mode: {experiment_mode}")
        if paper_datasets:
            print(f"   📋 Datasets from paper: {paper_datasets}")

        # RETRIEVE relevant context (paper analysis, previous attempts - exclude own to prevent self-referencing)
        previous_context = ""
        if self.hierarchical_context:
            previous_context = self.hierarchical_context.compile_context(
                query="paper analysis datasets experiments requirements environment",
                max_tokens=2000,
                exclude_sources=["planning"],
            )
            if previous_context:
                print(f"   📋 Retrieved {len(previous_context)} chars of previous context")

        # Select which experiments to include based on mode
        selected_experiments = self._select_experiments_by_mode(
            experiment_mode, custom_experiments, paper_datasets, paper_metrics
        )
        selected_datasets = [e.get("dataset") for e in selected_experiments if e.get("dataset")]
        print(f"   📋 Selected experiments: {selected_datasets}")

        # Format selected experiments for the prompt
        selected_exp_text = "\n".join([
            f"  - {exp.get('dataset', 'Unknown')}"
            for exp in selected_experiments
        ])

        # Build mode-specific instructions
        if experiment_mode == "single":
            mode_instructions = f"""
═══════════════════════════════════════════════════════════════
EXPERIMENT MODE: SINGLE (One Dataset Only)
═══════════════════════════════════════════════════════════════
From paper analysis, selected dataset: {selected_datasets[0] if selected_datasets else 'Unknown'}
Reason: {selected_experiments[0].get('selected_reason', '') if selected_experiments else ''}

YOU MUST:
- Include ONLY the selected dataset in the checklist's Experiments section
- Find the command that runs experiments on THIS dataset only
- Set **Strategy:** single in the checklist
- Set **Selected Dataset:** {selected_datasets[0] if selected_datasets else 'Unknown'}
- The "Experiments to Run" section should have ONLY ONE bullet point
"""
        elif experiment_mode == "custom":
            mode_instructions = f"""
═══════════════════════════════════════════════════════════════
EXPERIMENT MODE: CUSTOM (User-Selected Experiments)
═══════════════════════════════════════════════════════════════
User selected: {selected_datasets}

YOU MUST:
- Include ONLY these experiments/datasets in the checklist's Experiments section
- IGNORE all other experiments in README
- Set **Strategy:** custom in the checklist
- Set **Selected Datasets:** {', '.join(selected_datasets)}
- The "Experiments to Run" section should have ONLY {len(selected_datasets)} bullet point(s)
"""
        else:  # "all"
            mode_instructions = f"""
═══════════════════════════════════════════════════════════════
EXPERIMENT MODE: ALL (Full Reproduction)
═══════════════════════════════════════════════════════════════
From paper analysis, datasets to reproduce:
{selected_exp_text if selected_exp_text.strip() else "  - All experiments from README"}

YOU MUST:
- Include ALL these experiments/datasets in the checklist's Experiments section
- Set **Strategy:** all in the checklist
- List ALL experiments as bullet points in "Experiments to Run"
"""

        # Build the planning prompt
        planning_prompt = f"""Analyze this repository and create a comprehensive reproduction checklist.

{mode_instructions}

Repository Path: {code_path}
Paper Title: {paper_title}
Repository URL: {repo_url}

Paper Context (from analyzer):
{paper_context[:2000] if paper_context else "Not available"}

=== CONTEXT FROM PREVIOUS AGENTS ===
{previous_context if previous_context else "No previous context available"}
====================================

STEPS:
1. First, list the directory to see the structure
2. Read README.md thoroughly
3. Look for nested READMEs in subdirectories
4. Identify environment files and requirements
5. Create the reproduction_checklist.md file with the CORRECT experiment list based on mode above

IMPORTANT: The "Experiments" section of your checklist MUST reflect the experiment mode!
- For SINGLE mode: Only ONE experiment bullet point
- For CUSTOM mode: Only the user-specified experiments
- For ALL mode: All experiments from the README

Start by listing the repository contents."""

        # Create and run the ReAct agent
        agent = create_react_agent(
            self.llm,
            self.tools,
            prompt=self.system_prompt,
        )

        # Prepare callbacks
        print("\n" + "-" * 60)
        print(f"Planning Agent: Creating checklist for {code_path}")
        print("-" * 60)

        try:
            config = {"recursion_limit": self.max_iterations}
            if self.callbacks:
                config["callbacks"] = self.callbacks
            result = agent.invoke(
                {"messages": [HumanMessage(content=planning_prompt)]},
                config,
            )

            # Extract last_message for reasoning output
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

            # Extract the plan from agent's work
            plan = self._extract_plan_from_result(result, code_path)

            # Add experiment selection info to plan
            plan["experiment_mode"] = experiment_mode
            plan["selected_experiments"] = selected_experiments
            plan["selected_datasets"] = selected_datasets

            # Store FULL messages in hierarchical context (including tool calls)
            if self.hierarchical_context:
                from ..utils.context_utils import build_context_entry

                planning_result = {
                    "plan_created": True,
                    "skip_data_prep": plan.get("skip_data_prep", False),
                    "selected_datasets": selected_datasets,
                    "experiment_mode": experiment_mode,
                }

                context_entry = build_context_entry(
                    agent_name="planning",
                    result=planning_result,
                    messages=messages,
                    max_detail_tokens=4000,
                )

                self.hierarchical_context.add(
                    content=context_entry,
                    source="planning",
                    entry_type="result",
                    importance=0.85,
                    lazy=True,
                )

            # Ensure checklist was created
            checklist_path = os.path.join(code_path, "reproduction_checklist.md")
            if not os.path.exists(checklist_path):
                # Create a basic checklist if agent didn't
                # Skip data prep by default (reactive mode) unless explicitly required
                should_skip_data_prep = not plan.get("requires_data_prep", False)
                self._create_basic_checklist(
                    checklist_path, paper_title, repo_url,
                    experiment_mode, selected_experiments,
                    skip_data_prep=should_skip_data_prep
                )

            return {
                "reproduction_plan": plan,
                "checklist_path": checklist_path,
                "phase_status": {
                    "planning": "completed",
                    "environment": "pending",
                    "data_prep": "pending",
                    "execution": "pending",
                    "validation": "pending",
                },
                "planning_update_request": None,  # Clear any pending requests
                "last_message": last_message,  # Agent reasoning for verbose output
            }

        except Exception as e:
            print(f"⚠️  Planning Agent error: {e}")
            # Return minimal plan on error
            checklist_path = os.path.join(code_path, "reproduction_checklist.md")
            # Default to reactive mode (skip data prep) on error
            self._create_basic_checklist(
                checklist_path, paper_title, repo_url,
                experiment_mode, selected_experiments,
                skip_data_prep=True  # Reactive mode by default
            )

            return {
                "reproduction_plan": {
                    "error": str(e),
                    "basic_plan": True,
                    "experiment_mode": experiment_mode,
                    "selected_experiments": selected_experiments,
                    "selected_datasets": selected_datasets,
                },
                "checklist_path": checklist_path,
                "phase_status": {
                    "planning": "completed",  # Mark as completed even on error
                    "environment": "pending",
                    "data_prep": "pending",
                    "execution": "pending",
                    "validation": "pending",
                },
                "last_message": f"Exception: {str(e)}",  # Error as reasoning
            }

    def update_plan(self, state: Dict) -> Dict:
        """Update the plan based on sub-agent request.

        Called when a sub-agent sets planning_update_request in state.

        Args:
            state: Current state with planning_update_request

        Returns:
            Updated state with modified plan
        """
        request = state.get("planning_update_request", {})
        checklist_path = state.get("checklist_path", "")

        if not request or not checklist_path:
            return state

        source = request.get("source", "unknown")
        reason = request.get("reason", "")
        context = request.get("context", {})

        print(f"📋 Planning Agent: Updating plan (requested by {source})")
        print(f"   Reason: {reason}")

        # Read current checklist
        try:
            with open(checklist_path, "r", encoding="utf-8") as f:
                current_checklist = f.read()
        except Exception as e:
            print(f"⚠️  Could not read checklist: {e}")
            return state

        # Determine what section to update
        update_section = self._determine_update_section(source, context)

        # Build update prompt
        update_prompt = f"""Update the reproduction checklist based on new information.

Current checklist:
{current_checklist}

Update request from: {source}
Reason: {reason}
New context: {context}

Update the {update_section} section with this new information.
Preserve the existing structure and add the new items.
"""

        # Use LLM to generate updated content
        try:
            from langchain_core.messages import HumanMessage as HM

            response = self.llm.invoke([HM(content=update_prompt)])
            updated_content = response.content

            # Write updated checklist
            with open(checklist_path, "w", encoding="utf-8") as f:
                f.write(updated_content)

            print(f"✅ Checklist updated with new {update_section} information")

        except Exception as e:
            print(f"⚠️  Could not update checklist: {e}")

        # Clear the update request
        state["planning_update_request"] = None

        return state

    def _extract_plan_from_result(self, result: Dict, code_path: str) -> Dict:
        """Extract structured plan from agent result.

        Args:
            result: Agent execution result
            code_path: Repository path

        Returns:
            Structured plan dict
        """
        plan = {
            "repository_path": code_path,
            "created_at": datetime.now().isoformat(),
            "environment_files": [],
            "datasets": [],
            "experiments": [],
            "main_experiment_cmd": None,
            "notes": [],
            "skip_data_prep": False,  # Default to not skipping
            "requires_data_prep": False,  # Only True if README has explicit data prep steps
        }

        # Try to read checklist to extract skip_data_prep flag
        checklist_path = os.path.join(code_path, "reproduction_checklist.md")
        if os.path.exists(checklist_path):
            try:
                with open(checklist_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # Look for Skip Data Prep flag
                if "**Skip Data Prep:** YES" in content or "Skip Data Prep: YES" in content:
                    plan["skip_data_prep"] = True
                    print("   📋 Planning detected: Data prep can be skipped (auto-download)")
                # Check if explicit data prep is required (README has specific data instructions)
                if "**Skip Data Prep:** NO" in content or "Skip Data Prep: NO" in content:
                    plan["requires_data_prep"] = True
                    print("   📋 Planning detected: Explicit data prep required")
            except Exception:
                pass

        # Try to find environment files
        env_patterns = [
            "requirements.txt",
            "environment.yaml",
            "environment.yml",
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
        ]

        for pattern in env_patterns:
            if os.path.exists(os.path.join(code_path, pattern)):
                plan["environment_files"].append(pattern)

        # Try to detect main script from common patterns
        main_patterns = [
            "train.py",
            "main.py",
            "run.py",
            "run_experiment.py",
            "run_training.py",
        ]

        for pattern in main_patterns:
            if os.path.exists(os.path.join(code_path, pattern)):
                plan["main_experiment_cmd"] = f"python {pattern}"
                break

        return plan

    def _create_basic_checklist(
        self, checklist_path: str, paper_title: str, repo_url: str,
        experiment_mode: str = "all",
        selected_experiments: List[Dict] = None,
        skip_data_prep: bool = False
    ) -> None:
        """Create a basic checklist when agent fails.

        Args:
            checklist_path: Path to write checklist
            paper_title: Paper title
            repo_url: Repository URL
            experiment_mode: "single", "all", or "custom"
            selected_experiments: List of selected experiment dicts
            skip_data_prep: Whether data prep should be skipped (reactive mode)
        """
        # Build experiment list for checklist
        if selected_experiments:
            exp_lines = []
            for exp in selected_experiments:
                dataset = exp.get("dataset", "Unknown")
                metrics = exp.get("metrics", [])
                metric_str = ", ".join([f"{m.get('metric')}={m.get('value')}" for m in metrics[:3]]) if metrics else ""
                exp_lines.append(f"- [ ] {dataset}: [find command for {dataset}]")
                if metric_str:
                    exp_lines.append(f"      Expected: {metric_str}")
            experiments_section = "\n".join(exp_lines) if exp_lines else "- [ ] [find main experiment command]"
            datasets_list = ", ".join([e.get("dataset", "?") for e in selected_experiments])
            total_experiments = len(selected_experiments)
        else:
            experiments_section = "- [ ] [find main experiment command]"
            datasets_list = "[PENDING]"
            total_experiments = "[PENDING]"

        # Determine mode note
        if experiment_mode == "single":
            mode_note = "Single mode - run ONLY the listed experiment"
        elif experiment_mode == "custom":
            mode_note = "Custom mode - run ONLY the listed experiments"
        else:
            mode_note = "All mode - run ALL experiments"

        # Build data prep section based on skip_data_prep flag
        if skip_data_prep:
            data_prep_section = """**Status:** ⏭️ SKIPPED (Reactive Mode)
**Reason:** Data prep runs only if execution fails with data errors.

- [x] ~~Identify all required datasets~~ (handled by execution)
- [x] ~~Download datasets~~ (auto-download or already present)
- [x] ~~Verify data integrity~~ (smoke test validates data)

<!-- Data prep skipped - execution will route back if data issues occur -->"""
        else:
            data_prep_section = """**Datasets Required:** [PENDING]
**Download Method:** [PENDING]
**Skip Data Prep:** NO

- [ ] Identify all required datasets
- [ ] Download datasets
- [ ] Verify data integrity

<!-- EXPAND: data_prep_agent will add download commands here -->"""

        content = f"""# Reproduction Checklist

**Paper:** {paper_title}
**Repository:** {repo_url}
**Created:** {datetime.now().isoformat()}
**Status:** In Progress

---

## Environment Setup
**Tool Detected:** [PENDING - detect from README: conda/pip/poetry/uv/micromamba]
**Environment Name:** [PENDING - will be filled by environment agent]
**Python Version:** [PENDING]

- [ ] Read README and detect recommended tool (micromamba/conda/pip/etc)
- [ ] Check if recommended tool is installed
- [ ] Install tool if missing
- [ ] Analyze environment.yaml/requirements.txt (check for unpinned packages)
- [ ] Pin versions if necessary (e.g. numpy<2 for torch compatibility)
- [ ] Create isolated environment (NOT base!)
- [ ] Install dependencies
- [ ] Verify installation (list packages)
- [ ] Test imports work (torch, transformers, etc.)
- [ ] **SMOKE TEST**: Run main script for 60s to verify setup

<!-- environment_agent will update this section with results -->

---

## Data Preparation
{data_prep_section}

---

## Experiments
**Total Experiments:** {total_experiments}
**Strategy:** {experiment_mode}
**Selected Datasets:** {datasets_list}

### Experiments to Run:
{experiments_section}

<!-- NOTE: {mode_note} -->

---

## Verification
**Expected Metrics:** [PENDING]
**Success Threshold:** 5% relative error

- [ ] Extract metrics from result files
- [ ] Compare with paper values
- [ ] Report success rate

<!-- EXPAND: validation_agent will add comparison results here -->

---

## Notes
- Checklist created with basic template (planning agent could not complete full analysis)
- Please read README.md manually for additional instructions
"""

        try:
            os.makedirs(os.path.dirname(checklist_path), exist_ok=True)
            with open(checklist_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"📝 Created basic checklist at {checklist_path}")
        except Exception as e:
            print(f"⚠️  Could not create checklist: {e}")

    def _determine_update_section(self, source: str, context: Dict) -> str:
        """Determine which section to update based on request source.

        Args:
            source: Name of requesting agent
            context: Request context

        Returns:
            Section name to update
        """
        source_to_section = {
            "environment_agent": "Environment Setup",
            "environment_setup_agent": "Environment Setup",
            "data_prep_agent": "Data Preparation",
            "execution_agent": "Experiments",
            "validation_agent": "Verification",
        }

        return source_to_section.get(source, "Notes")

    def _select_experiments_by_mode(
        self,
        mode: str,
        custom_list: List[str],
        paper_datasets: List[str],
        paper_metrics: List[Dict]
    ) -> List[Dict]:
        """Select which experiments to include based on user mode and paper analysis.

        Args:
            mode: "single", "all", or "custom"
            custom_list: User-specified experiments for custom mode
            paper_datasets: Datasets extracted from paper by analyzer
            paper_metrics: Metrics extracted from paper by analyzer

        Returns:
            List of selected experiment dicts with dataset, metrics, etc.
        """
        if mode == "custom" and custom_list:
            # Use user-specified experiments
            return [{"dataset": exp, "from_user": True} for exp in custom_list]

        elif mode == "single":
            # Pick ONE experiment: prefer smallest/simplest dataset
            # Heuristic: smaller datasets like CIFAR-10, MNIST are easier than ImageNet
            small_datasets = ["mnist", "cifar", "cora", "citeseer", "pubmed", "synthetic", "ogb"]

            # Find first match from small datasets
            for small in small_datasets:
                for dataset in paper_datasets:
                    if small in dataset.lower():
                        # Get metrics for this dataset
                        metrics_for_dataset = [m for m in paper_metrics
                                               if dataset.lower() in m.get("dataset", "").lower()]
                        return [{
                            "dataset": dataset,
                            "metrics": metrics_for_dataset,
                            "selected_reason": "smallest dataset from paper"
                        }]

            # Fallback: use first dataset from paper
            if paper_datasets:
                first_dataset = paper_datasets[0]
                metrics_for_dataset = [m for m in paper_metrics
                                       if first_dataset.lower() in m.get("dataset", "").lower()]
                return [{
                    "dataset": first_dataset,
                    "metrics": metrics_for_dataset,
                    "selected_reason": "first dataset from paper"
                }]

            return [{"dataset": "main", "selected_reason": "no datasets found, use main experiment"}]

        else:  # "all" mode
            # Include ALL datasets/experiments from paper
            if paper_datasets:
                return [{"dataset": d, "metrics": [m for m in paper_metrics
                                                    if d.lower() in m.get("dataset", "").lower()]}
                        for d in paper_datasets]
            return [{"dataset": "all", "selected_reason": "run all experiments"}]

    def read_checklist(self, checklist_path: str) -> Optional[str]:
        """Read the current checklist content.

        Args:
            checklist_path: Path to checklist file

        Returns:
            Checklist content or None if not found
        """
        try:
            with open(checklist_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None

    def update_checklist_item(
        self, checklist_path: str, item: str, completed: bool = True
    ) -> bool:
        """Mark a checklist item as completed or pending.

        Args:
            checklist_path: Path to checklist file
            item: Item text to find and update
            completed: Whether to mark as completed

        Returns:
            True if item was found and updated
        """
        try:
            with open(checklist_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Find and replace the checkbox
            old_marker = "- [ ]" if not completed else "- [x]"
            new_marker = "- [x]" if completed else "- [ ]"

            if item in content:
                # Find the line with this item
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if item in line and old_marker in line:
                        lines[i] = line.replace(old_marker, new_marker, 1)
                        break

                content = "\n".join(lines)

                with open(checklist_path, "w", encoding="utf-8") as f:
                    f.write(content)

                return True

            return False

        except Exception as e:
            print(f"⚠️  Could not update checklist item: {e}")
            return False
