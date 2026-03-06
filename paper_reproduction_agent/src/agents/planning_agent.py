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
    write_file,
)
from ..tools.file_utils import grep_in_directory, find_files
from langchain_community.tools import DuckDuckGoSearchRun
from ..utils.llm_factory import create_llm
from ..utils.hierarchical_context import HierarchicalContextManager
from rich import print as rprint
from rich.panel import Panel
from rich.console import Console

class PlanningAgent:
    """Creates upfront reproduction plan based on README analysis."""

    def __init__(
        self,
        llm=None,
        max_iterations: int = 30,
        metrics_tracker=None,
        hierarchical_context: HierarchicalContextManager = None,
        callbacks: List = None,
        storage=None,
        base_dir: str = None,
    ):
        """Initialize the Planning Agent.

        Args:
            llm: Language model to use
            max_iterations: Maximum iterations for the ReAct agent
            metrics_tracker: Optional metrics tracker for observability
            hierarchical_context: Shared context manager for cross-agent knowledge
            storage: Optional StorageProvider for checklist persistence
            base_dir: Project base directory, used to compute relative storage keys
        """
        self.llm = llm or create_llm(temperature=0.1)
        self.max_iterations = max_iterations
        self.metrics_tracker = metrics_tracker
        self.hierarchical_context = hierarchical_context
        self.callbacks = callbacks or []
        self._storage = storage
        self._base_dir = base_dir
        self._setup_prompt_and_tools()

    # ── Storage helpers ──────────────────────────────────────────────────────

    def _storage_key(self, path: str) -> str:
        """Convert an absolute path to a storage key relative to base_dir."""
        from pathlib import Path as _Path
        if self._base_dir:
            try:
                return str(_Path(path).relative_to(self._base_dir)).replace("\\", "/")
            except ValueError:
                pass
        return _Path(path).name

    def _read_file(self, path: str) -> Optional[str]:
        """Read a file via StorageProvider or local filesystem fallback."""
        if self._storage is not None:
            return self._storage.read_text(self._storage_key(path))
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None

    def _write_file(self, path: str, content: str) -> bool:
        """Write a file via StorageProvider or local filesystem fallback."""
        if self._storage is not None:
            return self._storage.save_text(content, self._storage_key(path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return True

    def _setup_prompt_and_tools(self):
        """Set up system prompt and tools (called from __init__)."""
        from ..config.prompts import EFFICIENCY_RULES

        self.system_prompt = """You are a Planning Specialist for ML paper reproduction.

GOAL: REPRODUCE PAPER RESULTS
You are part of an automated system designed to reproduce the results of a scientific paper. Your individual tasks must always serve this ultimate goal.

Your job is to create a comprehensive reproduction_checklist.md that guides all subsequent agents.

{efficiency_rules}

""".replace("{efficiency_rules}", EFFICIENCY_RULES)

        self.system_prompt += """
⚠️  CRITICAL BOUNDARY RULE ⚠️
You MUST ONLY access files INSIDE the repository directory provided in the prompt.
NEVER use relative paths like "../" to navigate outside it.
NEVER access the parent directory or any sibling directories.

WEB SEARCH is allowed ONLY for:
  - Looking up the correct Python version when not specified in the repo
  - Nothing else — specifically do NOT search for datasets or commands

If a dataset is NOT found in the repo: write "NEEDS_DISCOVERY" in the checklist.
The data_prep agent will discover the download method via web search.

═══════════════════════════════════════════════════════════════
SEARCH BUDGET — READ THIS BEFORE STARTING
═══════════════════════════════════════════════════════════════
You have a LIMITED number of steps. Use them efficiently:

MANDATORY (always do these first):
  1. list root directory
  2. read README.md
  3. scan key subdirectories (scripts/, data/, configs/) — list only, don't read files unless needed

ALLOWED (only if the above leave gaps):
  - grep for specific keywords — max 2 times per keyword
  - read at most 3-4 specific files that look relevant based on directory listing

STOP AND WRITE THE CHECKLIST when ANY of the following is true:
  - You have found all commands and dataset sources
  - You have already done 2+ targeted searches for a dataset/command and found nothing
  - You have read more than 4 files beyond README and still haven't found what you need

NEVER:
  - Re-read a file you have already read
  - Run grep more than twice for the same keyword
  - Read implementation files to "understand the code" — you only need commands
  - Keep searching hoping the next file will have the answer

When a dataset or command is not found after exhausting README and scripts/:
write it as NEEDS_DISCOVERY in the checklist and move on.
The data_prep and execution agents will resolve these gaps.

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
**Download Source:** [auto / script:[filename] / NEEDS_DISCOVERY]
**Skip Data Prep:** [YES/NO - YES if scripts auto-download data]

- [ ] Identify all required datasets
- [ ] Download datasets (if needed)
- [ ] Verify data integrity

<!-- NOTE: Set "Skip Data Prep: YES" if README shows data is auto-downloaded by training scripts -->
<!-- NOTE: Set "Download Source: NEEDS_DISCOVERY" if dataset is mentioned but no download info found in repo -->
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

Set "**Skip Data Prep:** YES" and "**Download Source:** auto" if:
- Scripts handle data download automatically
- Data is fetched on first run from a standard library loader

Set "**Skip Data Prep:** NO" and "**Download Source:** script:[filename]" if:
- A download script exists in the repo
- Manual download via wget/curl is documented


Set "**Skip Data Prep:** NO" and "**Download Source:** NEEDS_DISCOVERY" if:
- A dataset is mentioned in the paper/README but NO download instructions exist in the repo
- Do NOT keep searching — write NEEDS_DISCOVERY and move on
- The data_prep agent will discover the download method automatically

After creating the checklist, respond with a summary of what you found.

═══════════════════════════════════════════════════════════════
PHASE 3: DETECT USER INPUT REQUIREMENTS
═══════════════════════════════════════════════════════════════

Based ONLY on info gathered in Phases 1-2 (do NOT read additional files),
determine if reproduction requires ANY of:
- API keys (OpenAI, HuggingFace tokens, W&B, Comet ML, etc.)
- Paid/licensed datasets requiring purchase or manual approval
- Manual account creation or credentials
- Private model weights requiring auth tokens

If NOTHING is required, skip this section entirely.
If user input IS required, add this section to the checklist:

```markdown
## User Input Required
**Status:** PENDING

**Step-by-step Guide:**
1. [First action — be specific with URLs, e.g. "Go to https://huggingface.co/settings/tokens"]
2. [Next action — exact steps to obtain each credential or resource]

**Items:**
- [ ] <ITEM_NAME> - Type: <api_key|dataset_access|credentials|other>
  Description: <what it is and why it's needed>
  Instructions: <exact steps with URLs>
  Environment Variable: <VAR_NAME>

**Sensitive Data Storage:**
For API keys/tokens/credentials, instruct the user to create a `.env` file:
- Path: <repo_root>/.env
- Template:
  # Required for reproduction
  <VAR_NAME_1>=your_value_here
  <VAR_NAME_2>=your_value_here
- The `.env` will be loaded by downstream agents automatically.
- Document the `.env` path in this checklist so agents can find it.
```

RULES:
- Be SPECIFIC — exact URLs, button names, page locations.
- For credentials: ALWAYS use .env (never ask user to paste secrets in terminal).
- If no user input is required, do NOT add this section.
"""

        # Wrap DuckDuckGo so network failures return an error string instead of crashing the agent
        _ddg = DuckDuckGoSearchRun()
        _ddg.handle_tool_error = True

        # Tools for planning: file reading + writing + grep for repo search + web search as last resort
        self.tools = [
            read_file,
            write_file,
            list_directory,
            grep_in_directory,
            find_files,
            _ddg,
        ]

        Console().print(Panel(f"Max Iterations: {self.max_iterations}", title="Planning Agent Initialized", border_style="cyan", expand=False))

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

        rprint("📋 Planning Agent: Creating reproduction checklist...")
        rprint(f"   📋 Experiment mode: {experiment_mode}")
        if paper_datasets:
            rprint(f"   📋 Datasets from paper: {paper_datasets}")

        # RETRIEVE relevant context (paper analysis, previous attempts - exclude own to prevent self-referencing)
        previous_context = ""
        if self.hierarchical_context:
            previous_context = self.hierarchical_context.compile_context(
                query="paper analysis datasets experiments requirements environment",
                max_tokens=2000,
                exclude_sources=["planning"],
            )
            if previous_context:
                rprint(f"   📋 Retrieved {len(previous_context)} chars of previous context")

        # Select which experiments to include based on mode
        selected_experiments = self._select_experiments_by_mode(
            experiment_mode, custom_experiments, paper_datasets, paper_metrics
        )
        selected_datasets = [e.get("dataset") for e in selected_experiments if e.get("dataset")]
        rprint(f"   📋 Selected experiments: {selected_datasets}")

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

⚠️  BOUNDARY: Only access files inside {code_path}. Never use "../" or go above this directory.
If something is not found inside {code_path}, write "not found in repo" — do NOT explore parent or sibling directories.

Paper Context (from analyzer):
{paper_context[:2000] if paper_context else "Not available"}

=== CONTEXT FROM PREVIOUS AGENTS ===
{previous_context if previous_context else "No previous context available"}
====================================

STEPS:
1. First, list the directory {code_path} to see the structure
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
        rprint(Panel(f"Creating checklist for [bold]{code_path}[/bold]", title="📋 Planning Agent", border_style="cyan"))

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
                from ..utils.context_utils import build_smart_context_entry

                planning_result = {
                    "plan_created": True,
                    "skip_data_prep": plan.get("skip_data_prep", False),
                    "selected_datasets": selected_datasets,
                    "experiment_mode": experiment_mode,
                }

                context_entry = build_smart_context_entry(
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

            # Ensure checklist was created and is complete
            checklist_path = os.path.join(code_path, "reproduction_checklist.md")
            should_skip_data_prep = not plan.get("requires_data_prep", False)
            if not os.path.exists(checklist_path):
                # File was never written — create basic checklist
                self._create_basic_checklist(
                    checklist_path, paper_title, repo_url,
                    experiment_mode, selected_experiments,
                    skip_data_prep=should_skip_data_prep
                )
            else:
                # File exists — validate it has all required sections (LLM may have written partial content)
                content = self._read_file(checklist_path) or ""
                required_sections = ["## Environment Setup", "## Data Preparation", "## Experiments"]
                missing = [s for s in required_sections if s.lower() not in content.lower()]
                if missing:
                    rprint(f"   ⚠️  Checklist incomplete (missing: {missing}), replacing with basic template")
                    self._create_basic_checklist(
                        checklist_path, paper_title, repo_url,
                        experiment_mode, selected_experiments,
                        skip_data_prep=should_skip_data_prep
                    )

            # Detect user input requirements from checklist
            user_input_req = self._detect_user_input_requirements(checklist_path)

            result_dict = {
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

            if user_input_req:
                result_dict["user_input_required"] = user_input_req

            return result_dict

        except Exception as e:
            rprint(f"⚠️  Planning Agent error: {e}")
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

        rprint(f"📋 Planning Agent: Updating plan (requested by {source})")
        rprint(f"   Reason: {reason}")

        # Read current checklist
        current_checklist = self._read_file(checklist_path)
        if current_checklist is None:
            rprint(f"⚠️  Could not read checklist: {checklist_path}")
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
            self._write_file(checklist_path, updated_content)
            rprint(f"✅ Checklist updated with new {update_section} information")

        except Exception as e:
            rprint(f"⚠️  Could not update checklist: {e}")

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
        if os.path.exists(checklist_path) or self._storage is not None:
            try:
                content = self._read_file(checklist_path)
                if content is None:
                    content = ""
                # Look for Skip Data Prep flag
                if "**Skip Data Prep:** YES" in content or "Skip Data Prep: YES" in content:
                    plan["skip_data_prep"] = True
                    rprint("   📋 Planning detected: Data prep can be skipped (auto-download)")
                # Check if explicit data prep is required (README has specific data instructions)
                if "**Skip Data Prep:** NO" in content or "Skip Data Prep: NO" in content:
                    plan["requires_data_prep"] = True
                    rprint("   📋 Planning detected: Explicit data prep required")
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

    def _detect_user_input_requirements(self, checklist_path: str) -> Optional[dict]:
        """Parse checklist for '## User Input Required' section written by the LLM.

        The planning agent's Phase 3 prompt instructs the LLM to analyze
        the README and write situation-specific instructions when it detects
        the paper needs API keys, paid datasets, credentials, etc.

        This method extracts the LLM's structured output into a dict
        so the CLI can display it clearly to the user.

        Args:
            checklist_path: Path to reproduction_checklist.md

        Returns:
            Dict with description and items, or None if no user input needed.
        """
        import re

        content = self._read_file(checklist_path)
        if content is None:
            return None

        # Look for the User Input Required section
        match = re.search(
            r"##\s*User Input Required\s*\n(.*?)(?=\n##\s|\Z)",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if not match:
            return None

        section = match.group(1)

        # Skip if the section says nothing is needed
        if "PENDING" not in section and "- [" not in section:
            return None

        # Parse individual items from the LLM-generated section
        # Expected format from Phase 3 prompt:
        #   - [ ] {name} - Type: {type}
        #     Description: {what and why}
        #     Instructions: {step-by-step}
        #     Environment Variable: {VAR_NAME}
        items = []
        item_blocks = re.findall(
            r"-\s*\[[ x]?\]\s*(.+?)(?=\n-\s*\[|\Z)",
            section,
            re.DOTALL,
        )

        for block in item_blocks:
            lines = block.strip().split("\n")
            if not lines:
                continue

            # First line: name - Type: type
            header = lines[0].strip()
            name_match = re.match(r"(.+?)\s*-\s*Type:\s*(\w+)", header)
            if name_match:
                name = name_match.group(1).strip()
                item_type = name_match.group(2).strip()
            else:
                name = header
                item_type = "other"

            description = ""
            instructions = ""
            env_var = ""

            for line in lines[1:]:
                line = line.strip()
                if line.lower().startswith("description:"):
                    description = line.split(":", 1)[1].strip()
                elif line.lower().startswith("instructions:"):
                    instructions = line.split(":", 1)[1].strip()
                elif line.lower().startswith("environment variable:"):
                    env_var = line.split(":", 1)[1].strip()

            items.append({
                "name": name,
                "type": item_type,
                "description": description or name,
                "instructions": instructions,
                "env_var": env_var,
            })

        if not items:
            return None

        return {
            "description": "The paper requires the following before reproduction can proceed:",
            "items": items,
        }

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
            self._write_file(checklist_path, content)
            rprint(f"📝 Created basic checklist at {checklist_path}")
        except Exception as e:
            rprint(f"⚠️  Could not create checklist: {e}")

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
        return self._read_file(checklist_path)

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
            content = self._read_file(checklist_path)
            if content is None:
                return False

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
                self._write_file(checklist_path, content)
                return True

            return False

        except Exception as e:
            rprint(f"⚠️  Could not update checklist item: {e}")
            return False
