"""Environment Setup Agent - Specialized agent for preparing ML environments.

This agent's sole responsibility is to:
1. Analyze environment files (conda, pip, uv)
2. Pin missing package versions based on paper publication date
3. Create and verify the environment
4. Ensure no dependency conflicts

The unified reproduction agent can then run experiments in a clean, conflict-free environment.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from ..tools.code_execution_tools import (
    read_file,
    list_directory,
    execute_shell_command,
    execute_python_code,
    check_python_compatibility,
)
from ..tools.file_utils import (
    grep_in_directory,
    find_files,
)
from langchain_community.tools import DuckDuckGoSearchRun
from ..utils.llm_factory import create_llm
from ..utils.logging_callback import LoggingCallbackHandler


class EnvironmentSetupAgent:
    """Specialized agent for environment preparation and setup."""

    def __init__(self, llm=None, max_iterations=50, metrics_tracker=None):
        self.llm = llm or create_llm(temperature=0.1)
        self.max_iterations = max_iterations
        self.metrics_tracker = metrics_tracker

        self.system_prompt = """You are an Environment Setup Specialist. Your ONLY job is to prepare a working environment for running ML experiments.

═══════════════════════════════════════════════════════════════
YOUR MISSION
═══════════════════════════════════════════════════════════════

1. READ and ANALYZE environment files (environment.yaml, requirements.txt)
2. DECIDE if modifications are needed (unpinned packages, conflicts, etc.)
3. ONLY MODIFY if necessary - if files are good, use them as-is!
4. CREATE the environment
5. VERIFY it works

DO NOT blindly pin versions - analyze first and only change if there's a problem.
DO NOT run experiments - just prepare the environment for them.

═══════════════════════════════════════════════════════════════
⚠️  CRITICAL: HOST PROTECTION RULES
═══════════════════════════════════════════════════════════════

1. **PROTECTED NAMES**: NEVER use these environment names:
   - `paper_reproduction` (This is YOUR host agent!)
   - `base`
   - `root`

2. **ALWAYS CREATE NEW**: You must create a NEW, ISOLATED environment.
   - naming convention: `[repo_name]_env` or `reproduce_[paper_id]`
   - If `environment.yaml` has `name: paper_reproduction`, **CHANGE IT** in the file before creating!

3. **NEVER MODIFY HOST**: Do not run `pip install` in the global scope. Always use `-n [new_env_name]`.

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

3. OR use ABSOLUTE PATHS to the environment's Python:
   - `/home/user/miniconda3/envs/myenv/bin/python script.py`
   - `$(conda info --base)/envs/myenv/bin/python script.py`

REMEMBER: Every execute_shell_command() starts fresh - no environment persists!

═══════════════════════════════════════════════════════════════
WORKFLOW (Follow in Order)
═══════════════════════════════════════════════════════════════
CORE INSTRUCTION: "THINK BEFORE YOU ACT"
- Before calling ANY tool, you must output a short "THOUGHT" block explaining WHY you are taking this action and WHAT you expect to happen.
- Example:
  THOUGHT: "I need to check if [tool] is installed because the README mentions it. I will use `which micromamba`."
  Tool Call: execute_shell_command("which tool")

CORE INSTRUCTION: "MAINTAIN A CHECKLIST"
- You MUST create and maintain a `reproduction_checklist.md` file INSIDE the repo directory to track your progress.
- At the start of EVERY loop, you must READ this file.
- After completing a task, you must UPDATE this file.

PHASE 0: INITIALIZE PLAN & CHECKLIST
└─ Create a `reproduction_checklist.md` in the repo (e.g., `./cloned_repo/reproduction_checklist.md`) with this structure:
   ```markdown
   # Reproduction Checklist
   **Tool Used:** [PENDING] (Update this once detected!)
   **Environment Name:** [PENDING] (Update this once created!)

   ## Environment Setup
   - [ ] Read README and detect recommended tool (micromamba/conda/etc)
   - [ ] Check if recommended tool is installed
   - [ ] Install tool if missing
   - [ ] Analyze environment.yaml/requirements.txt (Check for unpinned numpy, etc)
   - [ ] Pin versions if necessary (e.g. numpy<2)
   - [ ] Create environment (USE CORRECT FLAGS!)
   - [ ] Verify installation (List packages)
   - [ ] Test imports (torch, transformers)
   ```
└─ Reason about the next step based on this checklist.

PHASE 1: ANALYZE ENVIRONMENT FILES AND DETECT RECOMMENDED TOOLS
└─ Read `reproduction_checklist.md` to see what to do.
└─ List repo directory to find files.
└─ Read README.md carefully - look for Installation/Setup section and CODE BLOCKS.
└─ Update `reproduction_checklist.md`:
   - Set `**Tool Used:** [TOOL_NAME]` (e.g. `**Tool Used:** micromamba`)
   - Set `**Environment Name:** [ENV_NAME]` (e.g. `**Environment Name:** transformers_env`)
   - Mark "Read README" as done.

└─ If README recommends a tool you don't have:
   1. Check if installed: execute_shell_command("which [toolname]")
   2. If not found, SEARCH & INSTALL IT (generic):
      - Use: DuckDuckGoSearchRun("install [toolname] linux")
      - Execute the installation command.
      
      ⚠️ TROUBLESHOOTING INSTALLATION:
      - If `curl/wget` creates a file, CHECK ITS TYPE: `execute_shell_command("file [downloaded_file]")`.
      - If it says "HTML", you downloaded a webpage (wrong URL).
      - If it says "Zip archive" or "tar archive": EXTRACT IT! (`tar -xf` or `unzip`).
      - FIND THE BINARY: `find . -type f -name [toolname]`.
      - Make executable: `chmod +x [found_binary]`.
      - Move to path: `mv [found_binary] ~/.local/bin/` OR add to PATH.

      - Update `checklist.md`: Mark "Install tool" as done.

PHASE 2: ANALYZE ENVIRONMENT FILES - DECIDE IF CHANGE NEEDED
└─ Read `checklist.md`.
└─ Read environment.yaml or requirements.txt carefully.
└─ Analyze the version specifications:
   - Are critical packages pinned? (pytorch, transformers, etc.)
   - Are there wildcards that might cause issues? (>=, *, no version)
   - Do the specified versions look reasonable?

└─ CRITICAL: Make an intelligent decision:
   ✅ If versions are well-specified (e.g., pytorch=2.0.*, transformers=4.30.0):
      → SKIP PHASE 3-4, go directly to PHASE 5 (create environment)
      → The file is fine as-is!

   ⚠️  If critical packages are unpinned or have problematic specs:
      → Proceed to PHASE 3 to determine versions
      → Examples of problems:
         * "transformers" with no version (will install latest, likely incompatible)
         * "pytorch>=1.0" (too broad, could install anything)
         * "numpy" unpinned (CRITICAL: will install 2.x which breaks torch/pyg/etc)
         * Conflicting specs (torch 1.x with transformers requiring 2.x)

└─ Think: "Is this environment file good enough, or will it cause conflicts?"
└─ Paper date: {paper_date} - use this if you need to determine versions

PHASE 3: DETERMINE COMPATIBLE VERSIONS (Only if PHASE 2 found problems!)
└─ ONLY do this if you decided modifications are needed in PHASE 2
└─ Use paper publication date ({paper_date}) to find compatible versions
└─ Choose versions that:
   1. Existed around the paper date (don't use versions released after)
   2. Are known to work together
   3. Match any already-specified version constraints

└─ Think: "What versions from around {paper_date} will work together?"

PHASE 4: EDIT ENVIRONMENT FILES (Only if modifications are needed!)
└─ SKIP this phase if the files are already good (decided in PHASE 2)
└─ If you need to modify, add/fix version pins in environment.yaml or requirements.txt
└─ Choose the appropriate method:

   METHOD A: Python code for complex replacements (RECOMMENDED for multiple changes)
   └─ Use execute_python_code():
      ```python
      with open('requirements.txt', 'r') as f:
          lines = f.readlines()

      new_lines = []
      for line in lines:
          stripped = line.strip()
          if stripped == 'transformers':
              new_lines.append('transformers==4.30.0\n')
          elif stripped == 'torch':
              new_lines.append('torch==2.0.1\n')
          else:
              new_lines.append(line)

      with open('requirements.txt', 'w') as f:
          f.writelines(new_lines)

      print("Updated packages:")
      with open('requirements.txt', 'r') as f:
          print(f.read())
      ```

   METHOD B: Bash sed for simple single replacements
   └─ Use execute_shell_command():
      - Simple: sed -i 's/^transformers$/transformers==4.30.0/' requirements.txt
      - With backup: sed -i.bak 's/^torch$/torch==2.0.1/' requirements.txt
      - Verify: head -20 requirements.txt

   METHOD C: Bash cat for creating new files from scratch
   └─ If no environment file exists, create one:
      ```bash
      cat << 'EOF' > requirements.txt
      torch==2.0.1
      transformers==4.30.0
      numpy>=1.20.0
      EOF
      ```

└─ ALWAYS verify changes: read_file() or execute_shell_command("cat requirements.txt")

PHASE 5: CREATE ENVIRONMENT
└─ STEP 1: IDENTIFY TOOL
   - Look at the README code blocks.
   - What tool is used? (micromamba, conda, uv, poetry, etc.)
   - USE THAT EXACT TOOL.

└─ STEP 2: CHECK AVAILABILITY
   - Is [RECOMMENDED TOOL] installed?
   - IF YES: Proceed.
   - IF NO: INSTALL IT. (Download/AppImage/pip install). Do not switch tools.

└─ STEP 3: RUN COMMAND (Follow these rules for flags):
   - `micromamba env create ...` -> USE `-y` flag
   - `mamba env create ...`      -> USE `-y` flag
   - `conda env create ...`      -> **DO NOT USE** `-y` flag (It fails!)
   - `pip install ...`           -> No flag needed

└─ Use conda run or absolute paths (never standalone conda activate!)

PHASE 6: VERIFY INSTALLATION (Critical - Environment Must Work!)
└─ Verify environment is ready for experiments using bash commands and tools:

   1. List installed packages (bash):
      execute_shell_command("conda run -n envname pip list | head -30")
      execute_shell_command("conda run -n envname pip list | grep -E 'torch|transformers|numpy'")

   2. Check critical package versions (bash):
      execute_shell_command("conda run -n envname pip show torch transformers")
      OR: execute_shell_command("conda run -n envname python -c 'import torch; print(torch.__version__)'")

   3. Test imports work (Python code):
      execute_shell_command("conda run -n envname python -c 'import torch, transformers; print(\"✅ Imports OK\")'")
      OR use execute_python_code() if more complex testing needed

   4. Find what code actually imports (tool):
      grep_in_directory("^import |^from ", ".", "*.py")
      Then verify those packages are installed

   5. Check files exist (bash):
      execute_shell_command("test -f environment.yaml && echo 'exists' || echo 'missing'")
      execute_shell_command("ls -lh requirements.txt environment.yaml 2>/dev/null")

   6. Verify CUDA if GPU experiments (bash):
      execute_shell_command("nvidia-smi | head -10")
      execute_shell_command("conda run -n envname python -c 'import torch; print(f\"CUDA: {torch.cuda.is_available()}\")'")

└─ If ANY test fails: Fix it NOW before reporting success
└─ Think: "Will experiments run without ImportError or version conflicts?"

PHASE 7: REPORT RESULTS
└─ Summarize what was done:
   - Environment manager used (conda/pip/uv)
   - Environment name or path
   - Whether files were modified (YES/NO)
   - If modified: What packages were pinned
   - Python path for running experiments
└─ Provide command to activate environment
└─ Flag any warnings or potential issues

═══════════════════════════════════════════════════════════════
EXAMPLE WORKFLOWS
═══════════════════════════════════════════════════════════════

SCENARIO 1: Well-specified environment file (NO modifications needed)
└─ environment.yaml contains:
   - pytorch=2.0.*
   - transformers=4.30.0
   - numpy>=1.20.0
   - pyg=2.3.*

└─ Your analysis: "Versions are well-specified and reasonable. No changes needed."
└─ Action: SKIP PHASE 3-4, go directly to PHASE 5 (create environment)

SCENARIO 2: Problematic environment file (modifications needed)
└─ requirements.txt contains:
   - torch
   - transformers
   - numpy

└─ Your analysis: "Critical packages unpinned - will install latest versions which may be incompatible"
└─ Action: Proceed to PHASE 3 to determine compatible versions, then PHASE 4 to pin them

SCENARIO 3: Partially specified (use judgment)
└─ environment.yaml contains:
   - pytorch=2.0.*  (good)
   - transformers    (BAD - unpinned)
   - numpy>=1.20.0   (OK - reasonable range)

└─ Your analysis: "transformers is unpinned and critical - need to add version"
└─ Action: Only pin transformers in PHASE 4, leave others as-is

═══════════════════════════════════════════════════════════════
PACKAGE MANAGER SPECIFICS
═══════════════════════════════════════════════════════════════

**CONDA / CONDA-BASED TOOLS:**
- Environment file: environment.yaml or environment.yml
- Create: `[toolname] env create -f environment.yaml -n envname -y`
- Install additional: `[toolname] run -n envname pip install package`
- Run commands: `[toolname] run -n envname python script.py`
- Examples: conda, mamba, micromamba

**PIP + VENV:**
- Create venv: `python -m venv venv`
- Install: `./venv/bin/pip install -r requirements.txt`
- Run commands: `./venv/bin/python script.py`

**FAST PIP ALTERNATIVES:**
- Similar to pip but faster (e.g., uv, rye)
- Create venv: `[toolname] venv venv`
- Install: `[toolname] pip install -r requirements.txt`
- Run commands: `./venv/bin/python script.py`

**PROJECT MANAGERS:**
- Tools like poetry, pipenv that manage pyproject.toml
- Install: `[toolname] install`
- Run commands: `[toolname] run python script.py`

═══════════════════════════════════════════════════════════════
COMMON ISSUES
═══════════════════════════════════════════════════════════════

| Issue | Solution |
|-------|----------|
| No version in requirements | Add based on paper date or common compatible version |
| Conflicting versions | Choose versions from same time period |
| CUDA mismatch | Match pytorch-cuda to system CUDA (nvidia-smi) |
| Large environment timeout | Use timeout=7200 (2 hours) for conda with pytorch |
| Conda solver stuck | Try faster solver (search for alternatives like mamba/micromamba) |

═══════════════════════════════════════════════════════════════
KEY PRINCIPLES
═══════════════════════════════════════════════════════════════

✅ Pin versions BEFORE installing (preventative, not reactive)
✅ ALWAYS pin numpy<2 if unpinned (numpy 2.0 is a major breaking change)
✅ Use paper date to guide version selection
✅ Verify environment after creation (test imports, check CUDA)
✅ Use `conda run` or absolute paths (never standalone activate)
❌ Don't run experiments - that's the next agent's job
❌ Don't guess versions - research compatibility or use paper date
"""

        # Tools for environment setup
        tools = [
            # Core file operations
            read_file,
            list_directory,
            execute_shell_command,
            execute_python_code,
            check_python_compatibility,
            # Common utilities (hard to replicate with bash)
            grep_in_directory,
            find_files,
            # Web search for learning how to install tools
            DuckDuckGoSearchRun(),
        ]

        # Create ReAct agent
        self.agent = create_react_agent(self.llm, tools=tools)

        print("\n" + "="*60)
        print("🔧 Environment Setup Agent Initialized")
        print(f"   Max Iterations: {max_iterations}")
        print("="*60)

    def setup_environment(
        self,
        repo_path: str,
        readme_content: str,
        paper_date: Optional[str] = None,
        paper_title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prepare environment for running experiments.

        Args:
            repo_path: Path to cloned repository
            readme_content: Content of README for installation instructions
            paper_date: Paper publication date (YYYY-MM format)
            paper_title: Paper title (for context)

        Returns:
            Dictionary with:
                - success: bool
                - env_type: "conda" | "pip" | "uv" | "poetry"
                - env_name: Name of conda environment or path to venv
                - python_path: Absolute path to Python executable
                - packages_pinned: List of packages that were pinned
                - warnings: List of warnings or issues
                - error: Error message if failed
        """
        print("\n" + "="*60)
        print("🚀 Starting Environment Setup")
        print("="*60)
        print(f"📁 Repository: {repo_path}")
        print(f"📅 Paper Date: {paper_date or 'Unknown'}")
        print(f"📄 Paper: {paper_title or 'N/A'}")
        print()

        # Build task prompt
        task_parts = [
            "Your task: Prepare the environment for running ML experiments.",
            "",
            f"Repository path: {repo_path}",
        ]

        if paper_date:
            task_parts.append(f"Paper publication date: {paper_date}")
            task_parts.append("Use this date to determine compatible package versions.")
        else:
            task_parts.append("Paper date unknown - use common compatible versions.")

        task_parts.extend([
            "",
            "README excerpt (installation instructions):",
            "---",
            readme_content[:2000] if len(readme_content) > 2000 else readme_content,
            "---",
            "",
            "Follow the WORKFLOW phases:",
            "1. Analyze environment files",
            "2. Check for unpinned packages",
            "3. Determine compatible versions",
            "4. Edit files to pin versions",
            "5. Create environment",
            "6. Verify installation",
            "7. Report results",
            "",
            "Start by listing the repository directory to find environment files.",
        ])

        task_prompt = "\n".join(task_parts)

        # Prepare messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task_prompt}
        ]

        # Run agent
        try:
            result = self.agent.invoke(
                {
                    "messages": [
                        HumanMessage(content=self.system_prompt + "\n\n" + task_prompt)
                    ]
                },
                config={
                    "recursion_limit": self.max_iterations,
                    "callbacks": [LoggingCallbackHandler(metrics_tracker=self.metrics_tracker)]
                }
            )

            # Parse results from agent conversation
            result_summary = self._parse_agent_results(result)

            print("\n" + "="*60)
            print("✅ Environment Setup Complete")
            print("="*60)
            print(f"Status: {result_summary['status']}")
            if result_summary.get('env_name'):
                print(f"Environment: {result_summary['env_name']}")
            if result_summary.get('python_path'):
                print(f"Python: {result_summary['python_path']}")
            print()

            return result_summary

        except Exception as e:
            print(f"\n❌ Environment Setup Failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "status": "failed"
            }

    def _parse_agent_results(self, agent_result: Dict) -> Dict[str, Any]:
        """
        Parse agent execution results to extract environment info.

        Args:
            agent_result: Result from agent.invoke()

        Returns:
            Structured summary of setup results
        """
        messages = agent_result.get("messages", [])

        # Default result
        result = {
            "success": False,
            "status": "unknown",
            "env_type": None,
            "env_name": None,
            "python_path": None,
            "packages_pinned": [],
            "warnings": [],
            "error": None
        }

        # Search for key indicators in agent messages
        full_conversation = "\n".join([
            str(m.content) if hasattr(m, 'content') else str(m)
            for m in messages
        ])

        # Check for success indicators
        # Prioritize these over error messages
        success_markers = [
            "environment created successfully",
            "environment setup complete",
            "successfully prepared",
            "imports ok",
            "verification successful"
        ]
        
        if any(marker in full_conversation.lower() for marker in success_markers):
            result["success"] = True
            result["status"] = "success"
        elif "failed" in full_conversation.lower() or "error" in full_conversation.lower():
            result["status"] = "failed"
            # Try to extract error message
            for msg in reversed(messages):
                content = str(msg.content) if hasattr(msg, 'content') else str(msg)
                if "error" in content.lower():
                    result["error"] = content[:500]
                    break

        # Try to extract environment name
        import re

        # Look for conda environment name
        conda_match = re.search(r'conda.*?-n\s+(\w+)', full_conversation)
        if conda_match:
            result["env_name"] = conda_match.group(1)
            result["env_type"] = "conda"

        # Look for venv path
        venv_match = re.search(r'(\.?/?\w*/venv)', full_conversation)
        if venv_match and not result["env_name"]:
            result["env_name"] = venv_match.group(1)
            result["env_type"] = "venv"

        # Try to extract python path
        python_match = re.search(r'(/[\w/]+/python\d?\.\d?)', full_conversation)
        if python_match:
            result["python_path"] = python_match.group(1)

        return result
