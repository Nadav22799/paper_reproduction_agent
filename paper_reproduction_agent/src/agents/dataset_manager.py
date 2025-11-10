"""Dataset Manager Agent - Downloads and prepares datasets."""

import os
from typing import Dict
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from ..tools.code_execution_tools import (
    read_file,
    search_file,
    execute_shell_command,
    execute_python_script,
)
from ..utils.llm_factory import create_llm
from ..utils.message_utils import normalize_message_content


class DatasetManagerAgent:
    """Agent for downloading and preparing datasets."""

    def __init__(self, llm=None):
        self.llm = llm or create_llm(temperature=0.1)

        self.system_prompt = """You find and prepare datasets for ML experiments.

ALWAYS START BY READING README.md - it contains data instructions!

Your job is to:
1. Extract data/dataset instructions from README
2. Follow references to nested READMEs if mentioned
3. Try to execute them if they're commands
4. Report what you found and what happened

Common instruction patterns to look for:
- "Data is in examples/NLG/" → Check if directory exists AND read examples/NLG/README.md
- "Download from https://..." → Execute download with wget/curl
- "Run python download_data.py" → Execute the script
- "Register at ... then download" → Report as manual steps
- "Requires license" → Report as access restricted
- "See X/ for data instructions" → Read X/README.md

IMPORTANT - Nested READMEs:
If README says "Data is in X/" or "See X/ for data":
→ That subdirectory likely has its own README with detailed instructions
→ Read it: read_file(file_path="<repo>/X/README.md")
→ Use those detailed instructions for data preparation

How to handle different cases:
✅ Data in repo → Verify it exists, report location
✅ Download commands → Execute them (wget, curl, python script, etc.)
✅ Nested instructions → Follow references, read additional READMEs
⚠️  Manual steps → Extract and report instructions
🔒 Access restricted → Report restrictions

Be pragmatic: Try to execute what you can, report what you can't."""

        tools = [read_file, search_file, execute_shell_command, execute_python_script]
        # Don't pass custom prompt - let ReAct use its default optimized for tool calling
        self.agent = create_react_agent(self.llm, tools=tools)

    def prepare_datasets(self, code_path: str, paper_datasets: list = None,
                          agent_context: str = "") -> Dict:
        """
        Download and prepare datasets mentioned in paper.

        Args:
            code_path: Path to repository
            paper_datasets: Dataset names from paper (optional)
            agent_context: Context from previous agents (NEW!)

        Returns:
            Dataset preparation results
        """
        dataset_info = f"Expected datasets: {paper_datasets}" if paper_datasets else "Check code for dataset requirements"

        # Add context from previous agents if available
        context_note = ""
        if agent_context:
            context_note = f"\n\nContext from paper analysis: {agent_context}"

        # Prepend system context to task (since we can't override ReAct's prompt)
        task = f"""{self.system_prompt}

===== TASK: Prepare datasets for {code_path} =====

Expected datasets from paper: {dataset_info}{context_note}

===== YOUR WORKFLOW =====

STEP 1: Read root README.md for data instructions
   read_file(file_path="{code_path}/README.md")

   Look for sections like:
   - "Data" / "Dataset" / "Download"
   - Any instructions about getting/preparing data

STEP 2: Check if README points to subdirectory for data
   If README says things like:
   - "Data is in examples/NLG/"
   - "See X/ for data instructions"
   - "Refer to X/README.md for data"

   → Read that nested README: read_file(file_path="{code_path}/X/README.md")
   → Extract detailed data instructions from there

STEP 3: Extract data instructions from all READMEs you read
   What does README say about data? Copy the exact instructions.

   Examples:
   - "Data is in examples/NLG/" (then read examples/NLG/README.md)
   - "Download from https://example.com/data.zip"
   - "Run bash scripts/download.sh"
   - "Register at X, then download Y"

STEP 4: Try to execute instructions (if possible)

   If data is in repo → Check if directory exists
   If download URL → Use execute_shell_command to wget/curl
   If script to run → Use execute_shell_command or execute_python_script
   If manual steps → Cannot execute, report instructions
   If restricted → Cannot execute, report restriction

STEP 5: Verify data is ready
   Check if data directory exists and has files

STEP 6: Report clearly
   - What README(s) you read (root + nested if any)
   - What instructions you found
   - What you executed (if anything)
   - Where data is located OR what manual steps are needed

===== CRITICAL =====
- ALWAYS read README.md first!
- If README points to subdirectory → READ THAT README TOO
- Extract and report the ACTUAL instructions (don't paraphrase)
- Try to execute commands when possible
- Be clear about what succeeded vs what needs manual action"""

        messages = [HumanMessage(content=task)]

        try:
            result = self.agent.invoke(
                {"messages": messages},
                config={"recursion_limit": 30}
            )
        except Exception as e:
            print(f"\n❌ Dataset manager failed: {e}")
            result = {"messages": [], "error": str(e)}

        parsed_result = self._parse_dataset_result(result)

        # Validate datasets if download was reported as successful
        if parsed_result.get("datasets_downloaded"):
            validated = self._validate_datasets(code_path, parsed_result)
            parsed_result["datasets_validated"] = validated
            if not validated:
                print("\n⚠️  Warning: Datasets reported as downloaded but validation failed")
                parsed_result["errors"].append("Dataset validation failed - files may not exist")

        return parsed_result

    def _parse_dataset_result(self, result: Dict) -> Dict:
        """Extract dataset status and instructions from agent result."""
        messages = result.get("messages", [])

        dataset_info = {
            "datasets_identified": False,
            "datasets_downloaded": False,
            "dataset_locations": [],
            "download_instructions": "",  # Extracted instructions from README
            "executed_commands": [],  # Commands the agent tried to execute
            "manual_steps_needed": False,  # True if requires manual action
            "errors": [],
            "report": ""
        }

        # Handle agent errors
        if "error" in result:
            dataset_info["errors"].append(result["error"])
            dataset_info["report"] = f"Dataset preparation failed: {result['error']}"
            return dataset_info

        all_messages = []
        for msg in messages:
            if hasattr(msg, 'content') and msg.content:
                content_str = normalize_message_content(msg.content)
                all_messages.append(content_str)

        full_output = "\n".join(all_messages).lower()

        # Extract download instructions (look for README quotes or instruction mentions)
        for msg in all_messages:
            # Look for quoted instructions or lines that mention data/download
            if any(kw in msg.lower() for kw in ["data is in", "download from", "run python", "run bash", "execute", "wget", "curl"]):
                if dataset_info["download_instructions"]:
                    dataset_info["download_instructions"] += "\n"
                dataset_info["download_instructions"] += msg[:200]

        # Check if datasets were identified
        if any(kw in full_output for kw in ["dataset", "data", "mnist", "cifar", "imagenet", "squad", "wmt"]):
            dataset_info["datasets_identified"] = True

        # Check if download was attempted/successful
        if any(kw in full_output for kw in ["downloaded", "executing", "running", "executed"]):
            dataset_info["executed_commands"] = [
                msg[:150] for msg in all_messages
                if any(kw in msg.lower() for kw in ["execute", "running", "downloaded", "wget", "curl"])
            ]

        # Check if download succeeded
        if any(kw in full_output for kw in ["success", "complete", "finished", "downloaded successfully"]):
            dataset_info["datasets_downloaded"] = True

        # Check if manual steps are needed
        if any(kw in full_output for kw in ["manual", "manually", "you need to", "you must", "please download", "register", "sign up", "license", "restricted"]):
            dataset_info["manual_steps_needed"] = True

        # Extract data locations
        for msg in all_messages:
            if any(path in msg.lower() for path in ["data/", "/data", "examples/", "datasets/", ".data/"]):
                # Extract the path
                import re
                paths = re.findall(r'[\w/\-\.]+/(?:data|examples|datasets)[\w/\-\.]*', msg, re.IGNORECASE)
                dataset_info["dataset_locations"].extend(paths[:3])  # Limit to 3

        # Get final report
        if all_messages:
            dataset_info["report"] = all_messages[-1] if all_messages else ""

        return dataset_info

    def _validate_datasets(self, code_path: str, dataset_info: Dict) -> bool:
        """
        Validate that datasets were actually downloaded.

        Strategy:
        1. Read README to find data/example directories
        2. Check those directories
        3. Fallback: recursive search for data directories
        """
        directories_to_check = []

        # PHASE 1: Read README and extract data/example paths
        readme_path = os.path.join(code_path, "README.md")
        if os.path.exists(readme_path):
            try:
                with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                    readme_content = f.read()

                # Look for directory mentions in README
                # Common patterns: "examples/NLG/", "data/", "datasets/MNIST"
                import re
                # Match paths like: examples/NLG, data/train, datasets/cifar10
                path_pattern = r'(?:examples|data|datasets?|scripts)/[\w/\-.]+'
                found_paths = re.findall(path_pattern, readme_content)

                for path in found_paths:
                    full_path = os.path.join(code_path, path.strip('/'))
                    if full_path not in directories_to_check:
                        directories_to_check.append(full_path)
                        # Also check parent directory
                        parent = os.path.dirname(full_path)
                        if parent not in directories_to_check:
                            directories_to_check.append(parent)

                if directories_to_check:
                    print(f"📖 Found {len(directories_to_check)} data-related paths in README")
            except Exception as e:
                print(f"⚠️  Could not read README: {e}")

        # PHASE 2: Add common root-level directories
        common_dirs = [
            os.path.join(code_path, "data"),
            os.path.join(code_path, "datasets"),
            os.path.join(code_path, "examples"),
            os.path.join(code_path, ".data"),
            os.path.join(code_path, "downloads"),
        ]
        directories_to_check.extend(common_dirs)

        # PHASE 3: Add any locations mentioned in agent report
        for location in dataset_info.get("dataset_locations", []):
            if "/" in location:
                potential_dir = location.strip()
                if not potential_dir.startswith("/"):
                    potential_dir = os.path.join(code_path, potential_dir)
                directories_to_check.append(potential_dir)

        # Remove duplicates while preserving order
        seen = set()
        directories_to_check = [d for d in directories_to_check if not (d in seen or seen.add(d))]

        # Check if any directory exists and has content
        found_data = False
        for data_dir in directories_to_check:
            if os.path.exists(data_dir) and os.path.isdir(data_dir):
                try:
                    # Check for files recursively (up to 2 levels deep)
                    file_count = 0
                    for root, dirs, files in os.walk(data_dir):
                        file_count += len([f for f in files if not f.startswith('.')])
                        # Don't go too deep
                        if root.count(os.sep) - data_dir.count(os.sep) >= 2:
                            del dirs[:]  # Don't recurse further

                    if file_count > 0:
                        rel_path = os.path.relpath(data_dir, code_path)
                        print(f"✅ Found data/examples in: {rel_path} ({file_count} files)")
                        found_data = True
                        break
                except Exception as e:
                    print(f"⚠️  Could not access {data_dir}: {e}")
                    continue

        if not found_data:
            print(f"⚠️  No data/example files found")
            print(f"   Checked {len(directories_to_check)} locations from README and common paths")

        return found_data
