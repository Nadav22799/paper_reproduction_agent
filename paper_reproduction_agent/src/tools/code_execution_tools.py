"""Tools for executing and testing code."""

import os
import subprocess
import re
from typing import Dict, Any, Optional
from pathlib import Path
from langchain.tools import tool
from pydantic import BaseModel, Field


@tool
def search_file(path: str, query: str, max_results: int = 10) -> str:
    """Search for text within a file or directory.

    Args:
        path: File or directory path to search
        query: Text to search for
        max_results: Maximum number of results to return

    Returns:
        Search results showing matches with context
    """
    try:
        results = []

        if os.path.isfile(path):
            # Search in single file
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                for i, line in enumerate(lines, 1):
                    if query.lower() in line.lower():
                        results.append(f"{path}:{i}: {line.strip()}")
                        if len(results) >= max_results:
                            break
        elif os.path.isdir(path):
            # Search in directory
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(
                        (".py", ".md", ".txt", ".sh", ".yml", ".yaml", ".json")
                    ):
                        file_path = os.path.join(root, file)
                        try:
                            with open(
                                file_path, "r", encoding="utf-8", errors="ignore"
                            ) as f:
                                lines = f.readlines()
                                for i, line in enumerate(lines, 1):
                                    if query.lower() in line.lower():
                                        results.append(
                                            f"{file_path}:{i}: {line.strip()}"
                                        )
                                        if len(results) >= max_results:
                                            return "\n".join(results)
                        except:
                            pass

        if results:
            return "\n".join(results)
        else:
            return f"No matches found for '{query}' in {path}"
    except Exception as e:
        return f"Error searching: {str(e)}"


@tool
def read_file(file_path: str, max_lines: int = 500) -> str:
    """
    Read contents of a file.

    Args:
        file_path: Path to file to read
        max_lines: Maximum number of lines to read (default: 500)

    Returns:
        File contents or error message
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            if len(lines) > max_lines:
                content = "".join(lines[:max_lines])
                content += f"\n\n... (truncated, showing first {max_lines} of {len(lines)} lines)"
            else:
                content = "".join(lines)
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def write_file(file_path: str, content: str) -> str:
    """
    Write content to a file. Overwrites existing content.

    Args:
        file_path: Path to file to write
        content: Text content to write

    Returns:
        Status message
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"File written successfully to {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


@tool
def list_directory(
    dir_path: str = ".", recursive: bool = False, max_depth: int = 3
) -> str:
    """
    List contents of a directory.

    Args:
        dir_path: Path to directory (default: current directory)
        recursive: Whether to list recursively
        max_depth: Maximum depth for recursive listing

    Returns:
        Directory listing
    """
    try:
        result = []
        path = Path(dir_path)

        if not path.exists():
            return f"Directory {dir_path} does not exist"

        if not path.is_dir():
            return f"{dir_path} is not a directory"

        if recursive:
            for item in sorted(path.rglob("*")):
                rel_path = item.relative_to(path)
                depth = len(rel_path.parts)
                if depth <= max_depth:
                    indent = "  " * (depth - 1)
                    if item.is_dir():
                        result.append(f"{indent}{item.name}/")
                    else:
                        size = item.stat().st_size
                        result.append(f"{indent}{item.name} ({size} bytes)")
        else:
            for item in sorted(path.iterdir()):
                if item.is_dir():
                    result.append(f"{item.name}/")
                else:
                    size = item.stat().st_size
                    result.append(f"{item.name} ({size} bytes)")

        return "\n".join(result) if result else "Empty directory"

    except Exception as e:
        return f"Error listing directory: {str(e)}"


def _cleanup_distributed_training():
    """
    Safely clean up leftover distributed training processes for current user only.
    This prevents 'Address already in use' errors from previous failed runs.
    """
    import os
    import getpass

    try:
        current_user = getpass.getuser()

        # Kill only current user's distributed training processes
        subprocess.run(
            f"pkill -u {current_user} -f 'torch.distributed' 2>/dev/null || true",
            shell=True,
            capture_output=True,
            timeout=10,
        )

        # Free common distributed training ports (only for current user)
        for port in [29500, 29501, 29502]:
            subprocess.run(
                f"lsof -ti :{port} -u {current_user} 2>/dev/null | xargs -r kill -9 || true",
                shell=True,
                capture_output=True,
                timeout=10,
            )

        # Set random port for this run
        import random

        os.environ["MASTER_PORT"] = str(29500 + random.randint(0, 999))

    except Exception:
        pass  # Cleanup is best-effort, don't fail if it doesn't work


@tool
def execute_shell_command(
    command: str, cwd: str = ".", timeout: int = 3600, enable_oom_handling: bool = True
) -> Dict[str, Any]:
    """
    Execute a shell command and capture output with automatic OOM handling.

    Args:
        command: Shell command to execute
        cwd: Working directory for command execution
        timeout: Execution timeout in seconds (default: 3600 = 60 minutes)
        enable_oom_handling: Whether to automatically handle OOM errors (default: True)

    Returns:
        Dictionary with stdout, stderr, return code, and OOM handling info
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )

        print(f"Running: {command}")

        # Combine output for OOM detection
        full_output = result.stdout + result.stderr

        # If command failed, show more output to help diagnose
        stdout = result.stdout
        stderr = result.stderr

        # Check for OOM error if enabled
        oom_detected = False
        oom_info = {}

        if enable_oom_handling and result.returncode != 0:
            try:
                # Import OOM handler
                import sys
                from pathlib import Path

                sys.path.insert(0, str(Path(__file__).parent.parent))
                from utils.oom_handler import OOMHandler

                oom_handler = OOMHandler()
                oom_detected = oom_handler.detect_oom_error(full_output)

                if oom_detected:
                    # Extract script path from command
                    script_match = re.search(
                        r"(?:bash|sh|python)\s+([^\s]+\.(?:sh|py))", command
                    )
                    if script_match:
                        script_path = os.path.join(cwd, script_match.group(1))

                        # Handle OOM
                        oom_result = oom_handler.handle_oom(
                            script_path=script_path,
                            error_output=full_output,
                            attempt=1,
                            max_attempts=3,
                        )

                        oom_info = {
                            "detected": True,
                            "should_retry": oom_result["should_retry"],
                            "adjusted": oom_result["adjusted"],
                            "message": oom_result["message"],
                            "script_path": script_path,
                        }

                        if oom_result["adjusted"]:
                            print(f"\n🔧 OOM Handling: {oom_result['message']}")
                            print("   Script adjusted. Retry with the same command.")
                    else:
                        oom_info = {
                            "detected": True,
                            "should_retry": False,
                            "adjusted": False,
                            "message": "OOM detected but could not identify script to adjust",
                        }
            except Exception as e:
                print(f"⚠️  OOM handler error: {e}")
                oom_info = {"detected": oom_detected, "handler_error": str(e)}

        if result.returncode != 0:
            # Show last 3000 chars of output for failed commands
            if len(stdout) > 3000:
                stdout = "...(truncated)...\n" + stdout[-3000:]
            if len(stderr) > 3000:
                stderr = "...(truncated)...\n" + stderr[-3000:]
        else:
            # For successful commands, truncate more aggressively
            if len(stdout) > 10000:
                stdout = stdout[:10000] + "\n...(truncated, command succeeded)..."
            if len(stderr) > 2000:
                stderr = stderr[:2000] + "\n...(truncated)..."

        response = {
            "stdout": stdout,
            "stderr": stderr,
            "returncode": result.returncode,
            "success": result.returncode == 0,
        }

        if oom_info:
            response["oom_info"] = oom_info

        return response

    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
            "returncode": -1,
            "success": False,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Execution error: {str(e)}",
            "returncode": -1,
            "success": False,
        }


@tool
def create_python_file(file_path: str, content: str) -> str:
    """
    Create a Python file with given content.

    Args:
        file_path: Path where file should be created
        content: Python code content

    Returns:
        Status message
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"File created successfully at {file_path}"

    except Exception as e:
        return f"Error creating file: {str(e)}"


class ExecutePythonCodeInput(BaseModel):
    """Input schema for execute_python_code tool."""

    code: str = Field(description="Python code to execute")
    working_directory: str = Field(
        default=".", description="Directory to run the code from"
    )
    timeout: int = Field(default=300, description="Execution timeout in seconds")


@tool(args_schema=ExecutePythonCodeInput)
def execute_python_code(
    code: str, working_directory: str = ".", timeout: int = 300
) -> Dict[str, Any]:
    """
    Execute Python code written by the LLM directly.

    This is the PRIMARY tool for dynamic problem-solving. Instead of using specialized
    tools with fixed patterns, write custom Python code to:
    - Parse result files in ANY format (JSON, CSV, custom text, logs)
    - Extract metrics from experiment outputs
    - Compare results with paper values using custom logic
    - Generate reports in any format
    - Analyze logs and checkpoints
    - Handle edge cases specific to the repository

    The code is saved to a temporary file and executed. Use print() to output results.
    For structured data, print JSON that can be parsed.

    Example - Extract metrics from custom format:
    ```python
    import json
    from pathlib import Path

    results = {}
    for csv_file in Path("./results").glob("**/*.csv"):
        dataset = csv_file.parent.name
        with open(csv_file) as f:
            for line in f:
                if "accuracy" in line.lower():
                    # Parse your specific format here
                    value = float(line.split()[-1])
                    results[dataset] = value

    print(json.dumps(results, indent=2))
    ```

    Example - Compare with paper results:
    ```python
    extracted = {"mnist": 99.1, "cifar10": 92.5}
    expected = {"mnist": 99.0, "cifar10": 93.0}

    for ds, exp in expected.items():
        if ds in extracted:
            error = abs(extracted[ds] - exp) / exp * 100
            status = "MATCH" if error < 5 else "MISMATCH"
            print(f"{ds}: {extracted[ds]:.2f} vs {exp:.2f} ({error:.1f}%) - {status}")
    ```

    Args:
        code: Python code to execute
        working_directory: Directory to run the code from (default: current directory)
        timeout: Execution timeout in seconds (default: 300 = 5 minutes)

    Returns:
        Dictionary with:
            - stdout: Standard output from the code
            - stderr: Standard error (errors, warnings)
            - returncode: Exit code (0 = success)
            - success: Boolean indicating success
            - script_path: Path to the temporary script file (for debugging)
    """
    import uuid

    try:
        # Create a unique temporary script file
        script_name = f"llm_script_{uuid.uuid4().hex[:8]}.py"
        script_path = os.path.join(working_directory, script_name)

        # Write the code to the script file
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            # Execute the script
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_directory,
            )

            # Clean up - remove the temporary script
            try:
                os.remove(script_path)
            except:
                pass

            # Truncate output if too long
            stdout = result.stdout
            stderr = result.stderr

            if len(stdout) > 5000:
                stdout = stdout[:2500] + "\n...(truncated)...\n" + stdout[-2500:]
            if len(stderr) > 2000:
                stderr = stderr[:1000] + "\n...(truncated)...\n" + stderr[-1000:]

            return {
                "stdout": stdout,
                "stderr": stderr,
                "returncode": result.returncode,
                "success": result.returncode == 0,
                "script_path": script_path,
            }

        except subprocess.TimeoutExpired:
            # Clean up on timeout
            try:
                os.remove(script_path)
            except:
                pass
            return {
                "stdout": "",
                "stderr": f"Code execution timed out after {timeout} seconds",
                "returncode": -1,
                "success": False,
                "script_path": script_path,
            }

    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Execution error: {str(e)}",
            "returncode": -1,
            "success": False,
            "script_path": "",
        }


class ExecutePythonScriptInput(BaseModel):
    """Input schema for execute_python_script tool."""

    script_path: str = Field(description="Path to Python script")
    args: Optional[str] = Field(
        default=None,
        description="Command line arguments as space-separated string (e.g. '--batch-size 32 --epochs 10')",
    )
    timeout: int = Field(default=300, description="Execution timeout in seconds")


@tool(args_schema=ExecutePythonScriptInput)
def execute_python_script(
    script_path: str, args: Optional[str] = None, timeout: int = 300
) -> Dict[str, Any]:
    """
    Execute a Python script and capture output.
    Automatically uses the virtual environment (venv or conda) if it exists in the repo.

    Args:
        script_path: Path to Python script
        args: Command line arguments as space-separated string
        timeout: Execution timeout in seconds

    Returns:
        Dictionary with stdout, stderr, and return code
    """
    try:
        # Check for venv/conda in the script's directory or parent directories
        script_dir = Path(script_path).parent.resolve()
        env_python = None
        env_type = None

        # Look for environments in current dir and up to 2 parent levels
        for check_dir in [script_dir, script_dir.parent, script_dir.parent.parent]:
            # First check for conda environment (check if conda env was created for this repo)
            conda_env_name = _generate_conda_env_name(str(check_dir))
            conda_python = _get_conda_env_python(conda_env_name)
            if conda_python and Path(conda_python).exists():
                env_python = conda_python
                env_type = "conda"
                print(
                    f"🐍 Using conda environment Python: {env_python} (env: {conda_env_name})"
                )
                break

            # Then check for venv
            potential_venv = check_dir / "venv"
            if potential_venv.exists():
                if os.name == "nt":  # Windows
                    venv_python_path = potential_venv / "Scripts" / "python.exe"
                else:  # Linux/Mac
                    venv_python_path = potential_venv / "bin" / "python"

                if venv_python_path.exists():
                    env_python = str(venv_python_path)
                    env_type = "venv"
                    print(f"🐍 Using virtual environment Python: {env_python}")
                    break

        # Use env python if found, otherwise system python
        cmd = [env_python if env_python else "python", script_path]
        if args:
            # Split space-separated args string into list
            cmd.extend(args.split())

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(script_path) or ".",
        )

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "success": result.returncode == 0,
            "used_env": env_python is not None,
            "env_type": env_type,  # "conda", "venv", or None
        }

    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Execution timed out after {timeout} seconds",
            "returncode": -1,
            "success": False,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Execution error: {str(e)}",
            "returncode": -1,
            "success": False,
        }


@tool
def search_error_solution(error_message: str) -> Dict[str, Any]:
    """
    Use Gemini with Google Search grounding to find solutions for an error.

    Args:
        error_message: The error message to search for

    Returns:
        Dictionary with solutions and code fixes from web search
    """
    result = {
        "solutions": [],
        "raw_response": "",
    }

    try:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return {
                "solutions": [
                    "Google GenAI SDK not found. Please install `google-genai` to use this feature."
                ],
                "raw_response": "ImportError: google.genai module not found",
            }

        # Configure Gemini
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return {
                "solutions": ["GEMINI_API_KEY not found in environment"],
                "raw_response": "",
            }

        client = genai.Client(api_key=api_key)

        prompt = f"""Search for solutions to this Python error and provide actionable fixes:

Error: {error_message[:500]}

Provide:
1. What causes this error
2. How to fix it (with code examples if needed)
3. Common solutions from Stack Overflow or GitHub issues"""

        # Enable search grounding
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            ),
        )

        if response.text:
            result["solutions"].append(response.text)
            result["raw_response"] = response.text

    except Exception as e:
        result["solutions"].append(f"Search failed: {str(e)}")

    return result


@tool
def start_background_process(
    command: str, log_file: str, cwd: str = "."
) -> Dict[str, Any]:
    """
    Start a background process. PREFERRED for all training/evaluation experiments.
    ALWAYS use this instead of execute_shell_command for experiments.

    Args:
        command: Shell command to execute
        log_file: Path to log file (stdout/stderr will be redirected here)
        cwd: Working directory

    Returns:
        Dictionary with 'pid', 'log_file', and 'start_message'
    """
    try:

        # Open log file
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        f_log = open(log_path, "w", encoding="utf-8")

        # Start detached process
        # On Windows, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP/DETACHED_PROCESS might be needed
        # But for simple agent use, just Popen is usually enough if we don't kill it.
        # We use shell=True to support complex commands (pipes, etc)
        # We use shell=True to support complex commands (pipes, etc)
        full_log_path = str(log_path.absolute())
        print(f"Running background process: {command}")
        print(f"Logging to: {full_log_path}")
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            stdout=f_log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )

        return {
            "success": True,
            "pid": process.pid,
            "log_file": str(log_path.absolute()),
            "message": f"Process started with PID {process.pid}. Logs: {log_file}",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def wait_for_process(
    pid: int, log_file: str, timeout: int = 604800, check_interval: int = 60
) -> Dict[str, Any]:
    """
    Blocks and waits for a process to finish. Checks logs locally every minute.

    CRITICAL BEHAVIOR:
    - If process finishes (success or error) -> Returns IMMEDIATELY.
    - If process runs for 7 days -> Returns timeout.
    - DOES NOT use LLM tokens while waiting.

    Args:
        pid: Process ID to wait for
        log_file: Path to log file to read updates from
        timeout: Maximum wait time in seconds (default 7 days = 604800s)
        check_interval: How often to check status in seconds (default 60s)

    Returns:
        Final status and last logs
    """
    import time
    import psutil

    start_time = time.time()
    log_path = Path(log_file)
    last_pos = 0

    print(f"⏳ Waiting for PID {pid} (Timeout: {timeout}s)...")

    try:
        while True:
            # Check timeout
            if time.time() - start_time > timeout:
                return {
                    "success": False,
                    "status": "timeout",
                    "message": f"Process {pid} timed out after {timeout}s",
                    "wall_time_seconds": time.time() - start_time,
                }

            # Check if process is running and NOT a zombie
            try:
                proc = psutil.Process(pid)
                status = proc.status()
                if status in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD]:
                    print(
                        f"\n⚠️ Process {pid} is a zombie (finished but stuck). Treating as done."
                    )
                    break
            except psutil.NoSuchProcess:
                # Process definitely gone
                break
            except Exception:
                # Permission denied or other error - assume running if pid_exists was checked
                pass

            # Streaming logs to stdout for user visibility
            try:
                if log_path.exists():
                    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                        f.seek(last_pos)
                        new_data = f.read()
                        if new_data:
                            print(new_data, end="", flush=True)
                            last_pos = f.tell()
            except Exception:
                pass

            time.sleep(check_interval)

        # Process done - read final logs
        wall_time = time.time() - start_time
        tail_logs = ""
        try:
            if log_path.exists():
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    # Read last 2000 chars
                    f.seek(0, 2)
                    size = f.tell()
                    f.seek(max(0, size - 5000))
                    tail_logs = f.read()
        except Exception:
            pass

        return {
            "success": True,
            "status": "finished",
            "message": f"Process {pid} finished.",
            "tail_logs": tail_logs[-2000:] if tail_logs else "No logs found",
            "wall_time_seconds": wall_time,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "wall_time_seconds": time.time() - start_time,
        }


@tool
def stop_process(pid: int) -> str:
    """Stop a background process."""
    try:
        import psutil

        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
        return f"Process {pid} killed"
    except Exception as e:
        return f"Error killing process {pid}: {str(e)}"


@tool
def check_python_compatibility(repo_path: str) -> Dict[str, Any]:
    """
    Check if repository is compatible with current Python version.

    Args:
        repo_path: Path to repository

    Returns:
        Dictionary with compatibility info:
            - compatible: Whether repo is compatible
            - current_version: Current Python version
            - required_version: Required Python version if found
            - warnings: List of compatibility warnings
            - suggestions: List of suggestions for fixing issues
    """
    try:
        import sys

        # Try multiple paths to find the src directory
        possible_paths = [
            str(Path(__file__).parent.parent),  # From tools/ to src/
            str(
                Path(__file__).parent.parent.parent / "src"
            ),  # From paper_reproduction_agent/src/tools to src
            os.path.join(os.path.dirname(__file__), "..", ".."),  # Relative path
        ]

        # Add all possible paths
        for path in possible_paths:
            abs_path = str(Path(path).resolve())
            if abs_path not in sys.path:
                sys.path.insert(0, abs_path)

        from utils.python_compatibility import (
            check_python_compatibility as check_compat,
        )

        result = check_compat(repo_path)

        return {
            "compatible": result.get("compatible", True),
            "current_version": result.get("current_version", "unknown"),
            "required_version": result.get("required_version", "not specified"),
            "confidence": result.get("confidence", "low"),
            "warnings": result.get("warnings", []),
            "suggestions": result.get("suggestions", []),
            "success": True,
        }

    except ImportError as e:
        return {
            "compatible": True,  # Assume compatible if check fails
            "current_version": "unknown",
            "required_version": "unknown",
            "warnings": [
                f"Compatibility check module not found: {str(e)}. Paths tried: {sys.path[:5]}"
            ],
            "suggestions": ["Ensure paper_reproduction_agent/src is in PYTHONPATH"],
            "success": False,
        }
    except Exception as e:
        return {
            "compatible": True,  # Assume compatible if check fails
            "current_version": "unknown",
            "required_version": "unknown",
            "warnings": [f"Compatibility check failed: {str(e)}"],
            "suggestions": [],
            "success": False,
        }


# ============================================================================
# Python Version Management Helpers
# ============================================================================


def _parse_required_python_version(required_version: str) -> Optional[str]:
    """
    Parse required version string to get a specific Python version to install.

    Examples:
        ">=3.6,<3.10" -> "3.9"
        "3.6-3.8" -> "3.8"
        ">=3.7" -> None (use current)
        "3.9+" -> "3.9"

    Args:
        required_version: Version requirement string

    Returns:
        Specific Python version to install (e.g., "3.9") or None if current is fine
    """
    import re
    import sys

    if not required_version or required_version == "not specified":
        return None

    try:
        current_major, current_minor = sys.version_info.major, sys.version_info.minor

        # Handle range like "3.6-3.8"
        if "-" in required_version and not required_version.startswith(">="):
            parts = required_version.split("-")
            max_ver = parts[1].strip()
            # Extract version numbers
            match = re.search(r"(\d+)\.(\d+)", max_ver)
            if match:
                return f"{match.group(1)}.{match.group(2)}"

        # Handle ">=3.6,<3.10" - get the upper bound
        if "<" in required_version:
            upper_bound = required_version.split("<")[1].strip()
            match = re.search(r"(\d+)\.(\d+)", upper_bound)
            if match:
                major, minor = int(match.group(1)), int(match.group(2))
                # Suggest one version below the upper bound
                suggested_minor = max(0, minor - 1)
                suggested_version = f"{major}.{suggested_minor}"

                # Check if current Python is compatible
                if (current_major, current_minor) < (major, suggested_minor):
                    return suggested_version
                # If current is within range, use it
                return None

        # Handle ">=3.6" - use current if it satisfies, otherwise return minimum
        if ">=" in required_version:
            min_ver = required_version.split(">=")[1].split(",")[0].strip()
            match = re.search(r"(\d+)\.(\d+)", min_ver)
            if match:
                major, minor = int(match.group(1)), int(match.group(2))
                # If current Python is older than required, return the minimum
                if (current_major, current_minor) < (major, minor):
                    return f"{major}.{minor}"
            # Current Python is fine
            return None

        # Handle "3.9+" or "3.9"
        if "+" in required_version or re.match(r"^\d+\.\d+$", required_version):
            version_str = required_version.replace("+", "").strip()
            match = re.search(r"(\d+)\.(\d+)", version_str)
            if match:
                major, minor = int(match.group(1)), int(match.group(2))
                # If current is older, return the required version
                if (current_major, current_minor) < (major, minor):
                    return f"{major}.{minor}"
            # Current is fine
            return None

    except Exception as e:
        print(f"⚠️  Error parsing required version '{required_version}': {e}")

    return None


def _check_pyenv_available() -> bool:
    """Check if pyenv is available on the system."""
    import shutil

    return shutil.which("pyenv") is not None


def _get_or_install_pyenv_python(python_version: str) -> Optional[str]:
    """
    Get pyenv Python path, installing it if necessary.

    Args:
        python_version: Python version like "3.9" or "3.10"

    Returns:
        Path to Python executable or None if failed
    """
    try:
        # Check if pyenv has this version installed
        result = subprocess.run(
            ["pyenv", "versions", "--bare"], capture_output=True, text=True, timeout=10
        )

        if result.returncode != 0:
            print(f"⚠️  Failed to check pyenv versions: {result.stderr}")
            return None

        available_versions = result.stdout.strip().split("\n")

        # Find exact match or closest match
        matching_version = None
        for version in available_versions:
            if version.startswith(python_version):
                matching_version = version
                break

        # Install if not found
        if not matching_version:
            print(f"📥 Installing Python {python_version} via pyenv...")
            print("   This may take a few minutes on first install...")

            # Get latest patch version for this minor version
            versions_result = subprocess.run(
                ["pyenv", "install", "--list"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            # Find latest patch version (e.g., for 3.9, find 3.9.18)
            import re

            pattern = re.compile(rf"^\s*({python_version}\.\d+)\s*$", re.MULTILINE)
            matches = pattern.findall(versions_result.stdout)

            if matches:
                # Get the latest patch version
                latest_version = sorted(
                    matches, key=lambda v: tuple(map(int, v.split(".")))
                )[-1]
                matching_version = latest_version

                print(f"   Installing Python {matching_version}...")
                install_result = subprocess.run(
                    ["pyenv", "install", "-s", matching_version],  # -s = skip if exists
                    capture_output=True,
                    text=True,
                    timeout=600,
                )

                if install_result.returncode != 0:
                    print(f"❌ pyenv install failed: {install_result.stderr[:300]}")
                    return None

                print(f"✅ Python {matching_version} installed via pyenv")
            else:
                print(f"⚠️  Could not find Python {python_version} in pyenv repository")
                return None
        else:
            print(f"✅ Found Python {matching_version} in pyenv")

        # Get the Python executable path
        pyenv_prefix_result = subprocess.run(
            ["pyenv", "prefix", matching_version],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if pyenv_prefix_result.returncode != 0:
            print(f"⚠️  Failed to get pyenv prefix: {pyenv_prefix_result.stderr}")
            return None

        pyenv_prefix = pyenv_prefix_result.stdout.strip()
        pyenv_python = f"{pyenv_prefix}/bin/python"

        if not os.path.exists(pyenv_python):
            print(f"⚠️  Python executable not found at {pyenv_python}")
            return None

        # Verify the version
        version_check = subprocess.run(
            [pyenv_python, "--version"], capture_output=True, text=True, timeout=5
        )
        print(f"   Python executable: {pyenv_python}")
        print(f"   Version: {version_check.stdout.strip()}")

        return pyenv_python

    except Exception as e:
        print(f"❌ Error with pyenv: {str(e)}")
        return None


# ============================================================================
# Conda Environment Management Helpers
# ============================================================================


def _check_conda_available() -> bool:
    """Check if conda is available on the system."""
    import shutil

    return shutil.which("conda") is not None


def _detect_conda_requirements(repo_path: str) -> Optional[Path]:
    """
    Detect if repository uses conda by looking for environment files.

    Args:
        repo_path: Path to repository

    Returns:
        Path to conda environment file if found, None otherwise
    """
    repo = Path(repo_path)
    candidates = [
        "environment.yml",
        "environment.yaml",
        "conda_environment.yml",
        "conda_environment.yaml",
        "env.yml",
        "env.yaml",
        "requirements.yaml",
        "requirements.yml",
    ]

    for candidate in candidates:
        env_file = repo / candidate
        if env_file.exists():
            # For requirements.yaml/yml, verify it's conda format (has channels/dependencies)
            if "requirements" in candidate:
                try:
                    with open(env_file, "r") as f:
                        content = f.read()
                        if "channels:" in content or "dependencies:" in content:
                            return env_file
                except:
                    pass
            else:
                return env_file

    return None


def _generate_conda_env_name(repo_path: str) -> str:
    """
    Generate a unique conda environment name for a repository.

    Args:
        repo_path: Path to repository

    Returns:
        Environment name like "paper_repo_name_abc123"
    """
    import hashlib

    repo = Path(repo_path).resolve()
    repo_name = repo.name

    # Create short hash of full path to ensure uniqueness
    path_hash = hashlib.md5(str(repo).encode()).hexdigest()[:8]

    # Sanitize repo name (conda env names can't have special chars)
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", repo_name)

    return f"paper_{safe_name}_{path_hash}"


def _create_conda_env(
    env_name: str, python_version: Optional[str] = None, env_file: Optional[Path] = None
) -> bool:
    """
    Create conda environment with specific Python version or from environment file.

    Args:
        env_name: Name for the conda environment
        python_version: Python version like "3.9" (if creating from scratch)
        env_file: Path to environment.yml file (if creating from file)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if environment already exists
        list_result = subprocess.run(
            ["conda", "env", "list"], capture_output=True, text=True, timeout=30
        )

        if env_name in list_result.stdout:
            print(f"✅ Conda environment '{env_name}' already exists")
            return True

        if env_file and env_file.exists():
            # Create from environment.yml file
            print(f"📦 Creating conda environment '{env_name}' from {env_file.name}...")
            print("   This may take several minutes...")

            result = subprocess.run(
                ["conda", "env", "create", "-n", env_name, "-f", str(env_file)],
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minutes for conda install
            )
        else:
            # Create from scratch with Python version
            python_spec = f"python={python_version}" if python_version else "python"
            print(f"📦 Creating conda environment '{env_name}' with {python_spec}...")
            print("   This may take several minutes...")

            result = subprocess.run(
                ["conda", "create", "-n", env_name, python_spec, "-y"],
                capture_output=True,
                text=True,
                timeout=1800,
            )

        if result.returncode != 0:
            print("❌ Conda environment creation failed:")
            print(f"   {result.stderr[:500]}")
            return False

        print(f"✅ Conda environment '{env_name}' created successfully")
        return True

    except subprocess.TimeoutExpired:
        print("❌ Conda environment creation timed out (>30 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error creating conda environment: {str(e)}")
        return False


def _get_conda_env_python(env_name: str) -> Optional[str]:
    """
    Get Python executable path from conda environment.

    Args:
        env_name: Name of conda environment

    Returns:
        Path to Python executable or None if failed
    """
    try:
        # Get conda environment info
        result = subprocess.run(
            ["conda", "env", "list"], capture_output=True, text=True, timeout=10
        )

        if result.returncode != 0:
            print(f"⚠️  Failed to list conda environments: {result.stderr}")
            return None

        # Parse environment list to find path
        for line in result.stdout.split("\n"):
            if env_name in line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    env_path = parts[-1]  # Last part is the path

                    # Construct Python path
                    if os.name == "nt":  # Windows
                        python_path = Path(env_path) / "python.exe"
                    else:  # Linux/Mac
                        python_path = Path(env_path) / "bin" / "python"

                    if python_path.exists():
                        print(f"✅ Found conda Python: {python_path}")
                        return str(python_path)

        # Alternative method: use conda run to find python
        result = subprocess.run(
            ["conda", "run", "-n", env_name, "which", "python"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            python_path = result.stdout.strip()
            if python_path and Path(python_path).exists():
                print(f"✅ Found conda Python via 'conda run': {python_path}")
                return python_path

        print(f"⚠️  Could not find Python executable for conda env '{env_name}'")
        return None

    except Exception as e:
        print(f"❌ Error getting conda environment Python: {str(e)}")
        return None


def _get_conda_pip(env_name: str) -> Optional[str]:
    """
    Get pip executable path from conda environment.

    Args:
        env_name: Name of conda environment

    Returns:
        Path to pip executable or command to run pip
    """
    python_path = _get_conda_env_python(env_name)
    if not python_path:
        return None

    # Check if pip exists in conda environment
    if os.name == "nt":  # Windows
        pip_path = Path(python_path).parent / "pip.exe"
    else:  # Linux/Mac
        pip_path = Path(python_path).parent / "pip"

    if pip_path.exists():
        return str(pip_path)

    # Fallback: use python -m pip
    return f"{python_path} -m pip"


# ============================================================================
# Virtual Environment Management Helpers
# ============================================================================


def _create_venv_with_python(venv_path: Path, python_executable: str) -> bool:
    """
    Create virtual environment using specific Python executable.

    Args:
        venv_path: Path where venv should be created
        python_executable: Path to Python executable to use

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"📦 Creating venv at {venv_path}...")
        print(f"   Using: {python_executable}")

        result = subprocess.run(
            [python_executable, "-m", "venv", str(venv_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            print(f"❌ venv creation failed: {result.stderr[:300]}")
            return False

        # Verify venv was created
        if os.name == "nt":
            venv_python = venv_path / "Scripts" / "python.exe"
        else:
            venv_python = venv_path / "bin" / "python"

        if not venv_python.exists():
            print(f"❌ venv Python not found at {venv_python}")
            return False

        print("✅ Virtual environment created successfully")
        return True

    except Exception as e:
        print(f"❌ Error creating venv: {str(e)}")
        return False


def _install_with_conda(repo_path: str, conda_env_file: Path) -> Dict[str, Any]:
    """
    Install dependencies using conda in an isolated conda environment.

    Args:
        repo_path: Path to repository
        conda_env_file: Path to environment.yml file

    Returns:
        Installation result with status and any errors
    """
    try:

        # Generate unique conda environment name
        env_name = _generate_conda_env_name(repo_path)

        print("📦 Installing with Conda")
        print(f"   Environment name: {env_name}")
        print(f"   Environment file: {conda_env_file.name}")
        print(
            "   This creates an ISOLATED conda environment (won't affect your main env)\n"
        )

        # Step 1: Create conda environment from environment.yml
        success = _create_conda_env(env_name, env_file=conda_env_file)

        if not success:
            return {
                "success": False,
                "error": f"Failed to create conda environment from {conda_env_file.name}",
                "env_type": "conda",
                "env_name": env_name,
            }

        # Step 2: Get Python and pip paths from conda environment
        conda_python = _get_conda_env_python(env_name)
        conda_pip = _get_conda_pip(env_name)

        if not conda_python:
            return {
                "success": False,
                "error": f"Failed to locate Python in conda environment '{env_name}'",
                "env_type": "conda",
                "env_name": env_name,
            }

        # Step 3: Verify installation
        print("\n✅ Conda environment created successfully!")
        print(f"   Environment name: {env_name}")
        print(f"   Python: {conda_python}")
        print(f"   Pip: {conda_pip}")
        print(f"\n   To activate: conda activate {env_name}")
        print(f"   To remove: conda env remove -n {env_name}\n")

        # Get Python version from conda env
        version_result = subprocess.run(
            [conda_python, "--version"], capture_output=True, text=True, timeout=10
        )
        python_version = (
            version_result.stdout.strip()
            if version_result.returncode == 0
            else "unknown"
        )

        return {
            "success": True,
            "env_type": "conda",
            "env_name": env_name,
            "conda_python": conda_python,
            "conda_pip": conda_pip,
            "python_version_used": python_version,
            "environment_file": str(conda_env_file),
            "activation_command": f"conda activate {env_name}",
            "removal_command": f"conda env remove -n {env_name}",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Conda installation error: {str(e)}",
            "env_type": "conda",
        }


@tool
def smart_install_dependencies(repo_path: str) -> Dict[str, Any]:
    """
    Install dependencies using intelligent fallback strategies in an isolated environment.

    This tool will:
    1. Detect if repo uses conda (environment.yml) or pip (requirements.txt)
    2. Check required Python version for the repo
    3. Create isolated environment (conda env or venv) with correct Python version
    4. Try installing with original requirements
    5. If that fails, try with relaxed version constraints (pip only)
    6. If that fails, try with unpinned versions (pip only)

    Args:
        repo_path: Path to repository

    Returns:
        Installation result with status and any errors
    """
    try:
        import sys
        import venv
        import shutil

        # Step 0: Detect if repo uses conda
        print("🔍 Detecting dependency management system...")
        conda_env_file = _detect_conda_requirements(repo_path)
        use_conda = conda_env_file is not None and _check_conda_available()

        if conda_env_file and not _check_conda_available():
            print(f"⚠️  Found {conda_env_file.name} but conda is not installed")
            print("   Install conda: https://docs.conda.io/en/latest/miniconda.html")
            print("   Falling back to pip installation...\n")
            use_conda = False
        elif use_conda:
            print(f"✅ Detected conda requirements: {conda_env_file.name}")
            print("   Will use conda for installation\n")
        else:
            print("✅ Using pip/venv for installation\n")

        # =====================================================================
        # CONDA INSTALLATION PATH
        # =====================================================================
        if use_conda:
            return _install_with_conda(repo_path, conda_env_file)

        # =====================================================================
        # PIP/VENV INSTALLATION PATH (original logic)
        # =====================================================================
        venv_path = Path(repo_path) / "venv"

        # Step 1: Check Python version compatibility
        print("🔍 Checking Python version requirements...")
        compat_check = check_python_compatibility.invoke({"repo_path": repo_path})

        required_version = compat_check.get("required_version")
        current_version = compat_check.get(
            "current_version", f"{sys.version_info.major}.{sys.version_info.minor}"
        )
        is_compatible = compat_check.get("compatible", True)

        print(f"   Current Python: {current_version}")
        print(f"   Required Python: {required_version or 'not specified'}")

        # Step 2: Determine if we need a different Python version
        target_python_version = (
            _parse_required_python_version(required_version)
            if required_version
            else None
        )

        python_executable = None  # Will hold the Python to use for venv

        if target_python_version and target_python_version != current_version:
            print(
                f"\n⚠️  Repository requires Python {target_python_version}, but you have {current_version}"
            )
            print(
                f"   Attempting to create environment with Python {target_python_version}...\n"
            )

            # Try pyenv first
            if _check_pyenv_available():
                print("✅ pyenv is available")
                python_executable = _get_or_install_pyenv_python(target_python_version)

                if python_executable:
                    print(f"✅ Will use Python {target_python_version} from pyenv\n")
                else:
                    print(f"⚠️  Failed to get Python {target_python_version} from pyenv")
                    print(f"   Trying system Python {target_python_version}...\n")

                    # Try system Python as fallback
                    python_executable = shutil.which(f"python{target_python_version}")
                    if not python_executable:
                        print(f"❌ Python {target_python_version} not found on system")
                        print("   Install options:")
                        print("   1. Install pyenv: curl https://pyenv.run | bash")
                        print(
                            f"   2. Install Python {target_python_version} system-wide"
                        )
                        print(
                            f"   3. Use conda: conda create -n env python={target_python_version}"
                        )
                        print(
                            f"\n   Falling back to current Python {current_version} (may fail)...\n"
                        )
                        python_executable = sys.executable
            else:
                print("⚠️  pyenv not available")
                print("   To install pyenv: curl https://pyenv.run | bash")
                print(f"   Checking for system Python {target_python_version}...\n")

                # Try to find this Python version on the system
                python_executable = shutil.which(f"python{target_python_version}")
                if not python_executable:
                    print(f"❌ Python {target_python_version} not found")
                    print(
                        f"   Falling back to current Python {current_version} (may fail)...\n"
                    )
                    python_executable = sys.executable
                else:
                    print(
                        f"✅ Found system Python {target_python_version} at {python_executable}\n"
                    )
        else:
            # Current Python is compatible or no specific version required
            if not is_compatible:
                print(
                    f"⚠️  Warning: Compatibility check suggests issues, but continuing with Python {current_version}\n"
                )
            else:
                print(f"✅ Python {current_version} is compatible\n")
            python_executable = sys.executable

        # Step 3: Create virtual environment
        if venv_path.exists():
            print(f"📦 Virtual environment already exists at {venv_path}")
            print(
                f"   To recreate with different Python, delete it first: rm -rf {venv_path}\n"
            )
        else:
            # Create venv with the selected Python
            if python_executable == sys.executable:
                # Use built-in venv module
                print("📦 Creating virtual environment with current Python...")
                try:
                    venv.create(venv_path, with_pip=True)
                    print("✅ Virtual environment created!\n")
                except Exception as e:
                    print(f"❌ venv.create() failed: {e}")
                    print("   Trying subprocess method instead...\n")
                    success = _create_venv_with_python(venv_path, python_executable)
                    if not success:
                        return {
                            "success": False,
                            "error": f"Failed to create venv: {e}",
                            "python_version_required": target_python_version,
                            "python_version_current": current_version,
                        }
            else:
                # Use subprocess to call specific Python version
                success = _create_venv_with_python(venv_path, python_executable)
                if not success:
                    return {
                        "success": False,
                        "error": f"Failed to create venv with Python {python_executable}",
                        "python_version_required": target_python_version,
                        "python_version_current": current_version,
                    }
                print()

        # Determine venv python and pip paths
        if os.name == "nt":  # Windows
            venv_python = venv_path / "Scripts" / "python.exe"
            venv_pip = venv_path / "Scripts" / "pip.exe"
        else:  # Linux/Mac
            venv_python = venv_path / "bin" / "python"
            venv_pip = venv_path / "bin" / "pip"

        # Convert to absolute paths to avoid issues when cwd changes in subprocess
        venv_python = venv_python.resolve()
        venv_pip = venv_pip.resolve()

        # CRITICAL: Verify venv was created properly
        if not venv_python.exists():
            print("❌ Virtual environment creation failed!")
            print(f"   Expected Python at: {venv_python}")
            print("   File does not exist.")
            print("\n   Attempting to recreate with subprocess method...\n")

            # Remove broken venv
            import shutil

            if venv_path.exists():
                shutil.rmtree(venv_path)

            # Try subprocess method
            success = _create_venv_with_python(venv_path, python_executable)
            if not success or not venv_python.exists():
                return {
                    "success": False,
                    "error": f"Virtual environment creation failed - {venv_python} does not exist after creation",
                    "venv_path": str(venv_path),
                    "python_executable": python_executable,
                }

        # Verify pip exists
        if not venv_pip.exists():
            print(f"⚠️  Warning: pip not found at {venv_pip}")
            print("   Attempting to bootstrap pip...\n")

            # Try to bootstrap pip
            bootstrap_result = subprocess.run(
                [str(venv_python), "-m", "ensurepip", "--upgrade"],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if bootstrap_result.returncode != 0 or not venv_pip.exists():
                # Try alternative: use venv python with -m pip
                print(f"   Using '{venv_python} -m pip' instead of direct pip binary\n")
                # We'll use python -m pip for all subsequent commands
                venv_pip = f"{venv_python} -m pip"
            else:
                print("✅ Pip bootstrapped successfully\n")

        # Upgrade pip in the venv first
        print("📦 Upgrading pip in virtual environment...")
        upgrade_result = subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if upgrade_result.returncode == 0:
            print("✅ Pip upgraded successfully!\n")
        else:
            print(
                f"⚠️  Pip upgrade had issues (continuing anyway): {upgrade_result.stderr[:200]}\n"
            )

        # Try multiple paths to find the src directory
        possible_paths = [
            str(Path(__file__).parent.parent),  # From tools/ to src/
            str(
                Path(__file__).parent.parent.parent / "src"
            ),  # From paper_reproduction_agent/src/tools to src
            os.path.join(os.path.dirname(__file__), "..", ".."),  # Relative path
        ]

        # Add all possible paths
        for path in possible_paths:
            abs_path = str(Path(path).resolve())
            if abs_path not in sys.path:
                sys.path.insert(0, abs_path)

        from utils.dependency_resolver import DependencyResolver

        resolver = DependencyResolver(repo_path)
        result = resolver.resolve_and_install()

        # Actually execute the installation based on the strategy
        if result.get("attempts"):
            for attempt in result["attempts"]:
                if attempt.get("command"):
                    # Replace 'pip' with venv pip in the command
                    venv_command = attempt["command"].replace(
                        "pip install", f"{venv_pip} install"
                    )

                    print(
                        f"\n📦 Installing dependencies using strategy: {attempt['strategy']}"
                    )
                    print(f"   Command: {venv_command}")
                    print(f"   Virtual environment: {venv_path}")
                    print(
                        "   (This may take several minutes for large packages like PyTorch...)\n"
                    )

                    # Use Popen to show real-time output
                    process = subprocess.Popen(
                        venv_command,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,  # Combine stderr with stdout
                        text=True,
                        cwd=repo_path,
                        bufsize=1,  # Line buffered
                        universal_newlines=True,
                    )

                    # Stream output line by line
                    stdout_lines = []
                    for line in process.stdout:
                        print(line, end="")  # Show real-time progress
                        stdout_lines.append(line)

                    # Wait for completion
                    returncode = process.wait(timeout=600)
                    full_output = "".join(stdout_lines)

                    if returncode == 0:
                        print(
                            f"\n✅ Installation successful with {attempt['strategy']} strategy!\n"
                        )
                        print(f"🐍 Virtual environment created at: {venv_path}")
                        print(
                            f"   Python version: {target_python_version or current_version}"
                        )
                        print(f"   Python: {venv_python}")

                        # Check if we're using python -m pip or direct pip
                        if isinstance(venv_pip, str) and "-m pip" in venv_pip:
                            print(
                                f"   Pip: Using '{venv_python} -m pip' (no direct pip binary)"
                            )
                        else:
                            print(f"   Pip: {venv_pip}")

                        print(
                            f"   To activate: source {venv_path}/bin/activate (Linux/Mac) or {venv_path}\\Scripts\\activate (Windows)\n"
                        )
                        return {
                            "success": True,
                            "strategy_used": attempt["strategy"],
                            "venv_path": str(venv_path),
                            "venv_python": str(venv_python),
                            "venv_pip": str(venv_pip),
                            "python_version_used": target_python_version
                            or current_version,
                            "python_version_required": required_version,
                            "python_executable_used": python_executable,
                            "stdout": full_output,
                            "stderr": "",
                            "warnings": attempt.get("warning", ""),
                        }
                    else:
                        print(
                            f"\n❌ Installation failed with {attempt['strategy']} strategy, trying next...\n"
                        )
                        # Add error to resolver for pattern detection
                        resolver.error_detector.add_error(full_output)

                        # Check if we should stop
                        should_stop, reason = resolver.error_detector.should_stop()
                        if should_stop:
                            return {
                                "success": False,
                                "error": f"Stopped early: {reason}",
                                "stderr": full_output,
                            }

        return {
            "success": False,
            "error": "All installation strategies failed",
            "attempts": result.get("attempts", []),
        }

    except ImportError as e:
        return {
            "success": False,
            "error": f"Dependency resolver module not found: {str(e)}. Paths tried: {sys.path[:5]}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Installation error: {str(e)}",
        }
