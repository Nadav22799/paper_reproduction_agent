"""Tools for executing and testing code."""

import os
import subprocess
import json
from typing import Dict, List, Any, Optional
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
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
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
                    if file.endswith(('.py', '.md', '.txt', '.sh', '.yml', '.yaml', '.json')):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                lines = f.readlines()
                                for i, line in enumerate(lines, 1):
                                    if query.lower() in line.lower():
                                        results.append(f"{file_path}:{i}: {line.strip()}")
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
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            if len(lines) > max_lines:
                content = ''.join(lines[:max_lines])
                content += f"\n\n... (truncated, showing first {max_lines} of {len(lines)} lines)"
            else:
                content = ''.join(lines)
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def list_directory(dir_path: str = ".", recursive: bool = False, max_depth: int = 3) -> str:
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
            shell=True, capture_output=True, timeout=10
        )

        # Free common distributed training ports (only for current user)
        for port in [29500, 29501, 29502]:
            subprocess.run(
                f"lsof -ti :{port} -u {current_user} 2>/dev/null | xargs -r kill -9 || true",
                shell=True, capture_output=True, timeout=10
            )

        # Set random port for this run
        import random
        os.environ['MASTER_PORT'] = str(29500 + random.randint(0, 999))

    except Exception:
        pass  # Cleanup is best-effort, don't fail if it doesn't work


@tool
def execute_shell_command(command: str, cwd: str = ".", timeout: int = 1800) -> Dict[str, Any]:
    """
    Execute a shell command and capture output.

    Args:
        command: Shell command to execute
        cwd: Working directory for command execution
        timeout: Execution timeout in seconds (default: 1800 = 30 minutes)

    Returns:
        Dictionary with stdout, stderr, and return code
    """
    # # Auto-cleanup and setup before distributed training commands
    # distributed_keywords = ['torch.distributed', 'torchrun', 'nproc_per_node', 'distributed.launch']
    # training_script_patterns = ['_mnli', '_sst', '_mrpc', '_cola', '_qnli', 'train.sh', 'finetune']

    # is_distributed = any(kw in command for kw in distributed_keywords)
    # is_training_script = any(pat in command.lower() for pat in training_script_patterns)

    # if is_distributed or is_training_script:
    #     _cleanup_distributed_training()

    #     # For training scripts with long timeout (>1800s), do a quick test first
    #     if timeout > 1800:
    #         print(f"🔍 Running 60-second quick test to catch early errors...")

    #         # Run quick test
    #         try:
    #             quick_result = subprocess.run(
    #                 f"timeout 60 {command}",
    #                 shell=True,
    #                 capture_output=True,
    #                 text=True,
    #                 timeout=90,  # 90s to allow for timeout command
    #                 cwd=cwd
    #             )

    #             output = quick_result.stdout + quick_result.stderr

    #             # Check for fatal errors
    #             error_patterns = ['cuda error', 'invalid device', 'runtimeerror',
    #                               'address already in use', 'modulenotfounderror']
    #             has_error = any(pat in output.lower() for pat in error_patterns)

    #             if has_error:
    #                 print(f"❌ Quick test detected errors! Check output below.")
    #                 return {
    #                     "stdout": quick_result.stdout[-2000:] if len(quick_result.stdout) > 2000 else quick_result.stdout,
    #                     "stderr": quick_result.stderr[-2000:] if len(quick_result.stderr) > 2000 else quick_result.stderr,
    #                     "returncode": 1,
    #                     "success": False,
    #                     "quick_test": "FAILED - errors detected in first 60 seconds"
    #                 }
    #             else:
    #                 print(f"✅ Quick test passed - running full experiment...")

    #         except Exception as e:
    #             print(f"⚠️ Quick test error: {e} - proceeding with full run")

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd
        )

        # If command failed, show more output to help diagnose
        stdout = result.stdout
        stderr = result.stderr
        if result.returncode != 0:
            # Show last 3000 chars of output for failed commands
            if len(stdout) > 3000:
                stdout = "...(truncated)...\n" + stdout[-3000:]
            if len(stderr) > 3000:
                stderr = "...(truncated)...\n" + stderr[-3000:]
        else:
            # For successful commands, truncate more aggressively
            if len(stdout) > 1500:
                stdout = stdout[:1500] + "\n...(truncated, command succeeded)..."
            if len(stderr) > 500:
                stderr = stderr[:500] + "\n...(truncated)..."

        return {
            "stdout": stdout,
            "stderr": stderr,
            "returncode": result.returncode,
            "success": result.returncode == 0,
        }

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

        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

        return f"File created successfully at {file_path}"

    except Exception as e:
        return f"Error creating file: {str(e)}"


class ExecutePythonScriptInput(BaseModel):
    """Input schema for execute_python_script tool."""
    script_path: str = Field(description="Path to Python script")
    args: Optional[str] = Field(default=None, description="Command line arguments as space-separated string (e.g. '--batch-size 32 --epochs 10')")
    timeout: int = Field(default=300, description="Execution timeout in seconds")


@tool(args_schema=ExecutePythonScriptInput)
def execute_python_script(script_path: str, args: Optional[str] = None, timeout: int = 300) -> Dict[str, Any]:
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
                print(f"🐍 Using conda environment Python: {env_python} (env: {conda_env_name})")
                break

            # Then check for venv
            potential_venv = check_dir / "venv"
            if potential_venv.exists():
                if os.name == 'nt':  # Windows
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
            cwd=os.path.dirname(script_path) or "."
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
def install_dependencies(requirements: List[str], use_pip: bool = True) -> Dict[str, Any]:
    """
    Install Python dependencies.

    Args:
        requirements: List of package names or requirements.txt path
        use_pip: Use pip for installation (otherwise conda)

    Returns:
        Installation result
    """
    try:
        if len(requirements) == 1 and requirements[0].endswith('.txt'):
            # Install from requirements file
            cmd = ["pip", "install", "-r", requirements[0]]
        else:
            # Install individual packages
            cmd = ["pip", "install"] + requirements

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
        }

    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Installation error: {str(e)}",
            "success": False,
        }


@tool
def run_pytest(test_path: str, args: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Run pytest on a test file or directory.
    Automatically uses the virtual environment (venv or conda) if it exists in the repo.

    Args:
        test_path: Path to test file or directory
        args: Additional pytest arguments

    Returns:
        Test results
    """
    try:
        # Check for venv/conda in the test path's directory or parent directories
        test_dir = Path(test_path).resolve()
        if test_dir.is_file():
            test_dir = test_dir.parent

        env_python = None
        env_type = None

        # Look for environments in current dir and up to 2 parent levels
        for check_dir in [test_dir, test_dir.parent, test_dir.parent.parent]:
            # First check for conda environment
            conda_env_name = _generate_conda_env_name(str(check_dir))
            conda_python = _get_conda_env_python(conda_env_name)
            if conda_python and Path(conda_python).exists():
                env_python = conda_python
                env_type = "conda"
                print(f"🐍 Using conda environment Python for pytest: {env_python} (env: {conda_env_name})")
                break

            # Then check for venv
            potential_venv = check_dir / "venv"
            if potential_venv.exists():
                if os.name == 'nt':  # Windows
                    venv_python_path = potential_venv / "Scripts" / "python.exe"
                else:  # Linux/Mac
                    venv_python_path = potential_venv / "bin" / "python"

                if venv_python_path.exists():
                    env_python = str(venv_python_path)
                    env_type = "venv"
                    print(f"🐍 Using virtual environment Python for pytest: {env_python}")
                    break

        # Use env python with pytest module if found, otherwise system pytest
        if env_python:
            cmd = [env_python, "-m", "pytest", test_path, "-v", "--tb=short"]
        else:
            cmd = ["pytest", test_path, "-v", "--tb=short"]

        if args:
            cmd.extend(args)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "success": result.returncode == 0,
            "used_env": env_python is not None,
            "env_type": env_type,  # "conda", "venv", or None
        }

    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Test execution error: {str(e)}",
            "returncode": -1,
            "success": False,
        }


@tool
def check_python_syntax(file_path: str) -> Dict[str, Any]:
    """
    Check Python file for syntax errors.

    Args:
        file_path: Path to Python file

    Returns:
        Syntax check results
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        compile(code, file_path, 'exec')

        return {
            "valid": True,
            "message": "No syntax errors found",
        }

    except SyntaxError as e:
        return {
            "valid": False,
            "message": f"Syntax error at line {e.lineno}: {e.msg}",
            "line": e.lineno,
            "text": e.text,
        }
    except Exception as e:
        return {
            "valid": False,
            "message": f"Error checking syntax: {str(e)}",
        }


@tool
def run_linter(file_path: str, linter: str = "ruff") -> Dict[str, Any]:
    """
    Run a linter on Python code.

    Args:
        file_path: Path to Python file
        linter: Linter to use (ruff, pylint, flake8)

    Returns:
        Linting results
    """
    try:
        result = subprocess.run(
            [linter, file_path],
            capture_output=True,
            text=True,
            timeout=60
        )

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "issues_found": result.returncode != 0,
        }

    except FileNotFoundError:
        return {
            "stdout": "",
            "stderr": f"{linter} not found. Please install it.",
            "issues_found": False,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Linting error: {str(e)}",
            "issues_found": False,
        }


@tool
def capture_execution_metrics(script_path: str, metric_types: List[str]) -> Dict[str, Any]:
    """
    Execute script and capture performance metrics.

    Args:
        script_path: Path to script
        metric_types: Types of metrics to capture (time, memory, gpu)

    Returns:
        Execution metrics
    """
    import time
    import psutil

    try:
        process = psutil.Popen(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Track metrics
        start_time = time.time()
        peak_memory = 0

        while process.poll() is None:
            try:
                proc = psutil.Process(process.pid)
                memory_info = proc.memory_info()
                peak_memory = max(peak_memory, memory_info.rss)
            except psutil.NoSuchProcess:
                break
            time.sleep(0.1)

        end_time = time.time()
        stdout, stderr = process.communicate()

        return {
            "execution_time": end_time - start_time,
            "peak_memory_mb": peak_memory / (1024 * 1024),
            "returncode": process.returncode,
            "success": process.returncode == 0,
            "stdout": stdout.decode('utf-8', errors='ignore'),
            "stderr": stderr.decode('utf-8', errors='ignore'),
        }

    except Exception as e:
        return {
            "error": f"Metrics capture failed: {str(e)}",
            "success": False,
        }


@tool
def compare_outputs(expected: Any, actual: Any, tolerance: float = 1e-5) -> Dict[str, Any]:
    """
    Compare expected and actual outputs.

    Args:
        expected: Expected output
        actual: Actual output
        tolerance: Numerical tolerance for comparison

    Returns:
        Comparison results
    """
    import numpy as np

    try:
        # Handle different types
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            match = abs(expected - actual) < tolerance
            diff = abs(expected - actual)

        elif isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
            match = np.allclose(expected, actual, atol=tolerance)
            diff = np.mean(np.abs(expected - actual))

        elif isinstance(expected, list) and isinstance(actual, list):
            if all(isinstance(x, (int, float)) for x in expected + actual):
                expected_arr = np.array(expected)
                actual_arr = np.array(actual)
                match = np.allclose(expected_arr, actual_arr, atol=tolerance)
                diff = np.mean(np.abs(expected_arr - actual_arr))
            else:
                match = expected == actual
                diff = None

        else:
            match = expected == actual
            diff = None

        return {
            "match": bool(match),
            "expected": str(expected)[:200],
            "actual": str(actual)[:200],
            "difference": float(diff) if diff is not None else None,
        }

    except Exception as e:
        return {
            "match": False,
            "error": f"Comparison error: {str(e)}",
        }


@tool
def search_log_errors(log_path: str) -> Dict[str, Any]:
    """
    Search log files for error patterns and classify severity.

    Args:
        log_path: Path to log file or directory

    Returns:
        Dictionary with errors found, whether fatal, and error messages for web search
    """
    import re
    import glob

    result = {
        "errors": [],
        "fatal": False,
        "search_queries": [],
    }

    # Read log content
    log_content = ""
    try:
        log_path_obj = Path(log_path)
        if log_path_obj.is_file():
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                log_content = f.read()
        elif log_path_obj.is_dir():
            for pattern in ['*.log', 'quick_test*', 'output*']:
                for log_file in glob.glob(os.path.join(log_path, pattern)):
                    try:
                        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                            log_content += f.read() + "\n"
                    except:
                        pass
    except Exception as e:
        return {"errors": [f"Could not read log: {e}"], "fatal": False, "search_queries": []}

    if not log_content:
        return {"errors": ["No log content found"], "fatal": False, "search_queries": []}

    # Extract error lines
    error_patterns = [
        r'(?:error|exception|failed|fatal)[^\n]*',
        r'traceback[^\n]*',
        r'(?:modulenotfounderror|importerror|runtimeerror|oserror|valueerror|typeerror)[^\n]*',
    ]

    for pattern in error_patterns:
        matches = re.findall(pattern, log_content, re.IGNORECASE)
        for match in matches[:5]:  # Limit per pattern
            error_msg = match.strip()[:200]
            if error_msg and error_msg not in result["errors"]:
                result["errors"].append(error_msg)
                result["search_queries"].append(error_msg)

    # Mark as fatal if serious errors found
    fatal_keywords = ['modulenotfound', 'cuda error', 'runtimeerror', 'oserror', 'jsondecode', 'connection']
    log_lower = log_content.lower()
    if any(kw in log_lower for kw in fatal_keywords):
        result["fatal"] = True

    if not result["errors"]:
        result["errors"].append("No errors found")

    return result


@tool
def search_error_solution(error_message: str) -> Dict[str, Any]:
    """
    Use Gemini with Google Search grounding to find solutions for an error.

    Args:
        error_message: The error message to search for

    Returns:
        Dictionary with solutions and code fixes from web search
    """
    import google.generativeai as genai

    result = {
        "solutions": [],
        "raw_response": "",
    }

    try:
        # Configure Gemini
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return {"solutions": ["GEMINI_API_KEY not found in environment"], "raw_response": ""}

        genai.configure(api_key=api_key)

        # Use Gemini with search grounding
        model = genai.GenerativeModel('gemini-1.5-flash')

        prompt = f"""Search for solutions to this Python error and provide actionable fixes:

Error: {error_message[:500]}

Provide:
1. What causes this error
2. How to fix it (with code examples if needed)
3. Common solutions from Stack Overflow or GitHub issues"""

        # Enable search grounding
        response = model.generate_content(
            prompt,
            tools='google_search_retrieval'
        )

        if response.text:
            result["solutions"].append(response.text)
            result["raw_response"] = response.text

    except Exception as e:
        result["solutions"].append(f"Search failed: {str(e)}")

    return result


@tool
def extract_experiment_metrics(output_text: str, expected_metrics_context: str = "") -> Dict[str, Any]:
    """
    Extract numerical metrics from experiment output.

    Use this after running an experiment to extract metrics like accuracy, loss, F1, BLEU, etc.
    This helps verify if the experiment produced the expected results from the paper.

    Args:
        output_text: The experiment output text (from log files or stdout)
        expected_metrics_context: Optional context about what metrics to look for

    Returns:
        Dictionary with:
            - metrics: Dict of extracted metric names and values
            - extraction_method: How metrics were extracted (regex or llm)
            - success: Whether any metrics were found
    """
    try:
        # Try multiple paths to find the src directory
        import sys
        possible_paths = [
            str(Path(__file__).parent.parent),
            str(Path(__file__).parent.parent.parent / "src"),
            os.path.join(os.path.dirname(__file__), "..", ".."),
        ]

        for path in possible_paths:
            abs_path = str(Path(path).resolve())
            if abs_path not in sys.path:
                sys.path.insert(0, abs_path)

        from agents.metrics_extractor import MetricsExtractorAgent

        # Initialize metrics extractor
        extractor = MetricsExtractorAgent()

        # Extract metrics
        result = extractor.extract_metrics(output_text, expected_metrics_context)

        return {
            "metrics": result.get("metrics", {}),
            "extraction_method": result.get("extraction_method", "unknown"),
            "success": len(result.get("metrics", {})) > 0,
            "metric_count": len(result.get("metrics", {}))
        }

    except Exception as e:
        return {
            "metrics": {},
            "extraction_method": "failed",
            "success": False,
            "error": f"Extraction error: {str(e)}"
        }


@tool
def compare_with_paper_results(extracted_metrics: Dict[str, float], expected_results_str: str, tolerance: float = 0.05) -> Dict[str, Any]:
    """
    Compare extracted metrics with expected results from the paper.

    Use this after extracting metrics to determine if the experiment succeeded.

    SUCCESS CRITERIA (STRICT):
    - Uses RELATIVE error (percentage difference) - works for any metric scale
    - Relative error = |actual - expected| / expected
    - ALL metrics must have relative error < tolerance (default 5%)
    - If even ONE metric has error >= tolerance, experiment is marked as FAILURE
    - Always reports the portion of metrics that matched (e.g., "2/3 matched")

    Args:
        extracted_metrics: Dict of metric names to values (from extract_experiment_metrics)
        expected_results_str: String describing expected results from paper analysis
        tolerance: Relative error tolerance (default: 0.05 = 5%). Configurable per experiment.

    Returns:
        Dictionary with:
            - success: True ONLY if ALL metrics within tolerance
            - match_count: Number of metrics within tolerance
            - total_count: Total number of expected metrics
            - success_portion: String like "2/3" showing portion of success
            - matches: List of metrics within tolerance (with relative error %)
            - mismatches: List of metrics outside tolerance (with relative error %)
            - missing: List of expected metrics not found in output
            - details: Human-readable summary
    """
    try:
        import sys
        import re

        # Try multiple paths to find the src directory
        possible_paths = [
            str(Path(__file__).parent.parent),
            str(Path(__file__).parent.parent.parent / "src"),
            os.path.join(os.path.dirname(__file__), "..", ".."),
        ]

        for path in possible_paths:
            abs_path = str(Path(path).resolve())
            if abs_path not in sys.path:
                sys.path.insert(0, abs_path)

        # Parse expected results string to extract metric names and values
        expected_metrics = {}

        # Look for patterns like "accuracy = 0.95" or "BLEU: 28.4"
        patterns = [
            r'(\w+)\s*[:=]\s*(\d+\.?\d*)',
            r'(\w+)\s+(\d+\.?\d*)%?',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, expected_results_str, re.IGNORECASE)
            for match in matches:
                metric_name = match.group(1).lower().strip()
                metric_value = match.group(2).strip()
                try:
                    expected_metrics[metric_name] = float(metric_value)
                except ValueError:
                    pass

        # Compare each metric
        match_list = []
        mismatch_list = []
        missing_list = []

        for metric_name, expected_val in expected_metrics.items():
            found = False
            for extracted_name, extracted_val in extracted_metrics.items():
                # Fuzzy matching: check if names overlap
                if metric_name in extracted_name or extracted_name in metric_name:
                    found = True

                    # Calculate RELATIVE error (scale-invariant)
                    abs_diff = abs(extracted_val - expected_val)
                    if expected_val != 0:
                        relative_error = abs_diff / abs(expected_val)
                    else:
                        # For expected=0, use absolute error
                        relative_error = abs_diff

                    if relative_error < tolerance:
                        match_list.append({
                            "metric": metric_name,
                            "expected": expected_val,
                            "actual": extracted_val,
                            "relative_error": f"{relative_error*100:.2f}%",
                            "abs_diff": abs_diff
                        })
                    else:
                        mismatch_list.append({
                            "metric": metric_name,
                            "expected": expected_val,
                            "actual": extracted_val,
                            "relative_error": f"{relative_error*100:.2f}%",
                            "abs_diff": abs_diff
                        })
                    break

            if not found:
                missing_list.append(metric_name)

        # Calculate success
        total_count = len(expected_metrics)
        match_count = len(match_list)

        # STRICT: Success only if ALL metrics matched (no mismatches, no missing)
        all_matched = (match_count == total_count and len(mismatch_list) == 0 and len(missing_list) == 0)

        success_portion = f"{match_count}/{total_count}" if total_count > 0 else "0/0"

        # Generate human-readable details
        tolerance_pct = f"{tolerance*100:.0f}%"
        if all_matched:
            details = f"✅ SUCCESS: All {total_count} metrics within {tolerance_pct} relative error"
        elif match_count > 0:
            details = f"⚠️ PARTIAL: {success_portion} metrics matched ({tolerance_pct} tolerance). "
            if mismatch_list:
                details += f"{len(mismatch_list)} mismatch(es): "
                details += ", ".join([f"{m['metric']} ({m['relative_error']})" for m in mismatch_list[:2]])
                if len(mismatch_list) > 2:
                    details += f", +{len(mismatch_list)-2} more"
            if missing_list:
                details += f" | {len(missing_list)} missing: {', '.join(missing_list[:2])}"
        else:
            details = f"❌ FAILURE: 0/{total_count} metrics matched."

        return {
            "success": all_matched,
            "match_count": match_count,
            "total_count": total_count,
            "success_portion": success_portion,
            "matches": match_list,
            "mismatches": mismatch_list,
            "missing": missing_list,
            "tolerance_used": tolerance,
            "tolerance_type": "relative",
            "details": details
        }

    except Exception as e:
        return {
            "success": False,
            "match_count": 0,
            "total_count": 0,
            "success_portion": "0/0",
            "matches": [],
            "mismatches": [],
            "missing": [],
            "error": f"Comparison error: {str(e)}",
            "details": f"❌ ERROR: {str(e)}"
        }


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
            str(Path(__file__).parent.parent.parent / "src"),  # From paper_reproduction_agent/src/tools to src
            os.path.join(os.path.dirname(__file__), "..", ".."),  # Relative path
        ]

        # Add all possible paths
        for path in possible_paths:
            abs_path = str(Path(path).resolve())
            if abs_path not in sys.path:
                sys.path.insert(0, abs_path)

        from utils.python_compatibility import check_python_compatibility as check_compat

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
            "warnings": [f"Compatibility check module not found: {str(e)}. Paths tried: {sys.path[:5]}"],
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
        if '-' in required_version and not required_version.startswith('>='):
            parts = required_version.split('-')
            max_ver = parts[1].strip()
            # Extract version numbers
            match = re.search(r'(\d+)\.(\d+)', max_ver)
            if match:
                return f"{match.group(1)}.{match.group(2)}"

        # Handle ">=3.6,<3.10" - get the upper bound
        if '<' in required_version:
            upper_bound = required_version.split('<')[1].strip()
            match = re.search(r'(\d+)\.(\d+)', upper_bound)
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
        if '>=' in required_version:
            min_ver = required_version.split('>=')[1].split(',')[0].strip()
            match = re.search(r'(\d+)\.(\d+)', min_ver)
            if match:
                major, minor = int(match.group(1)), int(match.group(2))
                # If current Python is older than required, return the minimum
                if (current_major, current_minor) < (major, minor):
                    return f"{major}.{minor}"
            # Current Python is fine
            return None

        # Handle "3.9+" or "3.9"
        if '+' in required_version or re.match(r'^\d+\.\d+$', required_version):
            version_str = required_version.replace('+', '').strip()
            match = re.search(r'(\d+)\.(\d+)', version_str)
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
            ["pyenv", "versions", "--bare"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            print(f"⚠️  Failed to check pyenv versions: {result.stderr}")
            return None

        available_versions = result.stdout.strip().split('\n')

        # Find exact match or closest match
        matching_version = None
        for version in available_versions:
            if version.startswith(python_version):
                matching_version = version
                break

        # Install if not found
        if not matching_version:
            print(f"📥 Installing Python {python_version} via pyenv...")
            print(f"   This may take a few minutes on first install...")

            # Get latest patch version for this minor version
            versions_result = subprocess.run(
                ["pyenv", "install", "--list"],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Find latest patch version (e.g., for 3.9, find 3.9.18)
            import re
            pattern = re.compile(rf'^\s*({python_version}\.\d+)\s*$', re.MULTILINE)
            matches = pattern.findall(versions_result.stdout)

            if matches:
                # Get the latest patch version
                latest_version = sorted(matches, key=lambda v: tuple(map(int, v.split('.'))))[-1]
                matching_version = latest_version

                print(f"   Installing Python {matching_version}...")
                install_result = subprocess.run(
                    ["pyenv", "install", "-s", matching_version],  # -s = skip if exists
                    capture_output=True,
                    text=True,
                    timeout=600
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
            timeout=10
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
            [pyenv_python, "--version"],
            capture_output=True,
            text=True,
            timeout=5
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
                    with open(env_file, 'r') as f:
                        content = f.read()
                        if 'channels:' in content or 'dependencies:' in content:
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
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', repo_name)

    return f"paper_{safe_name}_{path_hash}"


def _create_conda_env(env_name: str, python_version: Optional[str] = None, env_file: Optional[Path] = None) -> bool:
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
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if env_name in list_result.stdout:
            print(f"✅ Conda environment '{env_name}' already exists")
            return True

        if env_file and env_file.exists():
            # Create from environment.yml file
            print(f"📦 Creating conda environment '{env_name}' from {env_file.name}...")
            print(f"   This may take several minutes...")

            result = subprocess.run(
                ["conda", "env", "create", "-n", env_name, "-f", str(env_file)],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes for conda install
            )
        else:
            # Create from scratch with Python version
            python_spec = f"python={python_version}" if python_version else "python"
            print(f"📦 Creating conda environment '{env_name}' with {python_spec}...")
            print(f"   This may take several minutes...")

            result = subprocess.run(
                ["conda", "create", "-n", env_name, python_spec, "-y"],
                capture_output=True,
                text=True,
                timeout=1800
            )

        if result.returncode != 0:
            print(f"❌ Conda environment creation failed:")
            print(f"   {result.stderr[:500]}")
            return False

        print(f"✅ Conda environment '{env_name}' created successfully")
        return True

    except subprocess.TimeoutExpired:
        print(f"❌ Conda environment creation timed out (>30 minutes)")
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
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            print(f"⚠️  Failed to list conda environments: {result.stderr}")
            return None

        # Parse environment list to find path
        for line in result.stdout.split('\n'):
            if env_name in line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    env_path = parts[-1]  # Last part is the path

                    # Construct Python path
                    if os.name == 'nt':  # Windows
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
            timeout=10
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
    if os.name == 'nt':  # Windows
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
            timeout=120
        )

        if result.returncode != 0:
            print(f"❌ venv creation failed: {result.stderr[:300]}")
            return False

        # Verify venv was created
        if os.name == 'nt':
            venv_python = venv_path / "Scripts" / "python.exe"
        else:
            venv_python = venv_path / "bin" / "python"

        if not venv_python.exists():
            print(f"❌ venv Python not found at {venv_python}")
            return False

        print(f"✅ Virtual environment created successfully")
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
        import sys

        # Generate unique conda environment name
        env_name = _generate_conda_env_name(repo_path)

        print(f"📦 Installing with Conda")
        print(f"   Environment name: {env_name}")
        print(f"   Environment file: {conda_env_file.name}")
        print(f"   This creates an ISOLATED conda environment (won't affect your main env)\n")

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
        print(f"\n✅ Conda environment created successfully!")
        print(f"   Environment name: {env_name}")
        print(f"   Python: {conda_python}")
        print(f"   Pip: {conda_pip}")
        print(f"\n   To activate: conda activate {env_name}")
        print(f"   To remove: conda env remove -n {env_name}\n")

        # Get Python version from conda env
        version_result = subprocess.run(
            [conda_python, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        python_version = version_result.stdout.strip() if version_result.returncode == 0 else "unknown"

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
            print(f"   Install conda: https://docs.conda.io/en/latest/miniconda.html")
            print(f"   Falling back to pip installation...\n")
            use_conda = False
        elif use_conda:
            print(f"✅ Detected conda requirements: {conda_env_file.name}")
            print(f"   Will use conda for installation\n")
        else:
            print(f"✅ Using pip/venv for installation\n")

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
        current_version = compat_check.get("current_version", f"{sys.version_info.major}.{sys.version_info.minor}")
        is_compatible = compat_check.get("compatible", True)

        print(f"   Current Python: {current_version}")
        print(f"   Required Python: {required_version or 'not specified'}")

        # Step 2: Determine if we need a different Python version
        target_python_version = _parse_required_python_version(required_version) if required_version else None

        python_executable = None  # Will hold the Python to use for venv

        if target_python_version and target_python_version != current_version:
            print(f"\n⚠️  Repository requires Python {target_python_version}, but you have {current_version}")
            print(f"   Attempting to create environment with Python {target_python_version}...\n")

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
                        print(f"   Install options:")
                        print(f"   1. Install pyenv: curl https://pyenv.run | bash")
                        print(f"   2. Install Python {target_python_version} system-wide")
                        print(f"   3. Use conda: conda create -n env python={target_python_version}")
                        print(f"\n   Falling back to current Python {current_version} (may fail)...\n")
                        python_executable = sys.executable
            else:
                print(f"⚠️  pyenv not available")
                print(f"   To install pyenv: curl https://pyenv.run | bash")
                print(f"   Checking for system Python {target_python_version}...\n")

                # Try to find this Python version on the system
                python_executable = shutil.which(f"python{target_python_version}")
                if not python_executable:
                    print(f"❌ Python {target_python_version} not found")
                    print(f"   Falling back to current Python {current_version} (may fail)...\n")
                    python_executable = sys.executable
                else:
                    print(f"✅ Found system Python {target_python_version} at {python_executable}\n")
        else:
            # Current Python is compatible or no specific version required
            if not is_compatible:
                print(f"⚠️  Warning: Compatibility check suggests issues, but continuing with Python {current_version}\n")
            else:
                print(f"✅ Python {current_version} is compatible\n")
            python_executable = sys.executable

        # Step 3: Create virtual environment
        if venv_path.exists():
            print(f"📦 Virtual environment already exists at {venv_path}")
            print(f"   To recreate with different Python, delete it first: rm -rf {venv_path}\n")
        else:
            # Create venv with the selected Python
            if python_executable == sys.executable:
                # Use built-in venv module
                print(f"📦 Creating virtual environment with current Python...")
                try:
                    venv.create(venv_path, with_pip=True)
                    print(f"✅ Virtual environment created!\n")
                except Exception as e:
                    print(f"❌ venv.create() failed: {e}")
                    print(f"   Trying subprocess method instead...\n")
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
        if os.name == 'nt':  # Windows
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
            print(f"❌ Virtual environment creation failed!")
            print(f"   Expected Python at: {venv_python}")
            print(f"   File does not exist.")
            print(f"\n   Attempting to recreate with subprocess method...\n")

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
            print(f"   Attempting to bootstrap pip...\n")

            # Try to bootstrap pip
            bootstrap_result = subprocess.run(
                [str(venv_python), "-m", "ensurepip", "--upgrade"],
                capture_output=True,
                text=True,
                timeout=120
            )

            if bootstrap_result.returncode != 0 or not venv_pip.exists():
                # Try alternative: use venv python with -m pip
                print(f"   Using '{venv_python} -m pip' instead of direct pip binary\n")
                # We'll use python -m pip for all subsequent commands
                venv_pip = f"{venv_python} -m pip"
            else:
                print(f"✅ Pip bootstrapped successfully\n")

        # Upgrade pip in the venv first
        print(f"📦 Upgrading pip in virtual environment...")
        upgrade_result = subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
            capture_output=True,
            text=True,
            timeout=120
        )
        if upgrade_result.returncode == 0:
            print(f"✅ Pip upgraded successfully!\n")
        else:
            print(f"⚠️  Pip upgrade had issues (continuing anyway): {upgrade_result.stderr[:200]}\n")

        # Try multiple paths to find the src directory
        possible_paths = [
            str(Path(__file__).parent.parent),  # From tools/ to src/
            str(Path(__file__).parent.parent.parent / "src"),  # From paper_reproduction_agent/src/tools to src
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
                    venv_command = attempt["command"].replace("pip install", f"{venv_pip} install")

                    print(f"\n📦 Installing dependencies using strategy: {attempt['strategy']}")
                    print(f"   Command: {venv_command}")
                    print(f"   Virtual environment: {venv_path}")
                    print(f"   (This may take several minutes for large packages like PyTorch...)\n")

                    # Use Popen to show real-time output
                    process = subprocess.Popen(
                        venv_command,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,  # Combine stderr with stdout
                        text=True,
                        cwd=repo_path,
                        bufsize=1,  # Line buffered
                        universal_newlines=True
                    )

                    # Stream output line by line
                    stdout_lines = []
                    for line in process.stdout:
                        print(line, end='')  # Show real-time progress
                        stdout_lines.append(line)

                    # Wait for completion
                    returncode = process.wait(timeout=600)
                    full_output = ''.join(stdout_lines)

                    if returncode == 0:
                        print(f"\n✅ Installation successful with {attempt['strategy']} strategy!\n")
                        print(f"🐍 Virtual environment created at: {venv_path}")
                        print(f"   Python version: {target_python_version or current_version}")
                        print(f"   Python: {venv_python}")

                        # Check if we're using python -m pip or direct pip
                        if isinstance(venv_pip, str) and "-m pip" in venv_pip:
                            print(f"   Pip: Using '{venv_python} -m pip' (no direct pip binary)")
                        else:
                            print(f"   Pip: {venv_pip}")

                        print(f"   To activate: source {venv_path}/bin/activate (Linux/Mac) or {venv_path}\\Scripts\\activate (Windows)\n")
                        return {
                            "success": True,
                            "strategy_used": attempt["strategy"],
                            "venv_path": str(venv_path),
                            "venv_python": str(venv_python),
                            "venv_pip": str(venv_pip),
                            "python_version_used": target_python_version or current_version,
                            "python_version_required": required_version,
                            "python_executable_used": python_executable,
                            "stdout": full_output,
                            "stderr": "",
                            "warnings": attempt.get("warning", ""),
                        }
                    else:
                        print(f"\n❌ Installation failed with {attempt['strategy']} strategy, trying next...\n")
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


# Tool list for easy import
code_execution_tools = [
    read_file,
    list_directory,
    execute_shell_command,
    create_python_file,
    execute_python_script,
    install_dependencies,
    run_pytest,
    check_python_syntax,
    run_linter,
    capture_execution_metrics,
    compare_outputs,
    check_python_compatibility,
    smart_install_dependencies,
    extract_experiment_metrics,
    compare_with_paper_results,
]
