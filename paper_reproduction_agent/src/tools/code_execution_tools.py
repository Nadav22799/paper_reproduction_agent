"""Tools for executing and testing code."""

import os
import subprocess
import json
import re
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
def execute_shell_command(command: str, cwd: str = ".", timeout: int = 1800,
                         enable_oom_handling: bool = True) -> Dict[str, Any]:
    """
    Execute a shell command and capture output with automatic OOM handling.

    Args:
        command: Shell command to execute
        cwd: Working directory for command execution
        timeout: Execution timeout in seconds (default: 1800 = 30 minutes)
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
            cwd=cwd
        )

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
                    script_match = re.search(r'(?:bash|sh|python)\s+([^\s]+\.(?:sh|py))', command)
                    if script_match:
                        script_path = os.path.join(cwd, script_match.group(1))

                        # Handle OOM
                        oom_result = oom_handler.handle_oom(
                            script_path=script_path,
                            error_output=full_output,
                            attempt=1,
                            max_attempts=3
                        )

                        oom_info = {
                            'detected': True,
                            'should_retry': oom_result['should_retry'],
                            'adjusted': oom_result['adjusted'],
                            'message': oom_result['message'],
                            'script_path': script_path
                        }

                        if oom_result['adjusted']:
                            print(f"\n🔧 OOM Handling: {oom_result['message']}")
                            print(f"   Script adjusted. Retry with the same command.")
                    else:
                        oom_info = {
                            'detected': True,
                            'should_retry': False,
                            'adjusted': False,
                            'message': 'OOM detected but could not identify script to adjust'
                        }
            except Exception as e:
                print(f"⚠️  OOM handler error: {e}")
                oom_info = {'detected': oom_detected, 'handler_error': str(e)}

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


def _normalize_metric_name(name: str) -> str:
    """Normalize metric name for comparison."""
    name = name.lower().strip()
    # Remove common prefixes/suffixes
    name = re.sub(r'^(test_|train_|val_|valid_|eval_|dev_|best_|final_)', '', name)
    name = re.sub(r'(_score|_rate|_ratio)$', '', name)
    # Normalize common variations
    replacements = {
        'accuracy': 'acc',
        'f1_score': 'f1',
        'f1-score': 'f1',
        'f_score': 'f1',
        'f-score': 'f1',
        'f_1': 'f1',
        'f-1': 'f1',
        'exact_match': 'em',
        'exactmatch': 'em',
        'perplexity': 'ppl',
        'bleu_score': 'bleu',
        'rouge_score': 'rouge',
    }
    for old, new in replacements.items():
        if name == old:
            name = new
    return name


def _normalize_metric_value(value: float, metric_name: str) -> float:
    """
    Normalize metric value to consistent scale.
    - Percentages (0-100) → decimal (0-1) for accuracy-like metrics
    - Keeps loss/perplexity as-is
    """
    metric_name = metric_name.lower()

    # Metrics that should be in 0-1 range
    decimal_metrics = ['acc', 'accuracy', 'f1', 'precision', 'recall', 'em', 'exact_match', 'auc', 'map', 'mrr', 'ndcg']

    # Metrics that can be 0-100 range
    percent_metrics = ['bleu', 'rouge', 'meteor']

    # Check if this is an accuracy-like metric that might be in percentage form
    is_decimal_metric = any(m in metric_name for m in decimal_metrics)
    is_percent_metric = any(m in metric_name for m in percent_metrics)

    if is_decimal_metric and value > 1.0 and value <= 100:
        # Convert percentage to decimal
        return value / 100.0
    elif is_percent_metric:
        # BLEU/ROUGE can be 0-100, keep as-is
        return value

    return value


def _fuzzy_match_metric(name1: str, name2: str) -> bool:
    """Check if two metric names are likely the same metric."""
    n1 = _normalize_metric_name(name1)
    n2 = _normalize_metric_name(name2)

    # Exact match after normalization
    if n1 == n2:
        return True

    # One contains the other
    if n1 in n2 or n2 in n1:
        return True

    # Common abbreviations
    abbrev_pairs = [
        ('acc', 'accuracy'),
        ('f1', 'f1_score'),
        ('em', 'exact_match'),
        ('ppl', 'perplexity'),
    ]
    for a, b in abbrev_pairs:
        if (n1 == a and b in n2) or (n2 == a and b in n1):
            return True
        if (n1 == b and a in n2) or (n2 == b and a in n1):
            return True

    return False


@tool
def compare_with_paper_results(extracted_metrics: Dict[str, float], expected_results_str: str, tolerance: float = 0.05) -> Dict[str, Any]:
    """
    Compare extracted metrics with expected results from the paper.

    FLEXIBLE COMPARISON:
    - Fuzzy metric name matching (test_accuracy matches accuracy)
    - Handles percentage vs decimal (94.5% matches 0.945)
    - Default 5% tolerance for relative error
    - Generates detailed comparison report

    SUCCESS CRITERIA:
    - Uses RELATIVE error: |actual - expected| / expected
    - Metrics within tolerance are considered "matched"
    - Reports partial success (e.g., "3/4 metrics matched")
    - Considers reproduction successful if most metrics match

    Args:
        extracted_metrics: Dict of metric names to values (from extract_experiment_metrics)
        expected_results_str: String describing expected results from paper
        tolerance: Relative error tolerance (default: 0.05 = 5%)

    Returns:
        Dictionary with comparison results and detailed breakdown
    """
    try:
        # Parse expected results string to extract metric names and values
        expected_metrics = {}

        # Multiple patterns to catch different formats
        patterns = [
            # "accuracy: 94.5%" or "accuracy = 0.945"
            r'(\w+(?:[-_]\w+)?)\s*[:=]\s*(\d+\.?\d*)\s*%?',
            # "BLEU-4: 28.4"
            r'(bleu-?\d*|rouge-?[12lL]?)\s*[:=]\s*(\d+\.?\d*)',
            # "F1 score: 0.89"
            r'(f1[\s-]?score)\s*[:=]\s*(\d+\.?\d*)',
            # "94.5% accuracy" (reversed)
            r'(\d+\.?\d*)\s*%?\s*(accuracy|acc|f1|precision|recall)',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, expected_results_str, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                # Handle reversed pattern
                try:
                    float(groups[0])
                    # First is value, second is name
                    metric_name = groups[1].lower().strip()
                    metric_value = float(groups[0])
                except ValueError:
                    # First is name, second is value
                    metric_name = groups[0].lower().strip()
                    try:
                        metric_value = float(groups[1])
                    except:
                        continue

                # Skip non-metric words
                skip_words = ['the', 'and', 'for', 'with', 'our', 'model', 'method', 'approach']
                if metric_name not in skip_words and len(metric_name) > 1:
                    expected_metrics[metric_name] = metric_value

        if not expected_metrics:
            return {
                "success": False,
                "error": "Could not parse expected results from paper",
                "expected_results_str": expected_results_str,
                "details": "❌ ERROR: No metrics found in expected results string"
            }

        # Compare each expected metric with extracted metrics
        match_list = []
        mismatch_list = []
        missing_list = []
        matched_extracted = set()

        for expected_name, expected_val in expected_metrics.items():
            best_match = None
            best_error = float('inf')

            # Find best matching extracted metric
            for extracted_name, extracted_val in extracted_metrics.items():
                if extracted_name in matched_extracted:
                    continue

                if _fuzzy_match_metric(expected_name, extracted_name):
                    # Normalize values for comparison
                    norm_expected = _normalize_metric_value(expected_val, expected_name)
                    norm_extracted = _normalize_metric_value(extracted_val, extracted_name)

                    # Calculate relative error
                    if norm_expected != 0:
                        rel_error = abs(norm_extracted - norm_expected) / abs(norm_expected)
                    else:
                        rel_error = abs(norm_extracted - norm_expected)

                    if rel_error < best_error:
                        best_error = rel_error
                        best_match = {
                            "extracted_name": extracted_name,
                            "extracted_val": extracted_val,
                            "norm_extracted": norm_extracted,
                            "norm_expected": norm_expected,
                            "rel_error": rel_error
                        }

            if best_match:
                matched_extracted.add(best_match["extracted_name"])

                result_item = {
                    "metric": expected_name,
                    "paper_value": expected_val,
                    "extracted_value": best_match["extracted_val"],
                    "normalized_paper": best_match["norm_expected"],
                    "normalized_extracted": best_match["norm_extracted"],
                    "relative_error": best_match["rel_error"],
                    "relative_error_pct": f"{best_match['rel_error']*100:.2f}%",
                    "abs_diff": abs(best_match["norm_extracted"] - best_match["norm_expected"]),
                    "within_tolerance": best_match["rel_error"] < tolerance
                }

                if result_item["within_tolerance"]:
                    match_list.append(result_item)
                else:
                    mismatch_list.append(result_item)
            else:
                missing_list.append({
                    "metric": expected_name,
                    "paper_value": expected_val,
                    "status": "not found in extracted metrics"
                })

        # Calculate success metrics
        total_expected = len(expected_metrics)
        matched_count = len(match_list)
        mismatched_count = len(mismatch_list)
        missing_count = len(missing_list)

        # Success criteria: at least 70% of metrics match OR all found metrics match
        found_metrics = matched_count + mismatched_count
        if found_metrics > 0:
            match_ratio = matched_count / found_metrics
        else:
            match_ratio = 0

        overall_ratio = matched_count / total_expected if total_expected > 0 else 0

        # Consider it successful if:
        # 1. All metrics within tolerance, OR
        # 2. At least 70% of found metrics match AND at least 50% of expected metrics found
        all_matched = (matched_count == total_expected and mismatched_count == 0)
        partial_success = (match_ratio >= 0.7 and found_metrics >= total_expected * 0.5)

        success = all_matched or (partial_success and mismatched_count == 0)

        # Generate summary
        success_portion = f"{matched_count}/{total_expected}"
        tolerance_pct = f"{tolerance*100:.0f}%"

        if all_matched:
            status = "✅ SUCCESS"
            details = f"All {total_expected} metrics within {tolerance_pct} tolerance!"
        elif matched_count > 0 and mismatched_count == 0:
            status = "✅ PARTIAL SUCCESS"
            details = f"{success_portion} metrics matched (missing: {missing_count})"
        elif matched_count > 0:
            status = "⚠️ PARTIAL MATCH"
            details = f"{matched_count} matched, {mismatched_count} outside tolerance, {missing_count} missing"
        else:
            status = "❌ NO MATCH"
            details = f"0/{total_expected} metrics matched within {tolerance_pct} tolerance"

        return {
            "success": success,
            "status": status,
            "match_count": matched_count,
            "mismatch_count": mismatched_count,
            "missing_count": missing_count,
            "total_expected": total_expected,
            "success_portion": success_portion,
            "match_ratio": f"{match_ratio*100:.1f}%",
            "overall_ratio": f"{overall_ratio*100:.1f}%",
            "matches": match_list,
            "mismatches": mismatch_list,
            "missing": missing_list,
            "tolerance_used": tolerance,
            "tolerance_pct": tolerance_pct,
            "details": f"{status}: {details}",
            "extracted_metrics_count": len(extracted_metrics),
            "expected_metrics_parsed": expected_metrics
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "details": f"❌ ERROR: {str(e)}"
        }


@tool
def generate_comparison_report(
    repo_path: str,
    extracted_metrics: Dict[str, float],
    paper_results: str,
    comparison_result: Dict[str, Any],
    output_filename: str = "reproduction_report.md"
) -> Dict[str, Any]:
    """
    Generate a detailed comparison report file showing extracted vs paper results.

    Creates a markdown report with:
    - Summary of reproduction success
    - Table comparing each metric
    - Visual indicators for match/mismatch
    - Error percentages
    - Recommendations

    Args:
        repo_path: Path to repository (report will be saved here)
        extracted_metrics: Metrics extracted from experiment
        paper_results: Expected results from paper (string)
        comparison_result: Result from compare_with_paper_results
        output_filename: Name for output file (default: reproduction_report.md)

    Returns:
        Dictionary with report path and summary
    """
    from datetime import datetime

    try:
        repo = Path(repo_path).resolve()
        report_path = repo / output_filename

        # Build the report
        lines = []

        # Header
        lines.append("# Paper Reproduction Report")
        lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Repository:** `{repo.name}`")
        lines.append(f"**Tolerance:** {comparison_result.get('tolerance_pct', '5%')}")
        lines.append("")

        # Summary
        status = comparison_result.get('status', 'Unknown')
        lines.append("## Summary")
        lines.append("")
        lines.append(f"**Status:** {status}")
        lines.append(f"**Metrics Matched:** {comparison_result.get('success_portion', 'N/A')}")
        lines.append(f"**Match Ratio:** {comparison_result.get('match_ratio', 'N/A')}")
        lines.append("")

        # Comparison Table
        lines.append("## Detailed Comparison")
        lines.append("")
        lines.append("| Metric | Paper Value | Extracted Value | Difference | Status |")
        lines.append("|--------|-------------|-----------------|------------|--------|")

        # Add matched metrics
        for m in comparison_result.get('matches', []):
            status_icon = "✅"
            lines.append(f"| {m['metric']} | {m['paper_value']} | {m['extracted_value']} | {m['relative_error_pct']} | {status_icon} Match |")

        # Add mismatched metrics
        for m in comparison_result.get('mismatches', []):
            status_icon = "❌"
            lines.append(f"| {m['metric']} | {m['paper_value']} | {m['extracted_value']} | {m['relative_error_pct']} | {status_icon} Mismatch |")

        # Add missing metrics
        for m in comparison_result.get('missing', []):
            if isinstance(m, dict):
                lines.append(f"| {m['metric']} | {m['paper_value']} | - | - | ⚠️ Not Found |")
            else:
                lines.append(f"| {m} | - | - | - | ⚠️ Not Found |")

        lines.append("")

        # All Extracted Metrics
        lines.append("## All Extracted Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for name, value in extracted_metrics.items():
            lines.append(f"| {name} | {value} |")
        lines.append("")

        # Expected Results (from paper)
        lines.append("## Expected Results (from paper)")
        lines.append("")
        lines.append("```")
        lines.append(paper_results)
        lines.append("```")
        lines.append("")

        # Interpretation Guide
        lines.append("## Interpretation Guide")
        lines.append("")
        lines.append("- **✅ Match**: Extracted value within tolerance of paper value")
        lines.append("- **❌ Mismatch**: Extracted value differs more than tolerance")
        lines.append("- **⚠️ Not Found**: Paper metric not found in extracted results")
        lines.append("")
        lines.append(f"**Tolerance Used:** {comparison_result.get('tolerance_pct', '5%')} relative error")
        lines.append("")
        lines.append("### Notes")
        lines.append("- Relative error = |extracted - paper| / paper")
        lines.append("- Values are normalized (e.g., 94.5% → 0.945) before comparison")
        lines.append("- Metric names are fuzzy-matched (test_accuracy matches accuracy)")
        lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")

        if comparison_result.get('success'):
            lines.append("✅ **Reproduction Successful!** Results match the paper within tolerance.")
        elif comparison_result.get('match_count', 0) > 0:
            lines.append("⚠️ **Partial Success.** Some metrics match. Consider:")
            lines.append("- Check if mismatched metrics use different evaluation methods")
            lines.append("- Verify hyperparameters match the paper exactly")
            lines.append("- Check for random seed differences")
            if comparison_result.get('missing'):
                lines.append("- Look for missing metrics in different output files")
        else:
            lines.append("❌ **Reproduction Failed.** Consider:")
            lines.append("- Verify the experiment completed successfully (check logs)")
            lines.append("- Check if result files are in a different location")
            lines.append("- Ensure correct dataset and preprocessing")
            lines.append("- Compare with paper's exact configuration")

        # Write the report
        report_content = "\n".join(lines)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"\n📄 Comparison report saved to: {report_path}")

        return {
            "success": True,
            "report_path": str(report_path),
            "report_content": report_content,
            "summary": comparison_result.get('details', ''),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
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


# =============================================================================
# Result Discovery and Extraction Tools
# =============================================================================

@tool
def discover_result_files(repo_path: str, experiment_name: str = "") -> Dict[str, Any]:
    """
    Discover result/output files from experiments - searches EVERYWHERE flexibly.

    This tool searches for result files FIRST before reading large log files.
    It handles many scenarios:
    - Standard naming: results.json, metrics.csv, eval_results.json
    - Custom directories: results/, outputs/, runs/, experiments/, logs/
    - Recently modified files (likely experiment output)
    - Files containing metric keywords (scans content)
    - Output paths mentioned in scripts/configs

    Args:
        repo_path: Path to repository root
        experiment_name: Optional experiment name to narrow search

    Returns:
        Dictionary with:
            - result_files: List of discovered result files with metadata
            - priority_files: Top files most likely to contain final results
            - log_files: Log files found (for error checking only)
            - output_dirs: Directories that likely contain outputs
            - script_output_paths: Output paths found in scripts
    """
    import glob
    from datetime import datetime

    result = {
        "result_files": [],
        "priority_files": [],
        "log_files": [],
        "checkpoint_dirs": [],
        "output_dirs": [],
        "script_output_paths": [],
        "recently_modified": [],
        "search_paths": [],
        "success": True
    }

    repo = Path(repo_path).resolve()
    if not repo.exists():
        return {"success": False, "error": f"Repository path not found: {repo_path}"}

    # =================================================================
    # STEP 1: Search common output directories
    # =================================================================
    output_dir_patterns = [
        "results", "Results", "RESULTS",
        "outputs", "output", "Outputs", "Output",
        "runs", "run", "Runs",
        "experiments", "experiment", "Experiments",
        "logs", "log", "Logs",
        "checkpoints", "checkpoint", "ckpt",
        "saved_models", "models",
        "eval", "evaluation", "evaluations",
        "metrics", "scores",
        "test_results", "train_results",
    ]

    for dir_name in output_dir_patterns:
        dir_path = repo / dir_name
        if dir_path.exists() and dir_path.is_dir():
            result["output_dirs"].append(str(dir_path.relative_to(repo)))

    # =================================================================
    # STEP 2: Search for result files with flexible patterns
    # =================================================================
    result_patterns = [
        # Any JSON/CSV/TXT in output directories
        "results/**/*.json",
        "results/**/*.csv",
        "results/**/*.txt",
        "outputs/**/*.json",
        "outputs/**/*.csv",
        "output/**/*.json",
        "output/**/*.csv",
        "runs/**/*.json",
        "experiments/**/*.json",
        "eval/**/*.json",
        "evaluation/**/*.json",
        "metrics/**/*.json",
        "metrics/**/*.csv",
        "logs/**/*.json",  # Some frameworks save JSON metrics in logs/

        # Standard naming patterns anywhere
        "**/results*.json",
        "**/result*.json",
        "**/eval*.json",
        "**/metrics*.json",
        "**/scores*.json",
        "**/test*.json",
        "**/train*.json",
        "**/output*.json",
        "**/final*.json",
        "**/summary*.json",
        "**/report*.json",
        "**/performance*.json",
        "**/accuracy*.json",
        "**/best*.json",

        # CSV patterns
        "**/results*.csv",
        "**/metrics*.csv",
        "**/eval*.csv",
        "**/scores*.csv",
        "**/log*.csv",
        "**/history*.csv",
        "**/training*.csv",

        # Text results
        "**/results*.txt",
        "**/eval*.txt",
        "**/scores*.txt",
        "**/accuracy*.txt",
        "**/metrics*.txt",
        "**/summary*.txt",
        "**/report*.txt",

        # Underscore/dash variants
        "**/*_results.json",
        "**/*_metrics.json",
        "**/*_eval.json",
        "**/*_scores.json",
        "**/*-results.json",
        "**/*-metrics.json",

        # Checkpoint evaluation results
        "**/checkpoint*/*.json",
        "**/checkpoint*/eval*.json",
        "**/ckpt*/*.json",

        # HuggingFace Trainer outputs
        "**/trainer_state.json",
        "**/all_results.json",
        "**/eval_results.json",
        "**/predict_results*.json",

        # PyTorch Lightning
        "**/lightning_logs/**/*.json",
        "**/lightning_logs/**/*.csv",

        # TensorBoard (event files - we note them but can't easily parse)
        "**/events.out.tfevents*",

        # Weights & Biases
        "**/wandb/**/*.json",

        # MLflow
        "**/mlruns/**/*.json",
    ]

    # Search for result files
    found_files = set()
    for pattern in result_patterns:
        try:
            matches = glob.glob(str(repo / pattern), recursive=True)
            for match in matches:
                file_path = Path(match)
                if file_path.is_file() and str(file_path) not in found_files:
                    # Skip very large files (likely not result files)
                    try:
                        size = file_path.stat().st_size
                        if size > 50 * 1024 * 1024:  # Skip files > 50MB
                            continue
                    except:
                        continue

                    found_files.add(str(file_path))
                    stat = file_path.stat()
                    modified_time = datetime.fromtimestamp(stat.st_mtime)

                    file_info = {
                        "path": str(file_path),
                        "relative_path": str(file_path.relative_to(repo)),
                        "name": file_path.name,
                        "size_bytes": stat.st_size,
                        "modified": modified_time.isoformat(),
                        "extension": file_path.suffix,
                        "is_recent": (datetime.now() - modified_time).total_seconds() < 86400,  # Last 24h
                        "is_very_recent": (datetime.now() - modified_time).total_seconds() < 3600,  # Last 1h
                    }

                    result["result_files"].append(file_info)
        except Exception:
            continue

    # =================================================================
    # STEP 3: Find recently modified files (likely experiment output)
    # =================================================================
    recent_patterns = ["**/*.json", "**/*.csv", "**/*.txt"]
    for pattern in recent_patterns:
        try:
            matches = glob.glob(str(repo / pattern), recursive=True)
            for match in matches[:100]:  # Limit search
                file_path = Path(match)
                if file_path.is_file():
                    try:
                        stat = file_path.stat()
                        modified_time = datetime.fromtimestamp(stat.st_mtime)
                        age_hours = (datetime.now() - modified_time).total_seconds() / 3600

                        # Very recently modified (last 2 hours) and reasonable size
                        if age_hours < 2 and 100 < stat.st_size < 10 * 1024 * 1024:
                            rel_path = str(file_path.relative_to(repo))
                            # Skip common non-result files
                            if not any(skip in rel_path.lower() for skip in [
                                'node_modules', '.git', '__pycache__', '.pyc',
                                'package-lock', 'yarn.lock', '.egg-info'
                            ]):
                                if rel_path not in [f["relative_path"] for f in result["recently_modified"]]:
                                    result["recently_modified"].append({
                                        "path": str(file_path),
                                        "relative_path": rel_path,
                                        "modified": modified_time.isoformat(),
                                        "age_hours": round(age_hours, 1),
                                        "size_bytes": stat.st_size
                                    })
                    except:
                        continue
        except Exception:
            continue

    # Sort recently modified by age
    result["recently_modified"] = sorted(
        result["recently_modified"],
        key=lambda x: x["age_hours"]
    )[:10]

    # =================================================================
    # STEP 4: Search scripts for output paths
    # =================================================================
    script_patterns = ["**/*.sh", "**/*.py", "**/config*.yaml", "**/config*.yml", "**/config*.json"]
    output_path_patterns = [
        r'--output[_-]?dir[=\s]+["\']?([^\s"\']+)',
        r'--save[_-]?dir[=\s]+["\']?([^\s"\']+)',
        r'--result[s]?[_-]?dir[=\s]+["\']?([^\s"\']+)',
        r'--log[_-]?dir[=\s]+["\']?([^\s"\']+)',
        r'output_dir[=:]\s*["\']?([^\s"\']+)',
        r'save_dir[=:]\s*["\']?([^\s"\']+)',
        r'result[s]?_dir[=:]\s*["\']?([^\s"\']+)',
        r'log_dir[=:]\s*["\']?([^\s"\']+)',
        r'SAVE_PATH[=:]\s*["\']?([^\s"\']+)',
        r'OUTPUT_PATH[=:]\s*["\']?([^\s"\']+)',
    ]

    for pattern in script_patterns[:3]:  # Limit to avoid long search
        try:
            matches = glob.glob(str(repo / pattern), recursive=True)
            for match in matches[:20]:
                try:
                    with open(match, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()[:5000]  # First 5KB

                    for path_pattern in output_path_patterns:
                        path_matches = re.findall(path_pattern, content, re.IGNORECASE)
                        for path_match in path_matches:
                            # Clean up the path
                            clean_path = path_match.strip('\'"').rstrip(',;')
                            if clean_path and len(clean_path) > 2:
                                if clean_path not in result["script_output_paths"]:
                                    result["script_output_paths"].append(clean_path)
                except:
                    continue
        except Exception:
            continue

    result["script_output_paths"] = result["script_output_paths"][:10]

    # =================================================================
    # STEP 5: Search for log files
    # =================================================================
    log_patterns = [
        "*.log",
        "**/train*.log",
        "**/eval*.log",
        "**/*.log",
        "**/logs/*.log",
        "**/log/*.log",
        "**/logs/*.txt",
    ]

    log_files = set()
    for pattern in log_patterns[:4]:
        try:
            matches = glob.glob(str(repo / pattern), recursive=True)
            for match in matches[:10]:
                file_path = Path(match)
                if file_path.is_file() and str(file_path) not in log_files:
                    log_files.add(str(file_path))
                    stat = file_path.stat()

                    result["log_files"].append({
                        "path": str(file_path),
                        "relative_path": str(file_path.relative_to(repo)),
                        "size_bytes": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        except Exception:
            continue

    # =================================================================
    # STEP 6: Find checkpoint directories
    # =================================================================
    checkpoint_patterns = ["**/checkpoint*", "**/checkpoints", "**/ckpt*", "**/saved_models"]
    for pattern in checkpoint_patterns:
        try:
            matches = glob.glob(str(repo / pattern), recursive=True)
            for match in matches[:5]:
                dir_path = Path(match)
                if dir_path.is_dir():
                    result["checkpoint_dirs"].append(str(dir_path.relative_to(repo)))
        except:
            continue

    # =================================================================
    # STEP 7: Prioritize result files
    # =================================================================
    priority_files = []

    # Priority 1: Very recent files (last hour) - highest priority
    for f in result["result_files"]:
        if f.get("is_very_recent"):
            priority_files.append(f)

    # Priority 2: Recent files with result/eval/metric in name
    if len(priority_files) < 5:
        for f in result["result_files"]:
            if f["is_recent"] and any(kw in f["name"].lower() for kw in [
                "result", "eval", "metric", "score", "accuracy", "final", "best", "test"
            ]):
                if f not in priority_files:
                    priority_files.append(f)

    # Priority 3: Files in output directories
    if len(priority_files) < 5:
        for f in result["result_files"]:
            if any(outdir in f["relative_path"] for outdir in result["output_dirs"]):
                if f not in priority_files:
                    priority_files.append(f)

    # Priority 4: Any recent JSON/CSV files
    if len(priority_files) < 5:
        for f in result["result_files"]:
            if f["is_recent"] and f["extension"] in [".json", ".csv"]:
                if f not in priority_files:
                    priority_files.append(f)

    # Priority 5: Add from recently_modified if still empty
    if not priority_files and result["recently_modified"]:
        for f in result["recently_modified"][:3]:
            priority_files.append({
                "path": f["path"],
                "relative_path": f["relative_path"],
                "name": Path(f["path"]).name,
                "size_bytes": f["size_bytes"],
                "modified": f["modified"],
                "extension": Path(f["path"]).suffix,
                "is_recent": True,
                "source": "recently_modified"
            })

    result["priority_files"] = priority_files[:10]  # Top 10
    result["total_result_files"] = len(result["result_files"])
    result["total_log_files"] = len(result["log_files"])

    # =================================================================
    # Summary output
    # =================================================================
    print(f"\n🔍 Result Discovery Summary:")
    print(f"   Output directories found: {result['output_dirs'][:3] if result['output_dirs'] else 'None'}")

    if priority_files:
        print(f"   🎯 Found {len(priority_files)} priority result files:")
        for f in priority_files[:5]:
            age_info = " (very recent)" if f.get("is_very_recent") else " (recent)" if f.get("is_recent") else ""
            print(f"      → {f['relative_path']}{age_info}")
    else:
        print(f"   ⚠️  No result files found by name patterns")

    if result["recently_modified"]:
        print(f"   📝 Recently modified files ({len(result['recently_modified'])}):")
        for f in result["recently_modified"][:3]:
            print(f"      → {f['relative_path']} ({f['age_hours']}h ago)")

    if result["script_output_paths"]:
        print(f"   📂 Output paths from scripts: {result['script_output_paths'][:3]}")

    if not priority_files and not result["recently_modified"]:
        print(f"   Found {len(result['log_files'])} log files (for error checking)")

    return result


def _get_llm_for_extraction():
    """Get LLM instance for result extraction."""
    try:
        from ..utils.llm_factory import create_llm
        return create_llm(temperature=0)
    except:
        # Fallback if import fails
        from langchain_anthropic import ChatAnthropic
        import os
        return ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            temperature=0
        )


def _llm_analyze_file_format(file_path: str, content: str, paper_context: str = "") -> Dict[str, Any]:
    """
    Use LLM to analyze a result file and extract metrics intelligently.

    The LLM understands the file format by looking at the content and extracts
    metrics accordingly - works with any format.
    """
    llm = _get_llm_for_extraction()

    # Truncate content if too long
    content_sample = content[:3000] if len(content) > 3000 else content

    prompt = f"""Analyze this experimental result file and extract all metrics/results.

FILE PATH: {file_path}
FILE CONTENT:
```
{content_sample}
```

PAPER CONTEXT (expected metrics/datasets):
{paper_context if paper_context else "Not provided"}

YOUR TASK:
1. First, understand the file format (CSV, JSON, custom text, table, etc.)
2. Identify what the columns/fields represent
3. Extract dataset names (may be in path, filename, or content)
4. Extract metric values (accuracy, F1, AUC, loss, etc.)

IMPORTANT:
- Dataset names might be in the file path (e.g., results/roman-empire/file.csv means dataset is "roman-empire")
- Look for numeric values that represent performance metrics (usually 0-1 or 0-100)
- Values before ± or $\\pm$ are usually the main metric, after is std deviation
- If multiple rows exist, extract the best result per dataset

Return your analysis as valid JSON in this exact format:
{{
    "format_detected": "description of the file format",
    "datasets_found": [
        {{
            "name": "dataset_name",
            "metrics": {{
                "metric_name": numeric_value,
                "another_metric": numeric_value
            }},
            "source": "how you identified this (path/content/filename)"
        }}
    ],
    "confidence": "high/medium/low",
    "notes": "any important observations"
}}

Return ONLY the JSON, no other text."""

    try:
        response = llm.invoke(prompt)
        response_text = response.content.strip()

        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            return json.loads(json_match.group())
        return {"error": "Could not parse LLM response", "raw": response_text[:500]}
    except Exception as e:
        return {"error": str(e)}


def _llm_extract_from_multiple_files(files_info: List[Dict], paper_context: str = "") -> Dict[str, Any]:
    """
    Use LLM to analyze multiple result files together and extract consolidated results.

    This is more efficient than analyzing files one by one and helps the LLM
    understand the overall structure better.
    """
    llm = _get_llm_for_extraction()

    # Build file summaries
    files_summary = []
    for i, finfo in enumerate(files_info[:10]):  # Limit to 10 files
        content_preview = finfo["content"][:1500] if len(finfo["content"]) > 1500 else finfo["content"]
        files_summary.append(f"""
=== FILE {i+1}: {finfo['path']} ===
{content_preview}
""")

    prompt = f"""Analyze these experimental result files and extract all metrics/results.

{chr(10).join(files_summary)}

PAPER CONTEXT (expected metrics/datasets):
{paper_context if paper_context else "Not provided"}

YOUR TASK:
1. Understand the format of each file
2. Extract dataset names - these might be:
   - In the file path (e.g., results/roman-empire/poly.csv → dataset is "roman-empire")
   - In the filename (e.g., cora_results.json → dataset is "cora")
   - Inside the file content
3. Extract the main performance metrics (accuracy, ROC-AUC, F1, etc.)
4. Match results to datasets appropriately

IMPORTANT PATTERNS TO RECOGNIZE:
- Space-separated values like "poly 0.3 0.001 92.32 ± nan" - the value before ± is the main metric
- CSV with headers - use headers to identify columns
- JSON with nested structures - look for metric keys
- Tables in text format - identify column meanings
- Log files - look for final metrics at the end

Return your analysis as valid JSON:
{{
    "files_analyzed": {len(files_info)},
    "format_notes": "overall observations about file formats",
    "results": {{
        "dataset_name_1": {{
            "accuracy": 92.55,
            "other_metric": value
        }},
        "dataset_name_2": {{
            "accuracy": 87.32
        }}
    }},
    "file_to_dataset_mapping": {{
        "path/to/file.csv": "dataset_name"
    }},
    "confidence": "high/medium/low"
}}

Return ONLY the JSON, no other text."""

    try:
        response = llm.invoke(prompt)
        response_text = response.content.strip()

        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            return json.loads(json_match.group())
        return {"error": "Could not parse LLM response", "raw": response_text[:500]}
    except Exception as e:
        return {"error": str(e)}


@tool
def smart_extract_results(repo_path: str, paper_metrics: str = "") -> Dict[str, Any]:
    """
    Intelligently extract results using LLM to understand ANY file format.

    This tool uses an LLM to:
    1. Analyze the structure of result files (CSV, JSON, custom text, tables, logs)
    2. Understand what columns/fields represent
    3. Extract dataset names from paths, filenames, or content
    4. Extract metric values regardless of format
    5. Handle custom formats like "poly 0.3 0.001 92.32 ± nan"

    The LLM-based approach is flexible and can adapt to any result format
    without needing hardcoded parsing rules.

    Args:
        repo_path: Path to repository containing result files
        paper_metrics: Expected metrics from paper (helps LLM understand context)

    Returns:
        Dictionary with organized results per dataset
    """
    import glob

    result = {
        "datasets": {},  # dataset_name -> {metric: value, ...}
        "raw_files": {},  # file_path -> parsed content
        "summary": {},   # Best results for each dataset
        "file_mapping": {},  # file -> dataset mapping
        "success": True,
        "method": "llm_extraction"
    }

    repo = Path(repo_path).resolve()
    if not repo.exists():
        return {"success": False, "error": f"Repository not found: {repo_path}"}

    print(f"\n🤖 LLM-Powered Smart Result Extraction")
    print(f"   Repository: {repo_path}")
    if paper_metrics:
        print(f"   Paper context provided: {len(paper_metrics)} chars")

    # Find all potential result files with broad patterns
    result_patterns = [
        "results/**/*", "outputs/**/*", "output/**/*",
        "**/*result*.*", "**/*metric*.*", "**/*eval*.*",
        "**/*.csv", "**/*.json",
        "**/logs/*.log", "**/logs/*.txt",
    ]

    found_files = set()
    for pattern in result_patterns:
        matches = glob.glob(str(repo / pattern), recursive=True)
        for m in matches:
            path = Path(m)
            # Filter to likely result files
            if path.is_file() and path.suffix.lower() in ['.csv', '.json', '.txt', '.log', '.md', '']:
                # Skip very large files and common non-result files
                try:
                    if path.stat().st_size < 500000:  # < 500KB
                        rel = str(path.relative_to(repo))
                        if not any(skip in rel.lower() for skip in [
                            'node_modules', '.git', '__pycache__', 'venv',
                            'requirements', 'setup.py', 'readme', 'license'
                        ]):
                            found_files.add(m)
                except:
                    pass

    found_files = list(found_files)[:20]  # Limit to 20 files
    print(f"   Found {len(found_files)} potential result files")

    if not found_files:
        print("   ⚠️ No result files found")
        return result

    # Read file contents
    files_info = []
    for file_path in found_files:
        try:
            path = Path(file_path)
            rel_path = str(path.relative_to(repo))

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if content.strip():  # Only include non-empty files
                files_info.append({
                    "path": rel_path,
                    "full_path": file_path,
                    "content": content,
                    "size": len(content)
                })
        except Exception as e:
            continue

    if not files_info:
        print("   ⚠️ No readable result files found")
        return result

    print(f"   Reading {len(files_info)} files for analysis...")
    for f in files_info[:5]:
        print(f"      • {f['path']} ({f['size']} bytes)")

    # Use LLM to analyze all files together
    print(f"\n🔍 Analyzing file formats with LLM...")
    llm_result = _llm_extract_from_multiple_files(files_info, paper_metrics)

    if "error" in llm_result:
        print(f"   ⚠️ LLM analysis error: {llm_result['error']}")
        # Fall back to basic parsing
        print("   Falling back to basic pattern matching...")
        return _fallback_extract_results(files_info, paper_metrics, result)

    # Process LLM results
    print(f"   Format notes: {llm_result.get('format_notes', 'N/A')[:100]}")
    print(f"   Confidence: {llm_result.get('confidence', 'unknown')}")

    # Extract results from LLM response
    llm_results = llm_result.get("results", {})
    file_mapping = llm_result.get("file_to_dataset_mapping", {})

    for dataset_name, metrics in llm_results.items():
        if isinstance(metrics, dict):
            # Normalize dataset name
            ds_name = dataset_name.lower().strip()
            result["datasets"][ds_name] = {}

            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    result["datasets"][ds_name][metric_name.lower()] = value

    result["file_mapping"] = file_mapping

    # Store raw file info
    for finfo in files_info:
        result["raw_files"][finfo["path"]] = {
            "content_preview": finfo["content"][:500],
            "size": finfo["size"]
        }

    # Build summary
    for dataset_name, metrics in result["datasets"].items():
        if metrics:
            # Find main metric
            main_value = None
            main_metric = None
            for m_name in ['accuracy', 'acc', 'roc_auc', 'auc', 'f1']:
                if m_name in metrics:
                    main_value = metrics[m_name]
                    main_metric = m_name
                    break

            if main_value is None and metrics:
                # Use first numeric value
                main_metric = list(metrics.keys())[0]
                main_value = metrics[main_metric]

            if main_value is not None:
                result["summary"][dataset_name] = {
                    "value": main_value,
                    "metric": main_metric
                }

    # Print results
    print(f"\n📊 Extracted Results ({len(result['datasets'])} datasets):")
    for dataset_name, metrics in result["datasets"].items():
        print(f"   {dataset_name}:")
        for metric_name, value in list(metrics.items())[:3]:
            print(f"      {metric_name}: {value}")

    if not result["datasets"]:
        print("   ⚠️ No results extracted")

    return result


def _fallback_extract_results(files_info: List[Dict], paper_metrics: str, result: Dict) -> Dict:
    """Fallback extraction using basic pattern matching when LLM fails."""
    result["method"] = "fallback_pattern_matching"

    for finfo in files_info:
        try:
            content = finfo["content"]
            rel_path = finfo["path"]

            # Try to extract dataset from path
            path_parts = rel_path.replace('\\', '/').split('/')
            dataset_name = None

            for part in path_parts:
                part_clean = part.lower().replace('.csv', '').replace('.json', '').replace('.txt', '')
                if part_clean not in ['results', 'outputs', 'output', 'logs', 'data', ''] and len(part_clean) > 2:
                    dataset_name = part_clean
                    break

            if not dataset_name:
                dataset_name = Path(rel_path).stem.lower()

            # Extract metrics using patterns
            metrics = _parse_with_patterns(content)

            if metrics and dataset_name:
                if dataset_name not in result["datasets"]:
                    result["datasets"][dataset_name] = {}
                result["datasets"][dataset_name].update(metrics)

        except Exception as e:
            continue

    return result


def _parse_with_patterns(content: str) -> Dict[str, float]:
    """Parse content using common patterns as fallback."""
    metrics = {}

    # Pattern 1: Key-value pairs
    kv_patterns = [
        r'(accuracy|acc|auc|roc_auc|f1|precision|recall|loss|error)[\s:=]+([0-9.]+)',
        r'"(accuracy|acc|auc|f1|loss)"[\s:]+([0-9.]+)',
    ]

    for pattern in kv_patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            metric_name = match.group(1).lower()
            try:
                value = float(match.group(2))
                metrics[metric_name] = value
            except:
                pass

    # Pattern 2: Numbers that look like accuracy (50-100 range)
    if not metrics:
        numbers = re.findall(r'(\d{2}\.\d+)', content)
        for num_str in numbers:
            try:
                val = float(num_str)
                if 50 <= val <= 100:
                    metrics["accuracy"] = val
                    break
            except:
                pass

    # Pattern 3: Decimal accuracy (0.5-1.0 range)
    if not metrics:
        numbers = re.findall(r'0\.(\d{2,})', content)
        for num_str in numbers:
            try:
                val = float(f"0.{num_str}")
                if 0.5 <= val <= 1.0:
                    metrics["accuracy"] = val * 100
                    break
            except:
                pass

    return metrics


@tool
def align_and_compare_results(
    extracted_results: Dict[str, Any],
    paper_metrics: str,
    tolerance: float = 0.05
) -> Dict[str, Any]:
    """
    Use LLM to intelligently align extracted results with paper expectations.

    This tool uses an LLM to:
    1. Understand the paper's expected results in ANY format
    2. Understand the extracted results structure
    3. Intelligently match datasets (handles name variations, abbreviations, etc.)
    4. Compare values with tolerance
    5. Generate detailed alignment report

    The LLM-based approach handles any format of paper metrics without
    needing hardcoded parsing rules.

    Args:
        extracted_results: Output from smart_extract_results
        paper_metrics: Expected metrics from paper (any format - tables, text, structured)
        tolerance: Tolerance for comparison (default 5%)

    Returns:
        Detailed comparison with alignment
    """
    result = {
        "aligned_comparisons": [],
        "matched": [],
        "mismatched": [],
        "missing_from_extracted": [],
        "extra_in_extracted": [],
        "summary": {},
        "success": True,
        "method": "llm_alignment"
    }

    extracted_datasets = extracted_results.get("datasets", {})

    if not extracted_datasets:
        result["summary"] = {
            "total_expected": 0,
            "matched_count": 0,
            "match_ratio": "0/0",
            "match_percentage": "0%",
            "status": "❌ FAILED - No extracted results"
        }
        return result

    print(f"\n🤖 LLM-Powered Result Alignment")
    print(f"   Extracted datasets: {list(extracted_datasets.keys())[:5]}")

    # Use LLM to parse paper metrics and align with extracted results
    llm = _get_llm_for_extraction()

    # Build extracted results summary for LLM
    extracted_summary = []
    for ds_name, metrics in extracted_datasets.items():
        metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
        extracted_summary.append(f"  - {ds_name}: {metrics_str}")

    prompt = f"""Align experimental results between paper expectations and extracted values.

PAPER EXPECTED RESULTS (may be in any format - tables, text, structured data):
{paper_metrics}

EXTRACTED RESULTS FROM EXPERIMENTS:
{chr(10).join(extracted_summary)}

YOUR TASK:
1. Parse the paper's expected results to identify dataset names and their metric values
2. Match each paper dataset to the corresponding extracted dataset
   - Handle name variations: "roman-empire" = "roman_empire" = "romanempire"
   - Handle abbreviations: "cora" might match "cora_ml" or "cora-full"
   - Use semantic understanding for matching
3. Compare the values and determine if they match within {tolerance*100:.0f}% tolerance
4. Classify each comparison as "matched" (within tolerance) or "mismatched"

Return your analysis as valid JSON:
{{
    "paper_datasets": [
        {{"name": "dataset_name", "metric": "accuracy", "value": 92.55}}
    ],
    "alignments": [
        {{
            "paper_dataset": "roman-empire",
            "extracted_dataset": "roman-empire",
            "paper_value": 92.55,
            "extracted_value": 92.32,
            "metric": "accuracy",
            "match_confidence": "high/medium/low",
            "within_tolerance": true,
            "relative_error_pct": 0.25
        }}
    ],
    "missing_datasets": ["datasets in paper but not in extracted"],
    "extra_datasets": ["datasets in extracted but not in paper"],
    "notes": "any observations about the alignment"
}}

IMPORTANT:
- Values might be in different scales (0-1 vs 0-100) - normalize before comparing
- The paper might use different metric names (acc vs accuracy, ROC-AUC vs auc)
- Focus on the main performance metric for each dataset

Return ONLY the JSON, no other text."""

    try:
        response = llm.invoke(prompt)
        response_text = response.content.strip()

        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            llm_result = json.loads(json_match.group())
        else:
            raise ValueError("Could not parse LLM response as JSON")

    except Exception as e:
        print(f"   ⚠️ LLM alignment error: {e}")
        # Fall back to basic matching
        return _fallback_align_results(extracted_datasets, paper_metrics, tolerance, result)

    # Process LLM alignments
    alignments = llm_result.get("alignments", [])

    for alignment in alignments:
        try:
            paper_value = float(alignment.get("paper_value", 0))
            extracted_value = float(alignment.get("extracted_value", 0))

            # Normalize values to same scale
            if extracted_value < 1 and paper_value > 1:
                extracted_value *= 100
            elif extracted_value > 1 and paper_value < 1:
                paper_value *= 100

            # Calculate relative error
            if paper_value != 0:
                rel_error = abs(extracted_value - paper_value) / paper_value
            else:
                rel_error = abs(extracted_value - paper_value)

            within_tol = rel_error <= tolerance

            comparison = {
                "expected_dataset": alignment.get("paper_dataset", "unknown"),
                "extracted_dataset": alignment.get("extracted_dataset", "unknown"),
                "expected_value": alignment.get("paper_value", 0),
                "extracted_value": extracted_value,
                "metric": alignment.get("metric", "accuracy"),
                "relative_error": rel_error,
                "relative_error_pct": f"{rel_error*100:.2f}%",
                "within_tolerance": within_tol,
                "match_confidence": alignment.get("match_confidence", "unknown")
            }

            result["aligned_comparisons"].append(comparison)

            if within_tol:
                result["matched"].append(comparison)
            else:
                result["mismatched"].append(comparison)

        except Exception as e:
            continue

    # Add missing and extra datasets
    result["missing_from_extracted"] = llm_result.get("missing_datasets", [])
    result["extra_in_extracted"] = llm_result.get("extra_datasets", [])

    # Build summary
    total = len(llm_result.get("paper_datasets", [])) or len(alignments)
    matched_count = len(result["matched"])

    result["summary"] = {
        "total_expected": total,
        "matched_count": matched_count,
        "mismatched_count": len(result["mismatched"]),
        "missing_count": len(result["missing_from_extracted"]),
        "match_ratio": f"{matched_count}/{total}" if total > 0 else "0/0",
        "match_percentage": f"{matched_count/total*100:.1f}%" if total > 0 else "0%",
        "status": "✅ SUCCESS" if matched_count == total and total > 0 else
                  "⚠️ PARTIAL" if matched_count > 0 else "❌ FAILED",
        "llm_notes": llm_result.get("notes", "")
    }

    # Print comparison
    print(f"\n📊 Comparison Results:")
    print(f"   Status: {result['summary']['status']}")
    print(f"   Matched: {result['summary']['match_ratio']} ({result['summary']['match_percentage']})")

    if result["matched"]:
        print(f"\n   ✅ Matched (within {tolerance*100:.0f}% tolerance):")
        for m in result["matched"][:5]:
            print(f"      {m['expected_dataset']}: {m['extracted_value']:.2f} vs {m['expected_value']} ({m['relative_error_pct']})")

    if result["mismatched"]:
        print(f"\n   ❌ Mismatched:")
        for m in result["mismatched"][:5]:
            print(f"      {m['expected_dataset']}: {m['extracted_value']:.2f} vs {m['expected_value']} ({m['relative_error_pct']})")

    if result["missing_from_extracted"]:
        print(f"\n   ⚠️  Missing from results: {result['missing_from_extracted'][:5]}")

    if llm_result.get("notes"):
        print(f"\n   📝 Notes: {llm_result['notes'][:200]}")

    return result


def _fallback_align_results(extracted_datasets: Dict, paper_metrics: str,
                            tolerance: float, result: Dict) -> Dict:
    """Fallback alignment using basic pattern matching when LLM fails."""
    result["method"] = "fallback_pattern_matching"

    # Basic parsing of paper metrics
    expected = {}
    for line in paper_metrics.split('\n'):
        # Try to find "dataset: value" or "dataset = value" patterns
        match = re.search(r'([a-zA-Z][a-zA-Z0-9_-]+)[:\s=]+(\d+\.?\d*)', line)
        if match:
            ds_name = match.group(1).lower()
            if ds_name not in ['accuracy', 'metric', 'value', 'dataset', 'result']:
                try:
                    expected[ds_name] = float(match.group(2))
                except:
                    pass

    # Normalize function for fuzzy matching
    def normalize(name):
        return re.sub(r'[_\-\s]', '', name.lower())

    # Try to match datasets
    for exp_ds, exp_val in expected.items():
        norm_exp = normalize(exp_ds)
        matched = False

        for ext_ds, metrics in extracted_datasets.items():
            norm_ext = normalize(ext_ds)

            if norm_exp == norm_ext or norm_exp in norm_ext or norm_ext in norm_exp:
                # Found a match
                ext_val = list(metrics.values())[0] if metrics else 0

                # Normalize scale
                if ext_val < 1 and exp_val > 1:
                    ext_val *= 100
                elif ext_val > 1 and exp_val < 1:
                    exp_val *= 100

                rel_error = abs(ext_val - exp_val) / exp_val if exp_val != 0 else abs(ext_val - exp_val)

                comparison = {
                    "expected_dataset": exp_ds,
                    "extracted_dataset": ext_ds,
                    "expected_value": exp_val,
                    "extracted_value": ext_val,
                    "metric": "accuracy",
                    "relative_error": rel_error,
                    "relative_error_pct": f"{rel_error*100:.2f}%",
                    "within_tolerance": rel_error <= tolerance
                }

                result["aligned_comparisons"].append(comparison)
                if comparison["within_tolerance"]:
                    result["matched"].append(comparison)
                else:
                    result["mismatched"].append(comparison)

                matched = True
                break

        if not matched:
            result["missing_from_extracted"].append(exp_ds)

    # Summary
    total = len(expected)
    matched_count = len(result["matched"])

    result["summary"] = {
        "total_expected": total,
        "matched_count": matched_count,
        "match_ratio": f"{matched_count}/{total}" if total > 0 else "0/0",
        "match_percentage": f"{matched_count/total*100:.1f}%" if total > 0 else "0%",
        "status": "✅ SUCCESS" if matched_count == total and total > 0 else
                  "⚠️ PARTIAL" if matched_count > 0 else "❌ FAILED"
    }

    return result


@tool
def read_result_files(file_paths: str, expected_metrics: str = "") -> Dict[str, Any]:
    """
    Read and extract metrics from result files - handles many formats flexibly.

    This tool reads result files and extracts numerical metrics. It handles:
    - JSON files (nested structures, arrays, flat dicts)
    - CSV/TSV files (gets best/last row)
    - Text/log files (regex extraction)
    - YAML files
    - Any file with metric-like content

    Args:
        file_paths: Comma-separated list of file paths to read
        expected_metrics: Optional context about expected metrics (e.g., "accuracy, F1, BLEU")

    Returns:
        Dictionary with:
            - metrics: Extracted metrics from all files
            - file_contents: Summary of each file's contents
            - best_results: Best values found for each metric
            - extraction_success: Whether extraction succeeded
    """
    import csv

    result = {
        "metrics": {},
        "file_contents": {},
        "best_results": {},
        "files_read": 0,
        "extraction_success": False,
        "raw_content": {}  # Store raw content for debugging
    }

    # Parse expected metrics for smarter extraction
    expected_metric_names = []
    if expected_metrics:
        # Extract metric names from the expected string
        expected_metric_names = re.findall(r'\b(accuracy|acc|f1|precision|recall|bleu|rouge|loss|perplexity|ppl|score|auc|map|mse|mae|rmse|error|em|exact_match)\b', expected_metrics.lower())

    paths = [p.strip() for p in file_paths.split(",") if p.strip()]

    for file_path in paths:
        path = Path(file_path)
        if not path.exists():
            result["file_contents"][file_path] = {"error": "File not found"}
            continue

        try:
            ext = path.suffix.lower()
            content = None
            file_metrics = {}
            raw_text = ""

            # Read file content
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_text = f.read()
            except Exception as e:
                result["file_contents"][file_path] = {"error": f"Read error: {e}"}
                continue

            # Store raw content (truncated)
            result["raw_content"][file_path] = raw_text[:1000]

            # Parse based on file type
            if ext == ".json":
                try:
                    content = json.loads(raw_text)
                    file_metrics = _extract_metrics_from_json(content)
                except json.JSONDecodeError:
                    # Maybe it's JSONL or malformed JSON, try line by line
                    for line in raw_text.split('\n'):
                        line = line.strip()
                        if line.startswith('{'):
                            try:
                                line_json = json.loads(line)
                                line_metrics = _extract_metrics_from_json(line_json)
                                file_metrics.update(line_metrics)
                            except:
                                pass
                    # Also try text extraction as fallback
                    file_metrics.update(_extract_metrics_from_text(raw_text))

            elif ext in [".csv", ".tsv"]:
                delimiter = '\t' if ext == ".tsv" else ','
                try:
                    lines = raw_text.strip().split('\n')
                    if len(lines) > 1:
                        reader = csv.DictReader(lines, delimiter=delimiter)
                        rows = list(reader)
                        if rows:
                            # Get last row (usually final results)
                            content = rows[-1]
                            file_metrics = _extract_metrics_from_dict(content)

                            # Also check for "best" row if exists
                            for row in rows:
                                if any(v and 'best' in str(v).lower() for v in row.values()):
                                    best_metrics = _extract_metrics_from_dict(row)
                                    for k, v in best_metrics.items():
                                        file_metrics[f"best_{k}"] = v
                except:
                    # Fallback to text extraction
                    file_metrics = _extract_metrics_from_text(raw_text)

            elif ext in [".yaml", ".yml"]:
                try:
                    import yaml
                    content = yaml.safe_load(raw_text)
                    if isinstance(content, dict):
                        file_metrics = _extract_metrics_from_json(content)
                except:
                    file_metrics = _extract_metrics_from_text(raw_text)

            elif ext == ".txt" or ext == ".log":
                content = raw_text[:3000]
                file_metrics = _extract_metrics_from_text(raw_text)

            else:
                # Try to auto-detect format
                content = raw_text[:2000]

                # Try JSON first
                if raw_text.strip().startswith('{') or raw_text.strip().startswith('['):
                    try:
                        parsed = json.loads(raw_text)
                        file_metrics = _extract_metrics_from_json(parsed)
                    except:
                        pass

                # Fallback to text extraction
                if not file_metrics:
                    file_metrics = _extract_metrics_from_text(raw_text)

            # If we have expected metrics, try targeted extraction
            if expected_metric_names and not file_metrics:
                for metric_name in expected_metric_names:
                    patterns = [
                        rf'{metric_name}\s*[:=]\s*(\d+\.?\d*)',
                        rf'"{metric_name}"\s*:\s*(\d+\.?\d*)',
                        rf"'{metric_name}'\s*:\s*(\d+\.?\d*)",
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, raw_text, re.IGNORECASE)
                        if match:
                            try:
                                file_metrics[metric_name] = float(match.group(1))
                            except:
                                pass

            result["file_contents"][file_path] = {
                "type": ext,
                "size_bytes": len(raw_text),
                "content_preview": str(content)[:300] if content else raw_text[:300],
                "metrics_found": len(file_metrics),
                "metrics": file_metrics
            }

            # Merge metrics (keep best values)
            for key, value in file_metrics.items():
                key_lower = key.lower()
                if key_lower not in result["metrics"]:
                    result["metrics"][key_lower] = value
                else:
                    # For accuracy-like metrics, keep higher; for loss, keep lower
                    if any(kw in key_lower for kw in ["loss", "error", "perplexity", "ppl", "mse", "mae", "rmse"]):
                        result["metrics"][key_lower] = min(result["metrics"][key_lower], value)
                    else:
                        result["metrics"][key_lower] = max(result["metrics"][key_lower], value)

            result["files_read"] += 1

        except Exception as e:
            result["file_contents"][file_path] = {"error": str(e)}

    # Set best results
    result["best_results"] = result["metrics"].copy()
    result["extraction_success"] = len(result["metrics"]) > 0

    # Summary
    print(f"\n📊 Result File Reading Summary:")
    print(f"   Files processed: {result['files_read']}/{len(paths)}")

    if result["metrics"]:
        print(f"   ✅ Extracted {len(result['metrics'])} metrics:")
        for key, val in list(result["metrics"].items())[:8]:
            print(f"      {key}: {val}")
    else:
        print(f"   ⚠️  No metrics extracted")
        # Show file previews to help debug
        for fp, info in result["file_contents"].items():
            if "error" not in info:
                print(f"   File: {Path(fp).name}")
                print(f"      Preview: {info.get('content_preview', '')[:100]}...")

    return result


def _extract_metrics_from_json(data: Any, prefix: str = "", depth: int = 0) -> Dict[str, float]:
    """Recursively extract numeric metrics from JSON data - very flexible."""
    metrics = {}

    # Prevent infinite recursion
    if depth > 10:
        return metrics

    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{prefix}_{key}" if prefix else key
            key_lower = key.lower()

            if isinstance(value, (int, float)) and not isinstance(value, bool):
                # Very flexible metric detection - capture almost any numeric field
                # Exclude obvious non-metrics
                skip_patterns = ["epoch", "step", "batch", "iter", "time", "seed", "gpu", "num_", "size", "len", "count", "id", "index", "version"]
                is_skip = any(skip in key_lower for skip in skip_patterns)

                # Metric-like patterns (very broad)
                metric_patterns = [
                    "acc", "accuracy", "f1", "precision", "recall", "bleu", "rouge",
                    "loss", "perplexity", "ppl", "score", "metric", "eval", "test",
                    "valid", "train", "mse", "mae", "rmse", "auc", "map", "mrr",
                    "ndcg", "hit", "rate", "ratio", "percent", "avg", "mean",
                    "best", "final", "result", "performance", "error", "em",
                    "exact", "match", "correct", "top", "micro", "macro", "weighted"
                ]
                is_metric = any(pattern in key_lower for pattern in metric_patterns)

                # Include if it's metric-like OR if it's a simple numeric field at the top level
                if is_metric or (not is_skip and depth <= 2):
                    metrics[full_key.lower()] = float(value)

            elif isinstance(value, str):
                # Try to parse numeric strings
                try:
                    num_val = float(value.strip().rstrip('%'))
                    metric_patterns = ["acc", "f1", "loss", "score", "precision", "recall", "bleu", "rouge", "error", "perplexity"]
                    if any(pattern in key_lower for pattern in metric_patterns):
                        metrics[full_key.lower()] = num_val
                except:
                    pass

            elif isinstance(value, dict):
                metrics.update(_extract_metrics_from_json(value, full_key, depth + 1))

            elif isinstance(value, list) and len(value) > 0:
                # Get last value if it's a list of numbers (likely training history)
                if isinstance(value[-1], (int, float)) and not isinstance(value[-1], bool):
                    metrics[f"{full_key}_final".lower()] = float(value[-1])
                    # Also get best value
                    if all(isinstance(v, (int, float)) for v in value):
                        # For loss-like metrics, get min; for others, get max
                        if any(kw in key_lower for kw in ["loss", "error", "perplexity"]):
                            metrics[f"{full_key}_best".lower()] = float(min(value))
                        else:
                            metrics[f"{full_key}_best".lower()] = float(max(value))
                elif isinstance(value[-1], dict):
                    # Last element of array is dict - common for results
                    metrics.update(_extract_metrics_from_json(value[-1], full_key, depth + 1))

    elif isinstance(data, list) and len(data) > 0:
        # If data is a list, try to extract from last element
        if isinstance(data[-1], dict):
            metrics.update(_extract_metrics_from_json(data[-1], prefix, depth + 1))

    return metrics


def _extract_metrics_from_dict(data: Dict) -> Dict[str, float]:
    """Extract numeric values from a flat dictionary."""
    metrics = {}
    for key, value in data.items():
        try:
            val = float(value)
            if not (val != val):  # Not NaN
                metrics[key.lower()] = val
        except (ValueError, TypeError):
            pass
    return metrics


def _extract_metrics_from_text(text: str) -> Dict[str, float]:
    """Extract metrics from text using regex patterns - very comprehensive."""
    metrics = {}

    # Common patterns - ordered by specificity
    patterns = [
        # "Test Accuracy: 94.5%" or "test_accuracy = 0.945"
        r'(test|train|val|valid|eval|dev|best|final)[_\s]*(accuracy|acc|loss|f1|score|precision|recall|bleu|rouge|perplexity|ppl|error|em|exact_match)\s*[:=]\s*(\d+\.?\d*)\s*%?',

        # "Accuracy: 0.95" or "accuracy = 95.2%"
        r'\b(accuracy|acc|f1|precision|recall|bleu|rouge|perplexity|ppl|loss|error|score|auc|mse|mae|rmse|em|exact_match|hit|ndcg|mrr|map)\s*[:=]\s*(\d+\.?\d*)\s*%?',

        # "BLEU-4: 28.4" or "ROUGE-L: 45.2"
        r'(bleu-?\d*|rouge-?[12lL]?|meteor)\s*[:=]\s*(\d+\.?\d*)',

        # "F1-score: 0.89" or "F1 score: 89%"
        r'(f1-?score|f-?score|f-?1|f-?measure)\s*[:=]\s*(\d+\.?\d*)\s*%?',

        # "top-1 accuracy: 76.5" or "top1: 76.5"
        r'(top-?\d+)\s*(?:accuracy|acc)?\s*[:=]\s*(\d+\.?\d*)\s*%?',

        # Sentence patterns: "achieved 94.5% accuracy" or "accuracy of 94.5%"
        r'(?:achieved|obtained|got|reached|is|was|:)\s*(\d+\.?\d*)\s*%?\s*(accuracy|acc|f1|precision|recall|bleu|rouge)',
        r'(accuracy|f1|precision|recall|bleu|rouge)\s+(?:of|is|was|:)\s*(\d+\.?\d*)\s*%?',

        # Table-like patterns: "| accuracy | 94.5 |" or "accuracy    94.5"
        r'\|\s*(accuracy|acc|f1|loss|precision|recall|bleu|rouge|score)\s*\|\s*(\d+\.?\d*)',
        r'\b(accuracy|acc|f1|loss|precision|recall|score)\s+(\d+\.?\d{2,})\b',

        # Final/Best results: "Best accuracy: 95.2" or "Final loss: 0.01"
        r'(best|final|peak|max|min)\s*(accuracy|acc|loss|f1|score)\s*[:=]?\s*(\d+\.?\d*)',

        # Epoch/step results: "Epoch 10: accuracy=94.5"
        r'(?:epoch|step)\s*\d+\s*[:\-]\s*(accuracy|acc|loss|f1)\s*[:=]\s*(\d+\.?\d*)',
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            groups = match.groups()
            if len(groups) >= 2:
                # Handle different group configurations
                if len(groups) == 3:  # Pattern with prefix like "test accuracy"
                    name = f"{groups[0]}_{groups[1]}".lower()
                    value = groups[2]
                elif len(groups) == 2:
                    # Check if first group is a number (reversed pattern)
                    try:
                        float(groups[0])
                        # First is value, second is name
                        name = groups[1].lower()
                        value = groups[0]
                    except ValueError:
                        # First is name, second is value
                        name = groups[0].lower()
                        value = groups[1]
                else:
                    continue

                try:
                    float_val = float(value)
                    # Skip obviously wrong values
                    if float_val < 0 or float_val > 10000:
                        continue
                    # Keep best value for each metric
                    if name not in metrics:
                        metrics[name] = float_val
                    else:
                        # For loss-like, keep lower; for others, keep higher
                        if any(kw in name for kw in ["loss", "error", "perplexity", "ppl"]):
                            metrics[name] = min(metrics[name], float_val)
                        else:
                            metrics[name] = max(metrics[name], float_val)
                except ValueError:
                    pass

    return metrics


@tool
def read_log_tail(log_path: str, num_lines: int = 30, check_errors: bool = True) -> Dict[str, Any]:
    """
    Read only the last N lines of a log file to check for completion/errors.

    Use this INSTEAD of reading the entire log file. This is efficient for:
    - Checking if experiment completed successfully
    - Finding error messages at the end
    - Verifying final status

    Args:
        log_path: Path to log file
        num_lines: Number of lines to read from end (default: 30)
        check_errors: Whether to scan for error patterns (default: True)

    Returns:
        Dictionary with:
            - tail: Last N lines of the log
            - completed: Whether experiment appears completed
            - has_errors: Whether errors were found
            - error_summary: Summary of errors if found
            - final_metrics: Any metrics in the tail section
    """
    result = {
        "tail": "",
        "completed": False,
        "has_errors": False,
        "error_summary": [],
        "final_metrics": {},
        "success": True
    }

    path = Path(log_path)
    if not path.exists():
        return {"success": False, "error": f"Log file not found: {log_path}"}

    try:
        # Read file efficiently from end
        with open(path, 'rb') as f:
            # Go to end of file
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()

            # Read last chunk (estimate ~100 chars per line)
            chunk_size = min(file_size, num_lines * 150)
            f.seek(max(0, file_size - chunk_size))

            content = f.read().decode('utf-8', errors='ignore')

        # Get last N lines
        lines = content.split('\n')
        tail_lines = lines[-num_lines:] if len(lines) > num_lines else lines
        result["tail"] = '\n'.join(tail_lines)

        tail_text = result["tail"].lower()

        # Check for completion indicators
        completion_patterns = [
            "training complete", "evaluation complete", "finished",
            "done", "completed successfully", "saved", "total time",
            "experiment finished", "all experiments done"
        ]
        result["completed"] = any(p in tail_text for p in completion_patterns)

        # Check for errors if requested
        if check_errors:
            error_patterns = [
                r'error[:\s]',
                r'exception[:\s]',
                r'traceback',
                r'failed',
                r'cuda.*error',
                r'out of memory',
                r'oom',
                r'killed',
                r'modulenotfounderror',
                r'importerror',
            ]

            for line in tail_lines:
                line_lower = line.lower()
                for pattern in error_patterns:
                    if re.search(pattern, line_lower):
                        result["has_errors"] = True
                        # Extract error message
                        if len(line.strip()) > 10 and len(line.strip()) < 500:
                            if line.strip() not in result["error_summary"]:
                                result["error_summary"].append(line.strip())
                        break

            # Limit error summary
            result["error_summary"] = result["error_summary"][:5]

        # Extract any metrics from tail
        result["final_metrics"] = _extract_metrics_from_text(result["tail"])

        # Summary output
        status = "✅ COMPLETED" if result["completed"] else "⚠️  Unknown status"
        if result["has_errors"]:
            status = "❌ ERRORS DETECTED"
        print(f"📋 Log tail ({num_lines} lines): {status}")
        if result["final_metrics"]:
            print(f"   Found metrics: {list(result['final_metrics'].keys())[:3]}")

        return result

    except Exception as e:
        return {"success": False, "error": f"Failed to read log: {str(e)}"}


@tool
def verify_experiment_results(
    repo_path: str,
    paper_expected_results: str,
    experiment_log: str = "",
    tolerance: float = 0.05
) -> Dict[str, Any]:
    """
    Complete verification workflow: discover results, extract metrics, compare with paper.

    This is a HIGH-LEVEL tool that combines:
    1. discover_result_files - Find result files
    2. read_result_files - Extract metrics from files
    3. read_log_tail - Check log for errors (only if needed)
    4. compare_with_paper_results - Verify against paper

    Args:
        repo_path: Path to repository
        paper_expected_results: Expected results from paper (e.g., "accuracy: 94.5%, F1: 0.89")
        experiment_log: Optional path to experiment log (for error checking)
        tolerance: Tolerance for metric comparison (default: 5%)

    Returns:
        Comprehensive verification result with:
            - verified: Whether results match paper expectations
            - extracted_metrics: All metrics found
            - comparison: Detailed comparison with paper
            - errors: Any errors found
            - recommendation: Next steps
    """
    result = {
        "verified": False,
        "extracted_metrics": {},
        "comparison": {},
        "errors": [],
        "source_files": [],
        "recommendation": "",
        "success": True
    }

    try:
        # Step 1: Discover result files
        print("\n🔍 Step 1: Discovering result files...")
        discovery = discover_result_files.invoke({"repo_path": repo_path})

        if not discovery.get("success", False):
            result["errors"].append(f"Discovery failed: {discovery.get('error', 'Unknown')}")
            result["recommendation"] = "Check if experiments produced output files"
            return result

        result_files = discovery.get("priority_files", [])

        # Step 2: Read and extract metrics from result files
        if result_files:
            print(f"\n📊 Step 2: Extracting metrics from {len(result_files)} files...")
            file_paths = ",".join([f["path"] for f in result_files])
            extraction = read_result_files.invoke({
                "file_paths": file_paths,
                "expected_metrics": paper_expected_results
            })

            result["extracted_metrics"] = extraction.get("metrics", {})
            result["source_files"] = [f["relative_path"] for f in result_files]

        # Step 3: If no metrics from files, check log tail
        if not result["extracted_metrics"] and experiment_log:
            print("\n📋 Step 3: Checking log file (no result files found)...")
            log_result = read_log_tail.invoke({
                "log_path": experiment_log,
                "num_lines": 50,
                "check_errors": True
            })

            if log_result.get("has_errors"):
                result["errors"].extend(log_result.get("error_summary", []))

            if log_result.get("final_metrics"):
                result["extracted_metrics"] = log_result.get("final_metrics", {})
                result["source_files"] = ["log_tail"]

        # Step 4: Compare with paper results
        if result["extracted_metrics"]:
            print("\n✅ Step 4: Comparing with paper results...")
            comparison = compare_with_paper_results.invoke({
                "extracted_metrics": result["extracted_metrics"],
                "expected_results_str": paper_expected_results,
                "tolerance": tolerance
            })

            result["comparison"] = comparison
            result["verified"] = comparison.get("success", False)

            # Generate recommendation
            if result["verified"]:
                result["recommendation"] = "✅ Results verified! Experiment successfully reproduced paper results."
            elif comparison.get("match_count", 0) > 0:
                result["recommendation"] = f"⚠️ Partial match: {comparison.get('success_portion', '0/0')}. Consider re-running with different hyperparameters."
            else:
                result["recommendation"] = "❌ No metrics matched. Check if experiment completed successfully."
        else:
            result["recommendation"] = "❌ No metrics found. Check experiment output or run experiments."

        print(f"\n{'='*50}")
        print(f"VERIFICATION RESULT: {'✅ PASSED' if result['verified'] else '❌ FAILED'}")
        print(f"{'='*50}")

        return result

    except Exception as e:
        result["errors"].append(str(e))
        result["success"] = False
        return result


@tool
def get_experiment_checkpoint_status(repo_path: str) -> Dict[str, Any]:
    """
    Check checkpoint status to determine which experiments can be resumed.

    This tool helps the agent:
    1. Find existing checkpoints from previous runs
    2. Determine which experiments completed vs need retry
    3. Identify the last successful checkpoint for resume

    Args:
        repo_path: Path to repository

    Returns:
        Dictionary with:
            - checkpoints_found: List of checkpoint files/directories
            - completed_experiments: List of completed experiment names
            - last_checkpoint: Path to most recent checkpoint
            - can_resume: Whether resume is possible
            - resume_from: Suggested starting point
    """
    import glob
    from datetime import datetime

    result = {
        "checkpoints_found": [],
        "completed_experiments": [],
        "last_checkpoint": None,
        "can_resume": False,
        "resume_from": None,
        "training_logs": [],
        "success": True
    }

    repo = Path(repo_path).resolve()

    # Common checkpoint patterns
    checkpoint_patterns = [
        "**/checkpoint*",
        "**/checkpoints/**",
        "**/ckpt*",
        "**/saved_models/**",
        "**/*.pt",
        "**/*.pth",
        "**/*.ckpt",
        "**/model_*.bin",
    ]

    checkpoints = []

    for pattern in checkpoint_patterns:
        matches = glob.glob(str(repo / pattern), recursive=True)
        for match in matches[:20]:  # Limit search
            path = Path(match)
            try:
                stat = path.stat()
                checkpoints.append({
                    "path": str(path),
                    "relative_path": str(path.relative_to(repo)),
                    "modified": datetime.fromtimestamp(stat.st_mtime),
                    "size_bytes": stat.st_size if path.is_file() else 0,
                    "is_dir": path.is_dir()
                })
            except:
                pass

    # Sort by modification time
    checkpoints.sort(key=lambda x: x["modified"], reverse=True)

    result["checkpoints_found"] = [
        {
            "path": c["relative_path"],
            "modified": c["modified"].isoformat(),
            "size_mb": c["size_bytes"] / (1024*1024) if c["size_bytes"] else 0
        }
        for c in checkpoints[:10]
    ]

    # Find last checkpoint
    if checkpoints:
        result["last_checkpoint"] = checkpoints[0]["relative_path"]
        result["can_resume"] = True

        # Try to determine experiment name from path
        last_path = checkpoints[0]["path"]
        # Common patterns: checkpoint-500, epoch_3, step_1000
        name_match = re.search(r'(checkpoint-?\d+|epoch[_-]?\d+|step[_-]?\d+)', last_path, re.IGNORECASE)
        if name_match:
            result["resume_from"] = name_match.group(1)

    # Look for training state files
    state_patterns = ["**/trainer_state.json", "**/training_state.json", "**/training_args.bin"]
    for pattern in state_patterns:
        matches = glob.glob(str(repo / pattern), recursive=True)
        for match in matches[:3]:
            try:
                with open(match, 'r') as f:
                    if match.endswith('.json'):
                        state = json.load(f)
                        # Extract useful info
                        if "global_step" in state:
                            result["completed_experiments"].append(f"Step {state['global_step']}")
                        if "epoch" in state:
                            result["completed_experiments"].append(f"Epoch {state['epoch']}")
            except:
                pass

    # Summary
    if result["can_resume"]:
        print(f"📂 Found {len(checkpoints)} checkpoints")
        print(f"   Last: {result['last_checkpoint']}")
        if result["resume_from"]:
            print(f"   Resume from: {result['resume_from']}")
    else:
        print("📂 No checkpoints found - starting fresh")

    return result


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
    # New result discovery and verification tools
    discover_result_files,
    read_result_files,
    read_log_tail,
    verify_experiment_results,
    get_experiment_checkpoint_status,
    generate_comparison_report,
]
