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


@tool
def execute_shell_command(command: str, cwd: str = ".", timeout: int = 300) -> Dict[str, Any]:
    """
    Execute a shell command and capture output.

    Args:
        command: Shell command to execute
        cwd: Working directory for command execution
        timeout: Execution timeout in seconds

    Returns:
        Dictionary with stdout, stderr, and return code
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

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
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
    Automatically uses the virtual environment if it exists in the repo.

    Args:
        script_path: Path to Python script
        args: Command line arguments as space-separated string
        timeout: Execution timeout in seconds

    Returns:
        Dictionary with stdout, stderr, and return code
    """
    try:
        # Check for venv in the script's directory or parent directories
        script_dir = Path(script_path).parent.resolve()
        venv_python = None

        # Look for venv in current dir and up to 2 parent levels
        for check_dir in [script_dir, script_dir.parent, script_dir.parent.parent]:
            potential_venv = check_dir / "venv"
            if potential_venv.exists():
                if os.name == 'nt':  # Windows
                    venv_python_path = potential_venv / "Scripts" / "python.exe"
                else:  # Linux/Mac
                    venv_python_path = potential_venv / "bin" / "python"

                if venv_python_path.exists():
                    venv_python = str(venv_python_path)
                    print(f"🐍 Using virtual environment Python: {venv_python}")
                    break

        # Use venv python if found, otherwise system python
        cmd = [venv_python if venv_python else "python", script_path]
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
            "used_venv": venv_python is not None,
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
    Automatically uses the virtual environment if it exists in the repo.

    Args:
        test_path: Path to test file or directory
        args: Additional pytest arguments

    Returns:
        Test results
    """
    try:
        # Check for venv in the test path's directory or parent directories
        test_dir = Path(test_path).resolve()
        if test_dir.is_file():
            test_dir = test_dir.parent

        venv_python = None

        # Look for venv in current dir and up to 2 parent levels
        for check_dir in [test_dir, test_dir.parent, test_dir.parent.parent]:
            potential_venv = check_dir / "venv"
            if potential_venv.exists():
                if os.name == 'nt':  # Windows
                    venv_python_path = potential_venv / "Scripts" / "python.exe"
                else:  # Linux/Mac
                    venv_python_path = potential_venv / "bin" / "python"

                if venv_python_path.exists():
                    venv_python = str(venv_python_path)
                    print(f"🐍 Using virtual environment Python for pytest: {venv_python}")
                    break

        # Use venv python with pytest module if found, otherwise system pytest
        if venv_python:
            cmd = [venv_python, "-m", "pytest", test_path, "-v", "--tb=short"]
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
            "used_venv": venv_python is not None,
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


@tool
def smart_install_dependencies(repo_path: str) -> Dict[str, Any]:
    """
    Install dependencies using intelligent fallback strategies in an isolated virtual environment.

    This tool will:
    1. Check required Python version for the repo
    2. Create a virtual environment with the correct Python version (using pyenv if needed)
    3. Try installing with original requirements
    4. If that fails, try with relaxed version constraints
    5. If that fails, try with unpinned versions

    Args:
        repo_path: Path to repository

    Returns:
        Installation result with status and any errors
    """
    try:
        import sys
        import venv
        import shutil

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
]
