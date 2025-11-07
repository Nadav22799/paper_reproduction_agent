"""Simple Environment Setup - No LLM/Agent needed, just run commands directly."""

import os
import subprocess
from pathlib import Path
from typing import Dict


def simple_setup_environment(code_path: str) -> Dict:
    """
    Install dependencies without using an LLM agent.

    This is a fallback for when the LLM isn't working properly.
    It follows a simple algorithm:
    1. Look for setup.py or requirements.txt
    2. Run pip install
    3. Report success/failure

    Args:
        code_path: Path to repository

    Returns:
        Setup results with status and errors
    """
    print(f"🔧 Simple environment setup for: {code_path}")

    code_path = Path(code_path)
    if not code_path.exists():
        return {
            "success": False,
            "dependencies_found": False,
            "dependencies_installed": False,
            "errors": [f"Directory {code_path} does not exist"],
            "report": f"Directory {code_path} does not exist"
        }

    # Look for dependency files
    setup_py = code_path / "setup.py"
    requirements_txt = code_path / "requirements.txt"
    pyproject_toml = code_path / "pyproject.toml"

    result = {
        "success": False,
        "dependencies_found": False,
        "dependencies_installed": False,
        "errors": [],
        "report": ""
    }

    # Try setup.py first
    if setup_py.exists():
        print(f"✅ Found setup.py")
        result["dependencies_found"] = True

        print(f"📦 Installing via: pip install -e .")
        try:
            process = subprocess.run(
                ["pip", "install", "-e", "."],
                cwd=str(code_path),
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes
            )

            if process.returncode == 0:
                print(f"✅ Installation successful!")
                result["success"] = True
                result["dependencies_installed"] = True
                result["report"] = f"Successfully installed dependencies from setup.py\n\nOutput:\n{process.stdout[-500:]}"
            else:
                print(f"❌ Installation failed!")
                print(f"Error: {process.stderr[:500]}")
                result["errors"].append(f"pip install failed: {process.stderr[:200]}")
                result["report"] = f"Failed to install from setup.py\n\nError:\n{process.stderr[:500]}"

        except subprocess.TimeoutExpired:
            print(f"❌ Installation timed out after 10 minutes")
            result["errors"].append("Installation timed out")
            result["report"] = "Installation timed out after 10 minutes"
        except Exception as e:
            print(f"❌ Error running pip: {e}")
            result["errors"].append(str(e))
            result["report"] = f"Error: {e}"

        return result

    # Try requirements.txt
    if requirements_txt.exists():
        print(f"✅ Found requirements.txt")
        result["dependencies_found"] = True

        print(f"📦 Installing via: pip install -r requirements.txt")
        try:
            process = subprocess.run(
                ["pip", "install", "-r", "requirements.txt"],
                cwd=str(code_path),
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes
            )

            if process.returncode == 0:
                print(f"✅ Installation successful!")
                result["success"] = True
                result["dependencies_installed"] = True
                result["report"] = f"Successfully installed dependencies from requirements.txt\n\nOutput:\n{process.stdout[-500:]}"
            else:
                print(f"❌ Installation failed!")
                print(f"Error: {process.stderr[:500]}")
                result["errors"].append(f"pip install failed: {process.stderr[:200]}")
                result["report"] = f"Failed to install from requirements.txt\n\nError:\n{process.stderr[:500]}"

        except subprocess.TimeoutExpired:
            print(f"❌ Installation timed out after 10 minutes")
            result["errors"].append("Installation timed out")
            result["report"] = "Installation timed out after 10 minutes"
        except Exception as e:
            print(f"❌ Error running pip: {e}")
            result["errors"].append(str(e))
            result["report"] = f"Error: {e}"

        return result

    # Try pyproject.toml
    if pyproject_toml.exists():
        print(f"✅ Found pyproject.toml")
        result["dependencies_found"] = True

        print(f"📦 Installing via: pip install -e .")
        try:
            process = subprocess.run(
                ["pip", "install", "-e", "."],
                cwd=str(code_path),
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes
            )

            if process.returncode == 0:
                print(f"✅ Installation successful!")
                result["success"] = True
                result["dependencies_installed"] = True
                result["report"] = f"Successfully installed dependencies from pyproject.toml\n\nOutput:\n{process.stdout[-500:]}"
            else:
                print(f"❌ Installation failed!")
                print(f"Error: {process.stderr[:500]}")
                result["errors"].append(f"pip install failed: {process.stderr[:200]}")
                result["report"] = f"Failed to install from pyproject.toml\n\nError:\n{process.stderr[:500]}"

        except subprocess.TimeoutExpired:
            print(f"❌ Installation timed out after 10 minutes")
            result["errors"].append("Installation timed out")
            result["report"] = "Installation timed out after 10 minutes"
        except Exception as e:
            print(f"❌ Error running pip: {e}")
            result["errors"].append(str(e))
            result["report"] = f"Error: {e}"

        return result

    # No dependency files found
    print(f"⚠️  No dependency files found (setup.py, requirements.txt, or pyproject.toml)")
    result["errors"].append("No dependency files found")
    result["report"] = "No dependency files found in repository"
    return result
