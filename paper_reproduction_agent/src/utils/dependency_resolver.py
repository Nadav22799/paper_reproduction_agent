"""Smart Dependency Resolver with fallback strategies.

Attempts to install dependencies with intelligent version fallbacks
to avoid getting stuck in dependency hell loops.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ErrorPatternDetector:
    """Detect repeated error patterns and suggest stopping early."""

    def __init__(self, max_same_error: int = 3):
        self.error_history: List[str] = []
        self.max_same_error = max_same_error

    def add_error(self, error_message: str) -> None:
        """Add an error to history."""
        # Normalize error message for comparison
        normalized = self._normalize_error(error_message)
        self.error_history.append(normalized)

    def should_stop(self) -> Tuple[bool, Optional[str]]:
        """
        Check if we should stop trying based on error patterns.

        Returns:
            (should_stop, reason)
        """
        if len(self.error_history) < 2:
            return False, None

        # Check for repeated identical errors
        recent_errors = self.error_history[-self.max_same_error:]
        if len(set(recent_errors)) == 1:
            return True, f"Same error repeated {self.max_same_error} times"

        # Check for Python version incompatibility patterns
        for error in self.error_history:
            if self._is_python_version_error(error):
                return True, "Python version incompatibility detected"

        # Check for unresolvable dependency conflicts
        if len(self.error_history) >= 5:
            conflict_count = sum(1 for e in self.error_history if 'conflict' in e.lower())
            if conflict_count >= 3:
                return True, "Multiple dependency conflicts - likely unresolvable"

        return False, None

    def _normalize_error(self, error: str) -> str:
        """Normalize error message for comparison."""
        # Remove version numbers and package names to detect pattern
        normalized = re.sub(r'\d+\.\d+\.\d+', 'X.X.X', error.lower())
        normalized = re.sub(r'["\']([^"\']+)["\']', 'PACKAGE', normalized)

        # Extract key error phrases
        key_phrases = [
            "could not find a version that satisfies",
            "no matching distribution found",
            "requires python",
            "syntax error",
            "module not found",
            "dependency conflict",
            "incompatible",
        ]

        for phrase in key_phrases:
            if phrase in normalized:
                return phrase

        return normalized[:100]

    def _is_python_version_error(self, error: str) -> bool:
        """Check if error indicates Python version incompatibility."""
        patterns = [
            r"requires python [<>=!]+\s*[\d.]+",
            r"python [<>=!]+\s*[\d.]+ is required",
            r"not available for python \d+\.\d+",
        ]

        for pattern in patterns:
            if re.search(pattern, error.lower()):
                return True

        return False

    def get_error_summary(self) -> str:
        """Get a summary of errors encountered."""
        if not self.error_history:
            return "No errors"

        unique_errors = list(set(self.error_history))
        return f"Encountered {len(self.error_history)} errors ({len(unique_errors)} unique patterns)"


class DependencyResolver:
    """Resolve dependencies with intelligent fallback strategies."""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.error_detector = ErrorPatternDetector()
        self.attempted_strategies: List[str] = []

    def resolve_and_install(self) -> Dict:
        """
        Attempt to install dependencies using multiple strategies.

        Returns:
            Dict with success status, strategy used, and any errors
        """
        result = {
            "success": False,
            "strategy_used": None,
            "dependencies_installed": False,
            "errors": [],
            "warnings": [],
            "attempts": []
        }

        # Strategy 0: Try setup.py installation first (most common for research repos)
        if (self.repo_path / "setup.py").exists():
            strategy_result = self._try_setup_py_install()
            result["attempts"].append(strategy_result)

            if strategy_result["success"]:
                result["success"] = True
                result["strategy_used"] = "setup.py"
                result["dependencies_installed"] = True
                return result

            self.error_detector.add_error(strategy_result.get("error", ""))

        # Strategy 1: Try original requirements as-is
        strategy_result = self._try_original_requirements()
        result["attempts"].append(strategy_result)

        if strategy_result["success"]:
            result["success"] = True
            result["strategy_used"] = "original"
            result["dependencies_installed"] = True
            return result

        self.error_detector.add_error(strategy_result.get("error", ""))

        # Check if we should stop early
        should_stop, reason = self.error_detector.should_stop()
        if should_stop:
            result["errors"].append(f"Stopping early: {reason}")
            return result

        # Strategy 2: Try relaxing version constraints
        strategy_result = self._try_relaxed_versions()
        result["attempts"].append(strategy_result)

        if strategy_result["success"]:
            result["success"] = True
            result["strategy_used"] = "relaxed_versions"
            result["dependencies_installed"] = True
            result["warnings"].append("Installed with relaxed version constraints")
            return result

        self.error_detector.add_error(strategy_result.get("error", ""))

        # Strategy 3: Try installing without version pins
        strategy_result = self._try_unpinned_versions()
        result["attempts"].append(strategy_result)

        if strategy_result["success"]:
            result["success"] = True
            result["strategy_used"] = "unpinned_versions"
            result["dependencies_installed"] = True
            result["warnings"].append("Installed without version pins - results may differ")
            return result

        self.error_detector.add_error(strategy_result.get("error", ""))

        # All strategies failed
        result["errors"].append(self.error_detector.get_error_summary())
        return result

    def _try_setup_py_install(self) -> Dict:
        """Try installing using setup.py (pip install -e .)."""
        setup_file = self.repo_path / "setup.py"

        if not setup_file.exists():
            return {
                "success": False,
                "strategy": "setup.py",
                "error": "No setup.py found"
            }

        try:
            # Read setup.py to check for Python 2
            setup_content = setup_file.read_text(encoding='utf-8', errors='ignore')

            # Check for Python 2 indicators
            if 'python_requires' in setup_content:
                # Extract python_requires value
                import re
                match = re.search(r'python_requires\s*=\s*["\']([^"\']+)["\']', setup_content)
                if match:
                    python_req = match.group(1)
                    # Check if it requires Python 2
                    if '2.' in python_req and '3.' not in python_req:
                        return {
                            "success": False,
                            "strategy": "setup.py",
                            "error": f"Requires Python 2: {python_req}"
                        }

            return {
                "success": False,  # Will be determined by actual installation
                "strategy": "setup.py",
                "setup_file": str(setup_file),
                "command": "pip install -e .",
                "note": "Installing in editable mode with setup.py"
            }

        except Exception as e:
            return {
                "success": False,
                "strategy": "setup.py",
                "error": str(e)
            }

    def _try_original_requirements(self) -> Dict:
        """Try installing with original requirements."""
        # Find requirements file
        req_file = self._find_requirements_file()

        if not req_file:
            return {
                "success": False,
                "strategy": "original",
                "error": "No requirements file found"
            }

        # Read and check requirements
        try:
            requirements = req_file.read_text(encoding='utf-8', errors='ignore')

            # Check for obvious Python 2 incompatibilities
            if self._has_python2_packages(requirements):
                return {
                    "success": False,
                    "strategy": "original",
                    "error": "Requirements contain Python 2-only packages"
                }

            return {
                "success": False,  # Will be determined by actual installation
                "strategy": "original",
                "requirements_file": str(req_file),
                "command": f"pip install -r {req_file.name}"
            }

        except Exception as e:
            return {
                "success": False,
                "strategy": "original",
                "error": str(e)
            }

    def _try_relaxed_versions(self) -> Dict:
        """Try installing with relaxed version constraints."""
        req_file = self._find_requirements_file()

        if not req_file:
            return {
                "success": False,
                "strategy": "relaxed",
                "error": "No requirements file found"
            }

        try:
            requirements = req_file.read_text(encoding='utf-8', errors='ignore')
            relaxed = self._relax_requirements(requirements)

            # Create temporary relaxed requirements file
            temp_req_file = self.repo_path / "requirements_relaxed.txt"
            temp_req_file.write_text(relaxed, encoding='utf-8')

            return {
                "success": False,  # Will be determined by actual installation
                "strategy": "relaxed",
                "requirements_file": str(temp_req_file),
                "command": f"pip install -r {temp_req_file.name}",
                "modifications": "Relaxed version constraints (e.g., ==1.15.0 -> >=1.15.0,<2.0)"
            }

        except Exception as e:
            return {
                "success": False,
                "strategy": "relaxed",
                "error": str(e)
            }

    def _try_unpinned_versions(self) -> Dict:
        """Try installing packages without version constraints."""
        req_file = self._find_requirements_file()

        if not req_file:
            return {
                "success": False,
                "strategy": "unpinned",
                "error": "No requirements file found"
            }

        try:
            requirements = req_file.read_text(encoding='utf-8', errors='ignore')

            # Extract just package names (remove versions)
            packages = []
            for line in requirements.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name before any operator
                    match = re.match(r'^([a-zA-Z0-9_-]+)', line)
                    if match:
                        pkg = match.group(1)
                        # Skip Python 2 only packages
                        if pkg not in ['futures', 'functools32']:
                            packages.append(pkg)

            if not packages:
                return {
                    "success": False,
                    "strategy": "unpinned",
                    "error": "No valid packages found"
                }

            # Create temporary unpinned requirements
            temp_req_file = self.repo_path / "requirements_unpinned.txt"
            temp_req_file.write_text('\n'.join(packages), encoding='utf-8')

            return {
                "success": False,  # Will be determined by actual installation
                "strategy": "unpinned",
                "requirements_file": str(temp_req_file),
                "command": f"pip install -r {temp_req_file.name}",
                "modifications": "Removed all version constraints - using latest compatible versions",
                "warning": "Results may differ significantly from paper due to version differences"
            }

        except Exception as e:
            return {
                "success": False,
                "strategy": "unpinned",
                "error": str(e)
            }

    def _find_requirements_file(self) -> Optional[Path]:
        """Find requirements file in repository."""
        candidates = [
            "requirements.txt",
            "requirements-dev.txt",
            "requirements-test.txt",
            "requirements/base.txt",
            "requirements/prod.txt"
        ]

        for candidate in candidates:
            req_path = self.repo_path / candidate
            if req_path.exists():
                return req_path

        return None

    def _has_python2_packages(self, requirements: str) -> bool:
        """Check if requirements contain Python 2-only packages."""
        python2_packages = ['futures', 'functools32', 'configparser']

        for pkg in python2_packages:
            if re.search(rf'\b{pkg}\b', requirements, re.IGNORECASE):
                return True

        return False

    def _relax_requirements(self, requirements: str) -> str:
        """Relax version constraints in requirements.

        Examples:
            tensorflow==1.15.0 -> tensorflow>=1.15.0,<2.0
            numpy==1.16.4 -> numpy>=1.16.0,<2.0
            package>=1.0,<=2.0 -> package>=1.0,<3.0
        """
        lines = []

        for line in requirements.strip().split('\n'):
            original_line = line
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                lines.append(original_line)
                continue

            # Handle == constraints
            if '==' in line:
                match = re.match(r'^([a-zA-Z0-9_-]+)==(\d+)\.(\d+)\.(\d+)', line)
                if match:
                    pkg, major, minor, patch = match.groups()
                    # Relax to allow any version in same major version
                    next_major = int(major) + 1
                    relaxed = f"{pkg}>={major}.{minor}.0,<{next_major}.0"
                    lines.append(relaxed)
                    continue

            # Handle >= with specific upper bound
            if '>=' in line and '<' in line:
                # Already has bounds, just keep it
                lines.append(line)
                continue

            # Handle >= without upper bound
            if '>=' in line and '<' not in line:
                match = re.match(r'^([a-zA-Z0-9_-]+)>=(\d+)\.(\d+)', line)
                if match:
                    pkg, major, minor = match.groups()
                    # Add reasonable upper bound
                    next_major = int(major) + 2
                    relaxed = f"{pkg}>={major}.{minor},<{next_major}.0"
                    lines.append(relaxed)
                    continue

            # Default: keep original line
            lines.append(line)

        return '\n'.join(lines)


def detect_installation_error_type(error_output: str) -> Dict:
    """
    Detect the type of installation error and suggest fixes.

    Args:
        error_output: Error output from pip install

    Returns:
        Dict with error_type and suggestions
    """
    result = {
        "error_type": "unknown",
        "suggestions": [],
        "is_fatal": False
    }

    error_lower = error_output.lower()

    # Python version incompatibility
    if re.search(r"requires python|python.*is required|not available for python", error_lower):
        result["error_type"] = "python_version_incompatibility"
        result["is_fatal"] = True
        result["suggestions"] = [
            "Create a virtual environment with a compatible Python version using conda",
            "Check the repository's Python version requirements",
        ]
        return result

    # Package not found
    if "could not find a version" in error_lower or "no matching distribution" in error_lower:
        result["error_type"] = "package_not_found"
        result["suggestions"] = [
            "Package may be deprecated or renamed - check PyPI for alternatives",
            "Try removing version constraints to allow pip to find compatible versions",
        ]
        return result

    # Dependency conflict
    if "conflict" in error_lower or "incompatible" in error_lower:
        result["error_type"] = "dependency_conflict"
        result["suggestions"] = [
            "Try relaxing version constraints in requirements.txt",
            "Install packages one by one to identify conflicting dependencies",
        ]
        return result

    # Build/compilation error
    if "error: command" in error_lower or "failed building wheel" in error_lower:
        result["error_type"] = "build_error"
        result["suggestions"] = [
            "Install required system dependencies (gcc, python-dev, etc.)",
            "Try installing pre-built wheels instead of building from source",
        ]
        return result

    return result
