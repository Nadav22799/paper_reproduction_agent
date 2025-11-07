"""Python Version Compatibility Checker.

Detects Python version compatibility issues early to avoid dependency hell loops.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class PythonCompatibilityChecker:
    """Check if a repository is compatible with current Python version."""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.current_python = f"{sys.version_info.major}.{sys.version_info.minor}"

    def check_compatibility(self) -> Dict:
        """
        Check Python version compatibility for the repository.

        Returns:
            Dict with:
                - compatible (bool): Whether repo is compatible with current Python
                - current_version (str): Current Python version
                - required_version (str): Required Python version if found
                - confidence (str): Confidence level (high, medium, low)
                - warnings (List[str]): List of compatibility warnings
                - suggestions (List[str]): Suggestions for resolving issues
        """
        result = {
            "compatible": True,
            "current_version": self.current_python,
            "required_version": None,
            "confidence": "low",
            "warnings": [],
            "suggestions": []
        }

        # Check setup.py for python_requires
        setup_py_version = self._check_setup_py()
        if setup_py_version:
            result["required_version"] = setup_py_version
            result["confidence"] = "high"
            if not self._is_version_compatible(setup_py_version):
                result["compatible"] = False
                result["warnings"].append(
                    f"setup.py requires Python {setup_py_version}, but you have {self.current_python}"
                )

        # Check pyproject.toml
        pyproject_version = self._check_pyproject_toml()
        if pyproject_version and not result["required_version"]:
            result["required_version"] = pyproject_version
            result["confidence"] = "high"
            if not self._is_version_compatible(pyproject_version):
                result["compatible"] = False
                result["warnings"].append(
                    f"pyproject.toml requires Python {pyproject_version}, but you have {self.current_python}"
                )

        # Check for old package versions in requirements
        old_packages = self._check_old_packages()
        if old_packages:
            result["warnings"].extend(old_packages)
            # If we found Python 2 packages, it's definitely incompatible with Python 3.10+
            if any("python 2" in w.lower() for w in old_packages):
                result["compatible"] = False
                result["confidence"] = "high"

        # Check README for Python version mentions
        readme_info = self._check_readme()
        if readme_info and not result["required_version"]:
            result["required_version"] = readme_info
            result["confidence"] = "medium"
            if not self._is_version_compatible(readme_info):
                result["compatible"] = False
                result["warnings"].append(
                    f"README mentions Python {readme_info}, but you have {self.current_python}"
                )

        # Check repository age (old repos often have compatibility issues)
        repo_age = self._estimate_repo_age()
        if repo_age and repo_age > 5:
            result["warnings"].append(
                f"Repository appears to be {repo_age}+ years old - may have compatibility issues"
            )
            if float(self.current_python.split('.')[1]) >= 10:  # Python 3.10+
                result["confidence"] = "medium"

        # Generate suggestions if incompatible
        if not result["compatible"]:
            result["suggestions"] = self._generate_suggestions(result["required_version"])

        return result

    def _check_setup_py(self) -> Optional[str]:
        """Check setup.py for python_requires field."""
        setup_py = self.repo_path / "setup.py"
        if not setup_py.exists():
            return None

        try:
            content = setup_py.read_text(encoding='utf-8', errors='ignore')

            # Look for python_requires
            match = re.search(r'python_requires\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                return match.group(1)

            # Look for classifiers
            classifiers = re.findall(r'Programming Language :: Python :: ([\d.]+)', content)
            if classifiers:
                return self._parse_version_from_classifiers(classifiers)
        except Exception:
            pass

        return None

    def _check_pyproject_toml(self) -> Optional[str]:
        """Check pyproject.toml for Python version requirements."""
        pyproject = self.repo_path / "pyproject.toml"
        if not pyproject.exists():
            return None

        try:
            content = pyproject.read_text(encoding='utf-8', errors='ignore')

            # Look for requires-python in [project]
            match = re.search(r'requires-python\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                return match.group(1)

            # Look for python in [tool.poetry.dependencies]
            match = re.search(r'\[tool\.poetry\.dependencies\].*?python\s*=\s*["\']([^"\']+)["\']',
                            content, re.DOTALL)
            if match:
                return match.group(1)
        except Exception:
            pass

        return None

    def _check_old_packages(self) -> List[str]:
        """Check for old package versions that indicate old Python."""
        warnings = []

        # Check requirements.txt
        req_files = ["requirements.txt", "requirements-dev.txt", "requirements-test.txt"]
        for req_file in req_files:
            req_path = self.repo_path / req_file
            if req_path.exists():
                try:
                    content = req_path.read_text(encoding='utf-8', errors='ignore')

                    # Check for Python 2 only packages
                    if re.search(r'(futures|functools32)', content, re.IGNORECASE):
                        warnings.append(
                            f"{req_file} contains Python 2-only packages"
                        )

                    # Check for very old TensorFlow versions
                    tf_match = re.search(r'tensorflow[^=]*==?\s*([0-9.]+)', content)
                    if tf_match:
                        tf_version = tf_match.group(1)
                        if tf_version.startswith('1.') or tf_version.startswith('0.'):
                            warnings.append(
                                f"Old TensorFlow version {tf_version} - requires Python 3.6-3.7"
                            )

                    # Check for very old PyTorch versions
                    torch_match = re.search(r'torch[^=]*==?\s*([0-9.]+)', content)
                    if torch_match:
                        torch_version = torch_match.group(1)
                        if torch_version.startswith('0.'):
                            warnings.append(
                                f"Old PyTorch version {torch_version} - may have compatibility issues"
                            )
                except Exception:
                    pass

        return warnings

    def _check_readme(self) -> Optional[str]:
        """Check README for Python version mentions."""
        readme_files = ["README.md", "README.rst", "README.txt", "README"]
        for readme_file in readme_files:
            readme_path = self.repo_path / readme_file
            if readme_path.exists():
                try:
                    content = readme_path.read_text(encoding='utf-8', errors='ignore')

                    # Look for Python version requirements
                    # Match patterns like "Python 3.6+", "Python 2.7-3.7", "requires Python 3.6"
                    patterns = [
                        r'(?:requires?|needs?|supports?)\s+Python\s+([\d.]+(?:\s*-\s*[\d.]+)?)',
                        r'Python\s+([\d.]+)\+',
                        r'Python\s+([\d.]+)\s*-\s*([\d.]+)',
                    ]

                    for pattern in patterns:
                        match = re.search(pattern, content, re.IGNORECASE)
                        if match:
                            return match.group(1)
                except Exception:
                    pass

        return None

    def _estimate_repo_age(self) -> Optional[int]:
        """Estimate repository age from file modification times."""
        try:
            # Check oldest Python file
            python_files = list(self.repo_path.rglob("*.py"))[:50]  # Sample first 50
            if not python_files:
                return None

            oldest_mtime = min(f.stat().st_mtime for f in python_files if f.exists())
            import time
            age_seconds = time.time() - oldest_mtime
            age_years = age_seconds / (365.25 * 24 * 3600)

            return int(age_years)
        except Exception:
            return None

    def _is_version_compatible(self, required: str) -> bool:
        """Check if current Python version satisfies requirement."""
        try:
            # Parse requirement string (e.g., ">=3.6", "3.6-3.8", ">=3.6,<3.10")
            current_major, current_minor = map(int, self.current_python.split('.'))

            # Handle ranges like "3.6-3.8"
            if '-' in required and not required.startswith('>='):
                parts = required.split('-')
                min_ver = parts[0].strip()
                max_ver = parts[1].strip() if len(parts) > 1 else None

                min_major, min_minor = self._parse_version(min_ver)
                if max_ver:
                    max_major, max_minor = self._parse_version(max_ver)
                    return (min_major, min_minor) <= (current_major, current_minor) <= (max_major, max_minor)
                else:
                    return (current_major, current_minor) >= (min_major, min_minor)

            # Handle operators like ">=3.6", ">=3.6,<3.10"
            if '>=' in required:
                version_str = required.split('>=')[1].split(',')[0].strip()
                min_major, min_minor = self._parse_version(version_str)

                # Check if there's an upper bound
                if '<' in required:
                    upper_bound = required.split('<')[1].strip()
                    max_major, max_minor = self._parse_version(upper_bound)
                    return (min_major, min_minor) <= (current_major, current_minor) < (max_major, max_minor)

                return (current_major, current_minor) >= (min_major, min_minor)

            # Handle exact version or version with +
            if '+' in required:
                version_str = required.replace('+', '').strip()
                min_major, min_minor = self._parse_version(version_str)
                return (current_major, current_minor) >= (min_major, min_minor)

            # Exact version match
            req_major, req_minor = self._parse_version(required)
            return (current_major, current_minor) == (req_major, req_minor)

        except Exception:
            # If parsing fails, assume compatible
            return True

    def _parse_version(self, version_str: str) -> Tuple[int, int]:
        """Parse version string to (major, minor) tuple."""
        version_str = re.sub(r'[^\d.]', '', version_str)  # Remove non-numeric
        parts = version_str.split('.')
        major = int(parts[0]) if len(parts) > 0 else 3
        minor = int(parts[1]) if len(parts) > 1 else 0
        return major, minor

    def _parse_version_from_classifiers(self, classifiers: List[str]) -> str:
        """Parse version range from classifier list."""
        versions = []
        for c in classifiers:
            try:
                parts = c.split('.')
                if len(parts) >= 2:
                    versions.append((int(parts[0]), int(parts[1])))
            except ValueError:
                pass

        if not versions:
            return None

        min_ver = min(versions)
        max_ver = max(versions)

        if min_ver == max_ver:
            return f"{min_ver[0]}.{min_ver[1]}"
        else:
            return f"{min_ver[0]}.{min_ver[1]}-{max_ver[0]}.{max_ver[1]}"

    def _generate_suggestions(self, required_version: Optional[str]) -> List[str]:
        """Generate suggestions for resolving compatibility issues."""
        suggestions = []

        if required_version:
            # Parse the required version to suggest specific Python version
            try:
                if '-' in required_version:
                    # Range like "3.6-3.8"
                    max_ver = required_version.split('-')[1].strip()
                    suggestions.append(
                        f"Create a virtual environment with Python {max_ver}: "
                        f"conda create -n old_env python={max_ver}"
                    )
                elif '>=' in required_version:
                    # Minimum version like ">=3.6"
                    min_ver = re.sub(r'[^\d.]', '', required_version.split('>=')[1].split(',')[0])
                    if '<' in required_version:
                        # Has upper bound
                        max_ver = re.sub(r'[^\d.]', '', required_version.split('<')[1])
                        major, minor = self._parse_version(max_ver)
                        # Suggest one version below the upper bound
                        suggested = f"{major}.{max(0, minor-1)}"
                        suggestions.append(
                            f"Create a virtual environment with Python {suggested}: "
                            f"conda create -n old_env python={suggested}"
                        )
                    else:
                        suggestions.append(
                            f"Try upgrading package versions to be compatible with Python {self.current_python}"
                        )
                else:
                    # Exact version or version with +
                    version = re.sub(r'[^\d.]', '', required_version)
                    suggestions.append(
                        f"Create a virtual environment with Python {version}: "
                        f"conda create -n old_env python={version}"
                    )
            except Exception:
                pass

        # Generic suggestions
        suggestions.append(
            "Try relaxing version constraints in requirements.txt (e.g., change tensorflow==1.15.0 to tensorflow>=1.15.0,<2.0)"
        )
        suggestions.append(
            "Consider using Docker with an older base image that matches the required Python version"
        )

        return suggestions


def check_python_compatibility(repo_path: str) -> Dict:
    """
    Convenience function to check Python compatibility.

    Args:
        repo_path: Path to repository

    Returns:
        Compatibility check results
    """
    checker = PythonCompatibilityChecker(repo_path)
    return checker.check_compatibility()
