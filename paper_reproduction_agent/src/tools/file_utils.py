"""Generic file utility tools - Most commonly used tools for file operations.

Only the 2 most essential tools that are hard to replicate with bash commands.
For other operations, use execute_shell_command() with bash commands.
"""

import re
from typing import Dict, Any, Optional
from pathlib import Path
from langchain_core.tools import tool


@tool
def grep_in_directory(pattern: str, directory: str = ".", file_pattern: str = "*", recursive: bool = True, max_results: int = 50) -> Dict[str, Any]:
    """
    Search for pattern across multiple files in a directory (like grep -r).

    Args:
        pattern: Regular expression pattern to search for
        directory: Directory to search in
        file_pattern: File pattern to match (e.g., "*.py", "*.txt")
        recursive: Search subdirectories (default: True)
        max_results: Maximum number of matches to return

    Returns:
        Dictionary with matches grouped by file

    Example:
        grep_in_directory("import torch", ".", "*.py")
        grep_in_directory("TODO", "./src", recursive=True)
    """
    try:
        directory = Path(directory).resolve()

        if recursive:
            files = list(directory.rglob(file_pattern))
        else:
            files = list(directory.glob(file_pattern))

        all_matches = {}
        total_matches = 0

        for file_path in files:
            if not file_path.is_file():
                continue

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()

                file_matches = []
                for i, line in enumerate(lines):
                    if re.search(pattern, line, re.IGNORECASE):
                        file_matches.append({
                            "line_number": i + 1,
                            "line": line.rstrip()
                        })
                        total_matches += 1

                        if total_matches >= max_results:
                            break

                if file_matches:
                    all_matches[str(file_path.relative_to(directory))] = file_matches

                if total_matches >= max_results:
                    break

            except Exception:
                continue

        return {
            "success": True,
            "directory": str(directory),
            "pattern": pattern,
            "files_with_matches": len(all_matches),
            "total_matches": total_matches,
            "matches": all_matches,
            "truncated": total_matches >= max_results,
            "summary": f"Found {total_matches} match(es) across {len(all_matches)} file(s)"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def find_files(directory: str = ".", pattern: str = "*", file_type: Optional[str] = None, max_depth: Optional[int] = None, max_results: int = 100) -> Dict[str, Any]:
    """
    Find files matching a pattern (like find command).

    Args:
        directory: Directory to search in
        pattern: File name pattern (supports wildcards: *, ?, [])
        file_type: Filter by type: "file", "dir", or None for both
        max_depth: Maximum directory depth to search
        max_results: Maximum number of results

    Returns:
        Dictionary with found files/directories

    Example:
        find_files(".", "*.yaml")
        find_files("./src", "test_*.py", file_type="file")
        find_files(".", "*requirements*.txt")
    """
    try:
        directory = Path(directory).resolve()

        if max_depth is None:
            paths = list(directory.rglob(pattern))
        else:
            # Implement depth limit
            paths = []
            for depth in range(max_depth + 1):
                search_pattern = "/".join(["*"] * depth) + "/" + pattern if depth > 0 else pattern
                paths.extend(directory.glob(search_pattern))

        # Filter by type
        if file_type == "file":
            paths = [p for p in paths if p.is_file()]
        elif file_type == "dir":
            paths = [p for p in paths if p.is_dir()]

        # Limit results
        paths = paths[:max_results]

        results = []
        for path in paths:
            results.append({
                "path": str(path.relative_to(directory)),
                "absolute_path": str(path),
                "type": "file" if path.is_file() else "dir",
                "size": path.stat().st_size if path.is_file() else None
            })

        return {
            "success": True,
            "directory": str(directory),
            "pattern": pattern,
            "found": results,
            "count": len(results),
            "truncated": len(paths) >= max_results,
            "summary": f"Found {len(results)} item(s) matching '{pattern}'"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


