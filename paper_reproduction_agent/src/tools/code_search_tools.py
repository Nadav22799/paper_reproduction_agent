"""Tools for searching and retrieving code implementations."""

import os
import re
import requests
from typing import Dict, List, Any, Optional
from github import Github, RateLimitExceededException
from langchain.tools import tool
import time


def check_and_display_rate_limit(g: Github) -> bool:
    """
    Check GitHub API rate limit and display current usage.

    Args:
        g: Github instance

    Returns:
        True if we have enough API calls remaining, False otherwise
    """
    try:
        rate_limit = g.get_rate_limit()
        remaining = rate_limit.core.remaining
        total = rate_limit.core.limit
        reset_time = rate_limit.core.reset.timestamp() - time.time()

        print(f"📊 GitHub API: {remaining}/{total} calls remaining (resets in {int(reset_time/60)}m)")

        return remaining > 10
    except Exception:
        return True  # If we can't check, assume it's ok


@tool
def search_github_repos(query: str, language: Optional[str] = None, max_results: int = 5, fetch_topics: bool = False) -> List[Dict[str, Any]]:
    """
    Search GitHub for repositories matching the query.

    Args:
        query: Search query (e.g., paper title, algorithm name)
        language: Programming language filter (e.g., "Python", "PyTorch")
        max_results: Maximum number of results to return (default: 5)
        fetch_topics: Whether to fetch repository topics (requires extra API call per repo, default: False)

    Returns:
        List of repository information dictionaries
    """
    try:
        token = os.getenv("GITHUB_TOKEN")
        g = Github(token) if token else Github()

        # Check and display rate limit
        if not check_and_display_rate_limit(g):
            rate_limit = g.get_rate_limit()
            reset_time = rate_limit.core.reset.timestamp() - time.time()
            if reset_time > 300:  # More than 5 minutes wait
                return [{"error": f"GitHub rate limit exceeded. Resets in {int(reset_time/60)} minutes. Skipping GitHub search."}]
            print(f"⚠️  Low on GitHub API calls. Proceeding carefully...")

        # Build search query
        search_query = query
        if language:
            search_query += f" language:{language}"

        repos = g.search_repositories(query=search_query, sort="stars", order="desc")

        results = []
        for i, repo in enumerate(repos[:max_results]):
            try:
                # Get topics only if requested (each topic fetch is an extra API call)
                topics = []
                if fetch_topics:
                    try:
                        topics = repo.get_topics()
                    except (RateLimitExceededException, Exception) as e:
                        print(f"⚠️  Skipping topics for {repo.full_name}: {str(e)[:50]}")

                results.append({
                    "name": repo.name,
                    "full_name": repo.full_name,
                    "description": repo.description or "",
                    "stars": repo.stargazers_count,
                    "url": repo.html_url,
                    "clone_url": repo.clone_url,
                    "language": repo.language or "Unknown",
                    "topics": topics if fetch_topics else [],
                    "has_readme": True,  # Most repos have README
                })
            except RateLimitExceededException:
                print(f"⚠️  Rate limit hit after {len(results)} repos. Returning what we have.")
                break
            except Exception as e:
                print(f"⚠️  Error processing repo {i}: {str(e)[:50]}")
                continue

        return results if results else [{"message": "No repositories found"}]

    except RateLimitExceededException as e:
        return [{"error": "GitHub rate limit exceeded. Please add GITHUB_TOKEN to .env or wait before retrying."}]
    except Exception as e:
        return [{"error": f"GitHub search failed: {str(e)}"}]


@tool
def get_repo_contents(repo_full_name: str, path: str = "") -> List[Dict[str, Any]]:
    """
    Get contents of a GitHub repository.

    Args:
        repo_full_name: Repository full name (owner/repo)
        path: Path within repository (default: root)

    Returns:
        List of file/directory information
    """
    try:
        token = os.getenv("GITHUB_TOKEN")
        g = Github(token) if token else Github()

        # Check rate limit
        rate_limit = g.get_rate_limit()
        if rate_limit.core.remaining < 5:
            return [{"error": "GitHub rate limit too low. Skipping repo contents fetch."}]

        repo = g.get_repo(repo_full_name)
        contents = repo.get_contents(path)

        if not isinstance(contents, list):
            contents = [contents]

        results = []
        for content in contents:
            results.append({
                "name": content.name,
                "path": content.path,
                "type": content.type,
                "size": content.size if hasattr(content, 'size') else 0,
                "download_url": content.download_url if content.type == "file" else None,
            })

        return results

    except RateLimitExceededException:
        return [{"error": "GitHub rate limit exceeded. Skipping repo contents."}]
    except Exception as e:
        return [{"error": f"Failed to get repo contents: {str(e)}"}]


@tool
def get_file_content(repo_full_name: str, file_path: str) -> str:
    """
    Get content of a specific file from GitHub repository.

    Args:
        repo_full_name: Repository full name (owner/repo)
        file_path: Path to file in repository

    Returns:
        File content as string
    """
    try:
        token = os.getenv("GITHUB_TOKEN")
        g = Github(token) if token else Github()

        # Check rate limit
        rate_limit = g.get_rate_limit()
        if rate_limit.core.remaining < 5:
            return "Error: GitHub rate limit too low. Skipping file fetch."

        repo = g.get_repo(repo_full_name)
        file_content = repo.get_contents(file_path)

        if hasattr(file_content, 'decoded_content'):
            return file_content.decoded_content.decode('utf-8')
        else:
            return "File is too large or binary"

    except RateLimitExceededException:
        return "Error: GitHub rate limit exceeded. Skipping file fetch."
    except Exception as e:
        return f"Error fetching file: {str(e)}"


@tool
def search_github_code(query: str, language: Optional[str] = None, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search for code snippets on GitHub.

    Args:
        query: Code search query
        language: Programming language filter
        max_results: Maximum number of results

    Returns:
        List of code search results
    """
    try:
        token = os.getenv("GITHUB_TOKEN")
        g = Github(token) if token else Github()

        # Check rate limit
        rate_limit = g.get_rate_limit()
        if rate_limit.core.remaining < 10:
            reset_time = rate_limit.core.reset.timestamp() - time.time()
            if reset_time > 300:
                return [{"error": f"GitHub rate limit exceeded. Resets in {int(reset_time/60)} minutes. Skipping code search."}]

        search_query = query
        if language:
            search_query += f" language:{language}"

        code_results = g.search_code(query=search_query)

        results = []
        for i, code in enumerate(code_results[:max_results]):
            try:
                results.append({
                    "name": code.name,
                    "path": code.path,
                    "repository": code.repository.full_name,
                    "url": code.html_url,
                    "repo_url": code.repository.html_url,
                })
            except RateLimitExceededException:
                print(f"⚠️  Rate limit hit after {len(results)} code results. Returning what we have.")
                break
            except Exception as e:
                print(f"⚠️  Error processing code result {i}: {str(e)[:50]}")
                continue

        return results if results else [{"message": "No code found"}]

    except RateLimitExceededException:
        return [{"error": "GitHub rate limit exceeded. Skipping code search."}]
    except Exception as e:
        return [{"error": f"Code search failed: {str(e)}"}]


@tool
def search_papers_with_code(paper_title: str) -> Dict[str, Any]:
    """
    Search Papers with Code for paper implementations.

    Args:
        paper_title: Title of the paper

    Returns:
        Dictionary with paper and implementation information
    """
    try:
        # Papers with Code API
        base_url = "https://paperswithcode.com/api/v1/papers/"

        # Search by title
        search_url = f"{base_url}?title={requests.utils.quote(paper_title)}"
        response = requests.get(search_url, timeout=10)

        if response.status_code == 200:
            data = response.json()

            if data.get("results"):
                paper = data["results"][0]
                paper_id = paper.get("id")

                # Get implementations
                impl_url = f"https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/"
                impl_response = requests.get(impl_url, timeout=10)

                implementations = []
                if impl_response.status_code == 200:
                    impl_data = impl_response.json()
                    implementations = impl_data.get("results", [])

                return {
                    "paper_id": paper_id,
                    "title": paper.get("title"),
                    "url": paper.get("url_abs"),
                    "arxiv_id": paper.get("arxiv_id"),
                    "implementations": [
                        {
                            "url": impl.get("url"),
                            "framework": impl.get("framework"),
                            "stars": impl.get("stars"),
                            "is_official": impl.get("is_official"),
                        }
                        for impl in implementations
                    ],
                }
            else:
                return {"message": "Paper not found on Papers with Code"}
        else:
            return {"error": f"API request failed with status {response.status_code}"}

    except Exception as e:
        return {"error": f"Papers with Code search failed: {str(e)}"}


@tool
def clone_repository(repo_url: str, target_dir: str) -> str:
    """
    Clone a Git repository to local directory.

    Args:
        repo_url: Repository URL to clone
        target_dir: Target directory for cloning

    Returns:
        Status message
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "clone", repo_url, target_dir],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            return f"Successfully cloned repository to {target_dir}"
        else:
            return f"Clone failed: {result.stderr}"

    except Exception as e:
        return f"Error cloning repository: {str(e)}"


# Tool list for easy import
code_search_tools = [
    search_github_repos,
    get_repo_contents,
    get_file_content,
    search_github_code,
    search_papers_with_code,
    clone_repository,
]
