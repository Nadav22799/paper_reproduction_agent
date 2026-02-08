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

        print(
            f"📊 GitHub API: {remaining}/{total} calls remaining (resets in {int(reset_time/60)}m)"
        )

        return remaining > 10
    except Exception:
        return True  # If we can't check, assume it's ok


@tool
def search_github_repos(
    query: str,
    language: Optional[str] = None,
    max_results: int = 5,
    fetch_topics: bool = False,
) -> List[Dict[str, Any]]:
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
                return [
                    {
                        "error": f"GitHub rate limit exceeded. Resets in {int(reset_time/60)} minutes. Skipping GitHub search."
                    }
                ]
            print("⚠️  Low on GitHub API calls. Proceeding carefully...")

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

                results.append(
                    {
                        "name": repo.name,
                        "full_name": repo.full_name,
                        "description": repo.description or "",
                        "stars": repo.stargazers_count,
                        "url": repo.html_url,
                        "clone_url": repo.clone_url,
                        "language": repo.language or "Unknown",
                        "topics": topics if fetch_topics else [],
                        "has_readme": True,  # Most repos have README
                    }
                )
            except RateLimitExceededException:
                print(
                    f"⚠️  Rate limit hit after {len(results)} repos. Returning what we have."
                )
                break
            except Exception as e:
                print(f"⚠️  Error processing repo {i}: {str(e)[:50]}")
                continue

        return results if results else [{"message": "No repositories found"}]

    except RateLimitExceededException:
        return [
            {
                "error": "GitHub rate limit exceeded. Please add GITHUB_TOKEN to .env or wait before retrying."
            }
        ]
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
            return [
                {"error": "GitHub rate limit too low. Skipping repo contents fetch."}
            ]

        repo = g.get_repo(repo_full_name)
        contents = repo.get_contents(path)

        if not isinstance(contents, list):
            contents = [contents]

        results = []
        for content in contents:
            results.append(
                {
                    "name": content.name,
                    "path": content.path,
                    "type": content.type,
                    "size": content.size if hasattr(content, "size") else 0,
                    "download_url": (
                        content.download_url if content.type == "file" else None
                    ),
                }
            )

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

        if hasattr(file_content, "decoded_content"):
            return file_content.decoded_content.decode("utf-8")
        else:
            return "File is too large or binary"

    except RateLimitExceededException:
        return "Error: GitHub rate limit exceeded. Skipping file fetch."
    except Exception as e:
        return f"Error fetching file: {str(e)}"


@tool
def search_github_code(
    query: str, language: Optional[str] = None, max_results: int = 5
) -> List[Dict[str, Any]]:
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
                return [
                    {
                        "error": f"GitHub rate limit exceeded. Resets in {int(reset_time/60)} minutes. Skipping code search."
                    }
                ]

        search_query = query
        if language:
            search_query += f" language:{language}"

        code_results = g.search_code(query=search_query)

        results = []
        for i, code in enumerate(code_results[:max_results]):
            try:
                results.append(
                    {
                        "name": code.name,
                        "path": code.path,
                        "repository": code.repository.full_name,
                        "url": code.html_url,
                        "repo_url": code.repository.html_url,
                    }
                )
            except RateLimitExceededException:
                print(
                    f"⚠️  Rate limit hit after {len(results)} code results. Returning what we have."
                )
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
                impl_url = (
                    f"https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/"
                )
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
            timeout=300,
        )

        if result.returncode == 0:
            return f"Successfully cloned repository to {target_dir}"
        else:
            return f"Clone failed: {result.stderr}"

    except Exception as e:
        return f"Error cloning repository: {str(e)}"


def search_github_for_arxiv_reference(
    arxiv_id: str, max_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Search GitHub for repositories that reference a specific arXiv paper.

    This is useful for finding implementations when the paper doesn't include
    a direct GitHub link, but the repo's README references the paper.

    Args:
        arxiv_id: The arXiv ID (e.g., "2301.12345" or "2301.12345v1")
        max_results: Maximum number of results to return

    Returns:
        List of repository information dictionaries with confidence scores
    """
    # Clean the arXiv ID (remove version suffix if present)
    clean_id = re.sub(r"v\d+$", "", arxiv_id)

    # Search patterns to try (in order of specificity)
    search_patterns = [
        f'"arxiv:{clean_id}"',  # Exact arXiv reference
        f'"arxiv.org/abs/{clean_id}"',  # Full URL reference
        f'"arxiv.org/pdf/{clean_id}"',  # PDF URL reference
    ]

    # Patterns to EXCLUDE (paper collections, awesome lists, not implementations)
    exclude_patterns = [
        "awesome",
        "paper",
        "list",
        "survey",
        "collection",
        "reading",
        "starred",
        "backup",
        "fork",
        "mirror",
        "copy",
        "clone",
    ]

    try:
        token = os.getenv("GITHUB_TOKEN")
        g = Github(token) if token else Github()

        # Check rate limit
        if not check_and_display_rate_limit(g):
            rate_limit = g.get_rate_limit()
            reset_time = rate_limit.core.reset.timestamp() - time.time()
            if reset_time > 300:
                return [
                    {
                        "error": f"GitHub rate limit exceeded. Resets in {int(reset_time/60)} minutes."
                    }
                ]

        all_repos = {}  # Use dict to deduplicate by repo full_name

        for pattern in search_patterns:
            try:
                # Search in README files specifically
                query = f"{pattern} filename:readme"
                code_results = g.search_code(query=query)

                # Check if we have results before iterating
                try:
                    total_count = code_results.totalCount
                    if total_count == 0:
                        continue  # No results for this pattern, try next
                except Exception:
                    pass  # If we can't check count, try iterating anyway

                for code in code_results[: max_results * 2]:  # Get more to filter
                    try:
                        repo = code.repository
                        repo_key = repo.full_name
                        repo_name_lower = repo.name.lower()
                        repo_desc_lower = (repo.description or "").lower()

                        # Skip if already processed
                        if repo_key in all_repos:
                            continue

                        # Check if this looks like a paper collection (not implementation)
                        is_paper_collection = any(
                            excl in repo_name_lower or excl in repo_desc_lower
                            for excl in exclude_patterns
                        )

                        # Check if README is in a subdirectory (likely a starred/cloned collection)
                        is_nested_readme = code.path.count("/") > 1

                        # Determine confidence level
                        if is_paper_collection:
                            confidence = "low"  # Paper collection, not implementation
                        elif is_nested_readme:
                            confidence = (
                                "low"  # Nested README, likely not the main repo
                            )
                        else:
                            confidence = "high"  # Likely the actual implementation

                        all_repos[repo_key] = {
                            "name": repo.name,
                            "full_name": repo.full_name,
                            "description": repo.description or "",
                            "stars": repo.stargazers_count,
                            "url": repo.html_url,
                            "clone_url": repo.clone_url,
                            "language": repo.language or "Unknown",
                            "confidence": confidence,
                            "match_pattern": pattern,
                            "match_file": code.path,
                            "is_paper_collection": is_paper_collection,
                        }
                    except RateLimitExceededException:
                        print("⚠️  Rate limit hit during arXiv search")
                        break
                    except Exception:
                        continue

            except RateLimitExceededException:
                print(f"⚠️  Rate limit hit for pattern: {pattern}")
                break
            except Exception as e:
                # Don't show error for empty results or index errors (common when no matches)
                error_msg = str(e)
                if "list index out of range" in error_msg.lower():
                    # This usually means no results for this pattern, which is fine
                    pass
                else:
                    print(f"⚠️  Search failed for pattern {pattern}: {error_msg[:50]}")
                continue

        results = list(all_repos.values())

        # Sort: high confidence first, then by stars
        results.sort(
            key=lambda x: (
                0 if x.get("confidence") == "high" else 1,
                -x.get("stars", 0),
            )
        )

        # Filter to return high confidence first, but include others if none found
        high_confidence = [r for r in results if r.get("confidence") == "high"]

        if high_confidence:
            print(
                f"🔍 Found {len(high_confidence)} high-confidence repo(s) referencing arXiv:{clean_id}"
            )
            return high_confidence[:max_results]
        elif results:
            print(
                f"🔍 Found {len(results)} repo(s) referencing arXiv:{clean_id} (mostly paper collections)"
            )
            return results[:max_results]
        else:
            return [{"message": f"No repos found referencing arXiv:{clean_id}"}]

    except RateLimitExceededException:
        return [
            {
                "error": "GitHub rate limit exceeded. Please add GITHUB_TOKEN to .env or wait."
            }
        ]
    except Exception as e:
        return [{"error": f"GitHub arXiv search failed: {str(e)}"}]


def search_github_by_paper_name(
    paper_title: str, max_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Search GitHub for repositories by extracting key terms from paper title.

    This searches for repos whose name matches the paper's method/acronym.

    Args:
        paper_title: Title of the paper
        max_results: Maximum number of results to return

    Returns:
        List of repository information dictionaries
    """
    # Extract potential repo names from paper title
    # Look for: acronyms (all caps), method names, key terms

    # Common patterns: "METHOD: Description" or "Description (METHOD)"
    import re

    search_terms = []

    # Extract acronyms (2+ capital letters)
    acronyms = re.findall(r"\b([A-Z][A-Z0-9]{1,})\b", paper_title)
    search_terms.extend(acronyms)

    # Extract terms from title before colon (often the method name)
    if ":" in paper_title:
        before_colon = paper_title.split(":")[0].strip()
        # Remove common words
        stop_words = {
            "a",
            "an",
            "the",
            "for",
            "of",
            "in",
            "on",
            "with",
            "via",
            "using",
            "towards",
        }
        words = [
            w
            for w in before_colon.split()
            if w.lower() not in stop_words and len(w) > 2
        ]
        if len(words) <= 3:  # Short enough to be a method name
            search_terms.append(before_colon)

    # Extract terms in parentheses (often acronyms)
    parens = re.findall(r"\(([^)]+)\)", paper_title)
    search_terms.extend(parens)

    # Remove duplicates and filter
    search_terms = list(set(t for t in search_terms if len(t) >= 2))

    if not search_terms:
        return [{"message": "Could not extract search terms from paper title"}]

    try:
        token = os.getenv("GITHUB_TOKEN")
        g = Github(token) if token else Github()

        if not check_and_display_rate_limit(g):
            rate_limit = g.get_rate_limit()
            reset_time = rate_limit.core.reset.timestamp() - time.time()
            if reset_time > 300:
                return [
                    {
                        "error": f"GitHub rate limit exceeded. Resets in {int(reset_time/60)} minutes."
                    }
                ]

        all_repos = {}

        for term in search_terms[:3]:  # Limit to top 3 terms
            try:
                # Search for repos with this name
                repos = g.search_repositories(
                    query=f"{term} in:name", sort="stars", order="desc"
                )

                for repo in repos[:max_results]:
                    try:
                        repo_key = repo.full_name

                        if repo_key in all_repos:
                            continue

                        # Check if repo name closely matches search term
                        name_lower = repo.name.lower()
                        term_lower = term.lower()
                        is_exact_match = (
                            name_lower == term_lower or term_lower in name_lower
                        )

                        all_repos[repo_key] = {
                            "name": repo.name,
                            "full_name": repo.full_name,
                            "description": repo.description or "",
                            "stars": repo.stargazers_count,
                            "url": repo.html_url,
                            "clone_url": repo.clone_url,
                            "language": repo.language or "Unknown",
                            "confidence": "medium",  # Name match, needs verification
                            "matched_term": term,
                            "is_exact_match": is_exact_match,
                        }
                    except RateLimitExceededException:
                        break
                    except Exception:
                        continue

            except RateLimitExceededException:
                print(f"⚠️  Rate limit hit for term: {term}")
                break
            except Exception as e:
                print(f"⚠️  Search failed for term {term}: {str(e)[:50]}")
                continue

        results = list(all_repos.values())

        # Sort by exact match first, then stars
        results.sort(
            key=lambda x: (0 if x.get("is_exact_match") else 1, -x.get("stars", 0))
        )

        if results:
            print(
                f"🔍 Found {len(results)} repo(s) matching paper name terms: {search_terms[:3]}"
            )

        return (
            results[:max_results]
            if results
            else [{"message": f"No repos found for terms: {search_terms}"}]
        )

    except RateLimitExceededException:
        return [{"error": "GitHub rate limit exceeded."}]
    except Exception as e:
        return [{"error": f"GitHub name search failed: {str(e)}"}]


def web_search_for_implementation(
    paper_title: str, arxiv_id: str = None, authors: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Use Gemini with Google Search to find GitHub implementations of a paper.

    This performs a web search and uses LLM evaluation to identify which results
    are likely official/authoritative implementations.

    Args:
        paper_title: Title of the paper
        arxiv_id: Optional arXiv ID for more specific search
        authors: Optional list of author names for validation

    Returns:
        List of candidate repositories with confidence levels
    """
    import json

    try:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return [
                {"error": "google-genai not installed. Run: pip install google-genai"}
            ]

    except ImportError:
        return [{"error": "google-genai not found. Please install `google-genai`."}]

    result = {
        "candidates": [],
        "raw_response": "",
    }

    try:
        # Configure Gemini
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return [{"error": "GEMINI_API_KEY not found in environment"}]

        client = genai.Client(api_key=api_key)

        # Build search query
        search_terms = [f'"{paper_title}"', "github", "implementation", "code"]
        if arxiv_id:
            search_terms.append(f"arxiv:{arxiv_id}")

        search_query = " ".join(search_terms)

        # Build author context for LLM
        author_context = ""
        if authors:
            author_context = f"\nPaper Authors: {', '.join(authors[:5])}"

        prompt = f"""Search for the official GitHub implementation of this research paper and evaluate the results.

Paper Title: {paper_title}
{"arXiv ID: " + arxiv_id if arxiv_id else ""}
{author_context}

Search Query: {search_query}

Instructions:
1. Search for GitHub repositories that implement this paper
2. For each potential repository found, evaluate:
   - Does the repo name/description match the paper's method or acronym?
   - Is it likely from the paper's authors (check GitHub usernames against author names)?
   - Does the repo explicitly reference this paper (in README, citation, etc.)?
   - Is this a paper-specific implementation (NOT a general library like huggingface/transformers)?

3. Return ONLY high-confidence candidates that are likely OFFICIAL implementations.
   Exclude: tutorials, course projects, third-party reimplementations, general ML libraries.

Return your response as a JSON object with this structure:
{{
    "candidates": [
        {{
            "url": "https://github.com/owner/repo",
            "name": "repo-name",
            "confidence": "high",
            "reason": "Brief explanation of why this is likely official"
        }}
    ],
    "search_performed": true
}}

If no high-confidence candidates are found, return an empty candidates array.
Only include repositories with GitHub URLs."""

        # Use Gemini with search grounding
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            ),
        )

        if response.text:
            result["raw_response"] = response.text

            # Try to parse JSON from response
            try:
                # Find JSON in response (may be wrapped in markdown code blocks)
                json_match = re.search(r"\{[\s\S]*\}", response.text)
                if json_match:
                    parsed = json.loads(json_match.group())
                    candidates = parsed.get("candidates", [])

                    # Filter to only include high confidence with valid GitHub URLs
                    valid_candidates = []
                    for c in candidates:
                        url = c.get("url", "")
                        if (
                            "github.com" in url.lower()
                            and c.get("confidence") == "high"
                        ):
                            valid_candidates.append(
                                {
                                    "url": url,
                                    "name": c.get("name", ""),
                                    "confidence": "high",
                                    "reason": c.get("reason", ""),
                                    "source": "web_search",
                                }
                            )

                    if valid_candidates:
                        print(
                            f"🌐 Web search found {len(valid_candidates)} high-confidence candidate(s)"
                        )

                    return (
                        valid_candidates
                        if valid_candidates
                        else [
                            {
                                "message": "No high-confidence implementations found via web search"
                            }
                        ]
                    )

            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract GitHub URLs from text
                github_urls = re.findall(
                    r"https?://github\.com/[\w\-\.]+/[\w\-\.]+", response.text
                )
                if github_urls:
                    return [
                        {
                            "url": url,
                            "confidence": "medium",
                            "source": "web_search_extracted",
                        }
                        for url in list(set(github_urls))[:3]
                    ]

        return [{"message": "Web search completed but no implementations found"}]

    except Exception as e:
        return [{"error": f"Web search failed: {str(e)}"}]


# Tool list for easy import
code_search_tools = [
    search_github_repos,
    get_repo_contents,
    get_file_content,
    search_github_code,
    search_papers_with_code,
    clone_repository,
    search_github_for_arxiv_reference,
    search_github_by_paper_name,
    web_search_for_implementation,
]
