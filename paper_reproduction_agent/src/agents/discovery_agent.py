"""Discovery Agent - Responsible for finding and selecting the best implementation for a paper."""

import re
import glob
import requests
from typing import Dict, List
from pathlib import Path
from datetime import datetime



class DiscoveryAgent:
    """Agent that handles repository discovery, selection, and validation."""

    def __init__(self, llm=None, metrics_tracker=None):
        self.llm = llm
        self.metrics_tracker = metrics_tracker

    def find_best_implementation(
        self,
        paper_title: str,
        paper_abstract: str = "",
        arxiv_id: str = None,
        authors: list = None,
        existing_repos: list = None,
    ) -> Dict:
        """Find the best implementation for a paper.

        Args:
            paper_title: Title of the paper
            paper_abstract: Abstract of the paper
            arxiv_id: Optional arXiv ID
            authors: Optional list of authors
            existing_repos: Optional list of repos already found (e.g. from paper text)

        Returns:
            Dict containing:
            - repo_url: The selected repository URL
            - confidence: "high", "medium", "low"
            - source: "paper_link", "papers_with_code", "github_search", etc.
            - all_candidates: List of all candidate repos found
        """
        all_candidates = existing_repos or []
        repo_metadata = []  # List of dicts {url, match_file, ...}

        # 1. Try Papers with Code if no repos found or just to augment
        pwc_repos = self._search_papers_with_code(paper_title)
        if pwc_repos:
            all_candidates.extend(pwc_repos)
            for repo in pwc_repos:
                repo_metadata.append(
                    {"url": repo, "match_file": "PapersWithCode", "confidence": "high"}
                )

        # 2. Enhanced Discovery (GitHub Search) if we still struggle
        if not all_candidates and (arxiv_id or paper_title):
            enhanced_repos = self._enhanced_discovery(arxiv_id, paper_title, authors)
            if enhanced_repos:
                for repo in enhanced_repos:
                    all_candidates.append(repo["url"])
                    repo_metadata.append(repo)

        # Deduplicate
        all_candidates = list(set(all_candidates))

        if not all_candidates:
            return {"repo_url": None, "confidence": "none", "all_candidates": []}

        # 3. Select the best one
        best_repo = self._select_best_repo(
            all_candidates, paper_title, paper_abstract, repo_metadata
        )

        return {
            "repo_url": best_repo,
            "confidence": "high" if best_repo else "low",  # simplified confidence
            "all_candidates": all_candidates,
        }

    def check_existing_results(self, repo_path: str) -> dict:
        """Check if results or model checkpoints already exist in the repository.

        This prevents re-running experiments when results are already available.

        Args:
            repo_path: Path to the cloned repository

        Returns:
            Dictionary with:
                - has_results: True if usable results found
                - result_files: List of result file paths
                - checkpoints: List of checkpoint paths
                - log_files: List of log files
        """
        result = {
            "has_results": False,
            "result_files": [],
            "checkpoints": [],
            "log_files": [],
            "recently_modified": [],
        }

        repo = Path(repo_path)
        if not repo.exists():
            return result

        # Check for result files (JSON, CSV with results/metrics in name)
        result_patterns = [
            "**/results*.json",
            "**/eval_results*.json",
            "**/metrics*.json",
            "**/results*.csv",
            "**/metrics*.csv",
            "**/all_results.json",
            "**/trainer_state.json",
            "results/**/*.json",
            "results/**/*.csv",
            "results/**/*.txt",
            "outputs/**/*.json",
            "outputs/**/*.csv",
            "output/**/*.json",
            "output/**/*.csv",
        ]

        for pattern in result_patterns:
            matches = glob.glob(str(repo / pattern), recursive=True)
            for match in matches:
                try:
                    path = Path(match)
                    stat = path.stat()
                    # Only consider files > 100 bytes (not empty)
                    if stat.st_size > 100:
                        result["result_files"].append(str(path.relative_to(repo)))
                except:
                    pass

        # Check for model checkpoints
        checkpoint_patterns = [
            "**/checkpoint-*",
            "**/checkpoint_*",
            "**/*.pt",
            "**/*.pth",
            "**/*.ckpt",
            "**/pytorch_model.bin",
            "**/model.safetensors",
        ]

        for pattern in checkpoint_patterns:
            matches = glob.glob(str(repo / pattern), recursive=True)
            for match in matches[:10]:  # Limit
                try:
                    path = Path(match)
                    result["checkpoints"].append(str(path.relative_to(repo)))
                except:
                    pass

        # Check for log files (for extracting metrics if no result files)
        log_patterns = ["*.log", "**/*.log", "**/logs/*.log"]
        for pattern in log_patterns[:2]:
            matches = glob.glob(str(repo / pattern), recursive=True)
            for match in matches[:5]:
                try:
                    path = Path(match)
                    stat = path.stat()
                    # Only logs > 1KB
                    if stat.st_size > 1024:
                        result["log_files"].append(str(path.relative_to(repo)))
                except:
                    pass

        # Check for recently modified files (last 24 hours) that might contain results
        recent_cutoff = datetime.now().timestamp() - 86400  # 24 hours
        recent_patterns = ["**/*.json", "**/*.csv"]
        for pattern in recent_patterns:
            matches = glob.glob(str(repo / pattern), recursive=True)
            for match in matches[:50]:
                try:
                    path = Path(match)
                    stat = path.stat()
                    if stat.st_mtime > recent_cutoff and stat.st_size > 100:
                        rel_path = str(path.relative_to(repo))
                        # Skip common non-result files
                        if not any(
                            skip in rel_path.lower()
                            for skip in [
                                "node_modules",
                                ".git",
                                "__pycache__",
                                "package",
                            ]
                        ):
                            result["recently_modified"].append(rel_path)
                except:
                    pass

        # Determine if we have usable results
        # Criteria: At least one result file OR (checkpoint + log file)
        has_result_files = len(result["result_files"]) > 0
        has_checkpoints_and_logs = (
            len(result["checkpoints"]) > 0 and len(result["log_files"]) > 0
        )
        len(result["recently_modified"]) > 0

        result["has_results"] = has_result_files or has_checkpoints_and_logs

        if result["has_results"]:
            print(f"\\n🔍 Checking for existing results in {repo_path}...")
            print(f"   Result files found: {len(result['result_files'])}")
            print(f"   Checkpoints found: {len(result['checkpoints'])}")
            print(f"   Log files found: {len(result['log_files'])}")

        return result

    def _search_papers_with_code(self, paper_title: str) -> List[str]:
        """Try Papers with Code API as fallback for finding implementations."""
        print("🔍 Searching Papers with Code API...")
        try:
            base_url = "https://paperswithcode.com/api/v1/papers/"
            search_url = f"{base_url}?title={requests.utils.quote(paper_title)}"
            response = requests.get(search_url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    paper = data["results"][0]
                    paper_id = paper.get("id")

                    impl_url = f"https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/"
                    impl_response = requests.get(impl_url, timeout=10)

                    if impl_response.status_code == 200:
                        impl_data = impl_response.json()
                        implementations = impl_data.get("results", [])

                        repos = [
                            impl["url"]
                            for impl in implementations
                            if impl.get("url") and impl.get("is_official")
                        ]
                        if not repos:
                            repos = [
                                impl["url"]
                                for impl in implementations
                                if impl.get("url")
                            ]

                        if repos:
                            print(
                                f"✅ Found {len(repos)} implementation(s) from Papers with Code"
                            )
                            return repos
        except Exception as e:
            print(f"⚠️  Papers with Code API failed: {str(e)[:50]}")

        return []

    def _enhanced_discovery(
        self, arxiv_id: str, paper_title: str, authors: list
    ) -> List[Dict]:
        """Try enhanced repo discovery methods (GitHub arXiv search + web search)."""
        print("🔎 Trying enhanced repository discovery...")

        if not arxiv_id and not paper_title:
            print("   ⚠️  No arXiv ID or paper title available for enhanced discovery")
            return []
            
        from ..tools.code_search_tools import (
            search_github_for_arxiv_reference,
            search_github_by_paper_name,
            web_search_for_implementation,
        )

        discovered_repos = []  # List of dicts with full metadata
        discovered_urls = set()  # Track URLs to avoid duplicates

        # Method 1: Search GitHub for repos that reference the arXiv paper
        if arxiv_id:
            print(
                f"🔍 Method 1: Searching GitHub for repos referencing arXiv:{arxiv_id}..."
            )
            try:
                arxiv_results = search_github_for_arxiv_reference(arxiv_id)

                for result in arxiv_results:
                    url = result.get("url")
                    if (
                        url
                        and result.get("confidence") == "high"
                        and url not in discovered_urls
                    ):
                        discovered_urls.add(url)
                        discovered_repos.append(
                            {
                                "url": url,
                                "match_file": result.get("match_file", "README"),
                                "confidence": result.get("confidence", "high"),
                                "stars": result.get(
                                    "stars", 0
                                ),  # Pass through star count
                                "source": "arxiv_reference",
                            }
                        )
                        stars = result.get("stars", 0)
                        stars_str = f" [{stars:,} ⭐]" if stars else ""
                        print(
                            f"   ✅ Found: {url}{stars_str} (arXiv reference in {result.get('match_file', 'README')})"
                        )

            except Exception as e:
                print(f"   ⚠️  GitHub arXiv search failed: {str(e)[:50]}")

        # If we found high-confidence repos from arXiv search, return them
        if discovered_repos:
            print(
                f"📚 Enhanced discovery found {len(discovered_repos)} repo(s) via arXiv reference"
            )
            return discovered_repos

        # Method 2: Search GitHub by paper/method name
        if paper_title:
            print("🔍 Method 2: Searching GitHub for repos matching paper name...")
            try:
                name_results = search_github_by_paper_name(paper_title)

                for result in name_results:
                    url = result.get("url")
                    if (
                        url
                        and result.get("is_exact_match")
                        and url not in discovered_urls
                    ):
                        discovered_urls.add(url)
                        discovered_repos.append(
                            {
                                "url": url,
                                "matched_term": result.get("matched_term", ""),
                                "confidence": result.get("confidence", "medium"),
                                "stars": result.get(
                                    "stars", 0
                                ),  # Pass through star count
                                "source": "name_match",
                            }
                        )
                        term = result.get("matched_term", "")
                        stars = result.get("stars", 0)
                        stars_str = f" [{stars:,} ⭐]" if stars else ""
                        print(f"   ✅ Found: {url}{stars_str} (name matches '{term}')")

            except Exception as e:
                print(f"   ⚠️  GitHub name search failed: {str(e)[:50]}")

        # If we found repos from name search, return them
        if discovered_repos:
            print(
                f"📚 Enhanced discovery found {len(discovered_repos)} repo(s) via name match"
            )
            return discovered_repos

        # Method 3: Web search with LLM evaluation
        if paper_title:
            print("🌐 Method 3: Searching web for implementations...")
            try:
                web_results = web_search_for_implementation(
                    paper_title=paper_title, arxiv_id=arxiv_id, authors=authors
                )

                for result in web_results:
                    url = result.get("url")
                    if (
                        url
                        and result.get("confidence") == "high"
                        and url not in discovered_urls
                    ):
                        discovered_urls.add(url)
                        discovered_repos.append(
                            {
                                "url": url,
                                "reason": result.get("reason", "LLM evaluation"),
                                "confidence": result.get("confidence", "high"),
                                "source": "web_search",
                            }
                        )
                        reason = result.get("reason", "LLM evaluation")
                        print(f"   ✅ Found: {url} ({reason[:50]})")

            except Exception as e:
                print(f"   ⚠️  Web search failed: {str(e)[:50]}")

        if discovered_repos:
            print(f"📚 Enhanced discovery found {len(discovered_repos)} repo(s)")
        else:
            print("📭 Enhanced discovery found no high-confidence repos")

        return discovered_repos

    def _extract_response_text(self, response) -> str:
        """Extract text content from LLM response (handles various formats)."""
        if hasattr(response, "content"):
            if isinstance(response.content, list):
                parts = []
                for item in response.content:
                    if isinstance(item, dict) and "text" in item:
                        parts.append(item["text"])
                    elif isinstance(item, str):
                        parts.append(item)
                    elif hasattr(item, "text"):
                        parts.append(item.text)
                return "".join(parts)
            else:
                return str(response.content)
        return str(response)

    def _extract_method_name(self, paper_title: str) -> str:
        """Extract method/algorithm name from paper title.

        Looks for patterns like:
        - "DreamerV3: ..." -> "DreamerV3"
        - "GPT-4 Technical Report" -> "GPT-4"
        - "Mastering Diverse Domains through World Models" -> looks for acronyms
        """
        # Try to find capitalized method names (DreamerV3, GPT-4, BERT, etc.)
        # Match patterns like: DreamerV3, GPT-4, BERT, RoBERTa, T5, etc.
        acronyms = re.findall(r"\b([A-Z][A-Za-z0-9\-]*(?:[Vv]\d+)?)\b", paper_title)
        if acronyms:
            # Filter out common words that aren't method names
            common = {
                "The",
                "A",
                "An",
                "In",
                "On",
                "For",
                "With",
                "And",
                "Or",
                "Is",
                "Are",
                "We",
                "Our",
                "This",
                "That",
                "From",
                "To",
                "By",
                "As",
                "At",
                "It",
                "Learning",
                "Training",
                "Using",
                "Through",
                "Toward",
                "Towards",
                "Model",
                "Models",
                "Method",
                "Methods",
                "Paper",
                "Report",
                "Technical",
            }
            filtered = [a for a in acronyms if a not in common and len(a) > 1]
            if filtered:
                return filtered[0]

        # Try text before colon (often the method name)
        if ":" in paper_title:
            before_colon = paper_title.split(":")[0].strip()
            # Only use if it's short (likely a method name, not a full sentence)
            if len(before_colon.split()) <= 3:
                return before_colon

        return ""

    def _parse_llm_number(self, response_text: str) -> int:
        """Parse a number from LLM response using multiple strategies.

        Returns the extracted number (1-indexed) or None if parsing fails.
        """
        # Strategy 1: Direct number after stripping quotes/formatting
        cleaned = response_text.strip().strip("'\"` \t\n\r")
        if cleaned.isdigit():
            return int(cleaned)

        # Strategy 2: Regex for standalone number
        match = re.search(r"\b(\d+)\b", cleaned)
        if match:
            return int(match.group(1))

        # Strategy 3: Handle "Option X", "Repository X", "#X" patterns
        match = re.search(
            r"(?:option|repository|repo|choice|number|#)\s*[:\-]?\s*(\d+)",
            cleaned,
            re.IGNORECASE,
        )
        if match:
            return int(match.group(1))

        # Strategy 4: Number at the very start or end
        match = re.match(r"^(\d+)", cleaned)
        if match:
            return int(match.group(1))
        match = re.search(r"(\d+)$", cleaned)
        if match:
            return int(match.group(1))

        return None

    def _select_best_repo(
        self,
        repos: list,
        paper_title: str,
        paper_abstract: str = "",
        repo_metadata: list = None,
    ) -> str:
        """Use LLM to select the best repository for the paper."""
        if len(repos) == 1:
            return repos[0]

        if not self.llm:
            return self._heuristic_select_repo(repos, paper_title, repo_metadata)

        # Build repo list with metadata if available
        repo_list_lines = []
        for i, repo in enumerate(repos):
            line = f"{i+1}. {repo}"

            # Add metadata about where arXiv reference was found
            if repo_metadata:
                for meta in repo_metadata:
                    if meta.get("url") == repo:
                        match_file = meta.get("match_file", "")
                        stars = meta.get("stars", 0)
                        if match_file:
                            if match_file == "README.md":
                                line += (
                                    " (arXiv cited in main README.md - STRONG signal)"
                                )
                            elif "example" in match_file.lower():
                                line += f" (arXiv cited in {match_file} - may be example usage, not official repo)"
                            else:
                                line += f" (arXiv cited in {match_file})"
                        if stars:
                            line += f" [{stars:,} stars]"
                        break

            repo_list_lines.append(line)

        prompt = f"""Given this paper and list of GitHub repositories, select the ONE repository that is most likely the official implementation.

Paper Title: {paper_title}

Abstract: {paper_abstract[:500] if paper_abstract else 'N/A'}

Repositories found:
{chr(10).join(repo_list_lines)}

IMPORTANT CRITERIA:
1. Repository NAME should match the paper's method/algorithm name (e.g., if paper is about "DreamerV3", prefer repos named "dreamerv3")
2. ArXiv reference in MAIN README.md is a STRONG signal it's the official repo
3. ArXiv reference in subdirectories like examples/ often means it's just an example usage of a library
4. AVOID general libraries (transformers, pytorch_geometric, etc.) unless the paper is ABOUT that library
5. Prefer repos from paper authors (if identifiable from repo name/paper title match)
6. High star count combined with matching name is a strong signal

Reply with ONLY the number (1, 2, 3, etc.) of the best repository.

Answer (number only):"""

        selected_idx = None

        try:
            # Setup callback for token tracking
            from ..utils.logging_callback import LoggingCallbackHandler

            callbacks = (
                [
                    LoggingCallbackHandler(
                        verbose=True, metrics_tracker=self.metrics_tracker
                    )
                ]
                if self.metrics_tracker
                else []
            )

            # Set a strict timeout to avoid hanging if provider is slow/down
            # This allows falling back to heuristics rather than waiting 15+ mins
            response = self.llm.invoke(prompt, config={"callbacks": callbacks})
            response_text = self._extract_response_text(response)

            # Try to parse the number
            selected_idx = self._parse_llm_number(response_text)

            if selected_idx is not None:
                idx = selected_idx - 1  # Convert to 0-indexed
                if 0 <= idx < len(repos):
                    print(f"🤖 LLM selected repo #{selected_idx}: {repos[idx]}")
                    return repos[idx]
                else:
                    print(
                        f"⚠️  LLM returned invalid index {selected_idx} (out of range 1-{len(repos)})"
                    )
                    selected_idx = None
            else:
                print(f"⚠️  LLM response couldn't be parsed: '{response_text[:80]}'")

        except Exception as e:
            print(f"⚠️  LLM repo selection failed: {e}")

        # Retry with simplified prompt if first attempt failed
        if selected_idx is None and self.llm:
            print("🔄 Retrying with simplified prompt...")
            retry_prompt = f"""Select the best repository number for paper: "{paper_title}"

Repositories:
{chr(10).join(f'{i+1}. {r}' for i, r in enumerate(repos))}

Reply with ONLY a single digit (1, 2, 3, etc.). Nothing else."""

            try:
                retry_response = self.llm.invoke(retry_prompt)
                retry_text = self._extract_response_text(retry_response)
                selected_idx = self._parse_llm_number(retry_text)

                if selected_idx is not None:
                    idx = selected_idx - 1
                    if 0 <= idx < len(repos):
                        print(
                            f"✅ Retry successful, LLM selected repo #{selected_idx}: {repos[idx]}"
                        )
                        return repos[idx]
                    else:
                        print(f"⚠️  Retry returned invalid index {selected_idx}")
                else:
                    print(
                        f"⚠️  Retry response also couldn't be parsed: '{retry_text[:50]}'"
                    )
            except Exception as e:
                print(f"⚠️  Retry also failed: {e}")

        # Fall back to heuristic selection
        return self._heuristic_select_repo(repos, paper_title, repo_metadata)

    def _heuristic_select_repo(
        self, repos: list, paper_title: str, repo_metadata: list = None
    ) -> str:
        """Select repository using heuristic scoring when LLM fails.

        Scoring factors:
        - Star count (high stars = likely official)
        - Method name match in repo name
        - Confidence level from metadata
        - README.md match location
        - Penalties for study guides, collections, generic libraries
        """
        print("🔍 Using heuristic fallback to select repository...")

        # Extract method name from paper title for matching
        paper_method = self._extract_method_name(paper_title)
        paper_keywords = set(paper_title.lower().split())

        best_score = -999
        best_repo = repos[0]
        scores_debug = []  # For debugging

        for repo in repos:
            score = 0
            repo.lower()

            # Extract repo name from URL
            repo_name = repo.split("/")[-1] if "/" in repo else repo
            repo_name_lower = repo_name.lower()

            # Get metadata for this repo
            repo_meta = None
            if repo_metadata:
                for meta in repo_metadata:
                    if meta.get("url") == repo:
                        repo_meta = meta
                        break

            # --- STAR COUNT SCORING (most reliable signal) ---
            if repo_meta:
                stars = repo_meta.get("stars", 0)
                if stars > 5000:
                    score += 20
                elif stars > 1000:
                    score += 15
                elif stars > 500:
                    score += 10
                elif stars > 100:
                    score += 5
                elif stars > 50:
                    score += 2

                # Confidence level from discovery
                if repo_meta.get("confidence") == "high":
                    score += 8

            # --- METHOD NAME MATCHING (very strong signal) ---
            if paper_method:
                method_lower = paper_method.lower().replace("-", "").replace("_", "")
                repo_name_normalized = repo_name_lower.replace("-", "").replace("_", "")

                # Exact match or close match
                if method_lower == repo_name_normalized:
                    score += 30  # Exact match is very strong
                elif method_lower in repo_name_normalized:
                    score += 25  # Contains method name
                elif repo_name_normalized in method_lower:
                    score += 20  # Repo name is part of method name

            # --- PAPER KEYWORD MATCHING ---
            for keyword in paper_keywords:
                if len(keyword) > 3 and keyword in repo_name_lower:
                    score += 5

            # --- FILE LOCATION SCORING ---
            if repo_meta:
                match_file = repo_meta.get("match_file", "")
                if match_file == "README.md":
                    score += 8  # Main README is strong signal
                elif match_file and "example" in match_file.lower():
                    score -= 10  # Example directories are weak signal
                elif match_file:
                    score += 3  # Other files still provide some signal

            # --- PENALTIES ---
            # Penalize generic library names
            generic_names = [
                "pytorch",
                "tensorflow",
                "transformers",
                "huggingface",
                "examples",
                "tutorials",
            ]
            for generic in generic_names:
                if generic in repo_name_lower:
                    score -= 10

            # Penalize study guides, paper collections, etc.
            collection_keywords = [
                "study",
                "awesome",
                "list",
                "collection",
                "survey",
                "papers",
                "reading",
                "notes",
                "resources",
                "curated",
                "deeplearning",
            ]
            for kw in collection_keywords:
                if kw in repo_name_lower:
                    score -= 20

            scores_debug.append((repo_name, score))

            if score > best_score:
                best_score = score
                best_repo = repo

        # Print top 3 scores for debugging
        scores_debug.sort(key=lambda x: x[1], reverse=True)
        print(
            f"   Top scores: {', '.join(f'{name}={s}' for name, s in scores_debug[:3])}"
        )
        print(f"🔍 Heuristic selected: {best_repo} (score: {best_score})")
        return best_repo
