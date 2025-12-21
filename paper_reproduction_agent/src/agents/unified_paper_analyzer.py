"""Unified Paper Analyzer Agent - Single LLM call to extract all key information."""

from typing import Dict, List
from langchain_core.messages import HumanMessage
from ..utils.llm_factory import create_llm
import re
import json


class UnifiedPaperAnalyzer:
    """
    Simplified paper analyzer that extracts all needed information in one LLM call.

    This replaces the fragmented approach of multiple extraction steps with a single,
    comprehensive analysis that lets the LLM understand the paper holistically.
    """

    def __init__(self, llm=None):
        """Initialize the unified paper analyzer."""
        self.llm = llm or create_llm(temperature=0.1)

    def analyze_paper(self, paper_text: str, paper_title: str = "Unknown") -> Dict:
        """
        Analyze paper and extract all key information in one pass.

        Args:
            paper_text: Full text of the paper
            paper_title: Title of the paper

        Returns:
            Dictionary containing:
            - github_repos: List of GitHub/GitLab/Bitbucket URLs found
            - results_to_reproduce: Dict with metrics, datasets, and expected values
            - core_contribution: Brief description of what the paper does
            - context_summary: Natural language summary for next agents
        """
        # First, do regex extraction for GitHub URLs (fast and reliable)
        github_urls = self._extract_code_urls_regex(paper_text)

        # Now ask LLM to extract results and understand the paper
        prompt = f"""Analyze this research paper and extract key information needed for reproduction.

Paper Title: {paper_title}

Paper Text:
{paper_text}

Please extract and provide in a clear, structured format:

1. **Main Results to Reproduce:**
   - What metrics are reported? (e.g., accuracy, F1, BLEU, perplexity, loss)
   - What datasets were used?
   - What are the specific numerical values reported?
   - Format: "Dataset: [name] | Metric: [metric] | Value: [value]"

2. **Core Contribution:**
   - In 1-2 sentences, what is the main contribution or claim of this paper?
   - What is the key innovation being tested?

3. **Additional GitHub/Code URLs:**
   - Are there any code repository URLs mentioned in the text?
   - Look for: github.com, gitlab.com, bitbucket.org
   - List full URLs if found

4. **Datasets Mentioned:**
   - List all datasets mentioned (e.g., MNIST, CIFAR-10, ImageNet, SQuAD, WMT)

5. **Key Implementation Details:**
   - Any specific model architectures mentioned?
   - Any crucial hyperparameters reported?
   - Training setup (epochs, batch size, etc.)?

Provide your response in a clear, structured format that can be easily parsed.
Focus on being specific and extracting actual numbers from the results section."""

        messages = [HumanMessage(content=prompt)]

        try:
            # Just invoke the LLM directly - no tool calling needed
            result = self.llm.invoke(messages)

            response_text = result.content if hasattr(result, 'content') else str(result)

            # Clean up any thinking tags
            clean_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
            clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', clean_response, flags=re.DOTALL)
            clean_response = re.sub(r'\n\s*\n', '\n', clean_response).strip()

            # Parse the LLM response
            parsed_results = self._parse_llm_response(clean_response)

            # Combine regex URLs with any URLs found by LLM
            all_repos = list(set(github_urls + parsed_results.get("github_repos", [])))

            return {
                "github_repos": all_repos,
                "results_to_reproduce": parsed_results.get("results_to_reproduce", {}),
                "core_contribution": parsed_results.get("core_contribution", ""),
                "datasets": parsed_results.get("datasets", []),
                "implementation_details": parsed_results.get("implementation_details", ""),
                "context_summary": self._create_context_summary(
                    paper_title,
                    all_repos,
                    parsed_results
                ),
                "raw_analysis": clean_response
            }

        except Exception as e:
            print(f"⚠️  LLM analysis failed: {str(e)[:200]}")
            return {
                "github_repos": github_urls,  # At least return regex results
                "results_to_reproduce": {},
                "core_contribution": "",
                "datasets": [],
                "implementation_details": "",
                "context_summary": f"Paper: {paper_title}. Found {len(github_urls)} code repositories.",
                "raw_analysis": "",
                "error": str(e)
            }

    def _extract_code_urls_regex(self, paper_text: str) -> List[str]:
        """
        Extract GitHub/GitLab/Bitbucket URLs using regex (fast and reliable).

        This is the same regex logic from the old extract_code_references tool.
        """
        all_urls = []
        text_lower = paper_text.lower()

        platforms = [
            ('github.com', r'https?://(?:www\.)?github\.com/\s*\n?\s*([\w\-\.]+)\s*/\s*\n?\s*([\w\-\.]+)'),
            ('gitlab.com', r'https?://(?:www\.)?gitlab\.com/\s*\n?\s*([\w\-\.]+)\s*/\s*\n?\s*([\w\-\.]+)'),
            ('bitbucket.org', r'https?://(?:www\.)?bitbucket\.org/\s*\n?\s*([\w\-\.]+)\s*/\s*\n?\s*([\w\-\.]+)'),
        ]

        for platform_name, url_pattern in platforms:
            idx = 0
            while True:
                idx = text_lower.find(platform_name, idx)
                if idx == -1:
                    break

                # Extract surrounding context
                start = max(0, idx - 500)
                end = min(len(paper_text), idx + 500)
                context = paper_text[start:end]

                # Find URLs in context
                matches = re.findall(url_pattern, context, re.IGNORECASE)

                for match in matches:
                    if isinstance(match, tuple) and len(match) == 2:
                        username, repo = match
                        username = username.strip()
                        repo = repo.strip()
                        full_url = f"https://{platform_name}/{username}/{repo}"
                        all_urls.append(full_url)

                idx += len(platform_name)

        # Remove duplicates
        return list(set(all_urls))

    def _parse_llm_response(self, response_text: str) -> Dict:
        """
        Parse the LLM's structured response.

        Extracts information from the formatted response.
        """
        parsed = {
            "github_repos": [],
            "results_to_reproduce": {
                "metrics": [],
                "summary": ""
            },
            "core_contribution": "",
            "datasets": [],
            "implementation_details": ""
        }

        # Extract GitHub URLs mentioned in response
        github_pattern = r'https?://(?:github|gitlab|bitbucket)\.(?:com|org)/[\w\-\.]+/[\w\-\.]+'
        github_urls = re.findall(github_pattern, response_text, re.IGNORECASE)
        parsed["github_repos"] = list(set(github_urls))

        # Extract datasets
        dataset_patterns = [
            r'\b(MNIST|CIFAR-?10|CIFAR-?100|ImageNet|COCO|Pascal VOC|MS COCO)\b',
            r'\b(SQuAD|GLUE|SuperGLUE|WikiText|Penn Treebank|WMT\d*)\b',
            r'\b(LibriSpeech|Common Voice|AudioSet)\b',
        ]
        for pattern in dataset_patterns:
            datasets = re.findall(pattern, response_text, re.IGNORECASE)
            parsed["datasets"].extend(datasets)
        parsed["datasets"] = list(set(parsed["datasets"]))

        # Extract metric lines (Dataset: X | Metric: Y | Value: Z)
        metric_pattern = r'Dataset:\s*([^\|]+)\s*\|\s*Metric:\s*([^\|]+)\s*\|\s*Value:\s*([^\n]+)'
        metric_matches = re.findall(metric_pattern, response_text, re.IGNORECASE)

        metrics_list = []
        for dataset, metric, value in metric_matches:
            metrics_list.append({
                "dataset": dataset.strip(),
                "metric": metric.strip(),
                "value": value.strip()
            })

        if metrics_list:
            parsed["results_to_reproduce"]["metrics"] = metrics_list

        # Store full response as summary
        parsed["results_to_reproduce"]["summary"] = response_text

        # Extract core contribution (look for section after "Core Contribution")
        contribution_match = re.search(
            r'(?:Core Contribution|Main Contribution)[:\s]*([^\n]+(?:\n(?!\#)[^\n]+)*)',
            response_text,
            re.IGNORECASE
        )
        if contribution_match:
            parsed["core_contribution"] = contribution_match.group(1).strip()

        # Extract implementation details
        impl_match = re.search(
            r'(?:Implementation Details|Key Implementation)[:\s]*([^\n]+(?:\n(?!\#)[^\n]+)*)',
            response_text,
            re.IGNORECASE
        )
        if impl_match:
            parsed["implementation_details"] = impl_match.group(1).strip()

        return parsed

    def _create_context_summary(self, paper_title: str, repos: List[str],
                                parsed_results: Dict) -> str:
        """
        Create a natural language summary for next agents.

        This summary will be stored in agent_contexts so other agents can understand
        what the paper is about without re-reading everything.
        """
        summary_parts = [f"Paper: {paper_title}"]

        if parsed_results.get("core_contribution"):
            summary_parts.append(f"Contribution: {parsed_results['core_contribution']}")

        if parsed_results.get("datasets"):
            datasets_str = ", ".join(parsed_results["datasets"][:3])  # First 3
            summary_parts.append(f"Datasets: {datasets_str}")

        metrics = parsed_results.get("results_to_reproduce", {}).get("metrics", [])
        if metrics:
            # Show first 2 metrics
            metric_strs = []
            for m in metrics[:2]:
                metric_strs.append(f"{m['metric']}={m['value']} on {m['dataset']}")
            summary_parts.append(f"Key Results: {'; '.join(metric_strs)}")

        if repos:
            summary_parts.append(f"Code: {repos[0]}")  # First repo
            if len(repos) > 1:
                summary_parts.append(f"(+{len(repos)-1} more repos)")

        return ". ".join(summary_parts)

    def enhanced_repo_discovery(
        self,
        arxiv_id: str = None,
        paper_title: str = None,
        authors: List[str] = None
    ) -> List[str]:
        """
        Enhanced repository discovery using multiple methods.

        This method is called when basic extraction (regex + LLM) doesn't find repos.
        It tries additional discovery methods in order of reliability:
        1. GitHub code search for arXiv references
        2. GitHub search by paper/method name
        3. Web search with LLM evaluation

        Args:
            arxiv_id: The arXiv paper ID (e.g., "2301.12345")
            paper_title: Title of the paper
            authors: List of author names (for validation)

        Returns:
            List of discovered GitHub repository URLs (high confidence only)
        """
        from ..tools.code_search_tools import (
            search_github_for_arxiv_reference,
            search_github_by_paper_name,
            web_search_for_implementation
        )

        discovered_repos = []

        # Method 1: Search GitHub for repos that reference the arXiv paper
        if arxiv_id:
            print(f"🔍 Method 1: Searching GitHub for repos referencing arXiv:{arxiv_id}...")
            try:
                arxiv_results = search_github_for_arxiv_reference(arxiv_id)

                for result in arxiv_results:
                    if result.get("url") and result.get("confidence") == "high":
                        url = result["url"]
                        if url not in discovered_repos:
                            discovered_repos.append(url)
                            print(f"   ✅ Found: {url} (arXiv reference in {result.get('match_file', 'README')})")

            except Exception as e:
                print(f"   ⚠️  GitHub arXiv search failed: {str(e)[:50]}")

        # If we found high-confidence repos from arXiv search, return them
        if discovered_repos:
            print(f"📚 Enhanced discovery found {len(discovered_repos)} repo(s) via arXiv reference")
            return discovered_repos

        # Method 2: Search GitHub by paper/method name
        if paper_title:
            print(f"🔍 Method 2: Searching GitHub for repos matching paper name...")
            try:
                name_results = search_github_by_paper_name(paper_title)

                for result in name_results:
                    url = result.get("url")
                    if url and result.get("is_exact_match"):
                        if url not in discovered_repos:
                            discovered_repos.append(url)
                            term = result.get("matched_term", "")
                            print(f"   ✅ Found: {url} (name matches '{term}')")

            except Exception as e:
                print(f"   ⚠️  GitHub name search failed: {str(e)[:50]}")

        # If we found repos from name search, return them
        if discovered_repos:
            print(f"📚 Enhanced discovery found {len(discovered_repos)} repo(s) via name match")
            return discovered_repos

        # Method 3: Web search with LLM evaluation
        if paper_title:
            print(f"🌐 Method 3: Searching web for implementations...")
            try:
                web_results = web_search_for_implementation(
                    paper_title=paper_title,
                    arxiv_id=arxiv_id,
                    authors=authors
                )

                for result in web_results:
                    url = result.get("url")
                    if url and result.get("confidence") == "high":
                        if url not in discovered_repos:
                            discovered_repos.append(url)
                            reason = result.get("reason", "LLM evaluation")
                            print(f"   ✅ Found: {url} ({reason[:50]})")

            except Exception as e:
                print(f"   ⚠️  Web search failed: {str(e)[:50]}")

        if discovered_repos:
            print(f"📚 Enhanced discovery found {len(discovered_repos)} repo(s)")
        else:
            print("📭 Enhanced discovery found no high-confidence repos")

        return discovered_repos
