"""Unified Paper Analyzer Agent - Generic extraction without forced structure."""

from typing import Dict, List
from langchain_core.messages import HumanMessage
from ..utils.llm_factory import create_llm
import re
import json


class UnifiedPaperAnalyzer:
    """
    Generic paper analyzer that lets the LLM intelligently extract information.

    No forced formats - the LLM identifies tables, results, and structures them naturally.
    """

    def __init__(self, llm=None, metrics_tracker=None):
        """Initialize the unified paper analyzer."""
        self.llm = llm or create_llm(temperature=0.1)
        self.metrics_tracker = metrics_tracker

    def _extract_text_from_response(self, response) -> str:
        """Extract text from LLM response, handling both string and list formats.

        Args:
            response: LLM response object

        Returns:
            Extracted text as string
        """
        response_text = ""
        if hasattr(response, "content"):
            # Handle case where content might be a list
            if isinstance(response.content, list):
                # Extract text from list of content blocks
                for item in response.content:
                    if isinstance(item, dict) and "text" in item:
                        response_text += item["text"]
                    elif isinstance(item, str):
                        response_text += item
                    elif hasattr(item, "text"):
                        response_text += item.text
            else:
                response_text = str(response.content)
        else:
            response_text = str(response)
        return response_text

    def analyze_paper(self, paper_text: str, paper_title: str = "Unknown") -> Dict:
        """
        Analyze paper and extract all key information in one pass.

        Args:
            paper_text: Full text of the paper
            paper_title: Title of the paper

        Returns:
            Dictionary containing:
            - github_repos: List of GitHub/GitLab/Bitbucket URLs found
            - results_to_reproduce: Dict with metrics and tables
            - core_contribution: Brief description of what the paper does
            - datasets: List of datasets mentioned
            - context_summary: Natural language summary for next agents
        """
        # First, do regex extraction for GitHub URLs (fast and reliable)
        github_urls = self._extract_code_urls_regex(paper_text)

        # Single-step extraction: Ask LLM to extract everything in structured format
        prompt = f"""Analyze this research paper and extract key information.

Paper Title: {paper_title}

Paper Text:
{paper_text}

Extract the following information and return it in JSON format:

1. **core_contribution**: Brief description of the main contribution (1-2 sentences)

2. **datasets**: List of dataset names mentioned in the paper (e.g., ["MNIST", "CIFAR-10", "ImageNet"])

3. **implementation_details**: Any implementation details mentioned (model architecture, hyperparameters, etc.)

4. **code_repositories**: Any GitHub/GitLab/Bitbucket URLs mentioned

5. **result_tables**: For each result table in the paper, extract:
   - Table name/number
   - The datasets tested
   - The metrics measured (accuracy, F1, BLEU, etc.)
   - The values achieved by the proposed method

   Structure each table as a list of results, where each result has:
   - "dataset": dataset name
   - "metric": metric name
   - "value": the value (as string, keep units like % or decimals)

Return ONLY valid JSON in this exact format:
```json
{{
  "core_contribution": "...",
  "datasets": ["dataset1", "dataset2", ...],
  "implementation_details": "...",
  "code_repositories": ["url1", "url2", ...],
  "result_tables": [
    {{
      "table_name": "Table 1",
      "results": [
        {{"dataset": "MNIST", "metric": "Accuracy", "value": "98.5%"}},
        {{"dataset": "CIFAR-10", "metric": "Accuracy", "value": "92.3%"}}
      ]
    }}
  ]
}}
```

Be thorough - extract ALL result tables and ALL datasets mentioned. If information is not found, use empty string or empty list."""

        messages = [HumanMessage(content=prompt)]

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

            # Get structured extraction
            result = self.llm.invoke(messages, config={"callbacks": callbacks})
            response_text = self._extract_text_from_response(result)

            # Clean up thinking tags and tool calls
            response_text = re.sub(
                r"<think>.*?</think>", "", response_text, flags=re.DOTALL
            )
            response_text = re.sub(
                r"<tool_call>.*?</tool_call>", "", response_text, flags=re.DOTALL
            )
            response_text = response_text.strip()

            print(f"\n📄 Paper analysis complete ({len(response_text)} chars)")
            print(f"Preview: {response_text[:500]}...\n")

            # Extract JSON from response (might be wrapped in markdown code blocks)
            json_match = re.search(
                r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL
            )
            if not json_match:
                json_match = re.search(
                    r"```\s*(\{.*?\})\s*```", response_text, re.DOTALL
                )
            if not json_match:
                json_match = re.search(r"(\{.*\})", response_text, re.DOTALL)

            structured_data = {}
            if json_match:
                try:
                    structured_data = json.loads(json_match.group(1))
                    print("✅ Successfully parsed JSON response")
                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON parsing failed: {e}")
                    structured_data = {}

            # Extract GitHub URLs from both regex and LLM
            llm_repos = structured_data.get("code_repositories", [])

            # Normalize LLM-provided URLs (add https:// if missing)
            normalized_llm_repos = []
            for repo in llm_repos:
                if repo and isinstance(repo, str):
                    if not repo.startswith("http"):
                        # Check if it looks like a valid repo URL
                        if any(
                            platform in repo.lower()
                            for platform in [
                                "github.com",
                                "gitlab.com",
                                "bitbucket.org",
                            ]
                        ):
                            repo = f"https://{repo}"
                    normalized_llm_repos.append(repo)

            # Also extract URLs from response text using regex
            github_pattern = (
                r"https?://(?:github|gitlab|bitbucket)\.(?:com|org)/[\w\-\.]+/[\w\-\.]+"
            )
            llm_found_urls = re.findall(github_pattern, response_text, re.IGNORECASE)

            # Combine all sources and deduplicate
            all_repos = list(set(github_urls + normalized_llm_repos + llm_found_urls))

            # Convert result_tables to the format expected by orchestrator
            result_tables = structured_data.get("result_tables", [])

            # Flatten all results from all tables into a single list
            all_results = []
            for table in result_tables:
                table_results = table.get("results", [])
                all_results.extend(table_results)

            # Return in expected format
            return {
                "github_repos": all_repos,
                "results_to_reproduce": {
                    "tables": result_tables,  # Full table structure
                    "metrics": all_results,  # Flattened list for easy access
                },
                "core_contribution": structured_data.get("core_contribution", ""),
                "datasets": structured_data.get("datasets", []),
                "implementation_details": structured_data.get(
                    "implementation_details", ""
                ),
                "context_summary": f"Paper: {paper_title}. Found {len(all_repos)} code repositories. Extracted {len(all_results)} metrics from {len(result_tables)} tables.",
                "raw_analysis": response_text,
                "structured_data": structured_data,
            }

        except Exception as e:
            print(f"⚠️  LLM analysis failed: {str(e)[:200]}")
            import traceback

            traceback.print_exc()
            return {
                "github_repos": github_urls,  # At least return regex results
                "results_to_reproduce": {"tables": [], "metrics": []},
                "core_contribution": "",
                "datasets": [],
                "implementation_details": "",
                "context_summary": f"Paper: {paper_title}. Found {len(github_urls)} code repositories.",
                "raw_analysis": "",
                "error": str(e),
            }



    def _extract_code_urls_regex(self, paper_text: str) -> List[str]:
        """
        Extract GitHub/GitLab/Bitbucket URLs using regex (fast and reliable).

        This is the same regex logic from the old extract_code_references tool.
        """
        all_urls = []
        text_lower = paper_text.lower()

        platforms = [
            (
                "github.com",
                r"https?://(?:www\.)?github\.com/\s*\n?\s*([\w\-\.]+)\s*/\s*\n?\s*([\w\-\.]+)",
            ),
            (
                "gitlab.com",
                r"https?://(?:www\.)?gitlab\.com/\s*\n?\s*([\w\-\.]+)\s*/\s*\n?\s*([\w\-\.]+)",
            ),
            (
                "bitbucket.org",
                r"https?://(?:www\.)?bitbucket\.org/\s*\n?\s*([\w\-\.]+)\s*/\s*\n?\s*([\w\-\.]+)",
            ),
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


