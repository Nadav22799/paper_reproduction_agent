"""Metrics Extractor Agent - Parses outputs and extracts numerical results."""

import re
from typing import Dict, List
from langchain_core.messages import HumanMessage
from ..utils.llm_factory import create_llm


class MetricsExtractorAgent:
    """Agent for extracting metrics from experiment outputs."""

    def __init__(self, llm=None):
        self.llm = llm or create_llm(temperature=0.0)

    def extract_metrics(self, output_text: str, expected_metrics: Dict = None) -> Dict:
        """
        Extract numerical metrics from experiment output.

        Args:
            output_text: Raw output from experiment
            expected_metrics: Expected metric names from paper

        Returns:
            Extracted metrics and comparison
        """
        # First try regex-based extraction (fast, no LLM needed)
        regex_metrics = self._extract_with_regex(output_text)

        # If we have expected metrics, use LLM for better parsing
        if expected_metrics and self.llm:
            llm_metrics = self._extract_with_llm(output_text, expected_metrics)
            # Merge both results (prefer LLM if it found more)
            if len(llm_metrics) > len(regex_metrics):
                return llm_metrics
            return regex_metrics

        return regex_metrics

    def _extract_with_regex(self, text: str) -> Dict:
        """Extract metrics using regex patterns."""
        metrics = {}

        # Common patterns: "accuracy: 0.95", "BLEU = 28.4", "Loss: 0.123"
        patterns = [
            r'(?P<name>accuracy|acc|precision|recall|f1|bleu|perplexity|loss|error)\s*[:=]\s*(?P<value>\d+\.?\d*)',
            r'(?P<name>Test|Train|Val|Validation)\s+(?P<metric>accuracy|loss|error)\s*[:=]\s*(?P<value>\d+\.?\d*)',
            r'(?P<name>\w+)\s*=\s*(?P<value>\d+\.?\d*)%?',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groupdict()
                if 'metric' in groups:
                    name = f"{groups['name']}_{groups['metric']}"
                    value = groups['value']
                else:
                    name = groups['name']
                    value = groups['value']

                try:
                    metrics[name.lower()] = float(value)
                except ValueError:
                    pass

        return {"metrics": metrics, "extraction_method": "regex"}

    def _extract_with_llm(self, text: str, expected_metrics: Dict) -> Dict:
        """Extract metrics using LLM for complex cases."""
        # Truncate very long outputs
        text_sample = text[-3000:] if len(text) > 3000 else text

        task = f"""Extract numerical metrics from this experiment output:

Output:
{text_sample}

Expected metrics from paper:
{expected_metrics}

Extract all numerical results. Format:
metric_name: value

Example:
accuracy: 0.95
bleu_score: 28.4
perplexity: 12.3

Be precise. Only include metrics you see in the output."""

        try:
            messages = [HumanMessage(content=task)]
            result = self.llm.invoke(messages)

            content = result.content if hasattr(result, 'content') else str(result)

            # Parse LLM response
            metrics = {}
            for line in content.split('\n'):
                match = re.search(r'(\w+)\s*[:=]\s*(\d+\.?\d*)', line)
                if match:
                    metrics[match.group(1).lower()] = float(match.group(2))

            return {"metrics": metrics, "extraction_method": "llm"}

        except Exception as e:
            return {"metrics": {}, "extraction_method": "llm_failed", "error": str(e)}

    def compare_metrics(self, extracted: Dict, expected: Dict) -> Dict:
        """
        Compare extracted metrics with paper results.

        Args:
            extracted: Metrics from running code
            expected: Metrics from paper

        Returns:
            Comparison results
        """
        comparison = {
            "matches": [],
            "mismatches": [],
            "missing": [],
            "overall_match": False
        }

        if not extracted.get("metrics"):
            comparison["missing"] = list(expected.keys()) if expected else []
            return comparison

        extracted_metrics = extracted.get("metrics", {})

        # Parse expected metrics if they're strings
        expected_values = {}
        if expected:
            for key, value in expected.items():
                if isinstance(value, str):
                    # Try to extract number from string
                    match = re.search(r'\d+\.?\d*', value)
                    if match:
                        expected_values[key.lower()] = float(match.group())
                else:
                    expected_values[key.lower()] = float(value)

        # Compare each metric
        tolerance = 0.05  # 5% tolerance
        matches = 0

        for metric_name, expected_val in expected_values.items():
            found = False
            for extracted_name, extracted_val in extracted_metrics.items():
                if metric_name in extracted_name or extracted_name in metric_name:
                    found = True
                    diff = abs(extracted_val - expected_val)
                    relative_diff = diff / expected_val if expected_val != 0 else diff

                    if relative_diff <= tolerance:
                        comparison["matches"].append({
                            "metric": metric_name,
                            "expected": expected_val,
                            "actual": extracted_val,
                            "diff": f"{relative_diff*100:.2f}%"
                        })
                        matches += 1
                    else:
                        comparison["mismatches"].append({
                            "metric": metric_name,
                            "expected": expected_val,
                            "actual": extracted_val,
                            "diff": f"{relative_diff*100:.2f}%"
                        })
                    break

            if not found:
                comparison["missing"].append(metric_name)

        # Determine overall match
        if expected_values:
            match_rate = matches / len(expected_values)
            comparison["overall_match"] = match_rate >= 0.7  # 70% of metrics should match
            comparison["match_rate"] = f"{match_rate*100:.1f}%"

        return comparison
