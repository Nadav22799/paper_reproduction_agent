"""Evaluation Framework for Paper Reproduction System.

This module provides comprehensive evaluation for both:
- Test A: Code Execution & Reproduction
- Test B: Code Generation from Paper
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.utils.storage import StorageProvider, StorageManager


class ReproductionEvaluator:
    """Evaluator for paper reproduction attempts."""

    def __init__(
        self,
        results_dir: str = "./evaluation_results",
        storage: Optional[StorageProvider] = None,
    ):
        """Initialize evaluator.

        Args:
            results_dir: Directory to store evaluation results locally.
            storage: Optional StorageProvider for cloud persistence.
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._storage = storage or StorageManager.get_provider(base_dir=results_dir)

    def evaluate_test_a(
        self,
        paper_id: str,
        repo_url: str,
        expected_results: Dict[str, float],
        actual_results: Dict[str, float],
        execution_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate Test A (Code Execution & Reproduction).

        Args:
            paper_id: Paper identifier (arXiv ID)
            repo_url: URL of the repository used
            expected_results: Results reported in paper
            actual_results: Results from execution
            execution_metadata: Metadata about the execution

        Returns:
            Evaluation results
        """
        from .tools.statistical_validation_tools import (
            batch_compare_metrics,
            compute_reproducibility_score,
            generate_comparison_report,
        )

        start_time = time.time()

        # Compare results
        comparison = batch_compare_metrics.invoke(
            {
                "actual_results": actual_results,
                "expected_results": expected_results,
                "tolerance_std": 1.0,
            }
        )

        if isinstance(comparison, str):
            comparison = json.loads(comparison)

        # Compute reproducibility score
        score = compute_reproducibility_score.invoke({"comparison_results": comparison})

        if isinstance(score, str):
            score = json.loads(score)

        # Generate report
        report = generate_comparison_report.invoke(
            {
                "comparison_results": comparison,
                "paper_title": paper_id,
                "reproduction_method": "Test A",
            }
        )

        # Evaluation results
        evaluation = {
            "test_type": "A",
            "paper_id": paper_id,
            "repo_url": repo_url,
            "timestamp": datetime.now().isoformat(),
            "expected_results": expected_results,
            "actual_results": actual_results,
            "comparison": comparison,
            "reproducibility_score": score.get("overall_score", 0),
            "grade": score.get("grade", "N/A"),
            "reproducible": score.get("reproducible", False),
            "execution_metadata": execution_metadata,
            "evaluation_time_seconds": time.time() - start_time,
            "report": report,
        }

        # Save results
        self._save_evaluation(paper_id, "test_a", evaluation)

        return evaluation

    def evaluate_test_b(
        self,
        paper_id: str,
        methodology: Dict[str, Any],
        generated_code_path: str,
        expected_results: Dict[str, float],
        actual_results: Optional[Dict[str, float]],
        generation_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate Test B (Code Generation from Paper).

        Args:
            paper_id: Paper identifier (arXiv ID)
            methodology: Extracted methodology
            generated_code_path: Path to generated code
            expected_results: Results reported in paper
            actual_results: Results from generated code (None if execution failed)
            generation_metadata: Metadata about code generation

        Returns:
            Evaluation results
        """
        from .tools.statistical_validation_tools import (
            batch_compare_metrics,
            compute_reproducibility_score,
            generate_comparison_report,
        )

        start_time = time.time()

        # Evaluate methodology extraction completeness
        methodology_score = self._evaluate_methodology_completeness(methodology)

        # Evaluate code generation quality
        code_quality = self._evaluate_code_quality(generated_code_path)

        # Compare results (if execution succeeded)
        if actual_results:
            comparison = batch_compare_metrics.invoke(
                {
                    "actual_results": actual_results,
                    "expected_results": expected_results,
                    "tolerance_std": 1.5,  # More lenient for generated code
                }
            )

            if isinstance(comparison, str):
                comparison = json.loads(comparison)

            score = compute_reproducibility_score.invoke(
                {"comparison_results": comparison}
            )

            if isinstance(score, str):
                score = json.loads(score)

            report = generate_comparison_report.invoke(
                {
                    "comparison_results": comparison,
                    "paper_title": paper_id,
                    "reproduction_method": "Test B",
                }
            )
        else:
            comparison = None
            score = {
                "overall_score": 0,
                "grade": "F (Execution Failed)",
                "reproducible": False,
            }
            report = "Code generation succeeded but execution failed."

        # Overall Test B score (weighted combination)
        weights = {
            "methodology_extraction": 0.3,
            "code_quality": 0.3,
            "results_reproduction": 0.4,
        }

        overall_score = (
            methodology_score * weights["methodology_extraction"]
            + code_quality["overall_score"] * weights["code_quality"]
            + score.get("overall_score", 0) * weights["results_reproduction"]
        )

        evaluation = {
            "test_type": "B",
            "paper_id": paper_id,
            "timestamp": datetime.now().isoformat(),
            "methodology_extraction_score": methodology_score,
            "code_quality_score": code_quality,
            "expected_results": expected_results,
            "actual_results": actual_results,
            "comparison": comparison,
            "reproducibility_score": score.get("overall_score", 0),
            "overall_test_b_score": overall_score,
            "grade": self._score_to_grade(overall_score),
            "reproducible": overall_score >= 70,
            "generation_metadata": generation_metadata,
            "generated_code_path": generated_code_path,
            "evaluation_time_seconds": time.time() - start_time,
            "report": report,
        }

        # Save results
        self._save_evaluation(paper_id, "test_b", evaluation)

        return evaluation

    def _evaluate_methodology_completeness(self, methodology: Dict[str, Any]) -> float:
        """Evaluate completeness of extracted methodology.

        Args:
            methodology: Extracted methodology dictionary

        Returns:
            Completeness score (0-100)
        """
        # Check for key components
        required_components = [
            "algorithm_core",
            "parameters",
            "procedure",
            "data_requirements",
            "evaluation",
        ]

        score = 0
        for component in required_components:
            if component in methodology and methodology[component]:
                score += 20  # 20 points per component

        return min(score, 100)

    def _evaluate_code_quality(self, code_path: str) -> Dict[str, Any]:
        """Evaluate quality of generated code.

        Args:
            code_path: Path to generated code directory

        Returns:
            Code quality metrics
        """
        code_path = Path(code_path)

        if not code_path.exists():
            return {"overall_score": 0, "error": "Code path does not exist"}

        quality_metrics = {
            "has_model": False,
            "has_training": False,
            "has_data_loader": False,
            "has_config": False,
            "has_requirements": False,
            "has_readme": False,
            "syntax_valid": True,
            "overall_score": 0,
        }

        # Check for essential files
        files_to_check = {
            "model.py": "has_model",
            "train.py": "has_training",
            "data.py": "has_data_loader",
            "config.py": "has_config",
            "config.yaml": "has_config",
            "requirements.txt": "has_requirements",
            "README.md": "has_readme",
        }

        for filename, metric_key in files_to_check.items():
            if (code_path / filename).exists():
                quality_metrics[metric_key] = True

        # Check syntax of Python files
        for py_file in code_path.glob("*.py"):
            try:
                with open(py_file, "r") as f:
                    compile(f.read(), str(py_file), "exec")
            except SyntaxError:
                quality_metrics["syntax_valid"] = False

        # Calculate overall score
        component_scores = {
            "has_model": 25,
            "has_training": 25,
            "has_data_loader": 15,
            "has_config": 10,
            "has_requirements": 10,
            "has_readme": 10,
            "syntax_valid": 5,
        }

        total_score = sum(
            component_scores[key]
            for key, value in quality_metrics.items()
            if isinstance(value, bool) and value
        )

        quality_metrics["overall_score"] = total_score

        return quality_metrics

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade.

        Args:
            score: Numeric score (0-100)

        Returns:
            Letter grade with description
        """
        if score >= 90:
            return "A (Excellent)"
        elif score >= 80:
            return "B (Good)"
        elif score >= 70:
            return "C (Acceptable)"
        elif score >= 50:
            return "D (Poor)"
        else:
            return "F (Failed)"

    def _save_evaluation(
        self, paper_id: str, test_type: str, evaluation: Dict[str, Any]
    ):
        """Save evaluation results to file.

        Args:
            paper_id: Paper identifier
            test_type: "test_a" or "test_b"
            evaluation: Evaluation results
        """
        # Create subdirectory for paper (used as key prefix)
        paper_subdir = paper_id.replace(":", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON
        json_key = f"{paper_subdir}/{test_type}_{ts}.json"
        self._storage.save_text(json.dumps(evaluation, indent=2), json_key)

        # Save markdown report
        if "report" in evaluation:
            report_key = f"{paper_subdir}/{test_type}_{ts}.md"
            self._storage.save_text(evaluation["report"], report_key)

    def generate_benchmark_report(self, test_type: str = "both") -> Dict[str, Any]:
        """Generate aggregate report across all evaluations.

        Args:
            test_type: "test_a", "test_b", or "both"

        Returns:
            Benchmark statistics
        """
        all_evaluations = []

        # Load all evaluations via StorageProvider (works for local and cloud)
        for key in self._storage.list_files("", "**/*.json"):
            # Skip benchmark files themselves
            if key.startswith("benchmark_"):
                continue
            text = self._storage.read_text(key)
            if not text:
                continue
            try:
                evaluation = json.loads(text)
                if (
                    test_type == "both"
                    or evaluation.get("test_type")
                    == test_type.split("_")[1].upper()
                ):
                    all_evaluations.append(evaluation)
            except Exception:
                continue

        if not all_evaluations:
            return {"error": "No evaluations found"}

        # Compute statistics
        total_count = len(all_evaluations)
        test_a_count = sum(1 for e in all_evaluations if e.get("test_type") == "A")
        test_b_count = sum(1 for e in all_evaluations if e.get("test_type") == "B")

        avg_score = (
            sum(e.get("reproducibility_score", 0) for e in all_evaluations)
            / total_count
        )
        success_count = sum(1 for e in all_evaluations if e.get("reproducible", False))
        success_rate = success_count / total_count * 100

        benchmark = {
            "total_papers_evaluated": total_count,
            "test_a_count": test_a_count,
            "test_b_count": test_b_count,
            "average_reproducibility_score": avg_score,
            "success_count": success_count,
            "success_rate_percent": success_rate,
            "grade_distribution": self._compute_grade_distribution(all_evaluations),
            "evaluations": all_evaluations,
        }

        # Save benchmark report
        benchmark_key = f"benchmark_{test_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self._storage.save_text(json.dumps(benchmark, indent=2), benchmark_key)

        return benchmark

    def _compute_grade_distribution(
        self, evaluations: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Compute distribution of grades.

        Args:
            evaluations: List of evaluation results

        Returns:
            Grade distribution
        """
        distribution = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}

        for evaluation in evaluations:
            grade = evaluation.get("grade", "F")[0]  # First letter
            if grade in distribution:
                distribution[grade] += 1

        return distribution
