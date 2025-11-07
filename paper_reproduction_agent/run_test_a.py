"""Run Test A: Code Execution & Reproduction.

This script:
1. Sets up the environment for a cloned repository
2. Resolves dependencies
3. Executes the code
4. Validates results against paper
5. Generates evaluation report
"""

import argparse
import logging
from pathlib import Path

from src.agents.environment_setup_agent import EnvironmentSetupAgent
from src.agents.code_execution_agent import CodeExecutionAgent
from src.evaluation_framework import ReproductionEvaluator
from src.utils.llm_factory import create_llm


def main():
    parser = argparse.ArgumentParser(description="Test A: Code Execution & Reproduction")
    parser.add_argument("paper_id", help="Paper ID (arXiv ID or identifier)")
    parser.add_argument("--repo_path", required=True, help="Path to cloned repository")
    parser.add_argument("--expected_results", required=True, help="JSON file with expected results")
    parser.add_argument("--quick_validation", action="store_true", help="Run quick validation (reduced epochs)")
    parser.add_argument("--timeout_minutes", type=int, default=120, help="Execution timeout in minutes")
    parser.add_argument("--output_dir", default="./evaluation_results", help="Output directory for results")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    logger.info(f"Starting Test A for paper: {args.paper_id}")
    logger.info(f"Repository: {args.repo_path}")

    # Initialize LLM
    llm = create_llm(temperature=0.1)

    # Initialize agents
    env_agent = EnvironmentSetupAgent(llm)
    exec_agent = CodeExecutionAgent(llm)

    # Initialize evaluator
    evaluator = ReproductionEvaluator(results_dir=args.output_dir)

    # Load expected results
    import json
    with open(args.expected_results, 'r') as f:
        expected_results = json.load(f)

    logger.info(f"Expected results: {expected_results}")

    # Step 1: Setup environment
    logger.info("=" * 60)
    logger.info("STEP 1: Setting up environment")
    logger.info("=" * 60)

    env_setup_result = env_agent.setup_environment(args.repo_path)

    if env_setup_result["status"] != "success":
        logger.error(f"Environment setup failed: {env_setup_result['message']}")
        return

    logger.info("Environment setup completed successfully")

    # Step 2: Execute and validate
    logger.info("=" * 60)
    logger.info("STEP 2: Executing code and validating results")
    logger.info("=" * 60)

    execution_result = exec_agent.execute_and_validate(
        repo_path=args.repo_path,
        paper_results=expected_results,
        quick_validation=args.quick_validation,
        timeout_minutes=args.timeout_minutes
    )

    logger.info(f"Execution status: {execution_result['status']}")
    logger.info(f"Results match: {execution_result['results_match']}")

    # Parse actual results from execution
    # (This would need to extract metrics from execution_result)
    # For now, using a placeholder
    actual_results = {}  # TODO: Extract from execution_result

    # Step 3: Evaluate
    logger.info("=" * 60)
    logger.info("STEP 3: Generating evaluation report")
    logger.info("=" * 60)

    execution_metadata = {
        "repo_path": args.repo_path,
        "quick_validation": args.quick_validation,
        "timeout_minutes": args.timeout_minutes,
        "execution_status": execution_result["status"],
        "env_setup_status": env_setup_result["status"]
    }

    evaluation = evaluator.evaluate_test_a(
        paper_id=args.paper_id,
        repo_url=args.repo_path,  # Would be actual URL in real scenario
        expected_results=expected_results,
        actual_results=actual_results,
        execution_metadata=execution_metadata
    )

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Reproducibility Score: {evaluation['reproducibility_score']:.1f}/100")
    logger.info(f"Grade: {evaluation['grade']}")
    logger.info(f"Reproducible: {evaluation['reproducible']}")
    logger.info(f"Results saved to: {args.output_dir}/{args.paper_id.replace(':', '_')}/")

    print("\n" + evaluation["report"])


if __name__ == "__main__":
    main()
