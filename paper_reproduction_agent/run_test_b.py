"""Run Test B: Code Generation from Paper.

This script:
1. Extracts methodology from paper (without seeing original code)
2. Generates code implementation from methodology
3. Executes generated code
4. Validates results against paper
5. Generates evaluation report
"""

import argparse
import logging
from pathlib import Path

from src.agents.methodology_extractor_agent import MethodologyExtractorAgent
from src.agents.code_generator_agent import CodeGeneratorAgent
from src.agents.code_execution_agent import CodeExecutionAgent
from src.evaluation_framework import ReproductionEvaluator
from src.utils.llm_factory import create_llm
from src.tools.paper_tools import fetch_arxiv_paper


def main():
    parser = argparse.ArgumentParser(description="Test B: Code Generation from Paper")
    parser.add_argument("paper_id", help="Paper ID (arXiv ID)")
    parser.add_argument("--expected_results", required=True, help="JSON file with expected results")
    parser.add_argument("--output_dir", default="./test_b_output", help="Output directory for generated code")
    parser.add_argument("--framework", default="pytorch", choices=["pytorch", "tensorflow"], help="Framework to use")
    parser.add_argument("--execute", action="store_true", help="Execute generated code")
    parser.add_argument("--eval_dir", default="./evaluation_results", help="Evaluation results directory")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    logger.info(f"Starting Test B for paper: {args.paper_id}")

    # Initialize LLM
    llm = create_llm(temperature=0.1)

    # Initialize agents
    methodology_agent = MethodologyExtractorAgent(llm)
    code_gen_agent = CodeGeneratorAgent(llm)
    exec_agent = CodeExecutionAgent(llm) if args.execute else None

    # Initialize evaluator
    evaluator = ReproductionEvaluator(results_dir=args.eval_dir)

    # Load expected results
    import json
    with open(args.expected_results, 'r') as f:
        expected_results = json.load(f)

    logger.info(f"Expected results: {expected_results}")

    # Step 1: Fetch paper
    logger.info("=" * 60)
    logger.info("STEP 1: Fetching paper")
    logger.info("=" * 60)

    paper_data = fetch_arxiv_paper(args.paper_id)

    if "error" in paper_data:
        logger.error(f"Failed to fetch paper: {paper_data['error']}")
        return

    paper_text = paper_data.get("full_text", "")
    logger.info(f"Paper fetched: {len(paper_text)} characters")

    # Step 2: Extract methodology
    logger.info("=" * 60)
    logger.info("STEP 2: Extracting methodology (WITHOUT seeing original code)")
    logger.info("=" * 60)

    methodology_result = methodology_agent.extract_full_methodology(
        paper_text=paper_text,
        arxiv_id=args.paper_id
    )

    methodology = methodology_result["methodology"]
    logger.info("Methodology extracted successfully")
    logger.info(f"Methodology preview:\n{str(methodology)[:500]}...")

    # Step 3: Generate code
    logger.info("=" * 60)
    logger.info("STEP 3: Generating code from methodology")
    logger.info("=" * 60)

    output_path = Path(args.output_dir) / args.paper_id.replace(":", "_")
    output_path.mkdir(parents=True, exist_ok=True)

    generation_result = code_gen_agent.generate_full_implementation(
        methodology={"extracted_methodology": methodology},
        output_dir=str(output_path),
        framework=args.framework
    )

    if generation_result["status"] != "success":
        logger.error(f"Code generation failed: {generation_result['message']}")
        generation_metadata = {
            "status": "failed",
            "error": generation_result["message"]
        }
        actual_results = None
    else:
        logger.info(f"Code generated successfully at: {output_path}")
        generation_metadata = {
            "status": "success",
            "framework": args.framework,
            "output_path": str(output_path)
        }

        # Step 4: Execute generated code (optional)
        actual_results = None
        if args.execute:
            logger.info("=" * 60)
            logger.info("STEP 4: Executing generated code")
            logger.info("=" * 60)

            execution_result = exec_agent.execute_and_validate(
                repo_path=str(output_path),
                paper_results=expected_results,
                quick_validation=True,  # Always use quick validation for Test B
                timeout_minutes=60
            )

            logger.info(f"Execution status: {execution_result['status']}")

            # TODO: Extract actual results from execution
            actual_results = {}  # Placeholder
            generation_metadata["execution_status"] = execution_result["status"]
            generation_metadata["execution_message"] = execution_result["message"]
        else:
            logger.info("Skipping execution (use --execute to run generated code)")

    # Step 5: Evaluate
    logger.info("=" * 60)
    logger.info("STEP 5: Generating evaluation report")
    logger.info("=" * 60)

    evaluation = evaluator.evaluate_test_b(
        paper_id=args.paper_id,
        methodology={"extracted_methodology": methodology},
        generated_code_path=str(output_path),
        expected_results=expected_results,
        actual_results=actual_results,
        generation_metadata=generation_metadata
    )

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Overall Test B Score: {evaluation['overall_test_b_score']:.1f}/100")
    logger.info(f"  - Methodology Extraction: {evaluation['methodology_extraction_score']:.1f}/100")
    logger.info(f"  - Code Quality: {evaluation['code_quality_score']['overall_score']:.1f}/100")
    logger.info(f"  - Results Reproduction: {evaluation['reproducibility_score']:.1f}/100")
    logger.info(f"Grade: {evaluation['grade']}")
    logger.info(f"Reproducible: {evaluation['reproducible']}")
    logger.info(f"Results saved to: {args.eval_dir}/{args.paper_id.replace(':', '_')}/")

    if "report" in evaluation and evaluation["report"]:
        print("\n" + evaluation["report"])


if __name__ == "__main__":
    main()
