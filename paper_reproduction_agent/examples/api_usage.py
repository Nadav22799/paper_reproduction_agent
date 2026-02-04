"""Example usage of the Paper Reproduction Agent system."""

import os
from dotenv import load_dotenv
from paper_reproduction_agent.src.orchestrator import PaperReproductionOrchestrator

# Load environment variables
load_dotenv()


def example_reproduce_paper_from_arxiv():
    """Example: Reproduce a paper from arXiv."""
    print("Example 1: Reproducing paper from arXiv ID")
    print("=" * 60)

    # Initialize orchestrator
    orchestrator = PaperReproductionOrchestrator()

    # Run workflow with arXiv ID
    # Example: Attention Is All You Need (Transformer paper)
    arxiv_id = "1706.03762"

    result = orchestrator.run(f"arxiv:{arxiv_id}")

    print("\nReproduction complete!")
    print(f"Status: {result['final_status']}")
    print(f"Results match paper: {result['results_match']}")


def example_reproduce_paper_from_pdf():
    """Example: Reproduce a paper from local PDF."""
    print("\nExample 2: Reproducing paper from PDF file")
    print("=" * 60)

    orchestrator = PaperReproductionOrchestrator()

    # Path to your PDF file
    pdf_path = "./papers/my_paper.pdf"

    if os.path.exists(pdf_path):
        result = orchestrator.run(pdf_path)
        print("\nReproduction complete!")
    else:
        print(f"PDF not found at {pdf_path}")


def example_use_individual_agents():
    """Example: Use individual agents separately."""
    print("\nExample 3: Using individual agents")
    print("=" * 60)

    from paper_reproduction_agent.src.agents.unified_paper_analyzer import UnifiedPaperAnalyzer
    from paper_reproduction_agent.src.utils.llm_factory import create_llm

    # Initialize analyzer with default LLM
    llm = create_llm()
    analyzer = UnifiedPaperAnalyzer(llm)

    # Analyze a paper
    arxiv_id = "2010.11929"  # Vision Transformer
    print(f"\nAnalyzing paper {arxiv_id}...")

    # Note: You'd need to fetch the paper content first
    # This is a simplified example
    paper_content = "Vision Transformer (ViT) paper content would go here..."
    analysis = analyzer.analyze_paper(paper_content, "Vision Transformer")

    print(f"Core contribution: {analysis.get('core_contribution', 'N/A')[:100]}...")
    print(f"Datasets found: {len(analysis.get('datasets', []))}")


def example_reproduce_with_custom_llm():
    """Example: Use custom LLM configuration via factory."""
    print("\nExample 4: Using custom LLM configuration")
    print("=" * 60)

    from paper_reproduction_agent.src.utils.llm_factory import create_llm

    # Use factory with custom temperature
    custom_llm = create_llm(temperature=0.0)  # More deterministic

    orchestrator = PaperReproductionOrchestrator(llm=custom_llm)

    # Run with custom LLM
    result = orchestrator.run("arxiv:2106.09685")  # LoRA paper

    print(f"Status: {result['final_status']}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("PAPER REPRODUCTION AGENT - EXAMPLES")
    print("=" * 60)

    # Check environment variables
    has_gemini = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    has_claude = bool(os.getenv("ANTHROPIC_API_KEY"))

    if not (has_gemini or has_claude):
        print("\nWarning: No LLM API key set!")
        print("Please set GOOGLE_API_KEY/GEMINI_API_KEY or ANTHROPIC_API_KEY")
        return

    try:
        # Run examples
        # Uncomment the examples you want to run

        # example_reproduce_paper_from_arxiv()
        # example_reproduce_paper_from_pdf()
        example_use_individual_agents()
        # example_reproduce_with_custom_llm()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
