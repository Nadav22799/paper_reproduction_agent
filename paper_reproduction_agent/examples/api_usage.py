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

    print("\n✅ Reproduction complete!")
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
        print("\n✅ Reproduction complete!")
    else:
        print(f"⚠️  PDF not found at {pdf_path}")


def example_use_individual_agents():
    """Example: Use individual agents separately."""
    print("\nExample 3: Using individual agents")
    print("=" * 60)

    from src.agents.paper_analyzer import PaperAnalyzerAgent
    from src.agents.code_searcher import CodeSearcherAgent

    # Initialize agents
    analyzer = PaperAnalyzerAgent()
    searcher = CodeSearcherAgent()

    # Analyze a paper
    arxiv_id = "2010.11929"  # Vision Transformer
    print(f"\n📄 Analyzing paper {arxiv_id}...")
    analysis = analyzer.analyze_paper(f"arxiv:{arxiv_id}")

    print(f"Paper title: {analysis.get('paper_metadata', {}).get('title', 'N/A')}")
    print(f"Algorithms found: {len(analysis.get('algorithms', []))}")

    # Search for implementations
    print("\n🔍 Searching for implementations...")
    paper_title = analysis.get('paper_metadata', {}).get('title', '')
    if paper_title:
        search_results = searcher.search_implementations(paper_title)
        print(f"Found {len(search_results.get('repositories', []))} repositories")


def example_verify_existing_code():
    """Example: Verify an existing implementation."""
    print("\nExample 4: Verifying existing code")
    print("=" * 60)

    from src.agents.code_verifier import CodeVerifierAgent

    verifier = CodeVerifierAgent()

    # Path to code you want to verify
    code_path = "./existing_implementation"

    # Expected results from paper
    paper_results = {
        "accuracy": 0.95,
        "f1_score": 0.93,
        "precision": 0.94,
    }

    if os.path.exists(code_path):
        print(f"Verifying code at {code_path}...")
        verification = verifier.verify_implementation(code_path, paper_results)

        print(f"\n{'✅' if verification['results_match_paper'] else '❌'} Verification result:")
        print(f"Execution successful: {verification['execution_successful']}")
        print(f"Results match paper: {verification['results_match_paper']}")
    else:
        print(f"⚠️  Code not found at {code_path}")


def example_reproduce_with_custom_llm():
    """Example: Use custom LLM configuration."""
    print("\nExample 5: Using custom LLM")
    print("=" * 60)

    from langchain_openai import ChatOpenAI

    # Use a different model or temperature
    custom_llm = ChatOpenAI(
        model="gpt-4-turbo-preview",
        temperature=0.0,  # More deterministic
    )

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
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OPENAI_API_KEY not set!")
        print("Please copy .env.example to .env and add your API key")
        return

    try:
        # Run examples
        # Uncomment the examples you want to run

        # example_reproduce_paper_from_arxiv()
        # example_reproduce_paper_from_pdf()
        example_use_individual_agents()
        # example_verify_existing_code()
        # example_reproduce_with_custom_llm()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
