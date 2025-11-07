#!/usr/bin/env python
"""Simple script to run paper reproduction for a specific paper."""

# IMPORTANT: Set multiprocessing start method BEFORE any other imports
# This fixes CUDA forking issues with vLLM
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src directory to Python path for module imports
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Load environment variables from the script's directory
env_path = script_dir / ".env"
load_dotenv(dotenv_path=env_path)


def check_environment():
    """Check if at least one LLM is configured (local or API)."""
    # Check for local LLM first
    use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"

    if use_local:
        backend = os.getenv("LOCAL_LLM_BACKEND", "ollama")
        model = os.getenv("LOCAL_LLM_MODEL", "gemma2:2b")
        print(f"✅ Using Local LLM ({backend}: {model})")
        return True

    # Check for API providers
    has_gemini = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_groq = bool(os.getenv("GROQ_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

    if not (has_gemini or has_openai or has_groq or has_anthropic):
        print("❌ No LLM configured!")
        print("\nOption 1 - Use Local LLM (No API key needed!):")
        print("  Set USE_LOCAL_LLM=true in .env file")
        print("\nOption 2 - Use API Provider:")
        print("  Set at least one of:")
        print("  - GOOGLE_API_KEY or GEMINI_API_KEY (Gemini)")
        print("  - OPENAI_API_KEY")
        print("  - GROQ_API_KEY")
        print("  - ANTHROPIC_API_KEY")
        return False

    # Show which API provider will be used (matches priority in llm_factory.py)
    if has_gemini:
        print("✅ Using Google Gemini")
    elif has_openai:
        print("✅ Using OpenAI")
    elif has_groq:
        print("✅ Using Groq (fast and free)")
    elif has_anthropic:
        print("✅ Using Anthropic")

    return True


def run_paper_reproduction(paper_input: str):
    """
    Run paper reproduction workflow.

    Args:
        paper_input: Can be:
            - arXiv ID: "1706.03762" or "arxiv:1706.03762"
            - PDF path: "/path/to/paper.pdf"
            - Paper title: "Attention Is All You Need"
    """
    from src.orchestrator import PaperReproductionOrchestrator

    print("\n" + "="*70)
    print("🚀 PAPER REPRODUCTION AGENT")
    print("="*70)
    print(f"\n📄 Input: {paper_input}\n")

    # Initialize orchestrator
    orchestrator = PaperReproductionOrchestrator()

    # Format input
    if paper_input.isdigit() or (len(paper_input.split()) == 1 and "." in paper_input):
        # Looks like arXiv ID
        if not paper_input.startswith("arxiv:"):
            paper_input = f"arxiv:{paper_input}"

    # Run workflow
    try:
        result = orchestrator.run(paper_input)

        print("\n" + "="*70)
        print("📊 FINAL REPORT")
        print("="*70)
        print(result['report'])

        return result

    except Exception as e:
        print(f"\n❌ Error during reproduction: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point."""
    print("Paper Reproduction Agent - Quick Run")
    print("="*70)

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Get paper input
    if len(sys.argv) > 1:
        # Paper provided as command line argument
        paper_input = " ".join(sys.argv[1:])
    else:
        # Interactive mode
        print("\nHow to specify a paper:")
        print("  1. arXiv ID: 1706.03762")
        print("  2. arXiv URL: arxiv:2010.11929")
        print("  3. PDF path: /path/to/paper.pdf")
        print("  4. Paper title: Vision Transformer")
        print()

        paper_input = input("Enter paper (arXiv ID, PDF path, or title): ").strip()

        if not paper_input:
            print("❌ No input provided. Exiting.")
            sys.exit(1)

    # Run reproduction
    result = run_paper_reproduction(paper_input)

    if result:
        print("\n" + "="*70)
        print("✅ Workflow completed successfully!")
        print("="*70)
    else:
        print("\n❌ Reproduction workflow failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
