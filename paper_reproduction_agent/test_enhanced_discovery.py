#!/usr/bin/env python3
"""
Test script for Enhanced GitHub Repository Discovery.

This script tests the new discovery methods that find GitHub repos
for papers that don't have direct code links in their text.

Usage:
    python test_enhanced_discovery.py

Requirements:
    - GITHUB_TOKEN in environment (optional but recommended)
    - GEMINI_API_KEY or GOOGLE_API_KEY in environment (for web search)
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# TEST CASES: Papers WITHOUT GitHub URLs in text, but WITH known implementations
# =============================================================================
# Format: (arxiv_id, paper_title, expected_repo_substring)
# expected_repo_substring: Part of the expected GitHub URL to verify correct discovery

TEST_PAPERS = [
    # TODO: Add your 3 test papers here
    # Example format:
    # ("2301.12345", "Paper Title Here", "github.com/author/repo"),
    ("2405.10292", "Fine-Tuning Large Vision-Language Models as Decision-Making Agents via Reinforcement Learning", "github.com/mehrdadsl254/Rl4vlm1"),
    ("2410.03321", "VISUAL-O1: UNDERSTANDING AMBIGUOUS IN STRUCTIONS VIA MULTI-MODAL MULTI-TURN CHAIN-OF-THOUGHTS REASONING", "github.com/kodenii/Visual-O1"),
    ("2403.01232", "POLYNORMER: POLYNOMIAL-EXPRESSIVE GRAPH TRANSFORMER IN LINEAR TIME", "github.com/cornell-zhang/Polynormer"),
]


def test_github_arxiv_search(arxiv_id: str, expected_substring: str = None) -> dict:
    """Test GitHub code search for arXiv reference."""
    from tools.code_search_tools import search_github_for_arxiv_reference

    print(f"\n{'='*60}")
    print(f"Testing GitHub arXiv Search for: {arxiv_id}")
    print('='*60)

    results = search_github_for_arxiv_reference(arxiv_id)

    found_repos = []
    for r in results:
        if r.get("url"):
            found_repos.append(r["url"])
            print(f"  ✅ Found: {r['url']}")
            if r.get("match_file"):
                print(f"     Match in: {r['match_file']}")
            if r.get("stars"):
                print(f"     Stars: {r['stars']}")

    if not found_repos:
        if results and results[0].get("message"):
            print(f"  ℹ️  {results[0]['message']}")
        elif results and results[0].get("error"):
            print(f"  ❌ Error: {results[0]['error']}")
        else:
            print("  ❌ No repos found")

    # Check if expected repo was found
    success = False
    if expected_substring and found_repos:
        for repo in found_repos:
            if expected_substring.lower() in repo.lower():
                success = True
                print(f"\n  🎯 Expected repo FOUND: {expected_substring}")
                break
        if not success:
            print(f"\n  ⚠️  Expected repo NOT in results: {expected_substring}")

    return {
        "method": "github_arxiv_search",
        "arxiv_id": arxiv_id,
        "found_repos": found_repos,
        "success": success if expected_substring else len(found_repos) > 0
    }


def test_github_name_search(paper_title: str, expected_substring: str = None) -> dict:
    """Test GitHub search by paper/method name."""
    from tools.code_search_tools import search_github_by_paper_name

    print(f"\n{'='*60}")
    print(f"Testing GitHub Name Search for: {paper_title[:50]}...")
    print('='*60)

    results = search_github_by_paper_name(paper_title)

    found_repos = []
    for r in results:
        if r.get("url"):
            found_repos.append(r["url"])
            print(f"  {'✅' if r.get('is_exact_match') else '🔸'} Found: {r['url']}")
            if r.get("matched_term"):
                print(f"     Matched term: {r['matched_term']}")
            if r.get("stars"):
                print(f"     Stars: {r['stars']}")

    if not found_repos:
        if results and results[0].get("message"):
            print(f"  ℹ️  {results[0]['message']}")
        elif results and results[0].get("error"):
            print(f"  ❌ Error: {results[0]['error']}")
        else:
            print("  ❌ No repos found")

    # Check if expected repo was found
    success = False
    if expected_substring and found_repos:
        for repo in found_repos:
            if expected_substring.lower() in repo.lower():
                success = True
                print(f"\n  🎯 Expected repo FOUND: {expected_substring}")
                break
        if not success:
            print(f"\n  ⚠️  Expected repo NOT in results: {expected_substring}")

    return {
        "method": "github_name_search",
        "paper_title": paper_title,
        "found_repos": found_repos,
        "success": success if expected_substring else len(found_repos) > 0
    }


def test_web_search(paper_title: str, arxiv_id: str = None, expected_substring: str = None) -> dict:
    """Test web search with LLM evaluation."""
    from tools.code_search_tools import web_search_for_implementation

    print(f"\n{'='*60}")
    print(f"Testing Web Search for: {paper_title[:50]}...")
    print('='*60)

    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("  ⚠️  GEMINI_API_KEY not set - skipping web search test")
        return {
            "method": "web_search",
            "paper_title": paper_title,
            "found_repos": [],
            "success": False,
            "skipped": True
        }

    results = web_search_for_implementation(
        paper_title=paper_title,
        arxiv_id=arxiv_id
    )

    found_repos = []
    for r in results:
        if r.get("url"):
            found_repos.append(r["url"])
            print(f"  ✅ Found: {r['url']}")
            if r.get("confidence"):
                print(f"     Confidence: {r['confidence']}")
            if r.get("reason"):
                print(f"     Reason: {r['reason'][:80]}")

    if not found_repos:
        if results and results[0].get("message"):
            print(f"  ℹ️  {results[0]['message']}")
        elif results and results[0].get("error"):
            print(f"  ❌ Error: {results[0]['error']}")
        else:
            print("  ❌ No repos found")

    # Check if expected repo was found
    success = False
    if expected_substring and found_repos:
        for repo in found_repos:
            if expected_substring.lower() in repo.lower():
                success = True
                print(f"\n  🎯 Expected repo FOUND: {expected_substring}")
                break
        if not success:
            print(f"\n  ⚠️  Expected repo NOT in results: {expected_substring}")

    return {
        "method": "web_search",
        "paper_title": paper_title,
        "found_repos": found_repos,
        "success": success if expected_substring else len(found_repos) > 0
    }


def test_enhanced_discovery(arxiv_id: str, paper_title: str, authors: list = None, expected_substring: str = None) -> dict:
    """Test the full enhanced discovery pipeline (calls all methods in sequence)."""
    from tools.code_search_tools import (
        search_github_for_arxiv_reference,
        search_github_by_paper_name,
        web_search_for_implementation
    )

    print(f"\n{'='*60}")
    print(f"Testing Full Enhanced Discovery Pipeline")
    print(f"Paper: {paper_title[:50]}...")
    print(f"arXiv: {arxiv_id}")
    print('='*60)

    discovered_repos = []

    # Method 1: arXiv reference search
    if arxiv_id:
        print("\n  📍 Method 1: arXiv reference search...")
        try:
            results = search_github_for_arxiv_reference(arxiv_id)
            for r in results:
                if r.get("url") and r.get("confidence") == "high":
                    if r["url"] not in discovered_repos:
                        discovered_repos.append(r["url"])
                        print(f"     ✅ {r['url']}")
        except Exception as e:
            print(f"     ⚠️ Error: {str(e)[:50]}")

    # Method 2: Name search (if no results yet)
    if not discovered_repos and paper_title:
        print("\n  📍 Method 2: Paper name search...")
        try:
            results = search_github_by_paper_name(paper_title)
            for r in results:
                if r.get("url") and r.get("is_exact_match"):
                    if r["url"] not in discovered_repos:
                        discovered_repos.append(r["url"])
                        print(f"     ✅ {r['url']}")
        except Exception as e:
            print(f"     ⚠️ Error: {str(e)[:50]}")

    # Method 3: Web search (if no results yet)
    if not discovered_repos and paper_title:
        print("\n  📍 Method 3: Web search...")
        try:
            results = web_search_for_implementation(paper_title, arxiv_id, authors)
            for r in results:
                if r.get("url") and r.get("confidence") == "high":
                    if r["url"] not in discovered_repos:
                        discovered_repos.append(r["url"])
                        print(f"     ✅ {r['url']}")
        except Exception as e:
            print(f"     ⚠️ Error: {str(e)[:50]}")

    print(f"\n📚 Total repos discovered: {len(discovered_repos)}")
    for repo in discovered_repos:
        print(f"  • {repo}")

    # Check if expected repo was found
    success = False
    if expected_substring and discovered_repos:
        for repo in discovered_repos:
            if expected_substring.lower() in repo.lower():
                success = True
                print(f"\n🎯 Expected repo FOUND: {expected_substring}")
                break
        if not success:
            print(f"\n⚠️  Expected repo NOT in results: {expected_substring}")

    return {
        "method": "enhanced_discovery",
        "arxiv_id": arxiv_id,
        "paper_title": paper_title,
        "found_repos": discovered_repos,
        "success": success if expected_substring else len(discovered_repos) > 0
    }


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("🧪 ENHANCED GITHUB REPOSITORY DISCOVERY - TEST SUITE")
    print("="*70)

    if not TEST_PAPERS:
        print("\n⚠️  No test papers defined!")
        print("Please add papers to TEST_PAPERS list in this script.")
        print("\nFormat: (arxiv_id, paper_title, expected_repo_substring)")
        print('Example: ("2106.09685", "LoRA: Low-Rank Adaptation...", "microsoft/LoRA")')
        return

    all_results = []

    for arxiv_id, paper_title, expected_repo in TEST_PAPERS:
        print(f"\n\n{'#'*70}")
        print(f"# TEST PAPER: {arxiv_id}")
        print(f"# Title: {paper_title[:60]}...")
        print(f"# Expected: {expected_repo}")
        print('#'*70)

        # Test 1: GitHub arXiv search
        result1 = test_github_arxiv_search(arxiv_id, expected_repo)
        all_results.append(result1)

        # Test 2: GitHub name search
        result2 = test_github_name_search(paper_title, expected_repo)
        all_results.append(result2)

        # Test 3: Web search (if API key available)
        result3 = test_web_search(paper_title, arxiv_id, expected_repo)
        all_results.append(result3)

        # Test 4: Full enhanced discovery pipeline
        result4 = test_enhanced_discovery(arxiv_id, paper_title, None, expected_repo)
        all_results.append(result4)

    # Summary
    print("\n\n" + "="*70)
    print("📊 TEST RESULTS SUMMARY")
    print("="*70)

    total_tests = len(all_results)
    passed = sum(1 for r in all_results if r.get("success"))
    skipped = sum(1 for r in all_results if r.get("skipped"))
    failed = total_tests - passed - skipped

    print(f"\nTotal Tests: {total_tests}")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  ⏭️  Skipped: {skipped}")

    # Per-paper summary
    print("\n" + "-"*50)
    print("Per-Paper Results:")
    print("-"*50)

    for arxiv_id, paper_title, expected_repo in TEST_PAPERS:
        paper_results = [r for r in all_results if r.get("arxiv_id") == arxiv_id or r.get("paper_title") == paper_title]
        paper_passed = any(r.get("success") for r in paper_results)
        status = "✅ FOUND" if paper_passed else "❌ NOT FOUND"
        print(f"\n{arxiv_id}: {status}")
        print(f"  Title: {paper_title[:50]}...")
        print(f"  Expected: {expected_repo}")

        for r in paper_results:
            method = r.get("method", "unknown")
            if r.get("skipped"):
                print(f"    {method}: ⏭️ SKIPPED")
            elif r.get("success"):
                print(f"    {method}: ✅ SUCCESS ({len(r.get('found_repos', []))} repos)")
            else:
                print(f"    {method}: ❌ FAILED")

    print("\n" + "="*70)

    # Return exit code
    return 0 if failed == 0 else 1


def test_single_paper(arxiv_id: str, paper_title: str = None, expected_repo: str = None):
    """Test a single paper interactively."""
    print(f"\n🔍 Testing discovery for: {arxiv_id}")

    if not paper_title:
        # Try to fetch title from arXiv
        try:
            import arxiv
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(search.results())
            paper_title = paper.title
            print(f"📄 Title: {paper_title}")
        except Exception as e:
            print(f"⚠️ Could not fetch paper title: {e}")
            paper_title = f"Paper {arxiv_id}"

    # Run all four methods
    test_github_arxiv_search(arxiv_id, expected_repo)
    test_github_name_search(paper_title, expected_repo)
    test_web_search(paper_title, arxiv_id, expected_repo)
    test_enhanced_discovery(arxiv_id, paper_title, None, expected_repo)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Enhanced GitHub Repository Discovery")
    parser.add_argument("--arxiv", type=str, help="Single arXiv ID to test")
    parser.add_argument("--title", type=str, help="Paper title (optional, fetched from arXiv if not provided)")
    parser.add_argument("--expected", type=str, help="Expected repo substring to verify")

    args = parser.parse_args()

    if args.arxiv:
        # Test single paper
        test_single_paper(args.arxiv, args.title, args.expected)
    else:
        # Run all predefined tests
        exit_code = run_all_tests()
        sys.exit(exit_code)
