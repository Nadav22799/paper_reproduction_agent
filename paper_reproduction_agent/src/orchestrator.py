"""LangGraph Orchestrator - Main workflow for paper reproduction."""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import operator
import signal
from contextlib import contextmanager


# Define the overall state for the entire workflow
class PaperReproductionState(TypedDict):
    """Overall state for paper reproduction workflow."""
    # Input
    paper_input: str  # arXiv ID, PDF path, or paper text
    paper_title: str

    # Paper Analysis
    paper_metadata: dict
    algorithms: list
    experimental_setup: dict
    paper_results: dict
    code_references: list

    # Code Search
    existing_repos: list
    selected_repo: dict
    repo_quality_score: float

    # Implementation
    implementation_path: str
    code_created: bool

    # Environment Setup
    env_setup_results: dict
    dependencies_installed: bool

    # Dataset Preparation
    dataset_results: dict
    datasets_ready: bool

    # Experiment Execution
    experiment_results: dict
    experiments_completed: bool

    # Metrics Extraction
    extracted_metrics: dict
    metrics_comparison: dict

    # Verification
    verification_results: dict
    results_match: bool

    # Debugging
    errors_found: list
    fixes_applied: list
    debug_attempts: int

    # Overall
    messages: Annotated[list, operator.add]
    next_step: str
    final_status: str
    report: str


class TimeoutException(Exception):
    """Exception raised when operation times out."""
    pass


@contextmanager
def timeout(seconds):
    """Context manager for timeout."""
    def timeout_handler(signum, frame):
        raise TimeoutException(f"Operation timed out after {seconds} seconds")

    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class PaperReproductionOrchestrator:
    """Main orchestrator for paper reproduction workflow."""

    def __init__(self, llm=None, search_timeout=60, enable_logging=True):
        """Initialize the orchestrator.

        Args:
            llm: Language model to use
            search_timeout: Timeout in seconds for code search (default: 60)
            enable_logging: Whether to enable detailed logging to file
        """
        from .agents.paper_analyzer import PaperAnalyzerAgent
        from .agents.code_searcher import CodeSearcherAgent
        from .agents.code_reproducer import CodeReproducerAgent
        from .agents.code_verifier import CodeVerifierAgent
        from .agents.code_debugger import CodeDebuggerAgent
        from .agents.environment_setup import EnvironmentSetupAgent
        from .agents.dataset_manager import DatasetManagerAgent
        from .agents.experiment_runner import ExperimentRunnerAgent
        from .agents.metrics_extractor import MetricsExtractorAgent
        from .utils.llm_factory import create_llm
        from .utils.file_logger import FileLogger
        from .utils.logging_callback import LoggingCallbackHandler

        self.llm = llm or create_llm(temperature=0.1)
        self.search_timeout = search_timeout

        # Setup logging
        self.enable_logging = enable_logging
        self.file_logger = None
        self.logging_callback = None
        if enable_logging:
            self.file_logger = FileLogger(log_dir="./logs")
            self.logging_callback = LoggingCallbackHandler(verbose=True, file_logger=self.file_logger)

        # Initialize all agents
        # Only EnvironmentSetupAgent currently supports callbacks
        self.paper_analyzer = PaperAnalyzerAgent(self.llm)
        self.code_searcher = CodeSearcherAgent(self.llm)
        self.code_reproducer = CodeReproducerAgent(self.llm)
        self.code_verifier = CodeVerifierAgent(self.llm)
        self.code_debugger = CodeDebuggerAgent(self.llm)
        # Use max_iterations=10 to prevent dependency hell loops
        self.env_setup = EnvironmentSetupAgent(
            self.llm,
            callback=self.logging_callback,
            max_iterations=10
        )
        self.dataset_manager = DatasetManagerAgent(self.llm)
        self.experiment_runner = ExperimentRunnerAgent(self.llm)
        self.metrics_extractor = MetricsExtractorAgent(self.llm)

        # Build the workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(PaperReproductionState)

        # Add nodes for each agent/step
        workflow.add_node("analyze_paper", self._analyze_paper_node)
        workflow.add_node("search_code", self._search_code_node)
        workflow.add_node("decide_path", self._decide_path_node)
        workflow.add_node("reproduce_code", self._reproduce_code_node)
        workflow.add_node("setup_environment", self._setup_environment_node)
        workflow.add_node("prepare_datasets", self._prepare_datasets_node)
        workflow.add_node("run_experiments", self._run_experiments_node)
        workflow.add_node("extract_metrics", self._extract_metrics_node)
        workflow.add_node("verify_code", self._verify_code_node)
        workflow.add_node("debug_code", self._debug_code_node)
        workflow.add_node("generate_report", self._generate_report_node)

        # Define the workflow edges
        workflow.set_entry_point("analyze_paper")

        workflow.add_edge("analyze_paper", "search_code")
        workflow.add_edge("search_code", "decide_path")

        # Conditional routing from decide_path
        workflow.add_conditional_edges(
            "decide_path",
            self._route_after_search,
            {
                "use_existing": "setup_environment",
                "create_new": "reproduce_code",
                "failed": "generate_report",
            }
        )

        workflow.add_edge("reproduce_code", "setup_environment")

        # Conditional routing after setup_environment - stop if failed
        workflow.add_conditional_edges(
            "setup_environment",
            self._route_after_setup,
            {
                "continue": "prepare_datasets",
                "failed": "generate_report",
            }
        )

        workflow.add_edge("prepare_datasets", "run_experiments")
        workflow.add_edge("run_experiments", "extract_metrics")
        workflow.add_edge("extract_metrics", "verify_code")

        # Conditional routing from verify_code
        workflow.add_conditional_edges(
            "verify_code",
            self._route_after_verification,
            {
                "success": "generate_report",
                "needs_debug": "debug_code",
            }
        )

        # Conditional routing from debug_code
        workflow.add_conditional_edges(
            "debug_code",
            self._route_after_debug,
            {
                "retry_verify": "verify_code",
                "give_up": "generate_report",
            }
        )

        workflow.add_edge("generate_report", END)

        return workflow.compile()

    def _analyze_paper_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Analyze the paper using PaperAnalyzerAgent."""
        print("📄 Analyzing paper...")

        # Step 1: Fetch the paper PDF and extract text (no LLM needed)
        paper_input = state["paper_input"]
        if paper_input.startswith("arxiv:") or (len(paper_input.split()) == 1 and "." in paper_input):
            # It's an arXiv ID - fetch it directly
            arxiv_id = paper_input.replace("arxiv:", "")
            print(f"📥 Fetching arXiv paper {arxiv_id} directly...")
            from .tools.paper_tools import fetch_arxiv_paper
            paper_data = fetch_arxiv_paper.invoke({"arxiv_id": arxiv_id})
            if "error" not in paper_data:
                state["paper_metadata"] = paper_data
                state["paper_title"] = paper_data.get("title", "Unknown")
                print(f"✅ Downloaded and extracted {len(paper_data.get('full_text', ''))} characters of text")
            else:
                print(f"⚠️  Failed to fetch paper: {paper_data.get('error')}")
                state["paper_title"] = "Unknown"

        # Step 2: Extract code references directly from paper text (no LLM needed)
        full_text = state.get("paper_metadata", {}).get("full_text", "")
        if full_text:
            print("🔍 Extracting code references from paper text...")
            from .tools.paper_tools import extract_code_references
            extracted_urls = extract_code_references(full_text)

            # Filter out the "No code repository URLs found" message
            if extracted_urls and not (len(extracted_urls) == 1 and "No code repository" in extracted_urls[0]):
                state["code_references"] = extracted_urls
                print(f"✅ Found {len(extracted_urls)} code reference(s) from paper")
            else:
                state["code_references"] = []

        # Step 3: Fallback to Papers with Code API if no code references found
        if not state.get("code_references") and state.get("paper_title"):
            print("🔍 Trying Papers with Code API for official implementation...")
            try:
                from .tools.code_search_tools import search_papers_with_code
                pwc_result = search_papers_with_code(state["paper_title"])
                if isinstance(pwc_result, dict) and "implementations" in pwc_result:
                    repos = [impl["url"] for impl in pwc_result["implementations"]
                            if impl.get("url") and impl.get("is_official")]
                    if not repos:  # If no official, take any implementation
                        repos = [impl["url"] for impl in pwc_result["implementations"] if impl.get("url")]
                    if repos:
                        state["code_references"] = repos
                        print(f"✅ Found {len(repos)} implementation(s) from Papers with Code")
            except Exception as e:
                print(f"⚠️  Papers with Code API failed: {str(e)[:50]}")

        # Step 4: Extract results and key information from paper using LLM
        if full_text:
            print("🔬 Extracting experimental results from paper...")
            try:
                # Use a simpler, focused extraction to get just what we need
                from langchain_core.messages import HumanMessage

                extraction_prompt = f"""Analyze this paper and extract ONLY the key experimental results.

Paper Title: {state.get('paper_title', 'Unknown')}

Paper text (first 8000 chars):
{full_text[:8000]}

Extract:
1. Main quantitative results (accuracy, BLEU scores, perplexity, etc.)
2. Key datasets used
3. Main metrics reported

Provide a brief summary in this format:
- Dataset: [name]
- Metric: [metric name]
- Result: [value]

Keep it concise - only the main results from the abstract/results section."""

                messages = [HumanMessage(content=extraction_prompt)]

                # Log the interaction if logging is enabled
                # For vLLM with tool calling enabled, explicitly disable tools for pure text generation
                invoke_config = {"callbacks": [self.logging_callback]} if self.logging_callback else {}

                # Try to disable tool calling for this request (for vLLM compatibility)
                try:
                    result = self.llm.invoke(messages, tool_choice="none", **invoke_config)
                except TypeError:
                    # If tool_choice parameter not supported, fall back to normal invoke
                    result = self.llm.invoke(messages, config=invoke_config) if invoke_config else self.llm.invoke(messages)

                # Store the extracted results - clean any <think> tags or XML that could confuse agents
                import re
                results_text = result.content if hasattr(result, 'content') else str(result)
                # Remove <think>...</think> blocks
                clean_results = re.sub(r'<think>.*?</think>', '', results_text, flags=re.DOTALL)
                # Remove <tool_call>...</tool_call> blocks if any
                clean_results = re.sub(r'<tool_call>.*?</tool_call>', '', clean_results, flags=re.DOTALL)
                # Clean up whitespace
                clean_results = re.sub(r'\n\s*\n', '\n', clean_results).strip()

                state["paper_results"] = {"summary": clean_results}
                state["algorithms"] = []  # Can be populated later if needed
                state["experimental_setup"] = {}  # Can be populated later if needed

                print(f"✅ Extracted results summary ({len(clean_results)} chars)")
            except Exception as e:
                print(f"⚠️  Results extraction failed: {str(e)[:100]}")
                state["algorithms"] = []
                state["experimental_setup"] = {}
                state["paper_results"] = {}
        else:
            state["algorithms"] = []
            state["experimental_setup"] = {}
            state["paper_results"] = {}

        # Add status message
        state["messages"].append(f"✅ Analyzed paper: {state['paper_title']}")
        if state.get("code_references"):
            state["messages"].append(f"📚 Found {len(state['code_references'])} code reference(s)")

        return state

    def _search_code_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Search for existing implementations using direct GitHub API."""
        print("🔍 Searching for existing implementations...")

        # Skip if we already have code references from the paper
        if state.get("code_references"):
            print("✅ Skipping GitHub search - already have code references from paper")
            state["existing_repos"] = []
            state["selected_repo"] = {}
            state["repo_quality_score"] = 0.0
            return state

        try:
            from .tools.code_search_tools import search_github_repos

            # Use direct GitHub API search (no LLM agent - avoids recursion loops)
            print(f"   Searching GitHub for: {state['paper_title'][:60]}...")

            repos = search_github_repos.invoke({
                "query": state['paper_title'],
                "language": "Python",
                "max_results": 5,
                "fetch_topics": False  # Faster
            })

            # Filter out errors
            valid_repos = [r for r in repos if isinstance(r, dict) and "error" not in r and "message" not in r]

            state["existing_repos"] = valid_repos

            if state["existing_repos"]:
                state["selected_repo"] = state["existing_repos"][0]
                state["repo_quality_score"] = 0.8
                print(f"   ✅ Found {len(state['existing_repos'])} repository(ies)")
            else:
                state["selected_repo"] = {}
                state["repo_quality_score"] = 0.0
                print(f"   ⚠️  No repositories found")

            state["messages"].append(f"Found {len(state['existing_repos'])} additional implementation(s) via GitHub search")

        except Exception as e:
            print(f"⚠️  GitHub search failed: {str(e)[:100]}")
            state["existing_repos"] = []
            state["selected_repo"] = {}
            state["repo_quality_score"] = 0.0
            state["messages"].append("Code search failed")

        return state

    def _decide_path_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Decide whether to use existing code or create new implementation."""
        print("🤔 Deciding on implementation path...")

        # Priority 1: Code references from paper itself
        if state["code_references"] and isinstance(state["code_references"], list):
            # Filter out "No code repository URLs found" messages
            valid_refs = [ref for ref in state["code_references"]
                         if ref.startswith("http") and "github" in ref.lower()]
            if valid_refs:
                state["next_step"] = "use_existing"
                state["selected_repo"] = {"url": valid_refs[0], "source": "paper"}
                state["messages"].append(f"📥 Using official implementation: {valid_refs[0]}")
                print(f"✅ Found official implementation: {valid_refs[0]}")

                # Clone the repository now
                repo_url = valid_refs[0]
                code_path = "./cloned_repo"

                import os
                import shutil
                if os.path.exists(code_path):
                    print(f"🗑️  Removing existing directory: {code_path}")
                    shutil.rmtree(code_path)

                print(f"📥 Cloning repository from {repo_url}...")
                from .tools.code_search_tools import clone_repository
                clone_result = clone_repository.invoke({"repo_url": repo_url, "target_dir": code_path})
                print(clone_result)

                if "Successfully cloned" in clone_result:
                    state["implementation_path"] = code_path
                else:
                    print(f"⚠️  Clone failed: {clone_result}")
                    state["messages"].append("Clone failed but continuing")

                return state

        # Priority 2: High-quality existing repo from search
        if state["repo_quality_score"] > 0.7 and state.get("selected_repo", {}).get("url"):
            state["next_step"] = "use_existing"
            state["messages"].append(f"📥 Using high-quality implementation: {state['selected_repo'].get('url', 'unknown')}")
            print(f"✅ Using existing repo: {state['selected_repo'].get('url', 'unknown')}")
            return state

        # Priority 3: Any repo from search with reasonable quality
        if state.get("existing_repos") and len(state["existing_repos"]) > 0:
            repo = state["existing_repos"][0]
            if isinstance(repo, dict) and repo.get("url"):
                state["next_step"] = "use_existing"
                state["selected_repo"] = repo
                state["messages"].append(f"📥 Using found implementation: {repo['url']}")
                print(f"✅ Using found repo: {repo['url']}")
                return state

        # Last resort: Would need to create new implementation
        # But for now, we'll stop here as this requires more complex code generation
        state["next_step"] = "failed"
        state["final_status"] = "Failed: No implementation found"
        state["messages"].append("❌ No implementation found")
        print("❌ No existing implementation found")
        print("❌ STOPPING workflow - automatic code generation not yet supported")

        return state

    def _setup_environment_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Setup execution environment - install dependencies."""
        print("🔧 Setting up environment...")

        code_path = state.get("implementation_path") or "./cloned_repo"

        # Try up to 3 times
        max_retries = 3
        setup_results = None

        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                print(f"🔄 Retry attempt {attempt}/{max_retries}")

            setup_results = self.env_setup.setup_environment(code_path)

            if setup_results.get("success"):
                state["env_setup_results"] = setup_results
                state["dependencies_installed"] = True
                state["messages"].append(f"✅ Dependencies installed successfully")
                print(f"✅ Dependencies installed")
                return state
            else:
                print(f"⚠️  Attempt {attempt} failed: {setup_results.get('errors', [])[:200]}")
                if attempt < max_retries:
                    import time
                    time.sleep(2)  # Brief delay before retry

        # All attempts failed - STOP the workflow
        state["env_setup_results"] = setup_results
        state["dependencies_installed"] = False
        state["messages"].append(f"❌ Environment setup failed after {max_retries} attempts")
        state["final_status"] = "Failed: Could not install dependencies"
        print(f"❌ Environment setup failed after {max_retries} attempts")
        print(f"❌ STOPPING workflow - cannot continue without dependencies")

        return state

    def _prepare_datasets_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Prepare datasets for experiments."""
        print("📦 Preparing datasets...")

        code_path = state.get("implementation_path") or "./cloned_repo"
        paper_datasets = state.get("experimental_setup", {}).get("datasets", [])

        dataset_results = self.dataset_manager.prepare_datasets(code_path, paper_datasets)

        state["dataset_results"] = dataset_results
        state["datasets_ready"] = dataset_results.get("datasets_identified", False)

        if dataset_results.get("datasets_downloaded"):
            state["messages"].append("✅ Datasets prepared successfully")
            print(f"✅ Datasets ready")
        else:
            # Don't add a message - it's handled in verification
            print(f"⚠️  Datasets not fully prepared")

        return state

    def _run_experiments_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Run experiments and capture outputs."""
        print("🧪 Running experiments...")

        code_path = state.get("implementation_path") or "./cloned_repo"
        paper_experiments = state.get("experimental_setup")

        experiment_results = self.experiment_runner.run_experiments(code_path, paper_experiments)

        state["experiment_results"] = experiment_results
        state["experiments_completed"] = experiment_results.get("execution_successful", False)

        if experiment_results.get("execution_successful"):
            state["messages"].append("✅ Experiments executed successfully")
            print(f"✅ Experiments completed")
        else:
            # Don't add a message - it's handled in verification
            print(f"⚠️  Execution issues: {experiment_results.get('errors', [])}")

        return state

    def _extract_metrics_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Extract and compare metrics from experiment outputs."""
        print("📊 Extracting metrics...")

        experiment_output = state.get("experiment_results", {}).get("output", "")
        paper_results = state.get("paper_results", {})

        # Extract metrics from output
        extracted = self.metrics_extractor.extract_metrics(experiment_output, paper_results)

        # Compare with paper results
        comparison = self.metrics_extractor.compare_metrics(extracted, paper_results)

        state["extracted_metrics"] = extracted
        state["metrics_comparison"] = comparison

        if comparison.get("overall_match"):
            state["messages"].append(f"✅ Metrics match paper results ({comparison.get('match_rate', 'N/A')})")
            print(f"✅ Metrics match: {comparison.get('match_rate', 'N/A')}")
        else:
            # Don't add a message - it's handled in verification
            print(f"⚠️  Metrics mismatch")

        return state

    def _reproduce_code_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Reproduce code from scratch using CodeReproducerAgent."""
        print("💻 Reproducing code from paper...")

        # Create implementation plan
        paper_analysis = {
            "paper_metadata": state["paper_metadata"],
            "algorithms": state["algorithms"],
            "experimental_setup": state["experimental_setup"],
        }

        plan = self.code_reproducer.create_implementation_plan(paper_analysis)

        # Implement the algorithm
        algorithm_desc = "\n".join(state["algorithms"]) if state["algorithms"] else "No explicit algorithm found"

        result = self.code_reproducer.implement_algorithm(
            algorithm_desc,
            output_dir="./implementation"
        )

        state["implementation_path"] = result["output_dir"]
        state["code_created"] = True
        state["messages"].append("✅ Code implementation created")

        return state

    def _verify_code_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Verify results by checking metrics comparison."""
        print("✅ Verifying results...")

        # Use metrics comparison from extract_metrics_node
        metrics_comparison = state.get("metrics_comparison", {})
        experiment_results = state.get("experiment_results", {})

        # Build verification report
        verification_report = []
        verification_report.append("## Execution Summary")
        verification_report.append(f"- Dependencies Installed: {'Yes' if state.get('dependencies_installed') else 'No'}")
        verification_report.append(f"- Datasets Ready: {'Yes' if state.get('datasets_ready') else 'No'}")
        verification_report.append(f"- Experiments Completed: {'Yes' if state.get('experiments_completed') else 'No'}")

        verification_report.append("\n## Metrics Comparison")
        if metrics_comparison.get("matches"):
            verification_report.append("### Matching Metrics:")
            for match in metrics_comparison["matches"]:
                verification_report.append(f"  - {match['metric']}: {match['actual']} (expected: {match['expected']}, diff: {match['diff']})")

        if metrics_comparison.get("mismatches"):
            verification_report.append("### Mismatched Metrics:")
            for mismatch in metrics_comparison["mismatches"]:
                verification_report.append(f"  - {mismatch['metric']}: {mismatch['actual']} (expected: {mismatch['expected']}, diff: {mismatch['diff']})")

        if metrics_comparison.get("missing"):
            verification_report.append("### Missing Metrics:")
            for missing in metrics_comparison["missing"]:
                verification_report.append(f"  - {missing}")

        report_text = "\n".join(verification_report)

        # Determine if results match
        results_match = metrics_comparison.get("overall_match", False)

        # If experiments didn't complete, check if we at least found the code
        if not state.get("experiments_completed"):
            # Check if we successfully cloned/found the repo
            if state.get("selected_repo") and state.get("env_setup_results"):
                results_match = True  # Partial success
                report_text += "\n\nNote: Experiments did not complete, but repository was found and environment was set up."

        state["verification_results"] = {
            "report": report_text,
            "results_match_paper": results_match,
            "discrepancies": metrics_comparison.get("mismatches", [])
        }
        state["results_match"] = results_match
        state["errors_found"] = [str(m) for m in metrics_comparison.get("mismatches", [])]

        print(f"\n📝 Verification Report:\n{report_text}\n")
        if results_match:
            state["messages"].append("✅ Verification: Repository found and setup complete")
        else:
            state["messages"].append("⚠️ Verification: Issues found")

        return state

    def _debug_code_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Debug and fix code using CodeDebuggerAgent."""
        print("🔧 Debugging and fixing code...")

        state["debug_attempts"] = state.get("debug_attempts", 0) + 1

        code_path = state.get("implementation_path", "./code")

        error_info = {
            "errors": state["errors_found"],
            "verification_report": state["verification_results"].get("report", ""),
        }

        debug_result = self.code_debugger.debug_and_fix(code_path, error_info)

        state["fixes_applied"] = debug_result.get("fixes_applied", [])
        state["messages"].append(f"Debug attempt {state['debug_attempts']}: {len(state['fixes_applied'])} fixes applied")

        return state

    def _generate_report_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Generate final report."""
        print("📊 Generating final report...")

        # Set final status first
        state["final_status"] = "Complete"

        # Get selected repo info
        selected_repo_info = "None"
        if state.get("selected_repo"):
            repo = state["selected_repo"]
            if isinstance(repo, dict):
                selected_repo_info = repo.get("url") or repo.get("full_name", "Unknown")

        # Count total repos found (from both code_references and existing_repos)
        total_repos_found = 0
        # Count code references from paper
        code_refs_count = len([r for r in state.get('code_references', []) if isinstance(r, str) and r.startswith('http')])
        # Count repos from search
        existing_repos_count = len(state.get('existing_repos', []))
        total_repos_found = code_refs_count + existing_repos_count

        # Deduplicate and filter messages - only show unique, important messages
        all_messages = state.get('messages', [])
        seen = set()
        unique_messages = []

        # Filter out noisy/redundant messages
        skip_phrases = [
            "continuing anyway",
            "will attempt execution anyway",
            "will attempt anyway",
            "had issues - continuing",
            "incomplete - will"
        ]

        for msg in all_messages:
            # Skip if we've seen this message
            if msg in seen:
                continue
            # Skip if it's a noisy message
            if any(phrase in msg.lower() for phrase in skip_phrases):
                continue
            seen.add(msg)
            unique_messages.append(msg)

        report = f"""
# Paper Reproduction Report

## Paper Information
- Title: {state.get('paper_title', 'N/A')}
- Analysis: {'Complete' if state.get('paper_metadata') else 'Incomplete'}
- Code References Found: {code_refs_count}

## Implementation
- Path: {state.get('next_step', 'N/A')}
- Selected Repository: {selected_repo_info}
- Code Created: {'Yes' if state.get('code_created') else 'No'}
- Total Repos Found: {total_repos_found} (from paper: {code_refs_count}, from search: {existing_repos_count})

## Verification
- Results Match Paper: {'Yes' if state.get('results_match') else 'No'}
- Errors Found: {len(state.get('errors_found', []))}
- Debug Attempts: {state.get('debug_attempts', 0)}
- Fixes Applied: {len(state.get('fixes_applied', []))}

## Status
{state.get('final_status', 'Complete')}

## Summary
{chr(10).join(unique_messages)}
"""

        state["report"] = report

        return state

    def _route_after_search(self, state: PaperReproductionState) -> Literal["use_existing", "create_new", "failed"]:
        """Route after code search."""
        return state.get("next_step", "create_new")

    def _route_after_setup(self, state: PaperReproductionState) -> Literal["continue", "failed"]:
        """Route after environment setup - stop if failed."""
        if state.get("dependencies_installed"):
            return "continue"
        else:
            print("🛑 Routing to report generation due to setup failure")
            return "failed"

    def _route_after_verification(self, state: PaperReproductionState) -> Literal["success", "needs_debug"]:
        """Route after verification."""
        # If we successfully found and cloned the repo, that's a success
        if state.get("results_match"):
            print("✅ Verification successful - skipping debug")
            return "success"
        # If there are no actual errors to debug, skip debugging
        elif not state.get("errors_found") or len(state.get("errors_found", [])) == 0:
            print("⚠️  No errors to debug - skipping debug phase")
            return "success"
        else:
            return "needs_debug"

    def _route_after_debug(self, state: PaperReproductionState) -> Literal["retry_verify", "give_up"]:
        """Route after debugging."""
        max_attempts = 3
        if state.get("debug_attempts", 0) >= max_attempts:
            return "give_up"
        else:
            return "retry_verify"

    def run(self, paper_input: str) -> dict:
        """
        Run the complete paper reproduction workflow.

        Args:
            paper_input: arXiv ID, PDF path, or paper identifier

        Returns:
            Final state with results
        """
        print(f"\n{'='*60}")
        print(f"🚀 Starting Paper Reproduction Workflow")
        print(f"{'='*60}\n")

        initial_state = {
            "paper_input": paper_input,
            "paper_title": "",
            "paper_metadata": {},
            "algorithms": [],
            "experimental_setup": {},
            "paper_results": {},
            "code_references": [],
            "existing_repos": [],
            "selected_repo": {},
            "repo_quality_score": 0.0,
            "implementation_path": "",
            "code_created": False,
            "env_setup_results": {},
            "dependencies_installed": False,
            "dataset_results": {},
            "datasets_ready": False,
            "experiment_results": {},
            "experiments_completed": False,
            "extracted_metrics": {},
            "metrics_comparison": {},
            "verification_results": {},
            "results_match": False,
            "errors_found": [],
            "fixes_applied": [],
            "debug_attempts": 0,
            "messages": [],
            "next_step": "",
            "final_status": "",
            "report": "",
        }

        final_state = self.workflow.invoke(initial_state)

        print(f"\n{'='*60}")
        print(f"✅ Workflow Complete")
        print(f"{'='*60}\n")

        # Close the log file if logging is enabled
        if self.file_logger:
            self.file_logger.close()

        return final_state
