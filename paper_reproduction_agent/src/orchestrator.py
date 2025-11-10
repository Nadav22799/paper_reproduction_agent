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

    # Agent Context History - NEW!
    agent_contexts: dict  # Stores summary from each agent for next agents to use

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
        """Analyze the paper using UnifiedPaperAnalyzer - simplified single-pass approach."""
        print("📄 Analyzing paper...")

        # Step 1: Fetch the paper PDF and extract text (do it ourselves, no @tool issues)
        paper_input = state["paper_input"]
        if paper_input.startswith("arxiv:") or (len(paper_input.split()) == 1 and "." in paper_input):
            # It's an arXiv ID - fetch it directly
            arxiv_id = paper_input.replace("arxiv:", "")
            print(f"📥 Fetching arXiv paper {arxiv_id} directly...")

            try:
                import arxiv
                import os
                from PyPDF2 import PdfReader

                # Fetch paper metadata
                search = arxiv.Search(id_list=[arxiv_id])
                paper = next(search.results())

                # Download PDF
                download_dir = "./downloads"
                os.makedirs(download_dir, exist_ok=True)
                pdf_path = paper.download_pdf(dirpath=download_dir)

                # Extract text from PDF
                reader = PdfReader(pdf_path)
                full_text = ""
                for page in reader.pages:
                    full_text += page.extract_text() + "\n"

                # Store metadata
                state["paper_metadata"] = {
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "abstract": paper.summary,
                    "published": paper.published.isoformat(),
                    "arxiv_id": arxiv_id,
                    "pdf_url": paper.pdf_url,
                    "full_text": full_text,
                    "categories": paper.categories,
                }
                state["paper_title"] = paper.title
                print(f"✅ Downloaded and extracted {len(full_text)} characters of text")

            except Exception as e:
                print(f"⚠️  Failed to fetch paper: {str(e)}")
                return state

        # Step 2: Use unified analyzer to extract EVERYTHING in one LLM call
        full_text = state.get("paper_metadata", {}).get("full_text", "")
        if full_text:
            print("🔬 Analyzing paper with unified analyzer...")
            from .agents.unified_paper_analyzer import UnifiedPaperAnalyzer

            analyzer = UnifiedPaperAnalyzer(self.llm)
            analysis = analyzer.analyze_paper(full_text, state.get("paper_title", "Unknown"))

            # Store results in state
            state["code_references"] = analysis.get("github_repos", [])
            state["paper_results"] = analysis.get("results_to_reproduce", {})
            state["experimental_setup"] = {
                "datasets": analysis.get("datasets", []),
                "implementation_details": analysis.get("implementation_details", "")
            }

            # Store agent context for future agents
            state["agent_contexts"]["paper_analyzer"] = analysis.get("context_summary", "")

            # Print detailed analysis results
            print("\n" + "="*80)
            print("📊 UNIFIED ANALYZER FINDINGS")
            print("="*80)

            # GitHub Repositories
            print(f"\n📚 GitHub Repositories Found: {len(state['code_references'])}")
            for i, repo in enumerate(state['code_references'], 1):
                print(f"   {i}. {repo}")
            if not state['code_references']:
                print("   (none found)")

            # Datasets
            datasets = analysis.get('datasets', [])
            print(f"\n📊 Datasets Identified: {len(datasets)}")
            if datasets:
                print(f"   {', '.join(datasets)}")
            else:
                print("   (none identified)")

            # Core Contribution
            print(f"\n💡 Core Contribution:")
            core = analysis.get('core_contribution', 'N/A')
            if core:
                # Wrap text at 70 chars
                import textwrap
                wrapped = textwrap.fill(core, width=74, initial_indent="   ", subsequent_indent="   ")
                print(wrapped)
            else:
                print("   (not extracted)")

            # Results to Reproduce
            metrics = state["paper_results"].get("metrics", [])
            print(f"\n🎯 Results to Reproduce: {len(metrics)} metric(s)")
            if metrics:
                for m in metrics[:5]:  # Show first 5
                    dataset = m.get('dataset', 'Unknown')
                    metric = m.get('metric', 'Unknown')
                    value = m.get('value', 'Unknown')
                    print(f"   - {dataset}: {metric} = {value}")
                if len(metrics) > 5:
                    print(f"   ... and {len(metrics) - 5} more")
            else:
                # Show summary if no structured metrics
                summary = state["paper_results"].get("summary", "")
                if summary:
                    print("   Summary from paper:")
                    summary_lines = summary.split('\n')[:3]  # First 3 lines
                    for line in summary_lines:
                        if line.strip():
                            print(f"   {line.strip()[:74]}")
                else:
                    print("   (no metrics extracted)")

            # Implementation Details
            impl_details = analysis.get('implementation_details', '')
            if impl_details:
                print(f"\n🔧 Implementation Details:")
                impl_lines = impl_details.split('\n')[:2]  # First 2 lines
                for line in impl_lines:
                    if line.strip():
                        print(f"   {line.strip()[:74]}")

            print("\n" + "="*80 + "\n")

            # Try Papers with Code API as fallback if no repos found
            if not state["code_references"] and state.get("paper_title"):
                print("🔍 No repos in paper, trying Papers with Code API...")
                try:
                    import requests
                    # Papers with Code API - do it ourselves
                    base_url = "https://paperswithcode.com/api/v1/papers/"
                    search_url = f"{base_url}?title={requests.utils.quote(state['paper_title'])}"
                    response = requests.get(search_url, timeout=10)

                    if response.status_code == 200:
                        data = response.json()
                        if data.get("results"):
                            paper = data["results"][0]
                            paper_id = paper.get("id")

                            # Get implementations
                            impl_url = f"https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/"
                            impl_response = requests.get(impl_url, timeout=10)

                            if impl_response.status_code == 200:
                                impl_data = impl_response.json()
                                implementations = impl_data.get("results", [])

                                # Get official repos first, otherwise any repo
                                repos = [impl["url"] for impl in implementations if impl.get("url") and impl.get("is_official")]
                                if not repos:
                                    repos = [impl["url"] for impl in implementations if impl.get("url")]

                                if repos:
                                    state["code_references"] = repos
                                    print(f"✅ Found {len(repos)} implementation(s) from Papers with Code")
                except Exception as e:
                    print(f"⚠️  Papers with Code API failed: {str(e)[:50]}")
        else:
            state["code_references"] = []
            state["paper_results"] = {}
            state["experimental_setup"] = {}

        # Add status message
        state["messages"].append(f"✅ Analyzed paper: {state['paper_title']}")
        if state.get("code_references"):
            state["messages"].append(f"📚 Found {len(state['code_references'])} code reference(s)")

        return state

    def _search_code_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """REMOVED: Generic GitHub search is unreliable and finds wrong repos."""
        print("🔍 Skipping generic GitHub search (relies on paper extraction only)")

        # We already extracted repos in _analyze_paper_node using:
        # 1. Regex extraction from paper text
        # 2. Papers with Code API fallback
        # No need for generic GitHub search - it finds wrong repos!

        state["existing_repos"] = []
        state["selected_repo"] = {}
        state["repo_quality_score"] = 0.0

        return state

    def _decide_path_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Decide whether to use existing code or create new implementation."""
        print("🤔 Deciding on implementation path...")

        import os
        import shutil
        import subprocess

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
                repo_marker = os.path.join(code_path, ".repo_url")

                # Check if we need to clone
                need_clone = True
                if os.path.exists(code_path):
                    # Check if it's the same repo
                    if os.path.exists(repo_marker):
                        try:
                            with open(repo_marker, 'r') as f:
                                existing_url = f.read().strip()
                            if existing_url == repo_url:
                                print(f"✅ Repository already cloned: {repo_url}")
                                need_clone = False
                            else:
                                print(f"🔄 Different repo detected (was: {existing_url})")
                                print(f"🗑️  Removing old repository...")
                                shutil.rmtree(code_path)
                        except Exception as e:
                            print(f"⚠️  Could not read repo marker: {e}")
                            print(f"🗑️  Removing directory to be safe...")
                            shutil.rmtree(code_path)
                    else:
                        print(f"🗑️  No repo marker found, removing directory...")
                        shutil.rmtree(code_path)

                if need_clone:
                    print(f"📥 Cloning repository from {repo_url}...")
                    # Clone repo ourselves (no @tool issues)
                    try:
                        result = subprocess.run(
                            ["git", "clone", repo_url, code_path],
                            capture_output=True,
                            text=True,
                            timeout=300
                        )
                        if result.returncode == 0:
                            clone_result = f"Successfully cloned repository to {code_path}"
                        else:
                            clone_result = f"Clone failed: {result.stderr}"
                    except Exception as e:
                        clone_result = f"Error cloning repository: {str(e)}"

                    print(clone_result)

                    if "Successfully cloned" in clone_result:
                        # Store repo URL marker
                        try:
                            with open(repo_marker, 'w') as f:
                                f.write(repo_url)
                        except Exception as e:
                            print(f"⚠️  Could not write repo marker: {e}")
                        state["implementation_path"] = code_path
                    else:
                        print(f"⚠️  Clone failed: {clone_result}")
                        state["messages"].append("Clone failed but continuing")
                else:
                    state["implementation_path"] = code_path

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

        # Get context from previous agents
        paper_context = state.get("agent_contexts", {}).get("paper_analyzer", "")

        dataset_results = self.dataset_manager.prepare_datasets(
            code_path, paper_datasets, agent_context=paper_context
        )

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

        # Build comprehensive context for experiment runner
        paper_results = state.get("paper_results", {})
        datasets = state.get("experimental_setup", {}).get("datasets", [])

        # Create detailed context with results to reproduce
        detailed_context = state.get("agent_contexts", {}).get("paper_analyzer", "")

        # Add detailed results if available
        if paper_results:
            detailed_context += "\n\nResults to Reproduce:\n"
            if isinstance(paper_results, dict):
                # From unified analyzer
                metrics = paper_results.get("metrics", [])
                for m in metrics:
                    detailed_context += f"  - {m.get('dataset', 'Unknown')}: {m.get('metric', 'Unknown')} = {m.get('value', 'Unknown')}\n"

                # Add summary if no structured metrics
                if not metrics and "summary" in paper_results:
                    detailed_context += f"{paper_results['summary']}\n"

        # Add dataset info
        if datasets:
            detailed_context += f"\nDatasets mentioned: {', '.join(datasets)}\n"

        experiment_results = self.experiment_runner.run_experiments(
            code_path, paper_experiments=None, agent_context=detailed_context
        )

        state["experiment_results"] = experiment_results
        state["experiments_completed"] = experiment_results.get("execution_successful", False)

        # Store experiment runner context for next agents
        exp_context = f"Ran: {experiment_results.get('executed_command', 'unknown command')}"
        if experiment_results.get("execution_successful"):
            exp_context += " (succeeded)"
        else:
            exp_context += " (failed)"
        state["agent_contexts"]["experiment_runner"] = exp_context

        if experiment_results.get("execution_successful"):
            state["messages"].append("✅ Experiments executed successfully")
            print(f"✅ Experiments completed")
            if experiment_results.get("used_readme"):
                print(f"   ✅ Used README command!")
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

        # Transform paper_results to flat dictionary format for comparison
        # unified_paper_analyzer returns: {"results_to_reproduce": {"metrics": [...]}}
        # compare_metrics expects: {"metric_name": value}
        expected_metrics_dict = {}
        if isinstance(paper_results, dict):
            results_to_repro = paper_results.get("results_to_reproduce", {})
            if isinstance(results_to_repro, dict):
                metrics_list = results_to_repro.get("metrics", [])
                if isinstance(metrics_list, list):
                    for m in metrics_list:
                        if isinstance(m, dict):
                            dataset = m.get("dataset", "").replace(" ", "_").replace("-", "_")
                            metric = m.get("metric", "").replace(" ", "_").replace("-", "_")
                            value = m.get("value", "")

                            # Create key like "MNLI_Accuracy" or just "Accuracy" if no dataset
                            if dataset and metric:
                                key = f"{dataset}_{metric}"
                            elif metric:
                                key = metric
                            else:
                                continue

                            expected_metrics_dict[key] = value

        # Compare with paper results (using transformed flat dictionary)
        comparison = self.metrics_extractor.compare_metrics(extracted, expected_metrics_dict)

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
        experiments_completed = state.get("experiments_completed", False)

        # More nuanced success criteria
        if experiments_completed:
            # Experiments ran - check if metrics match
            if results_match:
                success_level = "full"  # Full success
                status_msg = "✅ Verification: Experiments completed and results match paper"
            else:
                success_level = "partial"  # Partial - experiments ran but metrics don't match
                status_msg = "⚠️ Verification: Experiments completed but results don't match paper"
                report_text += "\n\n⚠️ Experiments completed but results differ from paper claims."
        else:
            # Experiments didn't run
            sanity_check_passed = state.get("experiment_results", {}).get("sanity_check_passed", False)

            if sanity_check_passed:
                # Sanity check passed but main experiment didn't run
                success_level = "minimal"
                results_match = False  # Mark as not matching since we didn't reproduce
                status_msg = "⚠️ Verification: Only sanity check completed, main experiment did not run"
                report_text += "\n\n⚠️ Only sanity check completed. Main experiments were not executed."
            elif state.get("selected_repo") and state.get("dependencies_installed"):
                # At least we found and set up the code
                success_level = "setup_only"
                results_match = False  # Definitely not matching - no experiments ran
                status_msg = "⚠️ Verification: Repository found and environment set up, but no experiments ran"
                report_text += "\n\n⚠️ Experiments did not run. Repository was found and environment was set up."
            else:
                # Complete failure
                success_level = "failed"
                results_match = False
                status_msg = "❌ Verification: Reproduction failed"
                report_text += "\n\n❌ Reproduction failed - could not run experiments."

        state["verification_results"] = {
            "report": report_text,
            "results_match_paper": results_match,
            "success_level": success_level,
            "discrepancies": metrics_comparison.get("mismatches", [])
        }
        state["results_match"] = results_match
        state["errors_found"] = [str(m) for m in metrics_comparison.get("mismatches", [])]

        print(f"\n📝 Verification Report:\n{report_text}\n")
        state["messages"].append(status_msg)

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

        # Determine final status based on verification results
        success_level = state.get("verification_results", {}).get("success_level", "failed")
        status_map = {
            "full": "✅ Complete - Results Match Paper",
            "partial": "⚠️ Partial - Experiments Ran, Results Differ",
            "minimal": "⚠️ Minimal - Only Sanity Check Passed",
            "setup_only": "⚠️ Setup Only - No Experiments Ran",
            "failed": "❌ Failed"
        }
        state["final_status"] = status_map.get(success_level, "❌ Failed")

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
            "agent_contexts": {},  # NEW: Agent context history
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
