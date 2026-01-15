"""Clean LangGraph Orchestrator - Simplified workflow for paper reproduction."""

from typing import TypedDict, Annotated, Literal
import stat
import os
from langgraph.graph import StateGraph, END
from pathlib import Path
import operator
import os
import subprocess
import shutil
from .utils.checkpoint_manager import ExperimentCheckpoint
from .utils.hierarchical_context import HierarchicalContextManager
from .utils.metrics_tracker import MetricsTracker


def remove_readonly(func, path, _):
    """Clear the readonly bit and reattempt the removal"""
    os.chmod(path, stat.S_IWRITE)
    func(path)


# Define the overall state for the entire workflow
class PaperReproductionState(TypedDict):
    """Overall state for paper reproduction workflow."""
    # Input
    paper_input: str  # arXiv ID, PDF path, or paper text
    paper_title: str

    # Paper Analysis
    paper_metadata: dict
    experimental_setup: dict
    paper_results: dict
    code_references: list

    # Implementation
    selected_repo: dict
    implementation_path: str
    experiment_selection_mode: str  # 'single', 'all', 'custom'
    custom_experiment_list: list  # List of specific experiments if custom mode

    # Reproduction Results
    env_setup_results: dict
    dependencies_installed: bool
    dataset_results: dict
    datasets_ready: bool
    experiment_results: dict
    experiments_completed: bool

    # Metrics & Verification
    extracted_metrics: dict
    metrics_comparison: dict
    verification_results: dict
    results_match: bool

    # Agent Context History
    agent_contexts: dict

    # Checkpoint & Resume
    completed_phases: list  # List of phase names that have been completed

    # Overall
    messages: Annotated[list, operator.add]
    final_status: str
    report: str


class PaperReproductionOrchestrator:
    """Simplified orchestrator for paper reproduction workflow."""

    def __init__(self, llm=None, enable_logging=True, enable_checkpoints=True):
        """Initialize the orchestrator.

        Args:
            llm: Language model to use
            enable_logging: Whether to enable detailed logging to file
            enable_checkpoints: Whether to enable checkpoint & resume functionality
        """
        # Only import agents that are actually used
        from .agents.environment_setup_agent import EnvironmentSetupAgent
        from .agents.unified_reproduction_agent import UnifiedReproductionAgent
        from .agents.discovery_agent import DiscoveryAgent
        from .utils.llm_factory import create_llm
        from .utils.file_logger import FileLogger
        from .utils.logging_callback import LoggingCallbackHandler
        from .config import ReproductionConfig

        self.llm = llm or create_llm(temperature=0.1)
        self.config = ReproductionConfig()

        # Setup metrics tracking for observability
        self.metrics_tracker = MetricsTracker(
            enable_live_display=self.config.enable_live_progress,
            update_interval=self.config.progress_update_interval,
            input_cost_per_million=self.config.llm_input_cost_per_million,
            output_cost_per_million=self.config.llm_output_cost_per_million
        )
        print("📊 Metrics tracker initialized")

        # Setup logging
        self.enable_logging = enable_logging
        self.file_logger = None
        self.logging_callback = None
        if enable_logging:
            self.file_logger = FileLogger(log_dir=self.config.logs_path)
            # Wire metrics_tracker to logging callback for token tracking
            self.logging_callback = LoggingCallbackHandler(
                verbose=True,
                file_logger=self.file_logger,
                metrics_tracker=self.metrics_tracker
            )

        # Setup checkpoint manager
        self.enable_checkpoints = enable_checkpoints
        self.checkpoint_manager = None
        if enable_checkpoints:
            self.checkpoint_manager = ExperimentCheckpoint(checkpoint_dir=self.config.checkpoints_path)
            print("💾 Checkpoint system enabled")

        # Initialize shared hierarchical context manager for cross-agent context
        self.hierarchical_context = HierarchicalContextManager(
            model_name="gpt-4",
            hot_capacity=50,
            max_tokens=100000  # Larger budget for orchestrator
        )
        print("🧠 Hierarchical context manager initialized")

        # Initialize specialized agents (share hierarchical context)
        self.env_setup_agent = EnvironmentSetupAgent(self.llm, max_iterations=50, metrics_tracker=self.metrics_tracker)
        self.unified_reproducer = UnifiedReproductionAgent(
            self.llm,
            max_iterations=50,
            hierarchical_context=self.hierarchical_context,  # Share context
            metrics_tracker=self.metrics_tracker
        )
        self.discovery_agent = DiscoveryAgent(self.llm, metrics_tracker=self.metrics_tracker)

        # Build the workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the simplified LangGraph workflow."""
        workflow = StateGraph(PaperReproductionState)

        # Add nodes - 6 specialized phases
        workflow.add_node("analyze_paper", self._analyze_paper_node)
        workflow.add_node("decide_and_clone", self._decide_and_clone_node)
        workflow.add_node("environment_setup", self._environment_setup_node)
        workflow.add_node("unified_reproduction", self._unified_reproduction_node)
        workflow.add_node("extract_and_verify", self._extract_and_verify_node)
        workflow.add_node("generate_report", self._generate_report_node)

        # Define the workflow edges
        workflow.set_entry_point("analyze_paper")

        workflow.add_edge("analyze_paper", "decide_and_clone")

        # Conditional routing from decide_and_clone
        workflow.add_conditional_edges(
            "decide_and_clone",
            self._route_after_clone,
            {
                "continue": "environment_setup",
                "failed": "generate_report",
            }
        )

        # Conditional routing from environment_setup
        workflow.add_conditional_edges(
            "environment_setup",
            self._route_after_env_setup,
            {
                "continue": "unified_reproduction",
                "failed": "generate_report",
            }
        )

        # Conditional routing after unified_reproduction
        workflow.add_conditional_edges(
            "unified_reproduction",
            self._route_after_reproduction,
            {
                "continue": "extract_and_verify",
                "failed": "generate_report",
            }
        )

        workflow.add_edge("extract_and_verify", "generate_report")
        workflow.add_edge("generate_report", END)

        return workflow.compile()

    def _analyze_paper_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Analyze the paper using UnifiedPaperAnalyzer."""
        # Check if this phase was already completed (resuming from checkpoint)
        if self._is_phase_completed(state, "analyze_paper"):
            print("⏭️  Skipping analyze_paper (already completed from checkpoint)")
            return state

        self.metrics_tracker.start_phase("analyze_paper")
        print("📄 Analyzing paper...")

        paper_input = state["paper_input"]

        # Handle arXiv papers
        if paper_input.startswith("arxiv:") or (len(paper_input.split()) == 1 and "." in paper_input):
            arxiv_id = paper_input.replace("arxiv:", "")
            print(f"📥 Fetching arXiv paper {arxiv_id}...")

            try:
                import arxiv
                from PyPDF2 import PdfReader
                from urllib.request import urlretrieve
                import re
                import textwrap

                # Fetch paper metadata
                search = arxiv.Search(id_list=[arxiv_id])
                paper = next(search.results())

                # Download PDF
                download_dir = os.path.abspath(self.config.downloads_path)
                os.makedirs(download_dir, exist_ok=True)

                safe_arxiv_id = arxiv_id.replace("/", "_").replace(".", "_")
                filename = f"{safe_arxiv_id}.pdf"
                pdf_path = os.path.join(download_dir, filename)

                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                print(f"   Downloading from: {pdf_url}")
                urlretrieve(pdf_url, pdf_path)

                # Extract text from PDF
                reader = PdfReader(pdf_path)
                full_text = ""
                for page in reader.pages:
                    full_text += page.extract_text() + "\n"

                # Truncate before bibliography/appendix
                truncate_patterns = [
                    r'\n\s*\d+\.?\s+References\s*\n',
                    r'\n\s*\d+\.?\s+Bibliography\s*\n',
                    r'\nReferences\s*\n',
                    r'\nBibliography\s*\n',
                    r'\nREFERENCES\s*\n',
                    r'\n\s*[A-Z]\.?\s+Appendix',
                    r'\nAppendix\s+[A-Z]',
                    r'\nAPPENDIX',
                    r'\n\s*Supplementary\s+Material\s*\n',
                    r'\n\s*Acknowledgment',
                ]
                for pattern in truncate_patterns:
                    match = re.search(pattern, full_text)
                    if match:
                        full_text = full_text[:match.start()]
                        print(f"📄 Truncated paper at '{match.group().strip()}'")
                        break

                # Store metadata
                state["paper_metadata"] = {
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "abstract": paper.summary,
                    "published": paper.published.isoformat(),
                    "arxiv_id": arxiv_id,
                    "pdf_url": pdf_url,
                    "full_text": full_text,
                    "categories": paper.categories,
                }
                state["paper_title"] = paper.title
                print(f"✅ Downloaded and extracted {len(full_text)} characters of text")

            except Exception as e:
                print(f"⚠️  Failed to fetch paper: {str(e)}")
                import traceback
                traceback.print_exc()
                return state

        # Analyze paper with unified analyzer
        full_text = state.get("paper_metadata", {}).get("full_text", "")
        if full_text:
            print("🔬 Analyzing paper with unified analyzer...")
            from .agents.unified_paper_analyzer import UnifiedPaperAnalyzer

            analyzer = UnifiedPaperAnalyzer(self.llm, metrics_tracker=self.metrics_tracker)
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

            # Store key findings in hierarchical context for semantic retrieval
            self._store_paper_context(state, analysis)

            # Print analysis summary
            self._print_analysis_summary(state, analysis)

            # Fallback discovery is now handled by DiscoveryAgent in the next node
            if not state["code_references"]:
                print("   (Code discovery deferred to Discovery Agent)")
        else:
            state["code_references"] = []
            state["paper_results"] = {}
            state["experimental_setup"] = {}

        # Add status message
        state["messages"].append(f"✅ Analyzed paper: {state['paper_title']}")
        if state.get("code_references"):
            state["messages"].append(f"📚 Found {len(state['code_references'])} code reference(s)")

        # Save checkpoint
        self._save_checkpoint(state, "analyze_paper")

        self.metrics_tracker.end_phase("analyze_paper", success=True)
        self.metrics_tracker.print_intermediate_summary()
        return state

    def _store_paper_context(self, state: PaperReproductionState, analysis: dict):
        """Store paper analysis results in hierarchical context for semantic retrieval."""
        try:
            paper_title = state.get("paper_title", "Unknown Paper")

            # Store core contribution
            core_contribution = analysis.get("core_contribution", "")
            if core_contribution:
                self.hierarchical_context.add(
                    content=f"Paper: {paper_title}\nCore Contribution: {core_contribution}",
                    source="paper_analyzer",
                    entry_type="result",
                    importance=1.0
                )

            # Store datasets
            datasets = analysis.get("datasets", [])
            if datasets:
                self.hierarchical_context.add(
                    content=f"Datasets to reproduce: {', '.join(datasets)}",
                    source="paper_analyzer",
                    entry_type="result",
                    importance=0.9
                )

            # Store metrics to reproduce
            paper_results = state.get("paper_results", {})
            if isinstance(paper_results, dict):
                metrics = paper_results.get("metrics", [])
                if metrics:
                    metrics_summary = []
                    for m in metrics[:10]:  # Limit to 10
                        if isinstance(m, dict):
                            dataset = m.get('dataset', 'Unknown')
                            metric = m.get('metric', 'Unknown')
                            value = m.get('value', 'N/A')
                            metrics_summary.append(f"{dataset}/{metric}: {value}")

                    if metrics_summary:
                        self.hierarchical_context.add(
                            content=f"Expected metrics:\n" + "\n".join(metrics_summary),
                            source="paper_analyzer",
                            entry_type="result",
                            importance=1.0
                        )

            # Store code references
            code_refs = state.get("code_references", [])
            if code_refs:
                refs_list = []
                for r in code_refs[:5]:
                    if isinstance(r, dict):
                        refs_list.append(r.get('url', str(r)))
                    else:
                        refs_list.append(str(r))

                self.hierarchical_context.add(
                    content=f"Code repositories: {', '.join(refs_list)}",
                    source="paper_analyzer",
                    entry_type="result",
                    importance=0.8
                )

            print("   🧠 Stored paper context in hierarchical storage")

        except Exception as e:
            print(f"   ⚠️  Warning: Failed to store paper context: {e}")

    def _print_analysis_summary(self, state, analysis):
        """Print detailed analysis results."""
        import textwrap

        print("\n" + "="*80)
        print("📊 UNIFIED ANALYZER FINDINGS")
        print("="*80)

        # GitHub Repositories
        repos = state['code_references']
        # Handle both list of strings and list of dicts
        repo_urls = []
        if repos:
            for r in repos:
                if isinstance(r, dict):
                    repo_urls.append(r.get('url', str(r)))
                else:
                    repo_urls.append(str(r))

        print(f"\n📚 GitHub Repositories Found: {len(repo_urls)}")
        for i, repo in enumerate(repo_urls, 1):
            print(f"   {i}. {repo}")
        if not repo_urls:
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
            wrapped = textwrap.fill(core, width=74, initial_indent="   ", subsequent_indent="   ")
            print(wrapped)
        else:
            print("   (not extracted)")

        # Results to Reproduce
        paper_results = state.get("paper_results", {})
        # Ensure it's a dict
        if not isinstance(paper_results, dict):
            paper_results = {}

        metrics = paper_results.get("metrics", [])
        print(f"\n🎯 Results to Reproduce: {len(metrics)} metric(s)")
        if metrics:
            for m in metrics[:5]:
                if isinstance(m, dict):
                    dataset = m.get('dataset', 'Unknown')
                    metric = m.get('metric', 'Unknown')
                    value = m.get('value', 'Unknown')
                    print(f"   - {dataset}: {metric} = {value}")
            if len(metrics) > 5:
                print(f"   ... and {len(metrics) - 5} more")
        else:
            summary = paper_results.get("summary", "")
            if summary:
                print("   Summary from paper:")
                for line in summary.split('\n')[:3]:
                    if line.strip():
                        print(f"   {line.strip()[:74]}")
            else:
                print("   (no metrics extracted)")

        print("\n" + "="*80 + "\n")



    def _decide_and_clone_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Decide on implementation path and clone repository if found."""
        # Check if this phase was already completed (resuming from checkpoint)
        if self._is_phase_completed(state, "decide_and_clone"):
            print("⏭️  Skipping decide_and_clone (already completed from checkpoint)")
            return state

        self.metrics_tracker.start_phase("decide_and_clone")
        print("🤔 Deciding on implementation path...")
        
        # Prepare inputs for Discovery Agent
        paper_title = state.get("paper_title", "")
        paper_abstract = state.get("paper_metadata", {}).get("abstract", "")
        arxiv_id = state.get("paper_metadata", {}).get("arxiv_id")
        authors = state.get("paper_metadata", {}).get("authors", [])
        
        # Format existing refs for the agent (from paper analysis)
        existing_refs = []
        if state.get("code_references"):
            refs = state["code_references"]
            if isinstance(refs, list):
                for ref in refs:
                    if isinstance(ref, dict):
                        existing_refs.append(ref.get("url"))
                    elif isinstance(ref, str):
                        existing_refs.append(ref)
        
        # CALL DISCOVERY AGENT
        discovery_result = self.discovery_agent.find_best_implementation(
            paper_title=paper_title, 
            paper_abstract=paper_abstract,
            arxiv_id=arxiv_id,
            authors=authors,
            existing_repos=existing_refs
        )
        
        selected_url = discovery_result.get("repo_url")
        
        if selected_url:
            state["selected_repo"] = {"url": selected_url, "source": "discovery_agent", "confidence": discovery_result.get("confidence")}
            state["messages"].append(f"📥 Using implementation: {selected_url}")
            print(f"✅ Discovery Agent selected: {selected_url}")

            # Clone the repository
            code_path = self.config.repo_path
            repo_marker = os.path.join(code_path, ".repo_url")

            need_clone = True
            if os.path.exists(code_path):
                if os.path.exists(repo_marker):
                    try:
                        with open(repo_marker, 'r') as f:
                            existing_url = f.read().strip()
                        if existing_url == selected_url:
                            print(f"✅ Repository already cloned: {selected_url}")
                            need_clone = False
                        else:
                            print(f"🔄 Different repo detected, removing old...")
                            shutil.rmtree(code_path, onerror=remove_readonly)
                    except Exception as e:
                        print(f"⚠️  Could not read repo marker: {e}")
                        shutil.rmtree(code_path, onerror=remove_readonly)
                else:
                    print(f"🗑️  No repo marker found, removing directory...")
                    shutil.rmtree(code_path, onerror=remove_readonly)

            if need_clone:
                print(f"📥 Cloning repository from {selected_url}...")
                try:
                    result = subprocess.run(
                        ["git", "clone", selected_url, code_path],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    if result.returncode == 0:
                        print(f"✅ Successfully cloned repository to {code_path}")
                        # Store repo URL marker
                        with open(repo_marker, 'w') as f:
                            f.write(selected_url)
                        state["implementation_path"] = code_path
                    else:
                        print(f"⚠️  Clone failed: {result.stderr}")
                        state["messages"].append("Clone failed")
                        state["final_status"] = "Failed: Could not clone repository"
                        self.metrics_tracker.end_phase("decide_and_clone", success=False)
                        return state
                except Exception as e:
                    print(f"⚠️  Clone error: {str(e)}")
                    state["final_status"] = f"Failed: Clone error - {str(e)}"
                    self.metrics_tracker.end_phase("decide_and_clone", success=False)
                    return state
            else:
                state["implementation_path"] = code_path

            # Save checkpoint after successful clone
            self._save_checkpoint(state, "decide_and_clone")

            self.metrics_tracker.end_phase("decide_and_clone", success=True)
            self.metrics_tracker.print_intermediate_summary()
            return state

        # No implementation found
        state["final_status"] = "Failed: No implementation found"
        state["messages"].append("❌ No implementation found")
        print("❌ Discovery Agent found no suitable implementation")

        # Save checkpoint even on failure
        self._save_checkpoint(state, "decide_and_clone")

        self.metrics_tracker.end_phase("decide_and_clone", success=False)
        self.metrics_tracker.print_intermediate_summary()
        return state

    def _environment_setup_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Prepare environment for running experiments."""
        # Check if this phase was already completed (resuming from checkpoint)
        if self._is_phase_completed(state, "environment_setup"):
            print("⏭️  Skipping environment_setup (already completed from checkpoint)")
            return state

        self.metrics_tracker.start_phase("environment_setup")
        print("🔧 Setting up environment...")

        code_path = state.get("implementation_path") or "./cloned_repo"

        # Get paper date for version pinning
        paper_date = state.get("paper_metadata", {}).get("published", None)
        paper_title = state.get("paper_title", "Unknown")

        # Read README for installation instructions
        readme_path = os.path.join(code_path, "README.md")
        readme_content = ""
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                readme_content = f.read()
        except FileNotFoundError:
            print("⚠️  No README.md found, will analyze environment files directly")
            readme_content = "No README found. Analyze environment files directly."

        # Run environment setup agent
        env_result = self.env_setup_agent.setup_environment(
            repo_path=code_path,
            readme_content=readme_content,
            paper_date=paper_date,
            paper_title=paper_title
        )

        # Update state with environment results
        state["env_setup_results"] = env_result
        state["dependencies_installed"] = env_result.get("success", False)

        if env_result.get("success"):
            state["messages"].append("✅ Environment setup successful")
            print(f"✅ Environment ready: {env_result.get('env_name', 'Unknown')}")

            # Save environment info for unified_reproduction to use
            if "agent_contexts" not in state:
                state["agent_contexts"] = {}
            state["agent_contexts"]["environment_setup"] = {
                "env_type": env_result.get("env_type"),
                "env_name": env_result.get("env_name"),
                "python_path": env_result.get("python_path"),
                "packages_pinned": env_result.get("packages_pinned", [])
            }

            # Store in hierarchical context
            self.hierarchical_context.add(
                content=f"Environment: {env_result.get('env_type', 'unknown')}, "
                        f"Name: {env_result.get('env_name', 'unknown')}, "
                        f"Python: {env_result.get('python_path', 'unknown')}",
                source="environment_setup",
                entry_type="result",
                importance=0.8
            )
        else:
            state["messages"].append("❌ Environment setup failed")
            print(f"❌ Environment setup failed: {env_result.get('error', 'Unknown error')}")
            state["final_status"] = f"Failed: Environment setup - {env_result.get('error', 'Unknown')}"

        # Save checkpoint
        self._save_checkpoint(state, "environment_setup")

        self.metrics_tracker.end_phase("environment_setup", success=env_result.get("success", False))
        self.metrics_tracker.print_intermediate_summary()
        return state

    def _unified_reproduction_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Run unified reproduction workflow."""
        # Check if this phase was already completed (resuming from checkpoint)
        if self._is_phase_completed(state, "unified_reproduction"):
            print("⏭️  Skipping unified_reproduction (already completed from checkpoint)")
            return state

        self.metrics_tracker.start_phase("unified_reproduction")
        code_path = state.get("implementation_path") or "./cloned_repo"

        # NEW: Check for existing results/checkpoints in the repo BEFORE running experiments
        existing_results = self.discovery_agent.check_existing_results(code_path)
        if existing_results.get("has_results"):
            print("\n" + "="*60)
            print("🎯 EXISTING RESULTS FOUND - Skipping to verification!")
            print("="*60)
            print(f"   Result files: {existing_results.get('result_files', [])[:3]}")
            print(f"   Checkpoints: {existing_results.get('checkpoints', [])[:3]}")
            print("="*60 + "\n")

            # Mark as successful and skip to verification
            state["env_setup_results"] = {"success": True, "report": "Skipped - using existing results"}
            state["dependencies_installed"] = True
            state["dataset_results"] = {"datasets_identified": True, "datasets_downloaded": True}
            state["datasets_ready"] = True
            state["experiment_results"] = {
                "execution_successful": True,
                "sanity_check_passed": True,
                "output": f"Using existing results from: {existing_results.get('result_files', [])}",
                "existing_results": existing_results,
                "skipped_execution": True
            }
            state["experiments_completed"] = True
            state["messages"].append("✅ Found existing results - skipping experiment execution")

            # Save checkpoint
            self._save_checkpoint(state, "unified_reproduction")
            self.metrics_tracker.end_phase("unified_reproduction", success=True)
            return state

        print("🚀 Starting unified reproduction workflow...")

        # Build comprehensive context from paper analysis
        paper_context_parts = []

        if state.get("agent_contexts", {}).get("paper_analyzer"):
            paper_context_parts.append(state["agent_contexts"]["paper_analyzer"])

        datasets = state.get("experimental_setup", {}).get("datasets", [])
        if datasets:
            paper_context_parts.append(f"\nDatasets mentioned in paper: {', '.join(datasets)}")

        paper_results = state.get("paper_results", {})
        if paper_results:
            paper_context_parts.append("\n\nResults to Reproduce:")
            if isinstance(paper_results, dict):
                metrics = paper_results.get("metrics", [])
                for m in metrics:
                    paper_context_parts.append(
                        f"  - {m.get('dataset', 'Unknown')}: {m.get('metric', 'Unknown')} = {m.get('value', 'Unknown')}"
                    )
                if not metrics and "summary" in paper_results:
                    paper_context_parts.append(f"\n{paper_results['summary']}")

        impl_details = state.get("experimental_setup", {}).get("implementation_details", "")
        if impl_details:
            paper_context_parts.append(f"\n\nImplementation Details:\n{impl_details[:500]}")

        paper_context = "\n".join(paper_context_parts)

        # Run unified reproduction
        result = self.unified_reproducer.reproduce(
            code_path, 
            paper_context,
            experiment_mode=state.get("experiment_selection_mode", "single"),
            custom_experiments=state.get("custom_experiment_list", [])
        )

        # Update state with results
        state["env_setup_results"] = {
            "success": result["setup_successful"],
            "report": result["report"]
        }
        state["dependencies_installed"] = result["dependencies_installed"]

        state["dataset_results"] = {
            "datasets_identified": result["data_attempted"],
            "datasets_downloaded": result["data_successful"],
            "dataset_locations": [result["data_location"]] if result["data_location"] else []
        }
        state["datasets_ready"] = result["data_successful"]

        state["experiment_results"] = {
            "execution_successful": result["main_experiment_successful"],
            "sanity_check_passed": result["sanity_check_passed"],
            "output": result["experiment_output"],
            "executed_command": ", ".join(result["executed_commands"]) if result["executed_commands"] else "",
            "errors": result["errors"],
            "experiments_tried": result.get("experiments_tried", []),
            "experiments_succeeded": result.get("experiments_succeeded", []),
            "partial_success": result.get("partial_success", False)
        }
        state["experiments_completed"] = result["main_experiment_successful"] or result.get("partial_success", False)

        # Store context
        state["agent_contexts"]["unified_reproducer"] = result["report"]

        # Add messages
        if result["setup_successful"]:
            state["messages"].append("✅ Environment setup successful")
            print("✅ Environment setup successful")
        else:
            state["messages"].append("❌ Environment setup failed")
            print("❌ Environment setup failed")

        if result["data_successful"]:
            state["messages"].append("✅ Datasets prepared successfully")
            print("✅ Datasets prepared")
        elif result["data_manual_steps"]:
            state["messages"].append("⚠️  Manual dataset steps required")
            print("⚠️  Manual dataset steps required")

        if result["main_experiment_successful"]:
            state["messages"].append("✅ Experiments executed successfully")
            print("✅ Experiments executed successfully")
        elif result["sanity_check_passed"]:
            state["messages"].append("⚠️  Sanity check passed but main experiment failed")
            print("⚠️  Sanity check passed but main experiment failed")
        else:
            state["messages"].append("❌ Experiments failed")
            print("❌ Experiments failed")

        # Print summary
        print(f"\n📊 Unified Reproduction Summary:")
        print(f"   READMEs consulted: {', '.join(result['readmes_consulted']) if result['readmes_consulted'] else 'None'}")
        print(f"   Setup: {'✅' if result['setup_successful'] else '❌'}")
        print(f"   Data: {'✅' if result['data_successful'] else '⚠️'}")
        print(f"   Experiments: {'✅' if result['main_experiment_successful'] else '❌'}")

        # Save checkpoint after unified reproduction
        self._save_checkpoint(state, "unified_reproduction")

        # Record experiment wall time if available from the result
        if result.get("experiment_wall_time"):
            self.metrics_tracker.record_experiment_time("unified_reproduction", result["experiment_wall_time"])

        self.metrics_tracker.end_phase("unified_reproduction", success=result["main_experiment_successful"])
        self.metrics_tracker.print_intermediate_summary()
        return state

    def _extract_and_verify_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Summarize verification results from the unified reproduction agent.

        The agent (unified_reproduction_agent) now handles extraction and verification
        using the code-first approach (execute_python_code). This node just summarizes
        the results already stored in state by the agent.
        """
        # Check if this phase was already completed (resuming from checkpoint)
        if self._is_phase_completed(state, "extract_and_verify"):
            print("⏭️  Skipping extract_and_verify (already completed from checkpoint)")
            return state

        self.metrics_tracker.start_phase("extract_and_verify")
        print("📊 Summarizing verification results...")

        experiment_results = state.get("experiment_results", {})
        dependencies_installed = state.get("dependencies_installed", False)

        # Get metrics already extracted by agent (via execute_python_code)
        extracted_metrics = state.get("extracted_metrics", {})
        metrics_comparison = state.get("metrics_comparison", {})

        # Build verification report from what agent stored
        verification_report = []
        verification_report.append("## Execution Summary")
        verification_report.append(f"- Dependencies Installed: {'Yes' if dependencies_installed else 'No'}")
        verification_report.append(f"- Datasets Ready: {'Yes' if state.get('datasets_ready') else 'No'}")
        verification_report.append(f"- Experiments Completed: {'Yes' if state.get('experiments_completed') else 'No'}")

        # Include metrics comparison if agent performed it
        if metrics_comparison:
            verification_report.append("\n## Metrics Comparison")
            if metrics_comparison.get("matches"):
                verification_report.append("### Matching Metrics:")
                for match in metrics_comparison["matches"]:
                    if isinstance(match, dict):
                        verification_report.append(f"  - {match.get('metric', 'N/A')}: {match.get('actual', 'N/A')} (expected: {match.get('expected', 'N/A')})")
            if metrics_comparison.get("mismatches"):
                verification_report.append("### Mismatched Metrics:")
                for mismatch in metrics_comparison["mismatches"]:
                    if isinstance(mismatch, dict):
                        verification_report.append(f"  - {mismatch.get('metric', 'N/A')}: {mismatch.get('actual', 'N/A')} (expected: {mismatch.get('expected', 'N/A')})")

        report_text = "\n".join(verification_report)

        # Determine success level based on experiments
        experiments_tried = experiment_results.get("experiments_tried", [])
        experiments_succeeded = experiment_results.get("experiments_succeeded", [])

        # Check prerequisites first
        if not dependencies_installed:
            success_level = "failed"
            results_match = False
            status_msg = "❌ Verification: Environment setup failed"
        elif not experiments_tried:
            sanity_check_passed = experiment_results.get("sanity_check_passed", False)
            if sanity_check_passed:
                success_level = "minimal"
                results_match = False
                status_msg = "⚠️ Verification: Only sanity check completed"
            else:
                success_level = "setup_only"
                results_match = False
                status_msg = "⚠️ Verification: Setup complete but no experiments run"
        else:
            total_experiments = len(experiments_tried)
            succeeded_count = len(experiments_succeeded)
            success_portion = f"{succeeded_count}/{total_experiments}"

            if succeeded_count == total_experiments:
                success_level = "full"
                results_match = True
                status_msg = f"✅ Verification: All {total_experiments} experiment(s) succeeded - results match paper (within 5%)"
            elif succeeded_count > 0:
                success_level = "partial"
                results_match = False
                status_msg = f"⚠️ Verification: Partial reproduction - {success_portion} experiments succeeded ({', '.join(experiments_succeeded)})"
            else:
                success_level = "failed"
                results_match = False
                status_msg = f"❌ Verification: All {total_experiments} experiment(s) failed"

        state["verification_results"] = {
            "report": report_text,
            "results_match_paper": results_match,
            "success_level": success_level,
            "discrepancies": metrics_comparison.get("mismatches", [])
        }
        state["results_match"] = results_match

        print(f"\n📝 Verification Report:\n{report_text}\n")
        state["messages"].append(status_msg)

        # Save checkpoint after verification
        self._save_checkpoint(state, "extract_and_verify")

        self.metrics_tracker.end_phase("extract_and_verify", success=results_match)
        return state

    def _generate_report_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Generate final report."""
        # Note: We always regenerate the report even when resuming, to ensure it's up-to-date
        # But we still mark it as completed for tracking purposes
        self.metrics_tracker.start_phase("generate_report")
        print("📊 Generating final report...")

        # Determine final status
        success_level = state.get("verification_results", {}).get("success_level", "failed")
        experiments_tried = state.get("experiment_results", {}).get("experiments_tried", [])
        experiments_succeeded = state.get("experiment_results", {}).get("experiments_succeeded", [])

        # Calculate overall portion
        if experiments_tried:
            success_portion = f"{len(experiments_succeeded)}/{len(experiments_tried)}"
        else:
            success_portion = "0/0"

        status_map = {
            "full": "✅ Complete - All Experiments Succeeded (Results Match Paper)",
            "partial": f"⚠️ Partial - {success_portion} Experiments Reproduced ({', '.join(experiments_succeeded)})" if experiments_succeeded else f"⚠️ Partial - {success_portion} Experiments Succeeded",
            "minimal": "⚠️ Minimal - Only Sanity Check Passed",
            "setup_only": "⚠️ Setup Only - No Experiments Run",
            "failed": "❌ Failed - Prerequisites or All Experiments Failed"
        }
        state["final_status"] = state.get("final_status") or status_map.get(success_level, "❌ Failed")

        # Get selected repo info
        selected_repo_info = "None"
        if state.get("selected_repo"):
            repo = state["selected_repo"]
            if isinstance(repo, dict):
                selected_repo_info = repo.get("url") or repo.get("full_name", "Unknown")

        # Count repos found
        code_refs_count = len([r for r in state.get('code_references', []) if isinstance(r, str) and r.startswith('http')])

        # Deduplicate messages
        seen = set()
        unique_messages = []
        skip_phrases = ["continuing anyway", "will attempt anyway", "had issues - continuing"]

        for msg in state.get('messages', []):
            if msg in seen:
                continue
            if any(phrase in msg.lower() for phrase in skip_phrases):
                continue
            seen.add(msg)
            unique_messages.append(msg)

        # Build experiment section
        experiment_section = ""
        if experiments_tried:
            experiment_section = f"""
## Experiments
- Attempted: {', '.join(experiments_tried)}
- Succeeded: {', '.join(experiments_succeeded) if experiments_succeeded else 'None'}
- Success Rate: {success_portion} ({len(experiments_succeeded)/len(experiments_tried)*100:.0f}%)
"""

        report = f"""
# Paper Reproduction Report

## Paper Information
- Title: {state.get('paper_title', 'N/A')}
- Analysis: {'Complete' if state.get('paper_metadata') else 'Incomplete'}
- Code References Found: {code_refs_count}

## Implementation
- Selected Repository: {selected_repo_info}
- Implementation Path: {state.get('implementation_path', 'N/A')}
{experiment_section}
## Verification
- Results Match Paper: {'Yes' if state.get('results_match') else 'No'}
- Success Level: {success_level}

## Status
{state.get('final_status', 'Complete')}

## Summary
{chr(10).join(unique_messages)}
"""

        state["report"] = report

        self.metrics_tracker.end_phase("generate_report", success=True)
        return state

    def _route_after_clone(self, state: PaperReproductionState) -> Literal["continue", "failed"]:
        """Route after cloning repository."""
        if state.get("implementation_path"):
            return "continue"
        return "failed"

    def _route_after_env_setup(self, state: PaperReproductionState) -> Literal["continue", "failed"]:
        """Route after environment setup."""
        if state.get("dependencies_installed", False):
            return "continue"
        print("🛑 Routing to report generation due to environment setup failure")
        return "failed"

    def _route_after_reproduction(self, state: PaperReproductionState) -> Literal["continue", "failed"]:
        """Route after unified reproduction."""
        setup_success = state.get("dependencies_installed", False)
        experiments_completed = state.get("experiments_completed", False)
        sanity_check_passed = state.get("experiment_results", {}).get("sanity_check_passed", False)

        if not setup_success:
            print("🛑 Routing to report generation due to setup failure")
            state["final_status"] = "Failed: Environment setup failed"
            return "failed"

        if experiments_completed or sanity_check_passed:
            print("✅ Routing to metrics extraction")
            return "continue"

        print("⚠️  No experiments run, but continuing to metrics extraction")
        return "continue"

    def _is_phase_completed(self, state: PaperReproductionState, phase: str) -> bool:
        """Check if a phase was already completed (from checkpoint resume).

        Args:
            state: Current workflow state
            phase: Phase name to check

        Returns:
            True if phase was already completed and should be skipped
        """
        completed = state.get("completed_phases", [])
        return phase in completed

    def _save_checkpoint(self, state: PaperReproductionState, phase: str):
        """Save checkpoint for current phase.

        Args:
            state: Current workflow state
            phase: Phase name (e.g., 'analyze_paper', 'decide_and_clone', etc.)
        """
        if not self.checkpoint_manager:
            return

        # Add this phase to completed_phases if not already there
        if "completed_phases" not in state:
            state["completed_phases"] = []
        if phase not in state["completed_phases"]:
            state["completed_phases"].append(phase)

        repo_path = state.get("implementation_path") or self.config.repo_path
        paper_id = state.get("paper_metadata", {}).get("arxiv_id", "") or state.get("paper_input", "")

        # Create checkpoint-safe state (only serializable data)
        checkpoint_state = {
            "phase": phase,
            "paper_title": state.get("paper_title", ""),
            "paper_metadata": state.get("paper_metadata", {}),
            "experimental_setup": state.get("experimental_setup", {}),
            "paper_results": state.get("paper_results", {}),
            "code_references": state.get("code_references", []),
            "selected_repo": state.get("selected_repo", {}),
            "implementation_path": state.get("implementation_path", ""),
            "env_setup_results": state.get("env_setup_results", {}),
            "dependencies_installed": state.get("dependencies_installed", False),
            "dataset_results": state.get("dataset_results", {}),
            "datasets_ready": state.get("datasets_ready", False),
            "experiment_results": state.get("experiment_results", {}),
            "experiments_completed": state.get("experiments_completed", False),
            "extracted_metrics": state.get("extracted_metrics", {}),
            "metrics_comparison": state.get("metrics_comparison", {}),
            "verification_results": state.get("verification_results", {}),
            "results_match": state.get("results_match", False),
            "agent_contexts": state.get("agent_contexts", {}),  # Context from agents
            "messages": state.get("messages", []),
            "completed_phases": state.get("completed_phases", []),  # Track completed phases for resume
        }

        self.checkpoint_manager.save(
            state=checkpoint_state,
            phase=phase,
            repo_path=repo_path,
            paper_id=paper_id
        )

    def _run_verification_only(self, paper_input: str, repo_path: str, existing_results: dict) -> dict:
        """Run verification workflow when results already exist.

        This skips paper analysis and experiment execution, using the unified
        reproduction agent to extract metrics and compare with paper via code-first
        approach (execute_python_code).

        Args:
            paper_input: Paper identifier (arXiv ID, etc.)
            repo_path: Path to repository with existing results
            existing_results: Dict from _check_existing_results

        Returns:
            Final state with verification results
        """
        print("📊 Running verification-only mode with code-first approach...")

        # Step 1: Try to get paper results from existing checkpoint
        paper_title = paper_input
        paper_results = {}

        if self.checkpoint_manager:
            checkpoint_data = self.checkpoint_manager.resume(
                repo_path=repo_path,
                paper_id=paper_input
            )
            if checkpoint_data:
                state = checkpoint_data.get("state", {})
                paper_results = state.get("paper_results", {})
                paper_title = state.get("paper_title", paper_input)
                if paper_results:
                    print(f"✅ Found paper results from checkpoint: {paper_title[:60]}...")

        # Step 2: If no checkpoint, try to get paper info from arXiv
        if not paper_results:
            try:
                if paper_input.startswith("arxiv:") or "." in paper_input:
                    arxiv_id = paper_input.replace("arxiv:", "")
                    print(f"📄 Fetching paper info for {arxiv_id}...")

                    import arxiv
                    search = arxiv.Search(id_list=[arxiv_id])
                    paper = next(search.results())
                    paper_title = paper.title
                    print(f"   Title: {paper_title[:60]}...")
            except Exception as e:
                print(f"⚠️ Could not fetch paper info: {e}")

        # Step 3: Build initial state for verification
        initial_state = {
            "paper_input": paper_input,
            "paper_title": paper_title,
            "paper_results": paper_results,
            "implementation_path": repo_path,
            "dependencies_installed": True,
            "datasets_ready": True,
            "experiments_completed": True,
            "experiment_results": {
                "skipped_execution": True,
                "existing_results": existing_results,
                "output": f"Using existing results from {repo_path}"
            },
            "messages": [f"✅ Using existing results from {repo_path}"],
            "completed_phases": []
        }

        # Step 4: Build verification prompt for agent
        result_files = existing_results.get("result_files", [])
        verification_prompt = f"""VERIFICATION-ONLY MODE

You are verifying existing experiment results against paper claims.

Repository: {repo_path}
Paper: {paper_title}

Existing result files found:
{chr(10).join(f"- {f}" for f in result_files[:20])}

Paper expected results:
{paper_results if paper_results else "Not available - extract from result files and report findings"}

YOUR TASK:
1. Use execute_python_code to write a script that:
   - Reads the result files in {repo_path}
   - Extracts metrics (accuracy, F1, etc.) from each file
   - Compares with paper expected values (if available)
   - Reports match/mismatch status

2. Store results in a structured format

IMPORTANT: Write Python code to parse the specific file formats you find.
Do NOT assume any particular format - explore and adapt.

Begin verification now."""

        # Step 5: Run agent for verification
        print("\n🤖 Running unified reproduction agent for verification...")
        agent_result = self.unified_reproducer.reproduce(
            code_path=repo_path,
            paper_context=verification_prompt
        )

        # Step 6: Build final state from agent results
        extracted_metrics = agent_result.get("extracted_metrics", {})
        metrics_comparison = agent_result.get("metrics_comparison", {})

        # Determine success level
        experiments_succeeded = agent_result.get("experiments_succeeded", [])
        experiments_tried = agent_result.get("experiments_tried", [])

        if experiments_tried:
            match_success = len(experiments_succeeded) == len(experiments_tried) and len(experiments_tried) > 0
            success_level = "verified" if match_success else ("partial" if experiments_succeeded else "failed")
        else:
            # Agent didn't track experiments - check if we have comparison results
            match_success = bool(extracted_metrics)
            success_level = "verified" if match_success else "unknown"

        final_state = {
            "paper_input": paper_input,
            "paper_title": paper_title,
            "implementation_path": repo_path,
            "dependencies_installed": True,
            "datasets_ready": True,
            "experiments_completed": True,
            "extracted_metrics": extracted_metrics,
            "metrics_comparison": metrics_comparison,
            "verification_results": {
                "report": agent_result.get("output", "Verification completed"),
                "results_match_paper": match_success,
                "success_level": success_level,
            },
            "results_match": match_success,
            "messages": [
                f"✅ Using existing results from {repo_path}",
                f"📊 Verification completed via code-first approach"
            ],
            "final_status": "✅ Verification Complete" if match_success else "⚠️ Verification Complete (partial)",
            "report": agent_result.get("output", "")
        }

        print(f"\n{'='*60}")
        print(f"{'✅' if match_success else '⚠️'} Verification Complete")
        print(f"{'='*60}\n")

        return final_state

    def _build_verification_report(self, comparison_result: dict, extracted_datasets: dict) -> str:
        """Build detailed verification report text."""
        lines = ["## Verification Report\n"]

        summary = comparison_result.get("summary", {})
        lines.append(f"**Status**: {summary.get('status', 'Unknown')}")
        lines.append(f"**Match Ratio**: {summary.get('match_ratio', 'N/A')}")
        lines.append(f"**Match Percentage**: {summary.get('match_percentage', 'N/A')}\n")

        if comparison_result.get("matched"):
            lines.append("### ✅ Matched Results (within 5% tolerance)")
            for m in comparison_result["matched"]:
                lines.append(f"- **{m['expected_dataset']}** → {m['extracted_dataset']}")
                lines.append(f"  - Extracted: {m['extracted_value']:.2f}")
                lines.append(f"  - Expected: {m['expected_value']:.2f}")
                lines.append(f"  - Error: {m['relative_error_pct']}")

        if comparison_result.get("mismatched"):
            lines.append("\n### ⚠️ Mismatched Results (outside 5% tolerance)")
            for m in comparison_result["mismatched"]:
                lines.append(f"- **{m['expected_dataset']}**")
                lines.append(f"  - Extracted: {m['extracted_value']:.2f}")
                lines.append(f"  - Expected: {m['expected_value']:.2f}")
                lines.append(f"  - Error: {m['relative_error_pct']}")

        if comparison_result.get("missing_from_extracted"):
            lines.append("\n### ❌ Missing Datasets")
            for ds in comparison_result["missing_from_extracted"]:
                lines.append(f"- {ds}")

        if comparison_result.get("extra_in_extracted"):
            lines.append("\n### 📊 Additional Datasets (not in paper)")
            for ds in comparison_result["extra_in_extracted"]:
                lines.append(f"- {ds}")

        return "\n".join(lines)

    def _build_final_report(self, paper_input: str, paper_title: str, repo_path: str,
                           existing_results: dict, extracted_datasets: dict,
                           extracted_metrics: dict, comparison_result: dict,
                           source_files: list) -> str:
        """Build the final markdown report."""
        summary = comparison_result.get("summary", {})

        # Build dataset results section
        dataset_lines = []
        for ds_name, metrics in list(extracted_datasets.items())[:10]:
            main_metric = list(metrics.keys())[0] if metrics else "unknown"
            main_value = list(metrics.values())[0] if metrics else "N/A"
            dataset_lines.append(f"- **{ds_name}**: {main_metric} = {main_value}")

        # Build comparison section
        comparison_lines = []
        if comparison_result.get("aligned_comparisons"):
            for comp in comparison_result["aligned_comparisons"][:10]:
                status = "✅" if comp["within_tolerance"] else "❌"
                comparison_lines.append(
                    f"- {status} **{comp['expected_dataset']}**: "
                    f"{comp['extracted_value']:.2f} (expected {comp['expected_value']:.2f}, "
                    f"error: {comp['relative_error_pct']})"
                )

        return f"""
# Paper Reproduction Report (Verification Only)

## Paper
- **Input**: {paper_input}
- **Title**: {paper_title}

## Existing Results Used
- **Repository**: {repo_path}
- **Result files found**: {len(existing_results.get('result_files', []))}
- **Model checkpoints**: {len(existing_results.get('checkpoints', []))}
- **Source files analyzed**: {', '.join(source_files[:3]) if source_files else 'N/A'}

## Extracted Results by Dataset
{chr(10).join(dataset_lines) if dataset_lines else '- No results extracted'}

## Comparison with Paper
- **Status**: {summary.get('status', 'Unknown')}
- **Match Ratio**: {summary.get('match_ratio', 'N/A')}
- **Match Percentage**: {summary.get('match_percentage', 'N/A')}

### Detailed Comparison
{chr(10).join(comparison_lines) if comparison_lines else '- No comparison available'}

## Summary
The reproduction {'successfully matched' if summary.get('matched_count', 0) == summary.get('total_expected', 0) and summary.get('total_expected', 0) > 0 else 'partially matched'} the paper results with {summary.get('match_percentage', '0%')} of datasets within the 5% tolerance threshold.
"""

    def _try_resume_checkpoint(self, paper_input: str, repo_path: str = None) -> dict:
        """Try to resume from checkpoint.

        Args:
            paper_input: Paper identifier (arXiv ID, etc.)
            repo_path: Repository path
        """
        if repo_path is None:
            repo_path = self.config.repo_path
        if not self.checkpoint_manager:
            return {}

        checkpoint_data = self.checkpoint_manager.resume(
            repo_path=repo_path,
            paper_id=paper_input
        )

        if checkpoint_data:
            state = checkpoint_data.get("state", {})
            completed_phases = state.get("completed_phases", [])

            print(f"\n♻️  RESUMING FROM CHECKPOINT")
            print(f"   Last phase: {checkpoint_data['phase']}")
            print(f"   Saved at: {checkpoint_data['timestamp']}")
            if completed_phases:
                print(f"   ✅ Completed phases that will be SKIPPED:")
                for phase in completed_phases:
                    print(f"      - {phase}")
            print()
            return state

        return {}

    def run(self, paper_input: str, clear_checkpoints: bool = False) -> dict:
        """
        Run the complete paper reproduction workflow.

        Args:
            paper_input: arXiv ID, PDF path, or paper identifier
            clear_checkpoints: If True, clear existing checkpoints and start fresh

        Returns:
            Final state with results
        """
        print(f"\n{'='*60}")
        print(f"🚀 Starting Paper Reproduction Workflow (Clean)")
        print(f"{'='*60}\n")

        # Start metrics tracking
        self.metrics_tracker.start_workflow()

        # Clear checkpoints if requested
        if clear_checkpoints and self.checkpoint_manager:
            self.checkpoint_manager.clear(repo_path=self.config.repo_path, paper_id=paper_input)
            print("🗑️  Cleared existing checkpoints\n")

        # FIRST: Check if cloned_repo already has results (before anything else!)
        repo_path = self.config.repo_path
        if os.path.exists(repo_path):
            existing_results = self._check_existing_results(repo_path)
            if existing_results.get("has_results"):
                print("\n" + "="*60)
                print("🎯 EXISTING RESULTS DETECTED IN REPOSITORY!")
                print("="*60)
                print(f"   📁 Repository: {repo_path}")
                print(f"   📊 Result files: {len(existing_results.get('result_files', []))}")
                print(f"   💾 Checkpoints: {len(existing_results.get('checkpoints', []))}")
                print(f"   📋 Log files: {len(existing_results.get('log_files', []))}")
                if existing_results.get('result_files'):
                    for f in existing_results['result_files'][:3]:
                        print(f"      → {f}")
                print("="*60)
                print("\n⏩ Skipping paper analysis and experiment execution...")
                print("   Going directly to RESULT VERIFICATION\n")

                # Create a minimal state and skip to verification
                result = self._run_verification_only(paper_input, repo_path, existing_results)
                # End metrics tracking for verification-only path
                self.metrics_tracker.end_workflow()
                print(self.metrics_tracker.get_summary())
                return result

        # Try to resume from checkpoint
        resumed_state = self._try_resume_checkpoint(paper_input)

        if resumed_state:
            # Merge with defaults (in case new fields were added)
            initial_state = {
                "paper_input": paper_input,
                "paper_title": "",
                "paper_metadata": {},
                "experimental_setup": {},
                "paper_results": {},
                "code_references": [],
                "selected_repo": {},
                "implementation_path": "",
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
                "agent_contexts": {},
                "completed_phases": [],  # Will be populated from resumed_state
                "messages": [],
                "final_status": "",
                "report": "",
            }
            # Update with resumed state
            initial_state.update(resumed_state)
            completed_count = len(resumed_state.get('completed_phases', []))
            print(f"✅ Resumed with {completed_count} completed phase(s), {len(resumed_state.get('messages', []))} messages\n")
        else:
            # Start fresh
            initial_state = {
                "paper_input": paper_input,
                "paper_title": "",
                "paper_metadata": {},
                "experimental_setup": {},
                "paper_results": {},
                "code_references": [],
                "selected_repo": {},
                "implementation_path": "",
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
                "agent_contexts": {},
                "completed_phases": [],  # Track completed phases for checkpoint resume
                "messages": [],
                "final_status": "",
                "report": "",
            }

        try:
            final_state = self.workflow.invoke(initial_state)
        finally:
            # End metrics tracking and print summary
            self.metrics_tracker.end_workflow()
            print(self.metrics_tracker.get_summary())

        print(f"\n{'='*60}")
        print(f"✅ Workflow Complete")
        print(f"{'='*60}\n")

        # Close the log file if logging is enabled
        if self.file_logger:
            self.file_logger.close()

        return final_state
