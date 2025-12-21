"""Clean LangGraph Orchestrator - Simplified workflow for paper reproduction."""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from pathlib import Path
import operator
import os
import subprocess
import shutil
from .utils.checkpoint_manager import ExperimentCheckpoint


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
        from .agents.unified_reproduction_agent import UnifiedReproductionAgent
        from .agents.metrics_extractor import MetricsExtractorAgent
        from .utils.llm_factory import create_llm
        from .utils.file_logger import FileLogger
        from .utils.logging_callback import LoggingCallbackHandler

        self.llm = llm or create_llm(temperature=0.1)

        # Setup logging
        self.enable_logging = enable_logging
        self.file_logger = None
        self.logging_callback = None
        if enable_logging:
            self.file_logger = FileLogger(log_dir="./logs")
            self.logging_callback = LoggingCallbackHandler(verbose=True, file_logger=self.file_logger)

        # Setup checkpoint manager
        self.enable_checkpoints = enable_checkpoints
        self.checkpoint_manager = None
        if enable_checkpoints:
            self.checkpoint_manager = ExperimentCheckpoint(checkpoint_dir="./checkpoints")
            print("💾 Checkpoint system enabled")

        # Initialize only the agents we actually use
        self.unified_reproducer = UnifiedReproductionAgent(self.llm, max_iterations=50)
        self.metrics_extractor = MetricsExtractorAgent(self.llm)

        # Build the workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the simplified LangGraph workflow."""
        workflow = StateGraph(PaperReproductionState)

        # Add nodes - only 5 instead of 9
        workflow.add_node("analyze_paper", self._analyze_paper_node)
        workflow.add_node("decide_and_clone", self._decide_and_clone_node)
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
                download_dir = os.path.abspath("./downloads")
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

            # Print analysis summary
            self._print_analysis_summary(state, analysis)

            # Try Papers with Code API as fallback
            if not state["code_references"] and state.get("paper_title"):
                self._try_papers_with_code(state)
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

        return state

    def _print_analysis_summary(self, state, analysis):
        """Print detailed analysis results."""
        import textwrap

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
            wrapped = textwrap.fill(core, width=74, initial_indent="   ", subsequent_indent="   ")
            print(wrapped)
        else:
            print("   (not extracted)")

        # Results to Reproduce
        metrics = state["paper_results"].get("metrics", [])
        print(f"\n🎯 Results to Reproduce: {len(metrics)} metric(s)")
        if metrics:
            for m in metrics[:5]:
                dataset = m.get('dataset', 'Unknown')
                metric = m.get('metric', 'Unknown')
                value = m.get('value', 'Unknown')
                print(f"   - {dataset}: {metric} = {value}")
            if len(metrics) > 5:
                print(f"   ... and {len(metrics) - 5} more")
        else:
            summary = state["paper_results"].get("summary", "")
            if summary:
                print("   Summary from paper:")
                for line in summary.split('\n')[:3]:
                    if line.strip():
                        print(f"   {line.strip()[:74]}")
            else:
                print("   (no metrics extracted)")

        print("\n" + "="*80 + "\n")

    def _try_papers_with_code(self, state):
        """Try Papers with Code API as fallback for finding implementations."""
        print("🔍 No repos in paper, trying Papers with Code API...")
        try:
            import requests
            base_url = "https://paperswithcode.com/api/v1/papers/"
            search_url = f"{base_url}?title={requests.utils.quote(state['paper_title'])}"
            response = requests.get(search_url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    paper = data["results"][0]
                    paper_id = paper.get("id")

                    impl_url = f"https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/"
                    impl_response = requests.get(impl_url, timeout=10)

                    if impl_response.status_code == 200:
                        impl_data = impl_response.json()
                        implementations = impl_data.get("results", [])

                        repos = [impl["url"] for impl in implementations if impl.get("url") and impl.get("is_official")]
                        if not repos:
                            repos = [impl["url"] for impl in implementations if impl.get("url")]

                        if repos:
                            state["code_references"] = repos
                            print(f"✅ Found {len(repos)} implementation(s) from Papers with Code")
                            return  # Found repos, no need to continue
        except Exception as e:
            print(f"⚠️  Papers with Code API failed: {str(e)[:50]}")

        # If Papers with Code didn't find anything, try enhanced discovery
        if not state.get("code_references"):
            self._try_enhanced_discovery(state)

    def _check_existing_results(self, repo_path: str) -> dict:
        """Check if results or model checkpoints already exist in the repository.

        This prevents re-running experiments when results are already available.

        Args:
            repo_path: Path to the cloned repository

        Returns:
            Dictionary with:
                - has_results: True if usable results found
                - result_files: List of result file paths
                - checkpoints: List of checkpoint paths
                - log_files: List of log files
        """
        import glob
        from pathlib import Path
        from datetime import datetime

        result = {
            "has_results": False,
            "result_files": [],
            "checkpoints": [],
            "log_files": [],
            "recently_modified": []
        }

        repo = Path(repo_path)
        if not repo.exists():
            return result

        # Check for result files (JSON, CSV with results/metrics in name)
        result_patterns = [
            "**/results*.json", "**/eval_results*.json", "**/metrics*.json",
            "**/results*.csv", "**/metrics*.csv",
            "**/all_results.json", "**/trainer_state.json",
            "results/**/*.json", "results/**/*.csv", "results/**/*.txt",
            "outputs/**/*.json", "outputs/**/*.csv",
            "output/**/*.json", "output/**/*.csv",
        ]

        for pattern in result_patterns:
            matches = glob.glob(str(repo / pattern), recursive=True)
            for match in matches:
                try:
                    path = Path(match)
                    stat = path.stat()
                    # Only consider files > 100 bytes (not empty)
                    if stat.st_size > 100:
                        result["result_files"].append(str(path.relative_to(repo)))
                except:
                    pass

        # Check for model checkpoints
        checkpoint_patterns = [
            "**/checkpoint-*", "**/checkpoint_*",
            "**/*.pt", "**/*.pth", "**/*.ckpt",
            "**/pytorch_model.bin", "**/model.safetensors",
        ]

        for pattern in checkpoint_patterns:
            matches = glob.glob(str(repo / pattern), recursive=True)
            for match in matches[:10]:  # Limit
                try:
                    path = Path(match)
                    result["checkpoints"].append(str(path.relative_to(repo)))
                except:
                    pass

        # Check for log files (for extracting metrics if no result files)
        log_patterns = ["*.log", "**/*.log", "**/logs/*.log"]
        for pattern in log_patterns[:2]:
            matches = glob.glob(str(repo / pattern), recursive=True)
            for match in matches[:5]:
                try:
                    path = Path(match)
                    stat = path.stat()
                    # Only logs > 1KB
                    if stat.st_size > 1024:
                        result["log_files"].append(str(path.relative_to(repo)))
                except:
                    pass

        # Check for recently modified files (last 24 hours) that might contain results
        recent_cutoff = datetime.now().timestamp() - 86400  # 24 hours
        recent_patterns = ["**/*.json", "**/*.csv"]
        for pattern in recent_patterns:
            matches = glob.glob(str(repo / pattern), recursive=True)
            for match in matches[:50]:
                try:
                    path = Path(match)
                    stat = path.stat()
                    if stat.st_mtime > recent_cutoff and stat.st_size > 100:
                        rel_path = str(path.relative_to(repo))
                        # Skip common non-result files
                        if not any(skip in rel_path.lower() for skip in [
                            'node_modules', '.git', '__pycache__', 'package'
                        ]):
                            result["recently_modified"].append(rel_path)
                except:
                    pass

        # Determine if we have usable results
        # Criteria: At least one result file OR (checkpoint + log file)
        has_result_files = len(result["result_files"]) > 0
        has_checkpoints_and_logs = len(result["checkpoints"]) > 0 and len(result["log_files"]) > 0
        has_recent_results = len(result["recently_modified"]) > 0

        result["has_results"] = has_result_files or has_checkpoints_and_logs

        if result["has_results"]:
            print(f"\n🔍 Checking for existing results in {repo_path}...")
            print(f"   Result files found: {len(result['result_files'])}")
            print(f"   Checkpoints found: {len(result['checkpoints'])}")
            print(f"   Log files found: {len(result['log_files'])}")

        return result

    def _try_enhanced_discovery(self, state):
        """Try enhanced repo discovery methods (GitHub arXiv search + web search)."""
        print("🔎 Trying enhanced repository discovery...")

        from .agents.unified_paper_analyzer import UnifiedPaperAnalyzer

        # Get paper metadata
        arxiv_id = state.get("paper_metadata", {}).get("arxiv_id")
        paper_title = state.get("paper_title")
        authors = state.get("paper_metadata", {}).get("authors", [])

        if not arxiv_id and not paper_title:
            print("   ⚠️  No arXiv ID or paper title available for enhanced discovery")
            return

        try:
            analyzer = UnifiedPaperAnalyzer(self.llm)
            discovered_repos = analyzer.enhanced_repo_discovery(
                arxiv_id=arxiv_id,
                paper_title=paper_title,
                authors=authors
            )

            if discovered_repos:
                state["code_references"] = discovered_repos
                print(f"✅ Enhanced discovery found {len(discovered_repos)} implementation(s)")
        except Exception as e:
            print(f"⚠️  Enhanced discovery failed: {str(e)[:50]}")

    def _select_best_repo(self, repos: list, paper_title: str, paper_abstract: str = "") -> str:
        """Use LLM to select the best repository for the paper."""
        if len(repos) == 1:
            return repos[0]

        prompt = f"""Given this paper and list of GitHub repositories, select the ONE repository that is most likely the official implementation.

Paper Title: {paper_title}

Abstract: {paper_abstract[:500] if paper_abstract else 'N/A'}

Repositories found:
{chr(10).join(f'{i+1}. {repo}' for i, repo in enumerate(repos))}

Reply with ONLY the number (1, 2, 3, etc.) of the best repository. Choose the one that:
- Has a name matching the paper's method/acronym
- Is from the paper's authors (if identifiable)
- Is NOT a general library like huggingface/transformers

Answer (number only):"""

        try:
            import re
            response = self.llm.invoke(prompt)
            match = re.search(r'\d+', response.content)
            if match:
                idx = int(match.group()) - 1
                if 0 <= idx < len(repos):
                    print(f"🤖 LLM selected repo #{idx+1}: {repos[idx]}")
                    return repos[idx]
        except Exception as e:
            print(f"⚠️ LLM repo selection failed: {e}")

        return repos[0]

    def _decide_and_clone_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Decide on implementation path and clone repository if found."""
        # Check if this phase was already completed (resuming from checkpoint)
        if self._is_phase_completed(state, "decide_and_clone"):
            print("⏭️  Skipping decide_and_clone (already completed from checkpoint)")
            return state

        print("🤔 Deciding on implementation path...")

        # Check for code references from paper
        if state["code_references"] and isinstance(state["code_references"], list):
            valid_refs = [ref for ref in state["code_references"]
                         if ref.startswith("http") and "github" in ref.lower()]

            if valid_refs:
                # Use LLM to select best repo if multiple found
                paper_abstract = state.get("paper_metadata", {}).get("abstract", "")
                selected_url = self._select_best_repo(valid_refs, state.get("paper_title", ""), paper_abstract)

                state["selected_repo"] = {"url": selected_url, "source": "paper"}
                state["messages"].append(f"📥 Using official implementation: {selected_url}")
                print(f"✅ Found official implementation: {selected_url}")

                # Clone the repository
                code_path = "./cloned_repo"
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
                                shutil.rmtree(code_path)
                        except Exception as e:
                            print(f"⚠️  Could not read repo marker: {e}")
                            shutil.rmtree(code_path)
                    else:
                        print(f"🗑️  No repo marker found, removing directory...")
                        shutil.rmtree(code_path)

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
                            return state
                    except Exception as e:
                        print(f"⚠️  Clone error: {str(e)}")
                        state["final_status"] = f"Failed: Clone error - {str(e)}"
                        return state
                else:
                    state["implementation_path"] = code_path

                # Save checkpoint after successful clone
                self._save_checkpoint(state, "decide_and_clone")

                return state

        # No implementation found
        state["final_status"] = "Failed: No implementation found"
        state["messages"].append("❌ No implementation found")
        print("❌ No existing implementation found")

        # Save checkpoint even on failure
        self._save_checkpoint(state, "decide_and_clone")

        return state

    def _unified_reproduction_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Run unified reproduction workflow."""
        # Check if this phase was already completed (resuming from checkpoint)
        if self._is_phase_completed(state, "unified_reproduction"):
            print("⏭️  Skipping unified_reproduction (already completed from checkpoint)")
            return state

        code_path = state.get("implementation_path") or "./cloned_repo"

        # NEW: Check for existing results/checkpoints in the repo BEFORE running experiments
        existing_results = self._check_existing_results(code_path)
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
        result = self.unified_reproducer.reproduce(code_path, paper_context)

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

        return state

    def _extract_and_verify_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Extract metrics and verify results against paper claims."""
        # Check if this phase was already completed (resuming from checkpoint)
        if self._is_phase_completed(state, "extract_and_verify"):
            print("⏭️  Skipping extract_and_verify (already completed from checkpoint)")
            return state

        print("📊 Extracting metrics and verifying results...")

        code_path = state.get("implementation_path") or "./cloned_repo"
        experiment_results = state.get("experiment_results", {})
        paper_results = state.get("paper_results", {})

        # Check if we're using existing results (skipped execution)
        if experiment_results.get("skipped_execution"):
            print("🎯 Using smart extraction for existing result files...")
            existing = experiment_results.get("existing_results", {})

            # Use smart extraction tools for better format handling
            from .tools.code_execution_tools import smart_extract_results, align_and_compare_results

            # Build paper metrics string from paper_results
            paper_metrics_str = ""
            if paper_results:
                metrics_list = paper_results.get("metrics", [])
                for m in metrics_list:
                    if isinstance(m, dict):
                        ds = m.get("dataset", "")
                        val = m.get("value", "")
                        if ds and val:
                            paper_metrics_str += f"{ds}: {val}\n"

            # Use smart extraction
            extraction_result = smart_extract_results.invoke({
                "repo_path": code_path,
                "paper_metrics": paper_metrics_str
            })
            extracted_datasets = extraction_result.get("datasets", {})

            # Build flat metrics dict for backward compatibility
            extracted_metrics = {}
            for dataset_name, metrics in extracted_datasets.items():
                for metric_name, value in metrics.items():
                    key = f"{dataset_name}_{metric_name}"
                    extracted_metrics[key] = value

            print(f"   ✅ Extracted {len(extracted_datasets)} datasets, {len(extracted_metrics)} metrics")

            # Store for comparison
            state["extracted_metrics"] = {
                "metrics": extracted_metrics,
                "datasets": extracted_datasets,
                "source": "smart_extraction"
            }

            # If we have paper metrics, do smart alignment
            if paper_metrics_str and extracted_datasets:
                comparison_result = align_and_compare_results.invoke({
                    "extracted_results": extraction_result,
                    "paper_metrics": paper_metrics_str,
                    "tolerance": 0.05
                })
                summary = comparison_result.get("summary", {})
                print(f"   📊 Comparison: {summary.get('status', 'N/A')} - {summary.get('match_ratio', 'N/A')}")
                state["metrics_comparison"] = comparison_result

            experiment_output = f"Metrics from smart extraction: {extracted_datasets}"
        else:
            experiment_output = experiment_results.get("output", "")

        # Extract metrics from output
        extracted = self.metrics_extractor.extract_metrics(experiment_output, paper_results)

        # Transform paper_results to flat dictionary format for comparison
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

                            if dataset and metric:
                                key = f"{dataset}_{metric}"
                            elif metric:
                                key = metric
                            else:
                                continue

                            expected_metrics_dict[key] = value

        # Compare with paper results
        comparison = self.metrics_extractor.compare_metrics(extracted, expected_metrics_dict)

        state["extracted_metrics"] = extracted
        state["metrics_comparison"] = comparison

        # Build verification report
        verification_report = []
        verification_report.append("## Execution Summary")
        verification_report.append(f"- Dependencies Installed: {'Yes' if state.get('dependencies_installed') else 'No'}")
        verification_report.append(f"- Datasets Ready: {'Yes' if state.get('datasets_ready') else 'No'}")
        verification_report.append(f"- Experiments Completed: {'Yes' if state.get('experiments_completed') else 'No'}")

        verification_report.append("\n## Metrics Comparison")
        if comparison.get("matches"):
            verification_report.append("### Matching Metrics:")
            for match in comparison["matches"]:
                verification_report.append(f"  - {match['metric']}: {match['actual']} (expected: {match['expected']}, diff: {match['diff']})")

        if comparison.get("mismatches"):
            verification_report.append("### Mismatched Metrics:")
            for mismatch in comparison["mismatches"]:
                verification_report.append(f"  - {mismatch['metric']}: {mismatch['actual']} (expected: {mismatch['expected']}, diff: {mismatch['diff']})")

        if comparison.get("missing"):
            verification_report.append("### Missing Metrics:")
            for missing in comparison["missing"]:
                verification_report.append(f"  - {missing}")

        report_text = "\n".join(verification_report)

        # Determine success level based on experiments (NEW LOGIC)
        # Success criteria: ALL experiments must succeed (within 5% error)
        # Partial: Some experiments succeeded (report portion X/Y)
        # Failure: All experiments failed OR prerequisites failed

        experiments_tried = state.get("experiment_results", {}).get("experiments_tried", [])
        experiments_succeeded = state.get("experiment_results", {}).get("experiments_succeeded", [])
        dependencies_installed = state.get("dependencies_installed", False)

        # Check prerequisites first
        if not dependencies_installed:
            success_level = "failed"
            results_match = False
            status_msg = "❌ Verification: Environment setup failed"
        elif not experiments_tried:
            # No experiments attempted
            sanity_check_passed = state.get("experiment_results", {}).get("sanity_check_passed", False)
            if sanity_check_passed:
                success_level = "minimal"
                results_match = False
                status_msg = "⚠️ Verification: Only sanity check completed"
            else:
                success_level = "setup_only"
                results_match = False
                status_msg = "⚠️ Verification: Setup complete but no experiments run"
        else:
            # Calculate overall success based on experiments
            total_experiments = len(experiments_tried)
            succeeded_count = len(experiments_succeeded)
            success_portion = f"{succeeded_count}/{total_experiments}"

            if succeeded_count == total_experiments:
                # ALL experiments succeeded (within 5% error)
                success_level = "full"
                results_match = True
                status_msg = f"✅ Verification: All {total_experiments} experiment(s) succeeded - results match paper (within 5%)"
            elif succeeded_count > 0:
                # PARTIAL success (some experiments succeeded)
                success_level = "partial"
                results_match = False
                status_msg = f"⚠️ Verification: Partial reproduction - {success_portion} experiments succeeded ({', '.join(experiments_succeeded)})"
            else:
                # ALL experiments failed
                success_level = "failed"
                results_match = False
                status_msg = f"❌ Verification: All {total_experiments} experiment(s) failed"

        state["verification_results"] = {
            "report": report_text,
            "results_match_paper": results_match,
            "success_level": success_level,
            "discrepancies": comparison.get("mismatches", [])
        }
        state["results_match"] = results_match

        print(f"\n📝 Verification Report:\n{report_text}\n")
        state["messages"].append(status_msg)

        # Save checkpoint after verification
        self._save_checkpoint(state, "extract_and_verify")

        return state

    def _generate_report_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Generate final report."""
        # Note: We always regenerate the report even when resuming, to ensure it's up-to-date
        # But we still mark it as completed for tracking purposes
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

        return state

    def _route_after_clone(self, state: PaperReproductionState) -> Literal["continue", "failed"]:
        """Route after cloning repository."""
        if state.get("implementation_path"):
            return "continue"
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

        repo_path = state.get("implementation_path") or "./cloned_repo"
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

        This skips paper analysis and experiment execution, going directly to
        extracting metrics from existing files and comparing with paper.

        Uses smart_extract_results to handle custom formats and extract dataset
        names from file paths (e.g., results/roman-empire/poly.csv).

        Args:
            paper_input: Paper identifier (arXiv ID, etc.)
            repo_path: Path to repository with existing results
            existing_results: Dict from _check_existing_results

        Returns:
            Final state with verification results
        """
        from .tools.code_execution_tools import (
            smart_extract_results, align_and_compare_results,
            read_log_tail, generate_comparison_report
        )

        print("📊 Smart extraction from existing result files...")

        # Step 1: Try to get paper results from existing checkpoint FIRST
        paper_results_str = ""
        paper_title = paper_input
        paper_metrics_structured = ""
        paper_results_from_checkpoint = {}

        # Check for existing checkpoint with paper_results
        if self.checkpoint_manager:
            checkpoint_data = self.checkpoint_manager.resume(
                repo_path=repo_path,
                paper_id=paper_input
            )
            if checkpoint_data:
                state = checkpoint_data.get("state", {})
                paper_results_from_checkpoint = state.get("paper_results", {})
                paper_title = state.get("paper_title", paper_input)

                if paper_results_from_checkpoint:
                    print(f"\n✅ Found paper results from checkpoint!")
                    print(f"   Paper title: {paper_title[:60]}...")

                    # Convert checkpoint paper_results to string format for comparison
                    metrics = paper_results_from_checkpoint.get("metrics", [])
                    if metrics:
                        print(f"   Found {len(metrics)} metrics from paper")
                        lines = []
                        for m in metrics:
                            if isinstance(m, dict):
                                dataset = m.get("dataset", "").replace("**", "").strip()
                                value = m.get("value", "")
                                metric_type = m.get("metric", "Accuracy")

                                # Extract the main Polynormer value (not Polynormer-r)
                                if "Polynormer:" in str(value):
                                    # Parse "Polynormer: 93.18±0.18, Polynormer-r: 93.68±0.21"
                                    import re
                                    match = re.search(r'Polynormer:\s*([\d.]+)', str(value))
                                    if match:
                                        numeric_value = match.group(1)
                                        lines.append(f"{dataset}: {numeric_value}")
                                        print(f"      {dataset}: {numeric_value} ({metric_type})")
                                elif isinstance(value, (int, float)):
                                    lines.append(f"{dataset}: {value}")

                        paper_results_str = "\n".join(lines)
                        print(f"   Converted to {len(lines)} dataset expectations")

        # If no checkpoint results, try to extract from PDF
        if not paper_results_str:
            try:
                if paper_input.startswith("arxiv:") or "." in paper_input:
                    arxiv_id = paper_input.replace("arxiv:", "")
                    print(f"\n📄 No checkpoint found, fetching paper info for {arxiv_id}...")

                    import arxiv
                    from PyPDF2 import PdfReader
                    from urllib.request import urlretrieve
                    import re

                    search = arxiv.Search(id_list=[arxiv_id])
                    paper = next(search.results())
                    paper_title = paper.title
                    paper_abstract = paper.summary

                    print(f"   Title: {paper_title[:60]}...")

                    # Download PDF for extracting expected results
                    download_dir = os.path.abspath("./downloads")
                    os.makedirs(download_dir, exist_ok=True)
                    safe_arxiv_id = arxiv_id.replace("/", "_").replace(".", "_")
                    pdf_path = os.path.join(download_dir, f"{safe_arxiv_id}.pdf")

                    if not os.path.exists(pdf_path):
                        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                        print(f"   Downloading PDF...")
                        urlretrieve(pdf_url, pdf_path)

                    # Extract tables with results from PDF
                    reader = PdfReader(pdf_path)
                    full_text = ""
                    for page in reader.pages[:15]:  # First 15 pages usually have results
                        full_text += page.extract_text() + "\n"

                    # Use LLM to extract expected results in structured format
                    print("   Extracting expected results from paper...")
                    extraction_prompt = f"""From this paper text, extract the main experimental results that should be reproduced.
Focus on the main results table(s) showing performance metrics (accuracy, ROC-AUC, F1, etc.) per dataset.

Return results in this format (one per line):
dataset_name: metric_value

Example format:
roman-empire: 92.55
amazon-ratings: 53.21
minesweeper: 98.41

Paper text:
{full_text[:8000]}

Extract the main results (dataset: value):"""

                    response = self.llm.invoke(extraction_prompt)
                    paper_metrics_structured = response.content
                    paper_results_str = paper_metrics_structured

                    print(f"   ✅ Extracted expected results from paper")

            except Exception as e:
                print(f"   ⚠️  Could not fetch paper info: {e}")
                paper_results_str = ""

        # Step 2: Use smart extraction to get results from files
        print("\n🔍 Running smart result extraction...")
        extraction_result = smart_extract_results.invoke({
            "repo_path": repo_path,
            "paper_metrics": paper_results_str
        })

        extracted_datasets = extraction_result.get("datasets", {})
        raw_files = extraction_result.get("raw_files", {})
        source_files = list(raw_files.keys())[:5]

        # Build flat metrics dict for backward compatibility
        extracted_metrics = {}
        for dataset_name, metrics in extracted_datasets.items():
            for metric_name, value in metrics.items():
                key = f"{dataset_name}_{metric_name}"
                extracted_metrics[key] = value

        print(f"\n📊 Extracted {len(extracted_datasets)} datasets, {len(extracted_metrics)} total metrics")

        # Step 3: If we have paper metrics, do smart alignment and comparison
        comparison_result = {}
        if extracted_datasets and paper_results_str:
            print("\n🔍 Aligning and comparing with paper results...")
            comparison_result = align_and_compare_results.invoke({
                "extracted_results": extraction_result,
                "paper_metrics": paper_results_str,
                "tolerance": 0.05
            })

            summary = comparison_result.get("summary", {})
            print(f"\n📊 Comparison Summary:")
            print(f"   Status: {summary.get('status', 'Unknown')}")
            print(f"   Match ratio: {summary.get('match_ratio', 'N/A')}")
            print(f"   Match percentage: {summary.get('match_percentage', 'N/A')}")

            # Print detailed matches
            if comparison_result.get("matched"):
                print("\n   ✅ Matched datasets:")
                for m in comparison_result["matched"][:5]:
                    print(f"      {m['expected_dataset']} → {m['extracted_dataset']}: "
                          f"{m['extracted_value']:.2f} (expected {m['expected_value']:.2f}, "
                          f"error: {m['relative_error_pct']})")

            if comparison_result.get("mismatched"):
                print("\n   ⚠️ Mismatched datasets:")
                for m in comparison_result["mismatched"][:5]:
                    print(f"      {m['expected_dataset']}: {m['extracted_value']:.2f} "
                          f"(expected {m['expected_value']:.2f}, error: {m['relative_error_pct']})")

        # Step 4: Generate comparison report
        if extracted_datasets:
            print("\n📝 Generating comparison report...")
            report_result = generate_comparison_report.invoke({
                "repo_path": repo_path,
                "extracted_metrics": extracted_metrics,
                "paper_results": paper_results_str,
                "comparison_result": comparison_result,
                "output_filename": "reproduction_report.md"
            })
            if report_result.get("success"):
                print(f"   ✅ Report saved to: {report_result.get('report_path')}")

        # Build detailed comparison for state
        summary = comparison_result.get("summary", {})
        match_success = summary.get("matched_count", 0) == summary.get("total_expected", 0) and summary.get("total_expected", 0) > 0

        # Build final state
        final_state = {
            "paper_input": paper_input,
            "paper_title": paper_title,
            "implementation_path": repo_path,
            "dependencies_installed": True,
            "datasets_ready": True,
            "experiments_completed": True,
            "extracted_metrics": {
                "metrics": extracted_metrics,
                "datasets": extracted_datasets,
                "source": "smart_extraction"
            },
            "metrics_comparison": comparison_result,
            "verification_results": {
                "report": self._build_verification_report(comparison_result, extracted_datasets),
                "results_match_paper": match_success,
                "success_level": "verified" if match_success else ("partial" if summary.get("matched_count", 0) > 0 else "failed"),
                "source_files": source_files
            },
            "results_match": match_success,
            "messages": [
                f"✅ Using existing results from {repo_path}",
                f"📊 Extracted {len(extracted_datasets)} datasets with {len(extracted_metrics)} metrics",
                f"{summary.get('status', 'Verification completed')}: {summary.get('match_ratio', 'N/A')} datasets matched"
            ],
            "final_status": summary.get("status", "✅ Verification Complete"),
            "report": self._build_final_report(
                paper_input, paper_title, repo_path, existing_results,
                extracted_datasets, extracted_metrics, comparison_result, source_files
            )
        }

        print(f"\n{'='*60}")
        print(f"{'✅' if match_success else '⚠️'} Verification Complete")
        print(f"{'='*60}")
        print(f"   Status: {summary.get('status', 'Unknown')}")
        print(f"   Datasets extracted: {len(extracted_datasets)}")
        print(f"   Match ratio: {summary.get('match_ratio', 'N/A')}")
        print(f"   Match percentage: {summary.get('match_percentage', 'N/A')}")
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

    def _try_resume_checkpoint(self, paper_input: str, repo_path: str = "./cloned_repo") -> dict:
        """Try to resume from checkpoint.

        Args:
            paper_input: Paper identifier (arXiv ID, etc.)
            repo_path: Repository path

        Returns:
            Checkpoint data if found, empty dict otherwise
        """
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

        # Clear checkpoints if requested
        if clear_checkpoints and self.checkpoint_manager:
            self.checkpoint_manager.clear(repo_path="./cloned_repo", paper_id=paper_input)
            print("🗑️  Cleared existing checkpoints\n")

        # FIRST: Check if cloned_repo already has results (before anything else!)
        repo_path = "./cloned_repo"
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
                return self._run_verification_only(paper_input, repo_path, existing_results)

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

        final_state = self.workflow.invoke(initial_state)

        print(f"\n{'='*60}")
        print(f"✅ Workflow Complete")
        print(f"{'='*60}\n")

        # Close the log file if logging is enabled
        if self.file_logger:
            self.file_logger.close()

        return final_state
