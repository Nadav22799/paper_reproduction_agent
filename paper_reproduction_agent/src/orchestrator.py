"""Clean LangGraph Orchestrator - Simplified workflow for paper reproduction."""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
import operator
import os
import subprocess
import shutil


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

    # Overall
    messages: Annotated[list, operator.add]
    final_status: str
    report: str


class PaperReproductionOrchestrator:
    """Simplified orchestrator for paper reproduction workflow."""

    def __init__(self, llm=None, enable_logging=True):
        """Initialize the orchestrator.

        Args:
            llm: Language model to use
            enable_logging: Whether to enable detailed logging to file
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
        except Exception as e:
            print(f"⚠️  Papers with Code API failed: {str(e)[:50]}")

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

                return state

        # No implementation found
        state["final_status"] = "Failed: No implementation found"
        state["messages"].append("❌ No implementation found")
        print("❌ No existing implementation found")

        return state

    def _unified_reproduction_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Run unified reproduction workflow."""
        print("🚀 Starting unified reproduction workflow...")

        code_path = state.get("implementation_path") or "./cloned_repo"

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

        return state

    def _extract_and_verify_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Extract metrics and verify results against paper claims."""
        print("📊 Extracting metrics and verifying results...")

        experiment_output = state.get("experiment_results", {}).get("output", "")
        paper_results = state.get("paper_results", {})

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

        return state

    def _generate_report_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Generate final report."""
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

    def run(self, paper_input: str) -> dict:
        """
        Run the complete paper reproduction workflow.

        Args:
            paper_input: arXiv ID, PDF path, or paper identifier

        Returns:
            Final state with results
        """
        print(f"\n{'='*60}")
        print(f"🚀 Starting Paper Reproduction Workflow (Clean)")
        print(f"{'='*60}\n")

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
