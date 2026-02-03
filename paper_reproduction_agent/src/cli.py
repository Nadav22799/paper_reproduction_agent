import click
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure we can import from src
sys.path.append(str(Path(__file__).parent.parent))

# Apply SQLite fix early
try:
    from src.utils.sqlite_fix import apply_sqlite_fix
    apply_sqlite_fix()
except ImportError:
    pass

from src.orchestrator import PaperReproductionOrchestrator
from src.config import ReproductionConfig


@click.group()
def cli():
    """Paper Reproduction Agent CLI - Reproduce research papers with AI."""
    pass


@cli.command()
@click.argument("paper_input")
@click.option("--no-logging", is_flag=True, help="Disable file logging")
@click.option("--no-checkpoints", is_flag=True, help="Disable checkpoint/resume")
@click.option("--max-iterations", default=50, help="Maximum tool iterations")
@click.option("--max-cycles", default=5, help="Maximum recovery/validation cycles")
def reproduce(paper_input, no_logging, no_checkpoints, max_iterations, max_cycles):
    """Reproduce a paper given its arXiv ID, URL, or Path.

    PAPER_INPUT can be:
    - arXiv ID (e.g., 2310.12345)
    - URL to PDF (e.g., https://arxiv.org/pdf/...)
    - Path to local PDF
    """
    # Force UTF-8 for stdout to support emojis on Windows
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    load_dotenv()

    # setup logging
    from src.utils.file_logger import TeeOutput
    from datetime import datetime

    tee = None
    if not no_logging:
        config = ReproductionConfig()
        log_dir = Path(config.logs_path)
        log_dir.mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"execution_{timestamp}.log"

        tee = TeeOutput(log_file)
        sys.stdout = tee
        click.echo(f"📝 Logging to: {log_file}")

    try:
        click.echo(f"🚀 Starting reproduction for: {paper_input}")
        click.echo(f"🔄 Max cycles: {max_cycles}")

        # Disable orchestrator's internal logging since we capture stdout
        orchestrator = PaperReproductionOrchestrator(
            enable_logging=False, enable_checkpoints=not no_checkpoints
        )

        # Run the workflow
        initial_state = {
            "paper_input": paper_input,
            "paper_title": "Unknown",
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
            "completed_phases": [],
            "messages": [],
            "final_status": "",
            "report": "",
            "max_cycles": max_cycles,
        }

        # Try to resume from checkpoint
        if not no_checkpoints:
             resumed_state = orchestrator._try_resume_checkpoint(paper_input)
             if resumed_state:
                 # Update initial state with resumed data
                 initial_state.update(resumed_state)
                 # Ensure max_cycles from CLI overrides resumed state if needed, or stick to CLI
                 initial_state["max_cycles"] = max_cycles
                 click.echo(f"✨ Resumed state contains {len(initial_state.get('completed_phases', []))} completed phases")
        
        # User Selection for Experiment Level (skip if already set in resumed state)
        if (
            not no_logging
            and not initial_state.get("experiment_selection_mode")
        ):  # Only prompt in interactive mode AND if not already decided
            click.echo("\n🎯 Select Reproduction Level:")
            click.echo("1. Single Experiment (Main Result)")
            click.echo("2. All Experiments (Full Reproduction)")
            click.echo("3. Custom Selection")
            selection = click.prompt(
                "Please choose",
                type=click.Choice(["1", "2", "3"]),
                default="1",
                show_default=True,
            )

            mode_map = {"1": "single", "2": "all", "3": "custom"}
            initial_state["experiment_selection_mode"] = mode_map[selection]

            if selection == "3":
                custom_input = click.prompt("Enter experiment names (comma separated)")
                initial_state["custom_experiment_list"] = [
                    e.strip() for e in custom_input.split(",")
                ]

            click.echo(
                f"✅ Selected mode: {initial_state['experiment_selection_mode'].upper()}"
            )
            if initial_state.get("custom_experiment_list"):
                click.echo(f"   Experiments: {initial_state['custom_experiment_list']}")

        # Start metrics tracking
        orchestrator.metrics_tracker.start_workflow()

        try:
            result = orchestrator.workflow.invoke(initial_state)
        finally:
            # Stop metrics tracking
            orchestrator.metrics_tracker.end_workflow()
            click.echo(orchestrator.metrics_tracker.get_summary())

        click.echo("\n" + "=" * 60)
        click.echo(f"🏁 Final Status: {result.get('final_status', 'Unknown')}")
        click.echo("=" * 60)

        if result.get("report"):
            click.echo("\n📄 Reproduction Report:")
            click.echo(result["report"])

    except Exception as e:
        click.echo(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        if tee:
            sys.stdout = tee.terminal
            tee.close()
            if not no_logging:
                click.echo(f"\n📝 Full execution log saved to: {log_file}")


@cli.command()
def verify():
    """Run self-verification tests."""
    # Force UTF-8 for stdout to support emojis on Windows
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    click.echo("Running internal verification...")

    checks = []

    # Check 1: Core Imports
    try:
        import torch
        import langchain
        import langgraph

        checks.append(("✅ Core Dependencies", True))
    except ImportError as e:
        checks.append((f"❌ Missing Dependency: {e.name}", False))

    # Check 2: Directory Structure
    req_dirs = ["logs", "downloads", "src"]
    for d in req_dirs:
        if os.path.exists(d):
            checks.append((f"✅ Directory found: {d}", True))
        else:
            checks.append((f"❌ Missing directory: {d}", False))

    # Report
    click.echo("\nSystem Health Check:")
    all_passed = True
    for msg, passed in checks:
        click.echo(msg)
        if not passed:
            all_passed = False

    if all_passed:
        click.echo("\n✨ System is ready for reproduction!")
    else:
        click.echo("\n⚠️ System has issues. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    cli()
