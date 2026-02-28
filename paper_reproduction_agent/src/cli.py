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
@click.option("--critic-mode", type=click.Choice(["auto", "critic"]), default=None,
              help="auto=fully autonomous, critic=ask before dangerous actions")
def reproduce(paper_input, no_logging, no_checkpoints, max_iterations, max_cycles, critic_mode):
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

        # Critic mode selection
        if critic_mode is None:
            click.echo("\n🛡️  Security Mode:")
            click.echo("1. Auto (fully autonomous, recommended for speed)")
            click.echo("2. Critic (ask before dangerous actions, recommended for safety)")
            mode_choice = click.prompt(
                "Please choose",
                type=click.Choice(["1", "2"]),
                default="1",
                show_default=True,
            )
            critic_mode = "auto" if mode_choice == "1" else "critic"
        os.environ["CRITIC_MODE"] = critic_mode
        click.echo(f"🛡️  Critic mode: {critic_mode.upper()}")

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
            "user_input_required": None,
            "user_input_response": None,
            "waiting_for_user": False,
        }

        # Try to resume from checkpoint
        resumed_state = None
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

        # Start/resume metrics tracking
        if not no_checkpoints and resumed_state and orchestrator.metrics_tracker.metrics.workflow_start:
            # Metrics were restored from checkpoint - resume (preserves original start time)
            orchestrator.metrics_tracker.resume_workflow()
            click.echo("   📊 Resumed metrics tracking (original start time preserved)")
        else:
            # Fresh start
            orchestrator.metrics_tracker.start_workflow()

        # Start experiment DB tracking
        run_id = orchestrator.start_run_tracking(
            paper_id=paper_input,
            config={
                "max_cycles": max_cycles,
                "experiment_selection_mode": initial_state.get("experiment_selection_mode", "single"),
            },
            critic_mode=critic_mode,
        )
        result = {}  # ensure defined for finally block

        try:
            result = orchestrator.workflow.invoke(initial_state)

            # Handle user input pause/resume loop
            while result.get("final_status") == "waiting_for_user_input":
                req = result.get("user_input_required", {})
                click.echo("\n" + "=" * 60)
                click.echo("  ACTION REQUIRED - Reproduction Paused")
                click.echo("=" * 60)
                click.echo(
                    f"\n{req.get('description', 'User action needed before continuing:')}\n"
                )

                for i, item in enumerate(req.get("items", []), 1):
                    click.echo(f"  {i}. [{item.get('name', 'Unknown')}]")
                    if item.get("description"):
                        click.echo(f"     {item['description']}")
                    if item.get("instructions"):
                        click.echo(f"     Instructions: {item['instructions']}")
                    if item.get("env_var"):
                        click.echo(f"     Set as: export {item['env_var']}=<your_value>")
                click.echo()

                # Collect user responses
                responses = {}
                for item in req.get("items", []):
                    if item.get("type") == "api_key":
                        value = click.prompt(
                            f"  Enter {item.get('name', 'value')}",
                            hide_input=True,
                        )
                        responses[item.get("env_var") or item.get("name", "key")] = value
                    else:
                        click.confirm(
                            f"  Have you completed: {item.get('name', 'this step')}?",
                            default=True,
                        )
                        responses[item.get("name", "step")] = "provided"

                # Resume workflow with user input
                result["user_input_response"] = responses
                result["waiting_for_user"] = False
                result["final_status"] = ""
                click.echo("\n🔄 Resuming reproduction...\n")
                result = orchestrator.workflow.invoke(result)

        finally:
            # Stop metrics tracking
            orchestrator.metrics_tracker.end_workflow()
            click.echo(orchestrator.metrics_tracker.get_summary())
            # Finalize experiment DB record
            orchestrator.finalize_run_tracking(run_id, result)

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

    load_dotenv()

    click.echo("Running internal verification...")

    checks = []

    # Check 1: Core Imports
    try:
        import importlib.util
        for _pkg in ("torch", "langchain", "langgraph"):
            if importlib.util.find_spec(_pkg) is None:
                raise ImportError(_pkg)

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

    # Check 3: Required environment variables (API keys)
    _required_env_groups = [
        ("LLM Keys", ["GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"]),
    ]
    for group_name, vars_in_group in _required_env_groups:
        found = [v for v in vars_in_group if os.getenv(v)]
        if found:
            checks.append((f"✅ {group_name}: {found[0]} is set", True))
        else:
            checks.append((f"❌ {group_name}: none of {vars_in_group} are set", False))

    # Check 4: ChromaDB availability
    try:
        import chromadb
        _client = chromadb.Client()
        checks.append(("✅ ChromaDB: embedded client initializes OK", True))
    except ImportError:
        checks.append(("❌ ChromaDB: not installed", False))
    except Exception as e:
        checks.append((f"⚠️  ChromaDB: installed but init failed ({e})", False))

    # Check 5: GPU availability (via resource_detector if present)
    try:
        from src.utils.resource_detector import detect_resources
        resources = detect_resources()
        gpu_info = resources.get("gpu", {})
        if gpu_info.get("available"):
            checks.append((f"✅ GPU: {gpu_info.get('name', 'available')}", True))
        else:
            checks.append(("ℹ️  GPU: not available (CPU-only mode)", True))
    except Exception:
        try:
            import torch
            if torch.cuda.is_available():
                checks.append((f"✅ GPU: {torch.cuda.get_device_name(0)}", True))
            else:
                checks.append(("ℹ️  GPU: not available (CPU-only mode)", True))
        except Exception:
            checks.append(("ℹ️  GPU: could not detect (torch not loaded)", True))

    # Check 6: Storage backend
    storage_backend = os.getenv("STORAGE_BACKEND", "local")
    checks.append((f"ℹ️  Storage backend: {storage_backend.upper()}", True))
    if storage_backend == "gcs":
        gcp_ok = bool(os.getenv("GCP_PROJECT_ID")) and bool(os.getenv("GCP_BUCKET_NAME"))
        if gcp_ok:
            checks.append(("✅ GCS: GCP_PROJECT_ID and GCP_BUCKET_NAME are set", True))
        else:
            checks.append(("❌ GCS: GCP_PROJECT_ID or GCP_BUCKET_NAME missing", False))
    elif storage_backend == "s3":
        s3_ok = bool(os.getenv("AWS_S3_BUCKET"))
        if s3_ok:
            checks.append(("✅ S3: AWS_S3_BUCKET is set", True))
        else:
            checks.append(("❌ S3: AWS_S3_BUCKET missing", False))

    # Report
    click.echo("\nSystem Health Check:")
    click.echo("-" * 55)
    all_passed = True
    for msg, passed in checks:
        click.echo(msg)
        if not passed:
            all_passed = False

    click.echo("-" * 55)
    if all_passed:
        click.echo("\n✨ System is ready for reproduction!")
    else:
        click.echo("\n⚠️ System has issues. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    cli()
