import click
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
console = Console()

# Ensure we can import from src
sys.path.append(str(Path(__file__).parent.parent))

# Apply SQLite fix early
try:
    from src.utils.sqlite_fix import apply_sqlite_fix
    apply_sqlite_fix()
except ImportError:
    pass

from src.orchestrator import PaperReproductionOrchestrator  # noqa: E402
from src.config import ReproductionConfig  # noqa: E402


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Paper Reproduction Agent CLI - Reproduce research papers with AI."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(reproduce)


@cli.command()
@click.argument("paper_input", required=False)
@click.option("--no-logging", is_flag=True, help="Disable file logging")
@click.option("--no-checkpoints", is_flag=True, help="Disable checkpoint/resume")
@click.option("--max-iterations", type=int, default=None, help="Maximum tool iterations")
@click.option("--max-cycles", type=int, default=None, help="Maximum recovery/validation cycles")
@click.option("--critic-mode", type=click.Choice(["auto", "critic"]), default=None,
              help="auto=fully autonomous, critic=ask before dangerous actions")
@click.option("--mode", type=click.Choice(["reproduce", "reproduce+generalize", "generalize"]),
              default=None,
              help="reproduce=standard, reproduce+generalize=with generalization test, generalize=generalization only (needs prior checkpoint)")
def reproduce(paper_input, no_logging, no_checkpoints, max_iterations, max_cycles, critic_mode, mode):
    """Reproduce a paper given its arXiv ID, URL, or Path.

    PAPER_INPUT can be:
    - arXiv ID (e.g., 2310.12345)
    - URL to PDF (e.g., https://arxiv.org/pdf/...)
    - Path to local PDF
    """
    import questionary
    
    # Interactive Menu if no paper is provided
    if not paper_input:
        console.print(Panel.fit(
            Text.assemble(
                ("Paper Reproduction Agent ", "bold cyan"),
                ("CLI", "italic yellow")
            ),
            subtitle="Interactive Setup",
            border_style="bright_blue"
        ))
        
        paper_input = questionary.text("📄 Enter the Paper identifier (arXiv ID, URL, or local path):").ask()
        if not paper_input:
            console.print("[red]❌ Paper input is required. Exiting.[/red]")
            sys.exit(1)
            
        if mode is None:
            mode = questionary.select(
                "📋 Select the Run Mode:",
                choices=[
                    "reproduce", 
                    "reproduce+generalize", 
                    "generalize"
                ]
            ).ask()
            
        if critic_mode is None:
            critic_mode = questionary.select(
                "🛡️  Security Mode (Critic pauses execution on dangerous commands for your approval):",
                choices=[
                    questionary.Choice("Auto (fully autonomous)", value="auto"),
                    questionary.Choice("Critic (ask before dangerous actions)", value="critic")
                ],
                default="auto"
            ).ask()
            
    # Set default values if not provided by CLI or interactive menu
    max_iterations = max_iterations if max_iterations is not None else 50
    max_cycles = max_cycles if max_cycles is not None else 5
    mode = mode or "reproduce"
    critic_mode = critic_mode or "auto"

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
        console.print(f"📝 Logging to: {log_file}")

    try:
        # Re-print title if we skipped the interactive menu
        if paper_input and 'settings' not in locals():
             console.print(Panel.fit(
                Text.assemble(
                    ("Paper Reproduction Agent ", "bold cyan"),
                    ("CLI", "italic yellow")
                ),
                subtitle="Reproduce research papers with AI",
                border_style="bright_blue"
            ))
            
        console.print(f"🚀 Starting reproduction for: [bold cyan]{paper_input}[/bold cyan]")
        console.print(f"🔄 Max cycles: [bold]{max_cycles}[/bold]")
        console.print(f"📋 Run mode: [bold]{mode}[/bold]")
        console.print(f"🛡️  Critic mode: [bold green]{critic_mode.upper()}[/bold green]")
        os.environ["CRITIC_MODE"] = critic_mode

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
            "run_mode": mode,
            "generalization_results": None,
            "generalization_success": False,
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
                 console.print(f"✨ Resumed state contains {len(initial_state.get('completed_phases', []))} completed phases")

        # Validate generalize-only mode
        if mode == "generalize":
            if not resumed_state or not initial_state.get("results_match"):
                console.print("[bold red]❌ Generalize-only mode requires a prior successful reproduction (with matching results).[/bold red]")
                console.print("   Run with --mode reproduce first, then re-run with --mode generalize.")
                sys.exit(1)
            console.print("🔬 Generalize-only mode: skipping reproduction, running generalization directly")

        if (
            not no_logging
            and not initial_state.get("experiment_selection_mode")
        ):  # Only prompt in interactive mode AND if not already decided
            import questionary
            selection = questionary.select(
                "🎯 Select Reproduction Level:",
                choices=[
                    questionary.Choice("1. Single Experiment (Main Result)", "1"),
                    questionary.Choice("2. All Experiments (Full Reproduction)", "2"),
                    questionary.Choice("3. Custom Selection", "3"),
                ]
            ).ask()
            
            if not selection:
                selection = "1"

            mode_map = {"1": "single", "2": "all", "3": "custom"}
            initial_state["experiment_selection_mode"] = mode_map[selection]

            if selection == "3":
                custom_input = questionary.text("Enter experiment names (comma separated):").ask() or ""
                initial_state["custom_experiment_list"] = [
                    e.strip() for e in custom_input.split(",") if e.strip()
                ]

            console.print(
                f"✅ Selected mode: [bold green]{initial_state['experiment_selection_mode'].upper()}[/bold green]"
            )
            if initial_state.get("custom_experiment_list"):
                console.print(f"   Experiments: {initial_state['custom_experiment_list']}")

        # Start/resume metrics tracking
        if not no_checkpoints and resumed_state and orchestrator.metrics_tracker.metrics.workflow_start:
            # Metrics were restored from checkpoint - resume (preserves original start time)
            orchestrator.metrics_tracker.resume_workflow()
            console.print("   📊 Resumed metrics tracking (original start time preserved)")
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
            # If resuming with pending user input, go straight into the input loop
            # (Planning was already completed & checkpointed but .env wasn't written yet)
            if resumed_state and initial_state.get("final_status") == "waiting_for_user_input":
                console.print("[bold yellow]⚠  Resumed with pending user input — collecting before continuing[/bold yellow]")
                result = dict(initial_state)
            else:
                result = orchestrator.workflow.invoke(initial_state)

            # Handle user input pause/resume loop
            while result.get("final_status") == "waiting_for_user_input":
                req = result.get("user_input_required", {})
                items = req.get("items", [])

                # Silence metrics while we own the terminal
                orchestrator.metrics_tracker.pause_display()

                import re as _re
                import questionary

                # Strip markdown bold/italic markers that LLMs write in checklist text
                def _clean(text: str) -> str:
                    return _re.sub(r"\*+|`", "", text or "").strip()

                # ── Summary panel ─────────────────────────────────────────────
                summary = Text()
                summary.append(f"\n{_clean(req.get('description', 'User action needed before continuing.'))}\n\n")
                for i, item in enumerate(items, 1):
                    summary.append(f"  {i}. ", style="bold yellow")
                    summary.append(f"{_clean(item.get('name', 'Unknown'))}", style="bold white")
                    item_type = item.get("type", "")
                    if item_type:
                        summary.append(f"  [{item_type}]", style="dim")
                    summary.append("\n")
                    if item.get("description"):
                        summary.append(f"     {_clean(item['description'])}\n", style="dim")
                    if item.get("instructions"):
                        summary.append(f"     ℹ  {_clean(item['instructions'])}\n", style="cyan")
                    if item.get("env_var"):
                        summary.append(f"     → {item['env_var']}=<value>  (will be saved to .env)\n", style="green")

                console.print()
                console.print(Panel(
                    summary,
                    title="[bold yellow]⚠  ACTION REQUIRED — Reproduction Paused[/bold yellow]",
                    border_style="yellow",
                    padding=(0, 2),
                ))

                # ── Per-item prompts, each wrapped in a mini panel ────────────
                responses = {}
                for item in items:
                    name = _clean(item.get("name", "Unknown"))
                    itype = item.get("type", "confirm")
                    env_var = item.get("env_var") or name
                    instructions = _clean(item.get("instructions", ""))

                    prompt_body = Text()
                    if instructions:
                        prompt_body.append(f"{instructions}\n\n", style="dim")

                    if itype == "api_key":
                        prompt_body.append("Enter the value below (input is hidden):", style="bold")
                        console.print(Panel(
                            prompt_body,
                            title=f"[bold cyan]🔑  {name}[/bold cyan]",
                            border_style="cyan",
                            padding=(0, 2),
                        ))
                        value = questionary.password(f"  {name}:").ask()
                        responses[env_var] = value or ""

                    elif itype == "text":
                        prompt_body.append("Type your response below:", style="bold")
                        console.print(Panel(
                            prompt_body,
                            title=f"[bold cyan]✏  {name}[/bold cyan]",
                            border_style="cyan",
                            padding=(0, 2),
                        ))
                        value = questionary.text(f"  {name}:").ask()
                        responses[env_var] = value or ""

                    else:
                        prompt_body.append("Confirm when ready to continue:", style="bold")
                        console.print(Panel(
                            prompt_body,
                            title=f"[bold cyan]✅  {name}[/bold cyan]",
                            border_style="cyan",
                            padding=(0, 2),
                        ))
                        ans = questionary.confirm(f"  Have you completed: {name}?").ask()
                        responses[name] = "provided" if ans else "skipped"

                # ── Optional free-form notes ───────────────────────────────────
                if req.get("requires_text_input", False):
                    console.print(Panel(
                        Text("Provide any additional context, notes, or instructions for the agent.", style="dim"),
                        title="[bold cyan]💬  Additional Information[/bold cyan]",
                        border_style="cyan",
                        padding=(0, 2),
                    ))
                    additional_info = questionary.text("  Your notes:").ask()
                    if additional_info:
                        responses["user_notes"] = additional_info

                # ── Write credentials to cloned_repo/.env ─────────────────────
                code_path = (
                    result.get("implementation_path")
                    or getattr(orchestrator.config, "repo_path", None)
                )
                env_entries = [
                    (item.get("env_var"), responses.get(item.get("env_var") or _clean(item.get("name", ""))))
                    for item in items
                    if item.get("env_var")
                    and responses.get(item.get("env_var") or _clean(item.get("name", "")))
                    not in (None, "", "provided", "skipped")
                ]
                if code_path and env_entries:
                    env_file = os.path.join(code_path, ".env")
                    mode = "a" if os.path.exists(env_file) else "w"
                    with open(env_file, mode, encoding="utf-8") as f:
                        if mode == "a":
                            f.write("\n")
                        f.write("# Credentials provided by user — Paper Reproduction Agent\n")
                        for env_var, value in env_entries:
                            f.write(f"{env_var}={value}\n")

                    # Also inject into current process so agents can use them immediately
                    for env_var, value in env_entries:
                        os.environ[env_var] = value

                    # Update the checklist to reference the .env file
                    checklist_path = result.get("checklist_path", "")
                    if checklist_path and os.path.exists(checklist_path):
                        with open(checklist_path, "r", encoding="utf-8") as f:
                            checklist = f.read()
                        # Use forward slashes in the note so it's valid on both Windows and WSL
                        env_file_display = env_file.replace("\\", "/")
                        env_note = (
                            f"\n> **Credentials saved:** `{env_file_display}`  \n"
                            f"> Load with `source {env_file_display}` (Linux/WSL/macOS) "
                            f"or `dotenv` (Windows) before running.\n"
                        )
                        if env_note.strip() not in checklist:
                            # Insert after the User Input Required section header.
                            # Use a lambda to avoid re interpreting backslashes in
                            # Windows paths (e.g. \U) as regex escape sequences.
                            checklist = _re.sub(
                                r"(##\s*User Input Required\s*\n)",
                                lambda m: m.group(1) + env_note,
                                checklist,
                                count=1,
                            )
                            with open(checklist_path, "w", encoding="utf-8") as f:
                                f.write(checklist)

                    console.print(Panel(
                        Text(f"Credentials written to: {env_file}", style="bold green"),
                        title="[bold green]💾  Saved[/bold green]",
                        border_style="green",
                        padding=(0, 2),
                    ))

                # ── Resume ────────────────────────────────────────────────────
                orchestrator.metrics_tracker.resume_display()

                result["user_input_response"] = responses
                result["waiting_for_user"] = False
                result["final_status"] = ""
                console.print()
                console.print(Panel(
                    Text("All inputs collected — continuing the workflow.", style="bold green"),
                    title="[bold green]🔄  Resuming Reproduction[/bold green]",
                    border_style="green",
                    padding=(0, 2),
                ))
                console.print()
                result = orchestrator.workflow.invoke(result)

        finally:
            # Stop metrics tracking
            orchestrator.metrics_tracker.end_workflow()
            console.print(Panel(orchestrator.metrics_tracker.get_summary(), title="📊 WORKFLOW METRICS SUMMARY", border_style="cyan"))
            # Finalize experiment DB record
            orchestrator.finalize_run_tracking(run_id, result)

        console.print(Panel(f"Final Status: [bold]{result.get('final_status', 'Unknown')}[/bold]", title="🏁 Workflow Complete", border_style="green"))

        if result.get("report"):
            console.print("\n📄 [bold]Reproduction Report:[/bold]")
            from rich.markdown import Markdown
            console.print(Markdown(result["report"]))

    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        if tee:
            sys.stdout = tee.terminal
            tee.close()
            if not no_logging:
                console.print(f"\n📝 Full execution log saved to: {log_file}")


@cli.command()
def verify():
    """Run self-verification tests."""
    from rich.console import Console
    console = Console()
    
    # Force UTF-8 for stdout to support emojis on Windows
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    load_dotenv()

    console.print("Running internal verification...")

    checks = []

    # Check 1: Core Imports
    try:
        import importlib.util
        for _pkg in ("langchain", "langgraph"):
            if importlib.util.find_spec(_pkg) is None:
                raise ImportError(name=_pkg)

        checks.append(("[green]✅ Core Dependencies[/green]", True))
    except ImportError as e:
        checks.append((f"[red]❌ Missing Dependency: {e.name}[/red]", False))

    # Check 2: Directory Structure
    req_dirs = ["logs", "downloads", "src"]
    for d in req_dirs:
        if os.path.exists(d):
            checks.append((f"[green]✅ Directory found: {d}[/green]", True))
        else:
            checks.append((f"[red]❌ Missing directory: {d}[/red]", False))

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
    console.print("\n[bold]System Health Check:[/bold]")
    console.print("-" * 55)
    all_passed = True
    for msg, passed in checks:
        console.print(msg)
        if not passed:
            all_passed = False

    click.echo("-" * 55)
    if all_passed:
        console.print("\n✨ [bold green]System is ready for reproduction![/bold green]")
    else:
        console.print("\n⚠️ [bold red]System has issues. Please check the errors above.[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    cli()
