"""Metrics tracking for observability - timing, costs, and progress display."""

import time
import sys
import os
from datetime import timedelta
from typing import Dict, Optional
from dataclasses import dataclass, field
from threading import Thread, Event


@dataclass
class PhaseMetrics:
    """Metrics for a single workflow phase."""

    name: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    accumulated_duration: float = 0.0  # Total duration across multiple execution cycles
    llm_tokens_input: int = 0
    llm_tokens_output: int = 0
    experiment_wall_time: float = 0.0  # Time spent waiting on experiments
    status: str = "pending"  # pending, running, completed, failed

    @property
    def duration(self) -> float:
        """Calculate phase duration in seconds (current + accumulated)."""
        # If currently running, add current elapsed time to accumulated
        if self.status == "running" and self.start_time is not None:
             return self.accumulated_duration + (time.time() - self.start_time)
        return self.accumulated_duration

    @property
    def llm_time(self) -> float:
        """Estimated LLM time (duration minus experiment wait time)."""
        return max(0, self.duration - self.experiment_wall_time)

    def to_dict(self) -> dict:
        """Serialize phase metrics to dictionary for checkpoint saving."""
        return {
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "accumulated_duration": self.accumulated_duration,
            "llm_tokens_input": self.llm_tokens_input,
            "llm_tokens_output": self.llm_tokens_output,
            "experiment_wall_time": self.experiment_wall_time,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PhaseMetrics":
        """Restore phase metrics from dictionary."""
        return cls(
            name=data["name"],
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            accumulated_duration=data.get("accumulated_duration", 0.0),
            llm_tokens_input=data.get("llm_tokens_input", 0),
            llm_tokens_output=data.get("llm_tokens_output", 0),
            experiment_wall_time=data.get("experiment_wall_time", 0.0),
            status=data.get("status", "pending"),
        )


@dataclass
class WorkflowMetrics:
    """Aggregated metrics for entire workflow."""

    phases: Dict[str, PhaseMetrics] = field(default_factory=dict)
    workflow_start: Optional[float] = None
    workflow_end: Optional[float] = None
    embedding_model: str = "Unknown"  # Track which model is used
    total_llm_tokens_input: int = 0
    total_llm_tokens_output: int = 0

    # Additional token tracking
    total_embedding_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0

    # Cost rates (per 1M tokens) - configurable
    input_cost_per_million: float = 3.00  # Default: GPT-4 Turbo input
    output_cost_per_million: float = 15.00  # Default: GPT-4 Turbo output
    embedding_cost_per_million: float = 0.10  # Default: OpenAI ada-002

    def to_dict(self) -> dict:
        """Serialize workflow metrics to dictionary for checkpoint saving."""
        return {
            "phases": {name: phase.to_dict() for name, phase in self.phases.items()},
            "workflow_start": self.workflow_start,
            "workflow_end": self.workflow_end,
            "embedding_model": self.embedding_model,
            "total_llm_tokens_input": self.total_llm_tokens_input,
            "total_llm_tokens_output": self.total_llm_tokens_output,
            "total_embedding_tokens": self.total_embedding_tokens,
            "total_reasoning_tokens": self.total_reasoning_tokens,
            "total_cache_creation_tokens": self.total_cache_creation_tokens,
            "total_cache_read_tokens": self.total_cache_read_tokens,
            "input_cost_per_million": self.input_cost_per_million,
            "output_cost_per_million": self.output_cost_per_million,
            "embedding_cost_per_million": self.embedding_cost_per_million,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowMetrics":
        """Restore workflow metrics from dictionary."""
        workflow = cls(
            input_cost_per_million=data.get("input_cost_per_million", 3.00),
            output_cost_per_million=data.get("output_cost_per_million", 15.00),
            embedding_cost_per_million=data.get("embedding_cost_per_million", 0.10),
        )
        workflow.workflow_start = data.get("workflow_start")
        workflow.workflow_end = data.get("workflow_end")
        workflow.embedding_model = data.get("embedding_model", "Unknown")
        workflow.total_llm_tokens_input = data.get("total_llm_tokens_input", 0)
        workflow.total_llm_tokens_output = data.get("total_llm_tokens_output", 0)
        workflow.total_embedding_tokens = data.get("total_embedding_tokens", 0)
        workflow.total_reasoning_tokens = data.get("total_reasoning_tokens", 0)
        workflow.total_cache_creation_tokens = data.get("total_cache_creation_tokens", 0)
        workflow.total_cache_read_tokens = data.get("total_cache_read_tokens", 0)

        # Restore phases
        for name, phase_data in data.get("phases", {}).items():
            workflow.phases[name] = PhaseMetrics.from_dict(phase_data)

        return workflow


class MetricsTracker:
    """Tracks timing, costs, and provides live progress display."""

    LEGACY_PHASES = [
        "analyze_paper",
        "decide_and_clone",
        "environment_setup",
        "unified_reproduction",
        "extract_and_verify",
        "generate_report",
    ]

    SUPERVISOR_PHASES = [
        "analyze_paper",
        "decide_and_clone",
        "planning",
        "supervisor",
        "critic",
        "environment_setup",
        "data_prep",
        "execution",
        "validation",
        "generate_report",
    ]

    def __init__(
        self,
        enable_live_display: bool = True,
        update_interval: int = 15,
        input_cost_per_million: float = 3.00,
        output_cost_per_million: float = 15.00,
        embedding_cost_per_million: float = 0.10,
        phases: Optional[list] = None,
    ):
        """
        Initialize MetricsTracker.

        Args:
            enable_live_display: Whether to show live progress updates
            update_interval: Seconds between live display updates
            input_cost_per_million: Cost per 1M input tokens
            output_cost_per_million: Cost per 1M output tokens
            embedding_cost_per_million: Cost per 1M embedding tokens
            phases: List of phases to track. Defaults to LEGACY_PHASES.
        """
        # Load defaults from env if not provided
        if input_cost_per_million == 3.00:  # Default value check
            input_cost_per_million = float(os.getenv("LLM_INPUT_COST_PER_M", "3.00"))
        if output_cost_per_million == 15.00:  # Default value check
            output_cost_per_million = float(os.getenv("LLM_OUTPUT_COST_PER_M", "15.00"))
        if embedding_cost_per_million == 0.10:  # Default value check
            embedding_cost_per_million = float(os.getenv("EMBEDDING_COST_PER_M", "0.10"))

        self.metrics = WorkflowMetrics(
            input_cost_per_million=input_cost_per_million,
            output_cost_per_million=output_cost_per_million,
            embedding_cost_per_million=embedding_cost_per_million,
        )
        self.enable_live_display = enable_live_display
        self.update_interval = update_interval
        self._stop_display = Event()
        self._display_thread: Optional[Thread] = None
        self._current_phase: Optional[str] = None
        
        # Determine which phases to track
        self.tracked_phases = phases if phases else self.LEGACY_PHASES

        # Initialize phase metrics
        for phase in self.tracked_phases:
            self.metrics.phases[phase] = PhaseMetrics(name=phase)

    def set_embedding_model(self, model_name: str):
        """Set the embedding model name for reporting."""
        self.metrics.embedding_model = model_name

    def start_workflow(self):
        """Mark workflow start and begin live display if enabled."""
        self.metrics.workflow_start = time.time()
        if self.enable_live_display:
            self._start_live_display()

    def end_workflow(self):
        """Mark workflow end and stop live display."""
        self.metrics.workflow_end = time.time()
        if self._display_thread:
            self._stop_display.set()
            self._display_thread.join(timeout=2)
            # Clear the live status line
            if self.enable_live_display:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()

    def start_phase(self, phase_name: str):
        """Mark phase start."""
        if phase_name in self.metrics.phases:
            phase = self.metrics.phases[phase_name]
            phase.start_time = time.time()
            phase.status = "running"
            self._current_phase = phase_name

    def end_phase(self, phase_name: str, success: bool = True):
        """Mark phase end."""
        if phase_name in self.metrics.phases:
            phase = self.metrics.phases[phase_name]
            
            # Accumulate duration from this run
            if phase.start_time:
                duration = time.time() - phase.start_time
                phase.accumulated_duration += duration
                phase.start_time = None  # Reset prevents double counting
            
            phase.end_time = time.time()
            phase.status = "completed" if success else "failed"
            if self._current_phase == phase_name:
                self._current_phase = None

    def record_experiment_time(self, phase_name: str, wall_time: float):
        """Record time spent waiting on experiment execution."""
        if phase_name in self.metrics.phases:
            self.metrics.phases[phase_name].experiment_wall_time += wall_time

    def record_tokens(
        self, input_tokens: int, output_tokens: int, phase_name: Optional[str] = None
    ):
        """Record token usage."""
        self.metrics.total_llm_tokens_input += input_tokens
        self.metrics.total_llm_tokens_output += output_tokens

        phase = phase_name or self._current_phase
        if phase and phase in self.metrics.phases:
            self.metrics.phases[phase].llm_tokens_input += input_tokens
            self.metrics.phases[phase].llm_tokens_output += output_tokens

    def record_embedding_tokens(
        self, tokens: int, phase_name: Optional[str] = None
    ):
        """Record embedding token usage."""
        self.metrics.total_embedding_tokens += tokens

    def record_reasoning_tokens(
        self, tokens: int, phase_name: Optional[str] = None
    ):
        """Record reasoning/thinking token usage (Claude, OpenAI o1/o3, Gemini)."""
        self.metrics.total_reasoning_tokens += tokens

    def record_cache_tokens(
        self, creation: int = 0, read: int = 0, phase_name: Optional[str] = None
    ):
        """Record cache token usage (Anthropic cache_creation/cache_read)."""
        self.metrics.total_cache_creation_tokens += creation
        self.metrics.total_cache_read_tokens += read

    def estimate_cost(self) -> float:
        """Estimate total cost from token usage with provider-aware cache pricing."""
        import os

        input_cost = (
            self.metrics.total_llm_tokens_input / 1_000_000
        ) * self.metrics.input_cost_per_million
        output_cost = (
            self.metrics.total_llm_tokens_output / 1_000_000
        ) * self.metrics.output_cost_per_million

        # Embedding cost
        embedding_cost = (
            self.metrics.total_embedding_tokens / 1_000_000
        ) * self.metrics.embedding_cost_per_million

        # Cache pricing varies by provider:
        # - Gemini: cache_read at 25% of input rate (75% discount)
        # - Claude: cache_read at 10% of input rate (90% discount)
        provider = os.getenv("LLM_PROVIDER", "").lower()
        if not provider:
            # Auto-detect from keys
            if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
                provider = "gemini"
            elif os.getenv("ANTHROPIC_API_KEY"):
                provider = "claude"

        if provider in ["claude", "anthropic"]:
            cache_read_discount = 0.10  # 90% off (Claude)
            cache_creation_multiplier = 1.25  # Claude charges 1.25x for cache creation
        else:
            cache_read_discount = 0.25  # 75% off (Gemini)
            cache_creation_multiplier = 1.0  # Gemini no extra charge for cache creation

        cache_creation_cost = (
            self.metrics.total_cache_creation_tokens / 1_000_000
        ) * self.metrics.input_cost_per_million * cache_creation_multiplier
        cache_read_cost = (
            self.metrics.total_cache_read_tokens / 1_000_000
        ) * self.metrics.input_cost_per_million * cache_read_discount

        # NOTE: reasoning_tokens are already included in total_llm_tokens_output
        # by Gemini and Claude. We track them for breakdown only.

        return (
            input_cost
            + output_cost
            + embedding_cost
            + cache_creation_cost
            + cache_read_cost
        )

    def get_total_duration(self) -> float:
        """Get total workflow duration in seconds."""
        if self.metrics.workflow_start is None:
            return 0.0
        end = self.metrics.workflow_end or time.time()
        return end - self.metrics.workflow_start

    def get_total_experiment_time(self) -> float:
        """Get total experiment wait time across all phases."""
        return sum(p.experiment_wall_time for p in self.metrics.phases.values())

    def get_total_llm_time(self) -> float:
        """Get estimated LLM/agent time (total - experiment time)."""
        return max(0, self.get_total_duration() - self.get_total_experiment_time())

    def _start_live_display(self):
        """Start background thread for live progress updates."""
        self._stop_display.clear()
        self._display_thread = Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()

    def _display_loop(self):
        """Background loop that prints progress updates."""
        while not self._stop_display.wait(timeout=self.update_interval):
            self._print_live_status()

    def _print_live_status(self):
        """Print current progress status."""
        elapsed = self.get_total_duration()
        cost = self.estimate_cost()
        total_tokens = (
            self.metrics.total_llm_tokens_input + self.metrics.total_llm_tokens_output
        )

        # Build status line
        status_parts = [
            f"[{self._format_duration(elapsed)}]",
            f"Phase: {self._current_phase or 'idle'}",
            f"Est. Cost: ${cost:.4f}",
            f"Tokens: {total_tokens:,}",
        ]

        # Print on new line with distinctive prefix (visible even with interleaved output)
        status_line = " | ".join(status_parts)
        status_line = " | ".join(status_parts)
        print(f"\n⏱️  PROGRESS: {status_line}", flush=True)

    def print_intermediate_summary(self):
        """Print a brief summary of cost and time so far."""
        if not self.enable_live_display:
            return

        elapsed = self.get_total_duration()
        cost = self.estimate_cost()
        print(
            f"\n💰 Current Metrics: Duration: {self._format_duration(elapsed)} | Cost: ${cost:.4f}\n"
        )

    def _format_duration(self, seconds: float) -> str:
        """Format duration as HH:MM:SS."""
        return str(timedelta(seconds=int(seconds)))

    def get_summary(self) -> str:
        """Generate comprehensive final summary report."""
        lines = []
        lines.append("")
        lines.append("=" * 70)
        lines.append("WORKFLOW METRICS SUMMARY")
        lines.append("=" * 70)

        # Total duration
        total_duration = self.get_total_duration()
        lines.append(f"\nTotal Duration: {self._format_duration(total_duration)}")

        # Phase breakdown with per-phase costs
        lines.append("\n--- Phase Breakdown ---")
        total_phase_cost = 0.0
        for phase_name in self.tracked_phases:
            phase = self.metrics.phases[phase_name]
            # Check status instead of start_time, because completed phases have start_time=None
            if phase.status != "pending":
                status_icon = {
                    "completed": " OK ",
                    "failed": "FAIL",
                    "running": " .. ",
                    "pending": " -- ",
                }
                icon = status_icon.get(phase.status, " ?? ")
                phase_cost = self.get_phase_cost(phase_name)
                total_phase_cost += phase_cost
                phase_tokens = phase.llm_tokens_input + phase.llm_tokens_output
                lines.append(
                    f"  {phase_name:20} [{icon}] "
                    f"Time: {self._format_duration(phase.duration):>8} | "
                    f"Tokens: {phase_tokens:>8,} | "
                    f"Cost: ${phase_cost:.4f}"
                )
            else:
                lines.append(f"  {phase_name:20} [ -- ] Not started")

        # Time breakdown
        total_experiment_time = self.get_total_experiment_time()
        llm_time = self.get_total_llm_time()
        lines.append("\n--- Time Breakdown ---")
        lines.append(f"  LLM/Agent Time:     {self._format_duration(llm_time)}")
        lines.append(
            f"  Experiment Time:    {self._format_duration(total_experiment_time)}"
        )
        lines.append(f"  Total:              {self._format_duration(total_duration)}")

        # Config info
        lines.append("\n--- Configuration ---")
        lines.append(f"  Embedding Model:    {self.metrics.embedding_model}")

        # Cost breakdown
        lines.append("\n--- Cost Estimate ---")
        lines.append(f"  Input Tokens:      {self.metrics.total_llm_tokens_input:>12,}")
        lines.append(f"  Output Tokens:     {self.metrics.total_llm_tokens_output:>12,}")
        if self.metrics.total_embedding_tokens > 0:
            lines.append(f"  Embedding Tokens:  {self.metrics.total_embedding_tokens:>12,}")
        if self.metrics.total_reasoning_tokens > 0:
            lines.append(f"  Reasoning Tokens:  {self.metrics.total_reasoning_tokens:>12,}")
        if self.metrics.total_cache_creation_tokens > 0:
            lines.append(f"  Cache Creation:    {self.metrics.total_cache_creation_tokens:>12,}")
        if self.metrics.total_cache_read_tokens > 0:
            lines.append(f"  Cache Read:        {self.metrics.total_cache_read_tokens:>12,}")
        lines.append(f"  Estimated Cost: ${self.estimate_cost():.4f}")
        lines.append(
            f"  (Rates: ${self.metrics.input_cost_per_million}/M input, "
            f"${self.metrics.output_cost_per_million}/M output)"
        )

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)

    def get_phase_cost(self, phase_name: str) -> float:
        """Calculate estimated cost for a specific phase based on its tokens."""
        if phase_name not in self.metrics.phases:
            return 0.0
        phase = self.metrics.phases[phase_name]
        input_cost = (phase.llm_tokens_input / 1_000_000) * self.metrics.input_cost_per_million
        output_cost = (phase.llm_tokens_output / 1_000_000) * self.metrics.output_cost_per_million
        return input_cost + output_cost

    def get_phase_stats(self, phase_name: str) -> Optional[Dict]:
        """Get stats for a specific phase."""
        if phase_name not in self.metrics.phases:
            return None

        phase = self.metrics.phases[phase_name]
        return {
            "name": phase.name,
            "status": phase.status,
            "duration": phase.duration,
            "experiment_wall_time": phase.experiment_wall_time,
            "llm_time": phase.llm_time,
            "llm_tokens_input": phase.llm_tokens_input,
            "llm_tokens_output": phase.llm_tokens_output,
        }

    def get_all_stats(self) -> Dict:
        """Get comprehensive stats dictionary."""
        return {
            "total_duration": self.get_total_duration(),
            "total_experiment_time": self.get_total_experiment_time(),
            "total_llm_time": self.get_total_llm_time(),
            "total_tokens_input": self.metrics.total_llm_tokens_input,
            "total_tokens_output": self.metrics.total_llm_tokens_output,
            "total_embedding_tokens": self.metrics.total_embedding_tokens,
            "total_reasoning_tokens": self.metrics.total_reasoning_tokens,
            "total_cache_creation_tokens": self.metrics.total_cache_creation_tokens,
            "total_cache_read_tokens": self.metrics.total_cache_read_tokens,
            "estimated_cost": self.estimate_cost(),
            "phases": {name: self.get_phase_stats(name) for name in self.tracked_phases},
        }

    def to_dict(self) -> dict:
        """
        Serialize metrics tracker state for checkpoint saving.

        Note: Thread state (_display_thread, _stop_display) is NOT serialized.
        The live display will be restarted on resume if enabled.

        Returns:
            Dictionary containing serializable metrics state
        """
        return {
            "metrics": self.metrics.to_dict(),
            "enable_live_display": self.enable_live_display,
            "update_interval": self.update_interval,
            "current_phase": self._current_phase,
            "tracked_phases": list(self.tracked_phases),
            "checkpoint_timestamp": time.time(),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        resume_workflow: bool = False,
    ) -> "MetricsTracker":
        """
        Restore metrics tracker from checkpoint data.

        Args:
            data: Dictionary from to_dict()
            resume_workflow: If True, restarts the live display thread

        Returns:
            Restored MetricsTracker instance
        """
        # Validate data structure
        if not isinstance(data, dict):
            raise ValueError("Invalid metrics data: expected dict")

        saved_metrics = data.get("metrics", {})
        if not saved_metrics:
            # No metrics data - return fresh tracker
            return cls(
                enable_live_display=data.get("enable_live_display", True),
                update_interval=data.get("update_interval", 15),
                phases=data.get("tracked_phases"),
            )

        # Extract cost rates from saved metrics
        tracker = cls(
            enable_live_display=data.get("enable_live_display", True),
            update_interval=data.get("update_interval", 15),
            input_cost_per_million=saved_metrics.get("input_cost_per_million", 3.00),
            output_cost_per_million=saved_metrics.get("output_cost_per_million", 15.00),
            embedding_cost_per_million=saved_metrics.get("embedding_cost_per_million", 0.10),
            phases=data.get("tracked_phases"),
        )

        # Restore the metrics object (overwrite the fresh one)
        tracker.metrics = WorkflowMetrics.from_dict(saved_metrics)

        # Restore current phase
        tracker._current_phase = data.get("current_phase")

        # Handle interrupted phases (were running when checkpoint was saved)
        checkpoint_time = data.get("checkpoint_timestamp", time.time())
        for phase_name, phase in tracker.metrics.phases.items():
            if phase.status == "running":
                # Phase was interrupted - accumulate partial duration
                if phase.start_time:
                    partial_duration = checkpoint_time - phase.start_time
                    phase.accumulated_duration += partial_duration
                    phase.start_time = None
                phase.status = "pending"  # Will be restarted

        # If resuming, restart live display if workflow was in progress
        if resume_workflow and tracker.enable_live_display:
            if tracker.metrics.workflow_start and not tracker.metrics.workflow_end:
                tracker._start_live_display()

        return tracker

    def resume_workflow(self):
        """
        Resume metrics tracking after checkpoint restore.

        Unlike start_workflow(), this preserves the original workflow_start time
        and only restarts the live display thread if needed.
        """
        if self.metrics.workflow_start is None:
            # No previous workflow - start fresh
            self.start_workflow()
            return

        # Workflow was previously started - just restart live display
        if self.enable_live_display and self._display_thread is None:
            self._start_live_display()

        print("   Resumed metrics from original start time")
