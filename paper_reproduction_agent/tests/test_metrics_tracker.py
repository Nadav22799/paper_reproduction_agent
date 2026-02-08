"""Unit tests for MetricsTracker."""

import sys
import os
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.metrics_tracker import MetricsTracker, PhaseMetrics


class TestPhaseMetrics:
    """Tests for PhaseMetrics dataclass."""

    def test_duration_not_started(self):
        """Phase not started should have zero duration."""
        phase = PhaseMetrics(name="test")
        assert phase.duration == 0.0

    def test_duration_running(self):
        """Running phase should calculate duration from start to now."""
        phase = PhaseMetrics(name="test", start_time=time.time() - 1.0, status="running")
        assert phase.duration >= 1.0

    def test_duration_completed(self):
        """Completed phase should have fixed accumulated duration."""
        phase = PhaseMetrics(name="test", accumulated_duration=5.0)
        assert phase.duration == 5.0

    def test_llm_time_calculation(self):
        """LLM time should be duration minus experiment time."""
        phase = PhaseMetrics(
            name="test",
            accumulated_duration=10.0,
            experiment_wall_time=7.0
        )
        assert phase.llm_time == 3.0

    def test_llm_time_minimum_zero(self):
        """LLM time should never be negative."""
        start = time.time()
        phase = PhaseMetrics(
            name="test",
            start_time=start,
            end_time=start + 5.0,
            experiment_wall_time=10.0  # More than duration
        )
        assert phase.llm_time == 0.0


class TestMetricsTracker:
    """Tests for MetricsTracker class."""

    def test_initialization(self):
        """Test basic initialization."""
        tracker = MetricsTracker(enable_live_display=False)
        assert len(tracker.metrics.phases) == 6
        assert all(p.status == "pending" for p in tracker.metrics.phases.values())

    def test_workflow_timing(self):
        """Test workflow start/end timing."""
        tracker = MetricsTracker(enable_live_display=False)
        tracker.start_workflow()
        time.sleep(0.1)
        tracker.end_workflow()

        assert tracker.metrics.workflow_start is not None
        assert tracker.metrics.workflow_end is not None
        assert tracker.get_total_duration() >= 0.1

    def test_phase_timing(self):
        """Test phase start/end timing."""
        tracker = MetricsTracker(enable_live_display=False)
        tracker.start_workflow()
        tracker.start_phase("analyze_paper")
        time.sleep(0.1)
        tracker.end_phase("analyze_paper")
        tracker.end_workflow()

        phase = tracker.metrics.phases["analyze_paper"]
        assert phase.status == "completed"
        assert phase.duration >= 0.1

    def test_phase_failure(self):
        """Test marking phase as failed."""
        tracker = MetricsTracker(enable_live_display=False)
        tracker.start_workflow()
        tracker.start_phase("analyze_paper")
        tracker.end_phase("analyze_paper", success=False)

        assert tracker.metrics.phases["analyze_paper"].status == "failed"

    def test_current_phase_tracking(self):
        """Test current phase is tracked correctly."""
        tracker = MetricsTracker(enable_live_display=False)
        tracker.start_workflow()

        assert tracker._current_phase is None
        tracker.start_phase("analyze_paper")
        assert tracker._current_phase == "analyze_paper"
        tracker.end_phase("analyze_paper")
        assert tracker._current_phase is None

    def test_experiment_time_recording(self):
        """Test recording experiment wall time."""
        tracker = MetricsTracker(enable_live_display=False)
        tracker.start_workflow()
        tracker.start_phase("unified_reproduction")
        tracker.record_experiment_time("unified_reproduction", 100.0)
        tracker.record_experiment_time("unified_reproduction", 50.0)  # Additional time
        tracker.end_phase("unified_reproduction")

        phase = tracker.metrics.phases["unified_reproduction"]
        assert phase.experiment_wall_time == 150.0

    def test_token_recording(self):
        """Test recording token usage."""
        tracker = MetricsTracker(enable_live_display=False)
        tracker.start_workflow()
        tracker.start_phase("analyze_paper")
        tracker.record_tokens(1000, 200)
        tracker.record_tokens(500, 100)  # Additional tokens
        tracker.end_phase("analyze_paper")

        assert tracker.metrics.total_llm_tokens_input == 1500
        assert tracker.metrics.total_llm_tokens_output == 300
        assert tracker.metrics.phases["analyze_paper"].llm_tokens_input == 1500
        assert tracker.metrics.phases["analyze_paper"].llm_tokens_output == 300

    def test_token_recording_explicit_phase(self):
        """Test recording tokens with explicit phase name."""
        tracker = MetricsTracker(enable_live_display=False)
        tracker.start_workflow()
        tracker.record_tokens(1000, 200, phase_name="decide_and_clone")

        assert tracker.metrics.phases["decide_and_clone"].llm_tokens_input == 1000
        assert tracker.metrics.phases["decide_and_clone"].llm_tokens_output == 200

    def test_cost_estimation(self):
        """Test cost estimation calculation."""
        tracker = MetricsTracker(
            enable_live_display=False,
            input_cost_per_million=3.0,
            output_cost_per_million=15.0
        )
        tracker.record_tokens(1_000_000, 100_000)  # 1M input, 100K output

        cost = tracker.estimate_cost()
        expected = 3.0 + 1.5  # $3/M input + $15/M * 0.1M output
        assert cost == expected

    def test_cost_estimation_zero_tokens(self):
        """Test cost estimation with no tokens."""
        tracker = MetricsTracker(enable_live_display=False)
        assert tracker.estimate_cost() == 0.0

    def test_total_experiment_time(self):
        """Test aggregating experiment time across phases."""
        tracker = MetricsTracker(enable_live_display=False)
        tracker.record_experiment_time("environment_setup", 30.0)
        tracker.record_experiment_time("unified_reproduction", 120.0)

        assert tracker.get_total_experiment_time() == 150.0

    def test_total_llm_time(self):
        """Test calculating total LLM time."""
        tracker = MetricsTracker(enable_live_display=False)
        tracker.start_workflow()
        time.sleep(0.2)
        tracker.record_experiment_time("unified_reproduction", 0.1)
        tracker.end_workflow()

        total = tracker.get_total_duration()
        experiment = tracker.get_total_experiment_time()
        llm = tracker.get_total_llm_time()

        assert llm >= 0.1  # At least 0.1s of non-experiment time
        assert abs(llm - (total - experiment)) < 0.01

    def test_format_duration(self):
        """Test duration formatting."""
        tracker = MetricsTracker(enable_live_display=False)

        assert tracker._format_duration(0) == "0:00:00"
        assert tracker._format_duration(65) == "0:01:05"
        assert tracker._format_duration(3661) == "1:01:01"

    def test_get_summary(self):
        """Test summary generation."""
        tracker = MetricsTracker(enable_live_display=False)
        tracker.start_workflow()
        tracker.start_phase("analyze_paper")
        tracker.record_tokens(10000, 1000)
        tracker.end_phase("analyze_paper")
        tracker.end_workflow()

        summary = tracker.get_summary()

        assert "WORKFLOW METRICS SUMMARY" in summary
        assert "analyze_paper" in summary
        assert "OK" in summary
        assert "Input Tokens" in summary
        assert "Output Tokens" in summary
        assert "Estimated Cost" in summary

    def test_get_phase_stats(self):
        """Test getting stats for a specific phase."""
        tracker = MetricsTracker(enable_live_display=False)
        tracker.start_workflow()
        tracker.start_phase("analyze_paper")
        tracker.record_tokens(1000, 200)
        tracker.record_experiment_time("analyze_paper", 5.0)
        tracker.end_phase("analyze_paper")

        stats = tracker.get_phase_stats("analyze_paper")

        assert stats["name"] == "analyze_paper"
        assert stats["status"] == "completed"
        assert stats["llm_tokens_input"] == 1000
        assert stats["llm_tokens_output"] == 200
        assert stats["experiment_wall_time"] == 5.0

    def test_get_phase_stats_invalid_phase(self):
        """Test getting stats for invalid phase returns None."""
        tracker = MetricsTracker(enable_live_display=False)
        assert tracker.get_phase_stats("invalid_phase") is None

    def test_get_all_stats(self):
        """Test getting comprehensive stats."""
        tracker = MetricsTracker(enable_live_display=False)
        tracker.start_workflow()
        tracker.start_phase("analyze_paper")
        tracker.record_tokens(1000, 200)
        tracker.end_phase("analyze_paper")
        tracker.end_workflow()

        stats = tracker.get_all_stats()

        assert "total_duration" in stats
        assert "total_experiment_time" in stats
        assert "total_llm_time" in stats
        assert "total_tokens_input" in stats
        assert "total_tokens_output" in stats
        assert "estimated_cost" in stats
        assert "phases" in stats
        assert len(stats["phases"]) == 6

    def test_custom_cost_rates(self):
        """Test custom cost rates."""
        tracker = MetricsTracker(
            enable_live_display=False,
            input_cost_per_million=0.5,  # Cheaper model
            output_cost_per_million=2.0
        )
        tracker.record_tokens(1_000_000, 1_000_000)

        cost = tracker.estimate_cost()
        assert cost == 0.5 + 2.0  # $0.5 + $2.0


class TestMetricsTrackerLiveDisplay:
    """Tests for live display functionality (without actually displaying)."""

    def test_live_display_disabled(self):
        """Test that display thread is not started when disabled."""
        tracker = MetricsTracker(enable_live_display=False)
        tracker.start_workflow()
        tracker.end_workflow()

        assert tracker._display_thread is None

    def test_live_display_enabled(self):
        """Test that display thread is started when enabled."""
        tracker = MetricsTracker(enable_live_display=True, update_interval=1)
        tracker.start_workflow()

        assert tracker._display_thread is not None
        assert tracker._display_thread.is_alive()

        tracker.end_workflow()
        time.sleep(0.1)  # Give thread time to stop
        assert not tracker._display_thread.is_alive()


class TestMultiplePhases:
    """Tests for multiple phase scenarios."""

    def test_all_phases_workflow(self):
        """Test running through all phases."""
        tracker = MetricsTracker(enable_live_display=False)
        tracker.start_workflow()

        phases = [
            "analyze_paper",
            "decide_and_clone",
            "environment_setup",
            "unified_reproduction",
            "extract_and_verify",
            "generate_report"
        ]

        for phase in phases:
            tracker.start_phase(phase)
            tracker.record_tokens(100, 50)
            if phase == "unified_reproduction":
                tracker.record_experiment_time(phase, 60.0)
            time.sleep(0.01)
            tracker.end_phase(phase)

        tracker.end_workflow()

        # All phases should be completed
        for phase_name in phases:
            assert tracker.metrics.phases[phase_name].status == "completed"

        # Total tokens should be accumulated
        assert tracker.metrics.total_llm_tokens_input == 600  # 100 * 6
        assert tracker.metrics.total_llm_tokens_output == 300  # 50 * 6

        # Experiment time should be recorded
        assert tracker.get_total_experiment_time() == 60.0

    def test_partial_workflow_failure(self):
        """Test workflow with a failed phase."""
        tracker = MetricsTracker(enable_live_display=False)
        tracker.start_workflow()

        tracker.start_phase("analyze_paper")
        tracker.end_phase("analyze_paper")

        tracker.start_phase("decide_and_clone")
        tracker.end_phase("decide_and_clone", success=False)

        # Remaining phases not started
        tracker.end_workflow()

        assert tracker.metrics.phases["analyze_paper"].status == "completed"
        assert tracker.metrics.phases["decide_and_clone"].status == "failed"
        assert tracker.metrics.phases["environment_setup"].status == "pending"
