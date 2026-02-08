
import sys
import os
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from utils.metrics_tracker import MetricsTracker

def test_legacy_metrics():
    print("\n=== Testing Legacy Metrics ===")
    tracker = MetricsTracker(enable_live_display=False, phases=MetricsTracker.LEGACY_PHASES)
    
    # Simulate some phases
    tracker.start_phase("analyze_paper")
    time.sleep(0.1)
    tracker.record_tokens(100, 50)
    tracker.end_phase("analyze_paper")
    
    tracker.start_phase("environment_setup")
    time.sleep(0.1)
    tracker.end_phase("environment_setup")
    
    tracker.start_phase("unified_reproduction") # Should be tracked
    tracker.end_phase("unified_reproduction")

    summary = tracker.get_summary()
    print(summary)
    
    assert "analyze_paper" in summary
    assert "supervisor" not in summary
    assert "unified_reproduction" in summary

def test_supervisor_metrics():
    print("\n=== Testing Supervisor Metrics ===")
    tracker = MetricsTracker(enable_live_display=False, phases=MetricsTracker.SUPERVISOR_PHASES)
    
    # Simulate specific supervisor phases
    tracker.start_phase("planning")
    time.sleep(0.1)
    tracker.end_phase("planning")
    
    tracker.start_phase("supervisor")
    time.sleep(0.05)
    tracker.end_phase("supervisor")
    
    tracker.start_phase("execution")
    time.sleep(0.1)
    tracker.end_phase("execution")

    summary = tracker.get_summary()
    print(summary)
    
    assert "planning" in summary
    assert "supervisor" in summary
    assert "unified_reproduction" not in summary
    assert "execution" in summary

if __name__ == "__main__":
    test_legacy_metrics()
    test_supervisor_metrics()
    print("\n✅ Verification successful!")
