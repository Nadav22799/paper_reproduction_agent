"""Specialized agents for paper reproduction."""

__all__ = [
    # Legacy agents
    "UnifiedPaperAnalyzer",
    "EnvironmentSetupAgent",
    "UnifiedReproductionAgent",
    "DiscoveryAgent",
    # Supervisor architecture agents
    "SupervisorAgent",
    "PlanningAgent",
    "CriticAgent",
    "DataPrepAgent",
    "ExecutionAgent",
    "ValidationAgent",
]

_AGENT_MODULES = {
    "UnifiedPaperAnalyzer": ".unified_paper_analyzer",
    "EnvironmentSetupAgent": ".environment_setup_agent",
    "UnifiedReproductionAgent": ".unified_reproduction_agent",
    "DiscoveryAgent": ".discovery_agent",
    "SupervisorAgent": ".supervisor_agent",
    "PlanningAgent": ".planning_agent",
    "CriticAgent": ".critic_agent",
    "DataPrepAgent": ".data_prep_agent",
    "ExecutionAgent": ".execution_agent",
    "ValidationAgent": ".validation_agent",
}


def __getattr__(name):
    """Lazy import to avoid pulling in heavy dependencies at package load time."""
    if name in _AGENT_MODULES:
        import importlib
        module = importlib.import_module(_AGENT_MODULES[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
