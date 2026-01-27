"""Specialized agents for paper reproduction."""

from .unified_paper_analyzer import UnifiedPaperAnalyzer
from .environment_setup_agent import EnvironmentSetupAgent
from .unified_reproduction_agent import UnifiedReproductionAgent
from .discovery_agent import DiscoveryAgent

# Supervisor architecture agents
from .supervisor_agent import SupervisorAgent
from .planning_agent import PlanningAgent
from .critic_agent import CriticAgent
from .data_prep_agent import DataPrepAgent
from .execution_agent import ExecutionAgent
from .validation_agent import ValidationAgent

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
