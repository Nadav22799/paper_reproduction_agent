"""Paper Reproduction Agent - Multi-agent system for reproducing academic papers."""

__version__ = "0.1.0"

__all__ = [
    "PaperReproductionOrchestrator",
    "UnifiedPaperAnalyzer",
    "UnifiedReproductionAgent",
]


def __getattr__(name):
    """Lazy import to avoid pulling in heavy dependencies at package load time."""
    if name == "PaperReproductionOrchestrator":
        from .orchestrator import PaperReproductionOrchestrator
        return PaperReproductionOrchestrator
    elif name == "UnifiedPaperAnalyzer":
        from .agents.unified_paper_analyzer import UnifiedPaperAnalyzer
        return UnifiedPaperAnalyzer
    elif name == "UnifiedReproductionAgent":
        from .agents.unified_reproduction_agent import UnifiedReproductionAgent
        return UnifiedReproductionAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
