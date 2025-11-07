"""Paper Reproduction Agent - Multi-agent system for reproducing academic papers."""

from .orchestrator import PaperReproductionOrchestrator
from .agents.paper_analyzer import PaperAnalyzerAgent
from .agents.code_searcher import CodeSearcherAgent
from .agents.code_reproducer import CodeReproducerAgent
from .agents.code_verifier import CodeVerifierAgent
from .agents.code_debugger import CodeDebuggerAgent

__version__ = "0.1.0"

__all__ = [
    "PaperReproductionOrchestrator",
    "PaperAnalyzerAgent",
    "CodeSearcherAgent",
    "CodeReproducerAgent",
    "CodeVerifierAgent",
    "CodeDebuggerAgent",
]
