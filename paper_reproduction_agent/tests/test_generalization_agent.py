"""Smoke tests for GeneralizationAgent."""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.generalization_agent import GeneralizationAgent


class TestGeneralizationAgentInit:
    """Tests for GeneralizationAgent construction."""

    def test_init_defaults(self):
        """Agent initializes with default parameters (no LLM call)."""
        # GeneralizationAgent creates an LLM via factory if none provided,
        # so we pass a mock to avoid needing API keys.
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        agent = GeneralizationAgent(llm=mock_llm, max_iterations=5)
        assert agent.max_iterations == 5
        assert agent.llm is mock_llm
        assert len(agent.tools) > 0

    def test_tools_include_guarded(self):
        """Agent should have guarded execution tools."""
        from unittest.mock import MagicMock
        agent = GeneralizationAgent(llm=MagicMock(), max_iterations=5)
        tool_names = [t.name for t in agent.tools]
        assert "read_file" in tool_names
        assert "list_directory" in tool_names
        assert "search_error_solution" in tool_names


class TestAnalyzeResult:
    """Tests for _analyze_result parsing logic."""

    def _make_agent(self):
        from unittest.mock import MagicMock
        return GeneralizationAgent(llm=MagicMock(), max_iterations=5)

    def test_parse_success(self):
        """Detect 'Generalization Status: PASSED' in output."""
        agent = self._make_agent()
        from unittest.mock import MagicMock
        msg = MagicMock()
        msg.content = "Generalization Status: ✅ PASSED\nExternal Dataset: CIFAR-100"
        result = agent._analyze_result([msg])
        assert result["generalization_success"] is True
        assert result["external_dataset"] == "cifar-100"
        assert result["phase_status"]["generalization"] == "completed"

    def test_parse_failure(self):
        """No success marker means failure."""
        agent = self._make_agent()
        from unittest.mock import MagicMock
        msg = MagicMock()
        msg.content = "The experiment failed to converge on external data."
        result = agent._analyze_result([msg])
        assert result["generalization_success"] is False
        assert result["phase_status"]["generalization"] == "completed"

    def test_parse_blocked(self):
        """Detect blocked state requiring user input."""
        agent = self._make_agent()
        from unittest.mock import MagicMock
        msg = MagicMock()
        msg.content = "BLOCKED: requires user input to adapt the model architecture."
        result = agent._analyze_result([msg])
        assert result["generalization_success"] is False
        assert result["phase_status"]["generalization"] == "blocked"
        assert "user_input_required" in result

    def test_empty_messages(self):
        """Empty message list returns failure gracefully."""
        agent = self._make_agent()
        result = agent._analyze_result([])
        assert result["generalization_success"] is False

    def test_return_keys(self):
        """Result dict has all expected keys."""
        agent = self._make_agent()
        result = agent._analyze_result([])
        expected_keys = {
            "generalization_success", "external_dataset",
            "novel_metrics", "baseline_metrics",
            "generalization_report", "phase_status",
        }
        assert expected_keys.issubset(set(result.keys()))
