"""
Unit tests for HierarchicalContextManager and token-based ContextManager.

Tests the 2025 best practices implementation:
- Three-tier storage (Hot → Warm → Cold)
- Token-based budget management
- Multi-factor relevance scoring
- Semantic retrieval
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest


class TestHierarchicalContextManager:
    """Tests for HierarchicalContextManager."""

    @pytest.fixture
    def context_manager(self):
        """Create a HierarchicalContextManager instance."""
        from src.utils.hierarchical_context import HierarchicalContextManager
        return HierarchicalContextManager(
            model_name="gpt-4",
            hot_capacity=5,
            max_tokens=10000
        )

    def test_initialization(self, context_manager):
        """Test manager initializes correctly."""
        assert context_manager.hot_capacity == 5
        assert context_manager.max_tokens == 10000
        assert len(context_manager.hot_context) == 0
        assert context_manager.hot_tokens == 0

    def test_token_counting(self, context_manager):
        """Test accurate token counting."""
        # Simple text
        text = "Hello world"
        tokens = context_manager.count_tokens(text)
        assert tokens > 0
        assert tokens < len(text)  # Tokens should be fewer than characters

        # Empty text
        assert context_manager.count_tokens("") == 0

        # Longer text
        long_text = "This is a longer sentence with multiple words." * 10
        long_tokens = context_manager.count_tokens(long_text)
        assert long_tokens > tokens

    def test_add_entry(self, context_manager):
        """Test adding entries to hot storage."""
        entry_id = context_manager.add(
            content="Paper uses BERT for text classification",
            source="paper_analyzer",
            entry_type="result",
            importance=1.0
        )

        assert entry_id != ""
        assert len(context_manager.hot_context) == 1
        assert context_manager.hot_tokens > 0

        # Check entry is stored correctly
        entry = context_manager.hot_context[entry_id]
        assert entry.content == "Paper uses BERT for text classification"
        assert entry.source == "paper_analyzer"
        assert entry.entry_type == "result"
        assert entry.importance == 1.0

    def test_add_empty_content_skipped(self, context_manager):
        """Test that empty content is skipped."""
        entry_id = context_manager.add(
            content="",
            source="test",
            entry_type="observation"
        )

        assert entry_id == ""
        assert len(context_manager.hot_context) == 0

    def test_auto_importance(self, context_manager):
        """Test automatic importance based on entry type."""
        # Result type should have high importance
        context_manager.add("Test result", "test", "result")
        entry = list(context_manager.hot_context.values())[0]
        assert entry.importance == 1.0

        context_manager.clear_hot()

        # Debug type should have low importance
        context_manager.add("Test debug", "test", "debug")
        entry = list(context_manager.hot_context.values())[0]
        assert entry.importance == 0.3

    def test_compaction(self, context_manager):
        """Test automatic compaction when hot capacity exceeded."""
        # Add more entries than hot_capacity
        for i in range(10):
            context_manager.add(f"Entry {i}", "test", "observation")

        # Hot storage should be compacted
        assert len(context_manager.hot_context) <= context_manager.hot_capacity

    def test_retrieve_basic(self, context_manager):
        """Test basic retrieval."""
        context_manager.add("BERT model for NLP", "paper_analyzer", "result", 1.0)
        context_manager.add("Dataset: GLUE benchmark", "paper_analyzer", "result", 0.9)
        context_manager.add("Debug: loaded config", "system", "debug", 0.3)

        results = context_manager.retrieve("What model is used?", max_tokens=1000)

        assert len(results) > 0
        # Results should be sorted by relevance
        assert results[0]["relevance"] >= results[-1]["relevance"]

    def test_retrieve_respects_token_budget(self, context_manager):
        """Test that retrieval respects token budget."""
        # Add many entries
        for i in range(20):
            context_manager.add(f"Entry {i} with some content " * 10, "test", "observation")

        results = context_manager.retrieve("test query", max_tokens=500)

        total_tokens = sum(r.get("tokens", 0) for r in results)
        assert total_tokens <= 500

    def test_relevance_scoring(self, context_manager):
        """Test multi-factor relevance scoring."""
        # High importance, high authority source
        context_manager.add("Accuracy: 92.5%", "paper_analyzer", "result", 1.0)

        # Low importance, low authority source
        context_manager.add("Debug: step 1", "system", "debug", 0.3)

        results = context_manager.retrieve("accuracy", max_tokens=1000)

        # Paper analyzer result should rank higher
        if len(results) >= 2:
            paper_result = next((r for r in results if "92.5" in r["content"]), None)
            debug_result = next((r for r in results if "Debug" in r["content"]), None)

            if paper_result and debug_result:
                assert paper_result["relevance"] > debug_result["relevance"]

    def test_compile_context(self, context_manager):
        """Test context compilation for LLM."""
        context_manager.add("Paper: BERT", "paper_analyzer", "result", 1.0)
        context_manager.add("Dataset: GLUE", "paper_analyzer", "result", 0.9)

        compiled = context_manager.compile_context(
            query="What is the paper about?",
            system_prompt="You are a helpful assistant.",
            max_tokens=5000
        )

        assert len(compiled) > 0
        assert "BERT" in compiled or "GLUE" in compiled

    def test_get_stats(self, context_manager):
        """Test statistics retrieval."""
        context_manager.add("Test entry", "test", "observation")

        stats = context_manager.get_stats()

        assert stats["hot_entries"] == 1
        assert stats["hot_tokens"] > 0
        assert stats["max_tokens"] == 10000
        assert stats["hot_capacity"] == 5

    def test_clear_hot(self, context_manager):
        """Test clearing hot storage."""
        context_manager.add("Test entry", "test", "observation")
        assert len(context_manager.hot_context) == 1

        context_manager.clear_hot()

        assert len(context_manager.hot_context) == 0
        assert context_manager.hot_tokens == 0


class TestContextManagerTokenBased:
    """Tests for token-based ContextManager."""

    @pytest.fixture
    def context_manager(self):
        """Create a ContextManager instance."""
        from src.utils.context_manager import ContextManager
        return ContextManager(
            max_tokens=10000,
            sliding_window_size=3
        )

    def test_initialization(self, context_manager):
        """Test manager initializes with token-based limits."""
        assert context_manager.max_tokens == 10000
        assert context_manager.sliding_window_size == 3
        # Legacy field should be approximate
        assert context_manager.max_context_chars == 10000 * 4

    def test_token_counting(self, context_manager):
        """Test accurate token counting."""
        text = "Hello world, this is a test."
        tokens = context_manager.count_tokens(text)

        assert tokens > 0
        assert tokens < len(text)

    def test_legacy_parameter_conversion(self):
        """Test that legacy max_context_chars is converted to tokens."""
        from src.utils.context_manager import ContextManager

        # Using legacy parameter (should be converted)
        cm = ContextManager(max_context_chars=200000, sliding_window_size=3)

        # Should be converted to tokens (200000 / 4 = 50000)
        assert cm.max_tokens == 50000

    def test_get_context_stats_tokens(self, context_manager):
        """Test that stats report tokens, not just chars."""
        from langchain_core.messages import HumanMessage

        messages = [
            HumanMessage(content="This is a test message with some content.")
        ]

        stats = context_manager.get_context_stats(messages)

        assert "total_tokens" in stats
        assert "tokens_remaining" in stats
        assert stats["total_tokens"] > 0
        assert stats["tokens_remaining"] < context_manager.max_tokens


class TestIntegration:
    """Integration tests for context management."""

    def test_hierarchical_and_context_manager_together(self):
        """Test that both managers can work together."""
        from src.utils.hierarchical_context import HierarchicalContextManager
        from src.utils.context_manager import ContextManager

        hierarchical = HierarchicalContextManager(hot_capacity=10, max_tokens=50000)
        pruning = ContextManager(max_tokens=50000, sliding_window_size=3)

        # Add to hierarchical
        hierarchical.add("Paper uses BERT", "paper_analyzer", "result", 1.0)

        # Both should use consistent token counting
        test_text = "This is a test sentence."
        h_tokens = hierarchical.count_tokens(test_text)
        p_tokens = pruning.count_tokens(test_text)

        # Should be similar (might differ slightly due to tokenizer init)
        assert abs(h_tokens - p_tokens) <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
