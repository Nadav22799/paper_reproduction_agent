"""Utility functions for the Paper Reproduction Agent."""

# Lazy imports to avoid requiring all dependencies upfront
# This allows individual modules to be imported without needing all dependencies

__all__ = ["create_llm", "get_available_providers", "create_specific_llm"]

# Import functions lazily to avoid dependency issues
def __getattr__(name):
    """Lazy import of module attributes."""
    if name == "create_llm":
        from .llm_factory import create_llm
        return create_llm
    elif name == "get_available_providers":
        from .llm_factory import get_available_providers
        return get_available_providers
    elif name == "create_specific_llm":
        from .llm_factory import create_specific_llm
        return create_specific_llm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
