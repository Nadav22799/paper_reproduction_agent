"""Tools for the Paper Reproduction Agent system."""

__all__ = [
    "read_file",
    "search_file",
    "list_directory",
    "execute_shell_command",
    "execute_python_script",
    "execute_python_code",
    "create_python_file",
    "check_python_compatibility",
    "smart_install_dependencies",
    "search_error_solution",
]


def __getattr__(name):
    """Lazy import to avoid pulling in heavy dependencies at package load time."""
    if name in __all__:
        from . import code_execution_tools
        return getattr(code_execution_tools, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
