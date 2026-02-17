"""Tool Guard - Per-tool-call safety validation for ReAct agents.

Provides a lightweight wrapper that intercepts dangerous tool calls
(execute_shell_command, execute_python_code, start_background_process)
with compiled regex checks. Two tiers:

1. ALWAYS_BLOCKED: Universally dangerous (rm -rf /, sudo, eval, etc.)
   → Blocked in both auto and critic modes.

2. ASK_OR_SUGGEST: Potentially legitimate but risky (git clone, curl|bash).
   → Auto mode: blocked with alternative suggestion.
   → Critic mode: pauses and asks user for approval.

Zero LLM cost. Sub-millisecond overhead per check.
"""

import re
import copy
from typing import List, Tuple, Optional
from functools import wraps


# === Tier 1: Always blocked (universally dangerous) ===
ALWAYS_BLOCKED: List[Tuple[str, str]] = [
    (r"rm\s+-rf\s+/", "Dangerous: recursive delete from root"),
    (r"rm\s+-rf\s+~", "Dangerous: recursive delete from home"),
    (r"rm\s+-rf\s+\*", "Dangerous: recursive delete with wildcard"),
    (r"rm\s+-rf\s+\.", "Dangerous: recursive delete of current directory"),
    (r"(?:kill\s+-9|pkill)\s+", "Process killing not allowed"),
    (r"sudo\s+", "No privilege escalation allowed"),
    (r"pip\s+install\s+--user", "Use virtual environment, not --user"),
    (r"conda\s+activate\s+base", "Do not modify base environment"),
    (r"conda\s+install\s+-n\s+base", "Do not modify base environment"),
    (r"pip\s+install.*--break-system-packages", "Do not break system packages"),
    (r"chmod\s+777", "Overly permissive file permissions"),
    (r"eval\s*\(", "No eval of arbitrary code"),
    (r"exec\s*\(", "No exec of arbitrary code"),
    (r"os\.system\s*\(", "Use subprocess instead of os.system"),
    (r"__import__\s*\(", "No dynamic imports"),
]

# === Tier 2: Ask in critic mode, suggest alternative in auto mode ===
ASK_OR_SUGGEST: List[Tuple[str, str, str]] = [
    (
        r"git\s+clone",
        "Clone operations are handled by orchestrator",
        "Download as archive instead: wget https://github.com/<user>/<repo>/archive/main.tar.gz",
    ),
    (
        r"curl.*\|\s*(?:bash|sh)",
        "Piping curl to shell is risky",
        "Download the file first with 'curl -O <url>', inspect it, then run separately",
    ),
    (
        r"wget.*\|\s*(?:bash|sh)",
        "Piping wget to shell is risky",
        "Download the file first with 'wget <url>', inspect it, then run separately",
    ),
    (
        r"(?:^|&&\s*|;\s*|\|\|\s*|\n)pip\s+install\s+",
        "Bare pip install may modify host environment",
        "Use 'conda run -n <env> pip install ...' or 'micromamba run -n <env> pip install ...' instead",
    ),
]

# Combined list for backward compatibility with CriticAgent
FORBIDDEN_PATTERNS: List[Tuple[str, str]] = [
    (p, r) for p, r in ALWAYS_BLOCKED
] + [
    (p, r) for p, r, _ in ASK_OR_SUGGEST
]

# Pre-compile all patterns at module load time
_compiled_always = [(re.compile(p, re.IGNORECASE), reason) for p, reason in ALWAYS_BLOCKED]
_compiled_ask = [(re.compile(p, re.IGNORECASE), reason, suggestion) for p, reason, suggestion in ASK_OR_SUGGEST]


def check_command(command: str) -> Optional[str]:
    """Check a command string against forbidden patterns.

    Returns None if safe, or a reason string if blocked.
    Only checks Tier 1 (always blocked) patterns.
    """
    for pattern, reason in _compiled_always:
        if pattern.search(command):
            return reason
    return None


def check_command_full(command: str) -> Optional[Tuple[str, str, int]]:
    """Check a command against both tiers.

    Returns None if safe, or (reason, suggestion, tier) if matched.
    tier=1 for always-blocked, tier=2 for ask-or-suggest.
    """
    for pattern, reason in _compiled_always:
        if pattern.search(command):
            return (reason, "", 1)
    for pattern, reason, suggestion in _compiled_ask:
        if pattern.search(command):
            return (reason, suggestion, 2)
    return None


def guard_tool(original_tool, mode: str = "auto"):
    """Wrap a LangChain tool with per-call safety guards.

    Args:
        original_tool: The LangChain tool to wrap.
        mode: "auto" (block silently with suggestions) or
              "critic" (ask user for Tier 2 commands).

    Returns:
        A copy of the tool with safety checks on every invocation.
    """
    guarded = copy.copy(original_tool)
    original_func = guarded.func

    @wraps(original_func)
    def safe_func(*args, **kwargs):
        # Extract the command/code content to check
        content = _extract_checkable_content(guarded.name, args, kwargs)

        if content:
            result = check_command_full(content)
            if result:
                reason, suggestion, tier = result

                if tier == 1:
                    # Always blocked — return error to agent
                    return f"BLOCKED: {reason}. Use a safe alternative."

                if tier == 2:
                    if mode == "critic":
                        # Ask user for approval
                        approved = _ask_user_approval(guarded.name, content, reason)
                        if approved:
                            return original_func(*args, **kwargs)
                        return f"BLOCKED by user: {reason}. Try a safe alternative."
                    else:
                        # Auto mode — block with helpful suggestion
                        return f"BLOCKED: {reason}. {suggestion}"

        return original_func(*args, **kwargs)

    guarded.func = safe_func
    return guarded


def _extract_checkable_content(tool_name: str, args: tuple, kwargs: dict) -> str:
    """Extract the command/code string from tool arguments."""
    if tool_name in ("execute_shell_command", "start_background_process"):
        return kwargs.get("command", args[0] if args else "")
    elif tool_name == "execute_python_code":
        return kwargs.get("code", args[0] if args else "")
    return ""


def _ask_user_approval(tool_name: str, content: str, reason: str) -> bool:
    """Pause and ask the user for approval in critic mode.

    Returns True if approved, False if denied.
    """
    print(f"\n{'='*60}")
    print(f"  CRITIC ALERT: {reason}")
    print(f"{'='*60}")
    print(f"  Tool: {tool_name}")
    print(f"  Command: {content[:300]}")
    if len(content) > 300:
        print(f"  ... ({len(content) - 300} more chars)")
    print()
    try:
        approval = input("  Approve execution? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        approval = "n"
    return approval == "y"


def guard_tools(tools: list, mode: str = "auto") -> list:
    """Convenience: wrap all dangerous tools in a list, leave safe ones unchanged.

    Only wraps tools named: execute_shell_command, execute_python_code,
    start_background_process.
    """
    DANGEROUS_TOOLS = {"execute_shell_command", "execute_python_code", "start_background_process"}
    result = []
    for tool in tools:
        if hasattr(tool, "name") and tool.name in DANGEROUS_TOOLS:
            result.append(guard_tool(tool, mode=mode))
        else:
            result.append(tool)
    return result
