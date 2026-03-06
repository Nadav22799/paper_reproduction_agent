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
import threading
from typing import List, Tuple, Optional, Protocol, runtime_checkable
from functools import wraps


# ---------------------------------------------------------------------------
# Cooperative cancellation — set by RunManager.stop_run(), checked before
# every guarded tool call.  Uses BaseException so it propagates through
# LangGraph / LangChain "except Exception" handlers and exits workflow.invoke().
# ---------------------------------------------------------------------------

class RunCancelled(BaseException):
    """Raised to cancel the current run.

    Inherits BaseException (not Exception) so it is NOT caught by the
    broad ``except Exception`` blocks inside LangGraph node implementations,
    and propagates cleanly out of ``workflow.invoke()``.
    """


_stop_event = threading.Event()


def request_stop() -> None:
    """Signal that the current run should be cancelled at the next tool call."""
    _stop_event.set()


def clear_stop() -> None:
    """Clear the stop signal.  Must be called before starting each new run."""
    _stop_event.clear()


@runtime_checkable
class ApprovalCallback(Protocol):
    """Callback protocol for requesting human approval in critic mode.

    Used to bridge the CLI ``input()`` flow with web-based approval UIs
    (or any other approval mechanism) without coupling tool_guard to
    either transport layer.
    """

    def request_approval(
        self, tool_name: str, content: str, reason: str
    ) -> Tuple[bool, str]:
        """Request approval for a tool call.

        Args:
            tool_name: Name of the tool being called.
            content: Command / code content (may be truncated for display).
            reason: Why the call was flagged.

        Returns:
            (approved, feedback) — approved=True means execute, feedback
            is an optional message from the reviewer (empty string if none).
        """
        ...


# Module-level approval callback — None means fall back to CLI input()
_approval_callback: Optional[ApprovalCallback] = None


def set_approval_callback(callback: Optional[ApprovalCallback]) -> None:
    """Register a callback for critic-mode approval requests.

    Pass ``None`` to revert to the default CLI ``input()`` behaviour.
    """
    global _approval_callback
    _approval_callback = callback


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
        # Check for cancellation before executing any tool
        if _stop_event.is_set():
            raise RunCancelled("Run stopped by user")

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
                        # Ask user for approval (via callback or CLI)
                        approved, feedback = _ask_user_approval(guarded.name, content, reason)
                        if approved:
                            return original_func(*args, **kwargs)
                        msg = f"BLOCKED by user: {reason}."
                        if feedback:
                            msg += f" Feedback: {feedback}."
                        return msg + " Try a safe alternative."
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


def _ask_user_approval(tool_name: str, content: str, reason: str) -> Tuple[bool, str]:
    """Pause and ask the user for approval in critic mode.

    If a callback is registered (e.g. WebApprovalCallback), delegates to it.
    Otherwise falls back to CLI using questionary.

    Returns:
        (approved, feedback) — approved=True means execute.
    """
    if _approval_callback is not None:
        return _approval_callback.request_approval(tool_name, content, reason)

    # CLI fallback
    from rich.panel import Panel
    from rich.console import Console
    import questionary

    console = Console()

    alert_text = f"Tool: {tool_name}\n"
    alert_text += f"Reason: {reason}\n"
    alert_text += f"Command:\n{content[:300]}"
    if len(content) > 300:
        alert_text += f"\n... ({len(content) - 300} more chars)"
        
    console.print(Panel(alert_text, title="🚨 CRITIC ALERT", border_style="red"))
    
    try:
        approval = questionary.confirm("Approve execution?", default=False).ask()
        feedback = ""
        if not approval:
            # If rejected, invite the human to provide feedback to the agent
            feedback = questionary.text("Provide feedback to the agent (optional):").ask()
    except (EOFError, KeyboardInterrupt):
        approval = False
        feedback = ""

    return (approval, feedback or "")


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
