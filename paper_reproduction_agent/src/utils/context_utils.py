"""Context utilities for formatting and storing agent messages.

This module provides helpers for:
1. Extracting key facts from agent results for quick reference
2. Formatting full messages including tool calls and outputs
3. Building structured context entries for hierarchical context storage
"""

from typing import Dict, Any


def extract_key_facts(agent_name: str, result: Dict[str, Any]) -> str:
    """Extract key facts from agent result for quick reference.

    Args:
        agent_name: Name of the agent (e.g., "environment_setup")
        result: Agent's result dictionary

    Returns:
        Formatted key facts string
    """
    facts = []

    if agent_name == "environment_setup":
        if result.get("env_type"):
            facts.append(f"- Environment Tool: {result['env_type']}")
        if result.get("env_name"):
            facts.append(f"- Environment Name: {result['env_name']}")
        if result.get("python_path"):
            facts.append(f"- Python Path: {result['python_path']}")
        if result.get("success"):
            facts.append("- Smoke Test: PASSED")
        else:
            facts.append(f"- Status: FAILED - {result.get('error', 'unknown')[:100]}")

    elif agent_name == "execution":
        if result.get("experiments_completed"):
            facts.append("- Experiments: COMPLETED")
        else:
            facts.append("- Experiments: FAILED")
        if result.get("experiment_results"):
            facts.append(f"- Results: {str(result['experiment_results'])[:200]}")
        if result.get("tool"):
            facts.append(f"- Tool Used: {result['tool']}")
        if result.get("env_name"):
            facts.append(f"- Environment: {result['env_name']}")

    elif agent_name == "data_prep":
        if result.get("datasets_ready"):
            facts.append("- Datasets: READY")
        else:
            facts.append("- Datasets: NOT READY")
        locations = result.get("dataset_results", {}).get("data_locations", [])
        if locations:
            facts.append(f"- Data Locations: {', '.join(str(loc) for loc in locations[:3])}")

    elif agent_name == "planning":
        if result.get("plan_created"):
            facts.append("- Plan: CREATED")
        if result.get("skip_data_prep"):
            facts.append("- Skip Data Prep: YES")
        if result.get("selected_datasets"):
            facts.append(f"- Selected Datasets: {', '.join(result['selected_datasets'][:3])}")

    elif agent_name == "validation":
        if result.get("results_match"):
            facts.append("- Validation: PASSED")
        else:
            facts.append("- Validation: FAILED")
        if result.get("match_ratio"):
            facts.append(f"- Match Ratio: {result['match_ratio']}")

    return "\n".join(facts) if facts else "No key facts extracted"


def format_messages_for_context(messages: list, max_tokens: int = 5000) -> str:
    """Format agent messages (including tool calls and outputs) for context storage.

    Args:
        messages: List of LangChain messages from agent.invoke()
        max_tokens: Maximum tokens to include (rough estimate: 4 chars/token)

    Returns:
        Formatted string with reasoning, tool calls, and outputs
    """
    if not messages:
        return "No messages available"

    formatted_parts = []
    estimated_tokens = 0
    chars_per_token = 4

    for msg in messages:
        content = ""
        msg_type = type(msg).__name__

        # Handle different message types
        if hasattr(msg, "content"):
            if isinstance(msg.content, str):
                content = msg.content
            elif isinstance(msg.content, list):
                # Handle list content (multimodal or structured)
                parts = []
                for p in msg.content:
                    if isinstance(p, str):
                        parts.append(p)
                    elif isinstance(p, dict):
                        if "text" in p:
                            parts.append(p["text"])
                        elif "type" in p:
                            parts.append(f"[{p['type']}]")
                content = "\n".join(parts)

        # Include tool calls if present (THIS IS KEY!)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.get("name", "unknown")
                tool_args = tc.get("args", {})
                # Format args nicely, truncate if too long
                args_str = str(tool_args)
                if len(args_str) > 300:
                    args_str = args_str[:300] + "..."
                content += f"\n[TOOL CALL: {tool_name}] Args: {args_str}"

        # Format the message
        if content and content.strip():
            # Truncate individual messages to keep things reasonable
            truncated = content.strip()
            if len(truncated) > 2000:
                truncated = truncated[:2000] + "..."

            formatted = f"[{msg_type}] {truncated}"
            msg_tokens = len(formatted) // chars_per_token

            if estimated_tokens + msg_tokens > max_tokens:
                formatted_parts.append("... (truncated due to token limit)")
                break

            formatted_parts.append(formatted)
            estimated_tokens += msg_tokens

    return "\n---\n".join(formatted_parts) if formatted_parts else "No content to format"


def build_context_entry(
    agent_name: str,
    result: Dict,
    messages: list,
    max_detail_tokens: int = 5000,
) -> str:
    """Build a complete context entry with key facts + detailed messages.

    This creates a structured entry that agents can quickly scan:
    1. KEY FACTS section - immediate important info (env_type, success, etc.)
    2. DETAILED TRACE section - full tool calls for debugging

    Args:
        agent_name: Name of the agent
        result: Agent's result dictionary
        messages: Full messages from agent.invoke()
        max_detail_tokens: Token budget for detailed section

    Returns:
        Formatted context entry string
    """
    key_facts = extract_key_facts(agent_name, result)
    detailed = format_messages_for_context(messages, max_tokens=max_detail_tokens)

    return f"""=== {agent_name.upper()} AGENT ===
KEY FACTS:
{key_facts}

DETAILED TRACE (tool calls & outputs):
{detailed}
{'=' * 40}"""


def estimate_tokens(text: str) -> int:
    """Estimate token count for text (rough: 4 chars per token).

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return len(text) // 4


# =============================================================================
# Smart Context Entry — Last N iterations + structured summary
# =============================================================================


def build_smart_context_entry(
    agent_name: str,
    result: Dict,
    messages: list,
    last_n: int = 3,
    max_detail_tokens: int = 5000,
) -> str:
    """Build a context entry focused on RECENT iterations + structured summary.

    Instead of iterating all messages from the start (and truncating early),
    this takes the LAST N ReAct iterations (where final outcomes live)
    plus a structured success/failure summary header.

    Args:
        agent_name: Name of the agent
        result: Agent's result dictionary
        messages: Full messages from agent.invoke()
        last_n: Number of recent iterations to include (default: 3)
        max_detail_tokens: Token budget for the detail section

    Returns:
        Formatted context entry string
    """
    # Section 1: Structured summary header
    key_facts = extract_key_facts(agent_name, result)
    success = _determine_success(agent_name, result)

    header = (
        f"=== {agent_name.upper()} AGENT ===\n"
        f"STATUS: {'SUCCESS' if success else 'FAILED'}\n"
        f"KEY FACTS:\n{key_facts}\n"
    )

    # Section 2: Extract and format last N iterations
    iterations = _extract_iterations(messages)
    recent = iterations[-last_n:] if iterations else []
    total_count = len(iterations)

    remaining_tokens = max_detail_tokens - estimate_tokens(header)
    detail = _format_recent_iterations(recent, total_count, remaining_tokens)

    return (
        f"{header}\n"
        f"RECENT TRACE (last {len(recent)} of {total_count} iterations):\n"
        f"{detail}\n"
        f"{'=' * 40}"
    )


def _determine_success(agent_name: str, result: Dict) -> bool:
    """Determine if agent succeeded based on its result dict."""
    mapping = {
        "environment_setup": "success",
        "execution": "experiments_completed",
        "data_prep": "datasets_ready",
        "validation": "results_match",
    }
    key = mapping.get(agent_name, "success")
    return bool(result.get(key, False))


def _extract_iterations(messages: list) -> list:
    """Group messages into ReAct iteration cycles.

    Each cycle is: AIMessage (reasoning + tool_calls) followed by ToolMessage(s).
    A new AIMessage after ToolMessage(s) starts a new cycle.

    Args:
        messages: List of LangChain messages from agent.invoke()

    Returns:
        List of lists, where each inner list is one ReAct iteration.
    """
    iterations = []
    current = []

    for msg in messages:
        msg_type = type(msg).__name__

        if msg_type == "HumanMessage" and not current:
            # Skip initial prompt — not part of iteration cycles
            continue
        elif msg_type == "AIMessage":
            # If we already have an AIMessage in the current cycle,
            # this starts a new iteration
            if current and any(type(m).__name__ == "AIMessage" for m in current):
                iterations.append(current)
                current = []
            current.append(msg)
        else:
            # ToolMessage or other — add to current cycle
            current.append(msg)

    # Don't forget the last iteration
    if current:
        iterations.append(current)

    return iterations


def _format_iteration(iteration: list, number: int) -> str:
    """Format a single ReAct iteration with full detail.

    Args:
        iteration: List of messages in this iteration
        number: Iteration number for display

    Returns:
        Formatted string for this iteration
    """
    parts = [f"[Iteration {number}]"]

    for msg in iteration:
        msg_type = type(msg).__name__
        content = ""

        # Extract text content
        if hasattr(msg, "content"):
            if isinstance(msg.content, str):
                content = msg.content
            elif isinstance(msg.content, list):
                text_parts = []
                for p in msg.content:
                    if isinstance(p, str):
                        text_parts.append(p)
                    elif isinstance(p, dict) and "text" in p:
                        text_parts.append(p["text"])
                content = "\n".join(text_parts)

        # Include tool calls with names and args
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.get("name", "unknown")
                args_str = str(tc.get("args", {}))
                if len(args_str) > 500:
                    args_str = args_str[:500] + "..."
                content += f"\n[TOOL: {tool_name}] {args_str}"

        if content and content.strip():
            truncated = content.strip()
            if len(truncated) > 3000:
                truncated = truncated[:3000] + "..."
            parts.append(f"  [{msg_type}] {truncated}")

    return "\n".join(parts)


def _format_recent_iterations(
    recent: list, total_count: int, max_tokens: int
) -> str:
    """Format last N iterations within a token budget.

    Args:
        recent: List of recent iteration message groups
        total_count: Total number of iterations (for numbering)
        max_tokens: Token budget for the entire section

    Returns:
        Formatted string of recent iterations
    """
    parts = []
    remaining = max_tokens

    for i, iteration in enumerate(recent):
        # Calculate actual iteration number
        number = total_count - len(recent) + i + 1
        text = _format_iteration(iteration, number)
        tokens = estimate_tokens(text)

        if tokens <= remaining:
            parts.append(text)
            remaining -= tokens
        else:
            # Truncate this iteration to fit remaining budget
            char_limit = remaining * 4  # rough tokens-to-chars
            if char_limit > 0:
                parts.append(text[:char_limit] + "... (truncated)")
            break

    return "\n---\n".join(parts) if parts else "No iterations available"
