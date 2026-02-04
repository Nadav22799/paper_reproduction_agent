"""Context utilities for formatting and storing agent messages.

This module provides helpers for:
1. Extracting key facts from agent results for quick reference
2. Formatting full messages including tool calls and outputs
3. Building structured context entries for hierarchical context storage
"""

from typing import Dict, Any, List, Optional


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
            facts.append(f"- Data Locations: {', '.join(str(l) for l in locations[:3])}")

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
