"""Message utilities for handling different LLM content formats."""


def normalize_message_content(content):
    """
    Normalize message content to string format.
    Handles both simple strings (OpenAI/Groq) and list format (Gemini/Anthropic).

    Different LLM providers return message content in different formats:
    - OpenAI/Groq: content="string"
    - Gemini/Anthropic: content=[{"type": "text", "text": "..."}]

    Args:
        content: Message content (string or list of content blocks)

    Returns:
        Normalized string content
    """
    if isinstance(content, list):
        # Extract text from content blocks (Gemini/Anthropic format)
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return content
