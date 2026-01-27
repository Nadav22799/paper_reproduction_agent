"""
vLLM Multi-Turn Wrapper - Enables multi-turn tool calling with vLLM.

This wrapper fixes the issue where vLLM returns empty responses after
processing tool results, enabling proper multi-turn tool calling workflows.
"""

from typing import Any, List, Optional
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel


class VLLMMultiTurnWrapper:
    """
    Minimal wrapper for vLLM ChatOpenAI that enables multi-turn tool calling.

    This wrapper simply delegates to the base LLM while maintaining
    compatibility with LangGraph's ReAct agent and tool binding.
    """

    def __init__(
        self, base_llm: BaseChatModel, debug: bool = False, enable_recovery: bool = True
    ):
        """
        Initialize wrapper.

        Args:
            base_llm: The base ChatOpenAI instance pointing to vLLM
            debug: Enable debug logging
            enable_recovery: Enable empty response recovery (set to False if causing issues)
        """
        self.base_llm = base_llm
        self.debug = debug
        self.enable_recovery = enable_recovery

    def invoke(
        self, messages: List[BaseMessage], config: Optional[dict] = None, **kwargs
    ) -> AIMessage:
        """
        Invoke the LLM with messages.

        Args:
            messages: List of messages
            config: Optional config
            **kwargs: Additional arguments

        Returns:
            AIMessage with content or tool_calls
        """
        if self.debug:
            print(f"[vLLM DEBUG] Invoking with {len(messages)} messages")

        # Simply delegate to base LLM
        result = self.base_llm.invoke(messages, config=config, **kwargs)

        # Check what we got back
        has_content = bool(result.content and len(str(result.content).strip()) > 0)
        has_tools = bool(hasattr(result, "tool_calls") and result.tool_calls)

        if self.debug:
            print(
                f"[vLLM DEBUG] Result: content={has_content}, tool_calls={len(result.tool_calls) if has_tools else 0}"
            )

            # Show raw content preview
            if result.content:
                preview = str(result.content)[:200]
                print(f"[vLLM DEBUG] Content preview: {preview}...")

            # Check response metadata
            if hasattr(result, "response_metadata"):
                metadata = result.response_metadata
                if "finish_reason" in metadata:
                    print(f"[vLLM DEBUG] finish_reason: {metadata['finish_reason']}")
                if "token_usage" in metadata:
                    print(f"[vLLM DEBUG] token_usage: {metadata['token_usage']}")

        # Try to extract content from GPT-OSS harmony format if empty
        if not has_content and not has_tools:
            # Check if this might be a harmony format response that wasn't parsed
            token_usage = (
                result.response_metadata.get("token_usage", {})
                if hasattr(result, "response_metadata")
                else {}
            )
            completion_tokens = token_usage.get("completion_tokens", 0)

            if completion_tokens > 0:
                print(
                    f"⚠️  WARNING: vLLM generated {completion_tokens} tokens but content is empty!"
                )
                print(
                    "    This suggests GPT-OSS harmony format isn't being parsed correctly."
                )
                print("    Attempting to extract from harmony channels...")

                # The content might be in the raw response - try to access it
                result = self._try_extract_harmony_content(result)

                # Check if extraction worked
                if result.content and len(str(result.content).strip()) > 0:
                    print(
                        f"    ✅ Successfully extracted {len(result.content)} chars from harmony format!"
                    )
                else:
                    print(
                        "    ❌ Failed to extract content - vLLM response format issue"
                    )
                    print(
                        f"    Response metadata: {result.response_metadata if hasattr(result, 'response_metadata') else 'N/A'}"
                    )

        # Check for empty response after tool results (multi-turn issue)
        if (
            self.enable_recovery
            and not has_content
            and not has_tools
            and self._has_recent_tool_result(messages)
        ):
            if self.debug:
                print(
                    "[vLLM DEBUG] Empty response detected after tool result - attempting recovery"
                )

            result = self._recover_from_empty(messages, result, config, **kwargs)

        return result

    def _try_extract_harmony_content(self, result: AIMessage) -> AIMessage:
        """
        Try to extract content from GPT-OSS harmony format channels.

        GPT-OSS uses special channels that vLLM's OpenAI parser might not extract:
        - <reasoning>...</reasoning>: Chain of thought
        - <tool_call>...</tool_call>: Tool calls
        - <response>...</response>: Final answer

        Args:
            result: AIMessage with potentially empty content

        Returns:
            AIMessage with extracted content if found
        """
        # Check if we have access to the raw response
        # LangChain's ChatOpenAI stores the raw OpenAI response in response_metadata
        if not hasattr(result, "response_metadata"):
            return result

        # Try to get the raw message content if available
        # Sometimes the harmony format content is in the original response but not extracted

        # Check if there's a raw response field we can access
        # This is a long shot but worth trying
        if hasattr(self.base_llm, "last_response"):
            str(self.base_llm.last_response)

        # If we can't access raw response, we can't extract harmony content
        # The issue is that by the time we get the AIMessage, LangChain has already
        # parsed it and lost the harmony channel content

        # Log what we tried
        if self.debug:
            print(
                "[vLLM DEBUG] Attempted harmony extraction but couldn't access raw response"
            )
            print(
                "[vLLM DEBUG] This means vLLM isn't including content in OpenAI-format response"
            )

        return result

    def _recover_from_empty(
        self,
        messages: List[BaseMessage],
        empty_result: AIMessage,
        config: Optional[dict],
        **kwargs,
    ) -> AIMessage:
        """
        Recover from empty response after tool results.

        This fixes the multi-turn tool calling issue where vLLM returns empty
        responses after processing tool results.

        Args:
            messages: Original messages
            empty_result: The empty response from vLLM
            config: Optional config
            **kwargs: Additional arguments

        Returns:
            Recovered AIMessage or original empty response
        """
        from langchain_core.messages import HumanMessage

        print("🔧 Recovering from empty vLLM response...")

        # Get recent tool results
        tool_results = self._get_recent_tool_results(messages)

        if not tool_results:
            print("  No tool results found - returning empty response")
            return empty_result

        # Create continuation prompt with tool result summary
        tool_summary = "\n".join(
            [
                f"- {msg.name if hasattr(msg, 'name') else 'unknown'}: {str(msg.content)[:200]}"
                for msg in tool_results
            ]
        )

        continuation = HumanMessage(
            content=f"""The tool(s) you called returned results:

{tool_summary}

Analyze these results and respond. You MUST either:
1. Explain what you learned from these results and call another tool if needed
2. Provide your conclusion based on the information

Your response should NOT be empty. Continue the task."""
        )

        extended_messages = list(messages) + [continuation]

        if self.debug:
            print("[vLLM DEBUG] Trying recovery with continuation prompt...")

        result = self.base_llm.invoke(extended_messages, config=config, **kwargs)

        if not self._is_empty_response(result):
            print("  ✅ Recovery successful!")
            return result

        # Recovery failed - try more explicit prompt
        print("  First attempt failed, trying stronger instruction...")
        explicit_continuation = HumanMessage(
            content="IMPORTANT: I need your response to the tool results shown above.\n\n"
            "Based on the tool results, write a response that either:\n"
            "1. Calls another tool with proper arguments, OR\n"
            "2. States your conclusion/decision based on what you learned\n\n"
            "Respond NOW - do not return an empty message."
        )
        extended_messages = list(messages) + [explicit_continuation]

        result = self.base_llm.invoke(extended_messages, config=config, **kwargs)

        if not self._is_empty_response(result):
            print("  ✅ Recovery successful with explicit instruction!")
            return result

        # All recovery attempts failed
        print("  ❌ All recovery attempts failed")
        return result

    def _is_empty_response(self, message: AIMessage) -> bool:
        """Check if response is empty (no content and no tool calls)."""
        has_content = bool(message.content and len(str(message.content).strip()) > 0)
        has_tool_calls = bool(hasattr(message, "tool_calls") and message.tool_calls)
        return not has_content and not has_tool_calls

    def _has_recent_tool_result(self, messages: List[BaseMessage]) -> bool:
        """Check if the last message is a tool result."""
        from langchain_core.messages import ToolMessage

        return len(messages) > 0 and isinstance(messages[-1], ToolMessage)

    def _get_recent_tool_results(
        self, messages: List[BaseMessage], limit: int = 5
    ) -> List:
        """Get recent tool results from message history."""
        from langchain_core.messages import ToolMessage, AIMessage

        tool_results = []
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                tool_results.insert(0, msg)
                if len(tool_results) >= limit:
                    break
            elif isinstance(msg, AIMessage):
                # Stop at the last AI message (before tool calls)
                break
        return tool_results

    def stream(
        self, messages: List[BaseMessage], config: Optional[dict] = None, **kwargs
    ):
        """Stream responses (delegates to base LLM)."""
        return self.base_llm.stream(messages, config=config, **kwargs)

    def bind_tools(self, tools: List[Any], **kwargs):
        """Bind tools and return wrapped version."""
        bound_llm = self.base_llm.bind_tools(tools, **kwargs)
        return VLLMMultiTurnWrapper(
            bound_llm, debug=self.debug, enable_recovery=self.enable_recovery
        )

    def with_structured_output(self, schema, **kwargs):
        """Support structured output."""
        structured_llm = self.base_llm.with_structured_output(schema, **kwargs)
        return VLLMMultiTurnWrapper(
            structured_llm, debug=self.debug, enable_recovery=self.enable_recovery
        )

    def __getattr__(self, name):
        """Delegate unknown attributes to base_llm."""
        return getattr(self.base_llm, name)

    def __or__(self, other):
        """Support pipe operator (|) for LangGraph compatibility."""
        # Let base_llm handle the pipe
        return self.base_llm | other

    def __call__(self, *args, **kwargs):
        """Make wrapper callable for LangGraph compatibility."""
        # LangGraph's ReAct agent needs the base_llm for binding
        return self.base_llm
