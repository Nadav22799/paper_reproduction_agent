"""Logging callback to show LLM responses in real-time."""

from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, Dict, List, Optional


class LoggingCallbackHandler(BaseCallbackHandler):
    """Callback handler that logs LLM responses and optionally tracks metrics."""

    def __init__(self, verbose: bool = True, file_logger=None, metrics_tracker=None):
        self.verbose = verbose
        self.iteration = 0
        self.file_logger = file_logger
        self.metrics_tracker = metrics_tracker  # Optional MetricsTracker for token tracking

    def _log(self, message: str):
        """Log to both console and file if file_logger is set."""
        if self.file_logger:
            self.file_logger.log(message)
        else:
            print(message, end='')

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Run when LLM starts."""
        self.iteration += 1
        if self.verbose:
            self._log(f"\n{'='*60}\n")
            self._log(f"🤖 Iteration {self.iteration}: Calling LLM...\n")
            self._log(f"{'='*60}\n")
            # Log the prompts
            for i, prompt in enumerate(prompts):
                self._log(f"\n📝 Prompt {i+1}:\n")
                self._log(f"{prompt[:1000]}\n")
                if len(prompt) > 1000:
                    self._log(f"... (truncated, total length: {len(prompt)} chars)\n")

    def on_llm_end(self, response, **kwargs) -> None:
        """Run when LLM ends."""
        # Track token usage if metrics_tracker is provided
        if self.metrics_tracker:
            try:
                # 1. Check llm_output (Standard LangChain/OpenAI)
                if hasattr(response, 'llm_output') and response.llm_output:
                    usage = response.llm_output.get('token_usage', {}) or response.llm_output.get('usage_metadata', {})
                    if self._record_usage_from_dict(usage):
                        return

                # 2. Check generations (Gemini/Anthropic/Others)
                if hasattr(response, 'generations') and response.generations:
                    for gen_list in response.generations:
                        for gen in gen_list:
                            # Check 2a: message.usage_metadata (Newer LangChain, verified for Gemini)
                            if hasattr(gen, 'message') and hasattr(gen.message, 'usage_metadata'):
                                if self._record_usage_from_dict(gen.message.usage_metadata):
                                    return

                            # Check 2b: generation_info (Older LangChain/Some wrappers)
                            if hasattr(gen, 'generation_info') and gen.generation_info:
                                usage = gen.generation_info.get('usage_metadata', {})
                                if self._record_usage_from_dict(usage):
                                    return
                            
                            # Check 2c: message.response_metadata (Fallback)
                            if hasattr(gen, 'message') and hasattr(gen.message, 'response_metadata'):
                                meta = gen.message.response_metadata
                                usage = meta.get('token_usage', {}) or meta.get('usage', {})
                                if self._record_usage_from_dict(usage):
                                    return

            except Exception:
                pass  # Token tracking is best-effort

    def _record_usage_from_dict(self, usage: dict) -> bool:
        """Helper to extract and record tokens from a usage dictionary."""
        if not usage:
            return False
            
        input_tokens = usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0) or usage.get('output_tokens', 0)
        
        # Handle total_tokens calculation if needed
        if not output_tokens and 'total_tokens' in usage:
            output_tokens = usage['total_tokens'] - input_tokens

        if input_tokens or output_tokens:
            self.metrics_tracker.record_tokens(input_tokens, output_tokens)
            return True
        return False

        if self.verbose:
            # Get the response text
            if hasattr(response, 'generations') and response.generations:
                for gen_list in response.generations:
                    for gen in gen_list:
                        if hasattr(gen, 'text'):
                            text = gen.text
                        elif hasattr(gen, 'message'):
                            message = gen.message
                            text = str(message.content) if hasattr(message, 'content') else str(message)

                            # Debug: Check for tool calls
                            if hasattr(message, 'tool_calls') and message.tool_calls:
                                self._log(f"\n🔧 Tool Calls Found: {len(message.tool_calls)}\n")
                                for tc in message.tool_calls:
                                    # Handle both dict and ToolCall object
                                    try:
                                        if isinstance(tc, dict):
                                            name = tc.get('name', 'unknown')
                                            args = tc.get('args', {})
                                        else:
                                            name = getattr(tc, 'name', 'unknown')
                                            args = getattr(tc, 'args', {})
                                        self._log(f"   - {name}: {args}\n")
                                    except Exception as e:
                                        self._log(f"   - {tc} (error logging: {e})\n")
                            elif hasattr(message, 'additional_kwargs') and message.additional_kwargs.get('tool_calls'):
                                tool_calls = message.additional_kwargs['tool_calls']
                                self._log(f"\n🔧 Tool Calls Found (additional_kwargs): {len(tool_calls)}\n")
                                for tc in tool_calls:
                                    self._log(f"   - {tc}\n")
                        else:
                            text = str(gen)

                        self._log(f"\n💬 LLM Response:\n")
                        self._log(f"{text}\n")  # Log full response, not truncated
                        self._log(f"\n(Response length: {len(text)} chars)\n")

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Run when LLM errors."""
        if self.verbose:
            self._log(f"\n❌ LLM Error: {error}\n")
            self._log(f"   This might be a rate limit or API error.\n")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Run when tool starts."""
        if self.verbose:
            tool_name = serialized.get("name", "unknown")
            self._log(f"\n🔧 Calling tool: {tool_name}\n")
            self._log(f"   Input: {input_str[:200]}\n")

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Run when tool ends."""
        if self.verbose:
            self._log(f"   ✅ Tool output: {str(output)[:300]}\n")

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Run when tool errors."""
        if self.verbose:
            self._log(f"   ❌ Tool error: {error}\n")
