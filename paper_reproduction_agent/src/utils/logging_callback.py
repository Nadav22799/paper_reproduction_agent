"""Logging callback to show LLM responses in real-time."""

from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown

console = Console()

class LoggingCallbackHandler(BaseCallbackHandler):
    """Callback handler that logs LLM responses and optionally tracks metrics."""

    def __init__(self, verbose: bool = True, file_logger=None, metrics_tracker=None):
        self.verbose = verbose
        self.iteration = 0
        self.file_logger = file_logger
        self.metrics_tracker = (
            metrics_tracker  # Optional MetricsTracker for token tracking
        )

    def _log(self, message: str, rich_renderable=None):
        """Log to both console and file if file_logger is set."""
        if self.file_logger:
            self.file_logger.log(message)
        
        if rich_renderable:
            console.print(rich_renderable)
        else:
            console.print(message, end="")

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs
    ) -> None:
        """Run when LLM starts."""
        self.iteration += 1
        if self.verbose:
            # Try to identify which agent/component is calling the LLM
            caller = "Agent"
            metadata = kwargs.get("metadata", {})

            # checkpoint_ns looks like "planning:uuid|agent:uuid" — first segment is the real agent name
            checkpoint_ns = metadata.get("checkpoint_ns", "")
            if checkpoint_ns:
                caller = checkpoint_ns.split(":")[0].replace("_", " ").title()
            elif "langgraph_node" in metadata:
                caller = metadata["langgraph_node"].replace("_", " ").title()

            msg = f"🤖 Iteration {self.iteration}: {caller} thinking..."
            
            self._log(f"\n{'='*60}\n{msg}\n{'='*60}\n",
                      rich_renderable=Panel(msg, border_style="cyan"))
            
            # Log the prompts (Disabled per user feedback requesting a cleaner CLI)
            # for i, prompt in enumerate(prompts):
            #     self._log(f"\n📝 Prompt {i+1}:\n{prompt[:1000]}\n")
            #     if len(prompt) > 1000:
            #         self._log(f"... (truncated, total length: {len(prompt)} chars)\n")

    def on_llm_end(self, response, **kwargs) -> None:
        """Run when LLM ends."""
        
        # 0. AGGRESSIVE REASONING LOGGING (Must happen before anything else)
        # Try to find reasoning in ANY generation or output field
        thoughts = None
        
        # Check deep inside generations (Standard location for Gemini/others)
        if hasattr(response, "generations") and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    if not hasattr(gen, "message"):
                        continue
                        
                    message = gen.message
                    
                    # Method A: Content Blocks (List of Dicts)
                    # Newer LangChain usage for structured models like Gemini
                    if isinstance(message.content, list):
                        for block in message.content:
                            if isinstance(block, dict):
                                # Check for blocks like {'type': 'thought', 'thought': '...'}
                                # or {'type': 'thinking', 'thinking': '...'} (Anthropic Claude)
                                if block.get("type") in ["thought", "thinking", "reasoning"]:
                                    thoughts = block.get("thought") or block.get("thinking") or block.get("content") or block.get("text")
                                elif block.get("type") == "text" and any(k in block.get("text", "").lower()[:50] for k in ["thought", "reasoning"]):
                                    thoughts = block.get("text")
                            if thoughts:
                                break
                    
                    # Method B: additional_kwargs (Thoughts, Reasoning, etc.)
                    if not thoughts and hasattr(message, "additional_kwargs"):
                        ak = message.additional_kwargs
                        # reasoning_content is used by OpenAI o1/o3
                        thoughts = ak.get("thoughts") or ak.get("reasoning_content") or ak.get("reasoning")
                        
                        # Special Case: Gemini placeholder tools
                        if not thoughts and "tool_calls" in ak:
                            for tc in ak["tool_calls"]:
                                func = tc.get("function", {})
                                if "placeholder_thoughts" in func.get("name", ""):
                                    thoughts = func.get("arguments")
                    
                    # Method C: generation_info
                    if not thoughts and hasattr(gen, "generation_info") and gen.generation_info:
                        gi = gen.generation_info
                        thoughts = gi.get("thoughts") or gi.get("reasoning")
                        
                    if thoughts:
                        self._log(f"\n💭 Reasoning:\n{thoughts}\n",
                                  rich_renderable=Panel(Markdown(thoughts), title="💭 Reasoning", border_style="magenta"))
                        break
                if thoughts:
                    break

        # Check top-level LLM output as a fallback
        if not thoughts and hasattr(response, "llm_output") and response.llm_output:
             if "thoughts" in response.llm_output:
                thoughts = response.llm_output['thoughts']
                self._log(f"\n💭 Reasoning (LLM Output):\n{thoughts}\n",
                          rich_renderable=Panel(Markdown(thoughts), title="💭 Reasoning (LLM Output)", border_style="magenta"))

        # Track token usage if metrics_tracker is provided
        # IMPORTANT: Record tokens only ONCE - LangChain often puts the same data
        # in multiple places (llm_output, usage_metadata, response_metadata, etc.)
        if self.metrics_tracker:
            recorded = False
            try:
                # Priority 1: llm_output (Standard LangChain/OpenAI)
                if hasattr(response, "llm_output") and response.llm_output:
                    usage = response.llm_output.get(
                        "token_usage", {}
                    ) or response.llm_output.get("usage_metadata", {})
                    recorded = self._record_usage_from_dict(usage)

                # Priority 2: generations (Gemini/Anthropic/Others) - only if not already recorded
                if not recorded and hasattr(response, "generations") and response.generations:
                    for gen_list in response.generations:
                        if recorded:
                            break
                        for gen in gen_list:
                            if recorded:
                                break
                            # Try message.usage_metadata first (Newer LangChain, verified for Gemini)
                            if hasattr(gen, "message") and hasattr(gen.message, "usage_metadata"):
                                recorded = self._record_usage_from_dict(gen.message.usage_metadata)
                            # Then generation_info (Older LangChain/Some wrappers)
                            if not recorded and hasattr(gen, "generation_info") and gen.generation_info:
                                usage = gen.generation_info.get("usage_metadata", {})
                                recorded = self._record_usage_from_dict(usage)
                            # Finally response_metadata (Fallback)
                            if not recorded and hasattr(gen, "message") and hasattr(gen.message, "response_metadata"):
                                meta = gen.message.response_metadata
                                usage = meta.get("token_usage", {}) or meta.get("usage", {})
                                recorded = self._record_usage_from_dict(usage)

            except Exception:
                pass  # Token tracking is best-effort

        if self.verbose:
            # Get the response text
            if hasattr(response, "generations") and response.generations:
                for gen_list in response.generations:
                    for gen in gen_list:
                        if hasattr(gen, "text"):
                            text = gen.text
                        elif hasattr(gen, "message"):
                            message = gen.message
                            # Correctly handle list-based content for display
                            if isinstance(message.content, list):
                                text_parts = []
                                for block in message.content:
                                    if isinstance(block, dict) and block.get("type") == "text":
                                        text_parts.append(block.get("text", ""))
                                    elif isinstance(block, str):
                                        text_parts.append(block)
                                text = "".join(text_parts)
                            else:
                                text = (
                                    str(message.content)
                                    if hasattr(message, "content")
                                    else str(message)
                                )

                            if hasattr(message, "tool_calls") and message.tool_calls:
                                self._log(
                                    f"\n🔧 Tool Calls Found: {len(message.tool_calls)}\n"
                                )
                                for tc in message.tool_calls:
                                    # Handle both dict and ToolCall object
                                    try:
                                        if isinstance(tc, dict):
                                            name = tc.get("name", "unknown")
                                            args = tc.get("args", {})
                                        else:
                                            name = getattr(tc, "name", "unknown")
                                            args = getattr(tc, "args", {})
                                        
                                        msg = f"   - {name}: {args}\n"
                                        self._log(msg, rich_renderable=Panel(
                                            Text.assemble(
                                                ("Tool: ", "bold magenta"),
                                                (str(name), "italic cyan"),
                                                ("\nArgs: ", "bold green"),
                                                (str(args), "white")
                                            ),
                                            title="External Action",
                                            border_style="green"
                                        ))
                                    except Exception as e:
                                        self._log(f"   - {tc} (error logging: {e})\n")
                            elif hasattr(
                                message, "additional_kwargs"
                            ) and message.additional_kwargs.get("tool_calls"):
                                tool_calls = message.additional_kwargs["tool_calls"]
                                self._log(
                                    f"\n🔧 Tool Calls Found (additional_kwargs): {len(tool_calls)}\n"
                                )
                                for tc in tool_calls:
                                    self._log(f"   - {tc}\n", rich_renderable=Panel(str(tc), title="External Action", border_style="green"))
                            
                            # Reasoning is already logged at the top, no need to duplicate here.
                            # We just log the text response itself now.

                            self._log("\n💬 LLM Response:\n")
                            self._log(f"{text}\n", rich_renderable=Panel(Markdown(text), title="💬 LLM Response", border_style="blue"))  # Log full response, not truncated
                            self._log(f"\n(Response length: {len(text)} chars)\n")

    def _record_usage_from_dict(self, usage: dict) -> bool:
        """Helper to extract and record tokens from a usage dictionary."""
        if not usage:
            return False

        input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0) or usage.get(
            "output_tokens", 0
        )

        # Handle total_tokens calculation if needed


        self.metrics_tracker.record_tokens(input_tokens, output_tokens)

        # Extract reasoning tokens from multiple provider formats
        reasoning_tokens = 0

        # Gemini format (with include_thoughts=True)
        output_token_details = usage.get("output_token_details", {})
        if isinstance(output_token_details, dict):
            reasoning_tokens = output_token_details.get("reasoning", 0) or 0

        # OpenAI o1/o3 and Anthropic format
        if not reasoning_tokens:
            reasoning_tokens = usage.get("reasoning_tokens", 0) or 0

        if reasoning_tokens and hasattr(self.metrics_tracker, "record_reasoning_tokens"):
            self.metrics_tracker.record_reasoning_tokens(reasoning_tokens)

        # Extract cache tokens from multiple provider formats
        cache_creation = 0
        cache_read = 0

        # Gemini format
        input_token_details = usage.get("input_token_details", {})
        if isinstance(input_token_details, dict):
            cache_read = input_token_details.get("cache_read", 0) or 0

        # Anthropic/Claude format
        if not cache_read:
            cache_read = usage.get("cache_read_input_tokens", 0) or 0
        cache_creation = usage.get("cache_creation_input_tokens", 0) or 0

        if (cache_creation or cache_read) and hasattr(self.metrics_tracker, "record_cache_tokens"):
            self.metrics_tracker.record_cache_tokens(cache_creation, cache_read)

        return bool(input_tokens or output_tokens)

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Run when LLM errors."""
        if self.verbose:
            self._log(f"\n❌ LLM Error: {error}\n   This might be a rate limit or API error.\n",
                      rich_renderable=Panel(f"❌ LLM Error: {error}\nThis might be a rate limit or API error.", border_style="red"))

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs
    ) -> None:
        """Run when tool starts."""
        if self.verbose:
            tool_name = serialized.get("name", "unknown")
            self._log(f"\n🔧 Calling tool: {tool_name}\n   Input: {input_str[:200]}\n",
                      rich_renderable=Panel(f"🔧 Calling tool: {tool_name}\nInput: {input_str[:200]}", border_style="yellow"))

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Run when tool ends."""
        if self.verbose:
            self._log(f"   ✅ Tool output: {str(output)[:300]}\n", rich_renderable=Text(f"✅ Tool output: {str(output)[:300]}", style="dim"))

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Run when tool errors."""
        if self.verbose:
            self._log(f"   ❌ Tool error: {error}\n", rich_renderable=Text(f"❌ Tool error: {error}", style="bold red"))
