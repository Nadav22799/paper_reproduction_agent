"""Context management utilities for preventing context window explosion."""

import re
from typing import List, Dict, Any, Tuple
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage


class ContextManager:
    """Manages conversation context to prevent explosion with aggressive pruning."""

    def __init__(self, max_context_chars: int = 200000, sliding_window_size: int = 2):
        """
        Initialize context manager.

        Args:
            max_context_chars: Maximum total characters in context (hard limit)
            sliding_window_size: Number of recent tool calls to keep in detail
        """
        self.max_context_chars = max_context_chars
        self.sliding_window_size = sliding_window_size
        self.seen_errors = set()
        self.error_summaries = []  # Store summarized errors for context

    def prune_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Aggressively prune message list to stay under context limit.

        Strategy:
        1. Always keep system messages and initial task
        2. Extract and summarize ALL errors (keep detailed version of latest error only)
        3. Summarize old tool interactions aggressively
        4. Keep only last N tool calls in detail
        5. If still over limit, reduce to last 1 tool call
        6. Hard enforcement of character limit

        Args:
            messages: List of conversation messages

        Returns:
            Pruned list of messages
        """
        if not messages:
            return messages

        # Calculate current size
        current_size = self._get_total_size(messages)

        # If under 50% of limit, do light pruning only
        if current_size < self.max_context_chars * 0.5:
            return self._light_prune(messages)

        # If under limit but over 50%, do normal pruning
        if current_size < self.max_context_chars:
            return self._normal_prune(messages)

        # Over limit - aggressive pruning
        return self._aggressive_prune(messages)

    def _get_total_size(self, messages: List[BaseMessage]) -> int:
        """Get total character count of messages."""
        return sum(len(str(getattr(msg, 'content', ''))) for msg in messages)

    def _light_prune(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Light pruning when under 50% capacity."""
        # Just deduplicate errors and truncate extreme outputs
        messages = self._truncate_extreme_outputs(messages, limit=8000)
        messages = self._deduplicate_errors(messages)
        return messages

    def _normal_prune(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Normal pruning when between 50-100% capacity."""
        # Separate messages by type
        system_msgs, initial_msgs, tool_interactions = self._categorize_messages(messages)

        # Group tool interactions
        grouped_interactions = self._group_tool_interactions(tool_interactions)

        # Keep last N interactions in detail, summarize rest
        if len(grouped_interactions) > self.sliding_window_size:
            old_interactions = grouped_interactions[:-self.sliding_window_size]
            recent_interactions = grouped_interactions[-self.sliding_window_size:]

            # Summarize old interactions
            summary_msg = self._summarize_interactions(old_interactions)

            # Flatten recent interactions
            recent_msgs = []
            for interaction in recent_interactions:
                recent_msgs.extend(interaction)

            pruned_messages = system_msgs + initial_msgs + [summary_msg] + recent_msgs
        else:
            all_interaction_msgs = []
            for interaction in grouped_interactions:
                all_interaction_msgs.extend(interaction)
            pruned_messages = system_msgs + initial_msgs + all_interaction_msgs

        # Truncate outputs and deduplicate errors
        pruned_messages = self._truncate_extreme_outputs(pruned_messages, limit=5000)
        pruned_messages = self._deduplicate_errors(pruned_messages)

        return pruned_messages

    def _aggressive_prune(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Aggressive pruning when over limit."""
        print("   ⚠️  AGGRESSIVE PRUNING - Over context limit!")

        # Separate messages
        system_msgs, initial_msgs, tool_interactions = self._categorize_messages(messages)

        # Group tool interactions
        grouped_interactions = self._group_tool_interactions(tool_interactions)

        # Extract all errors and create a summary
        all_errors = self._extract_all_errors(messages)
        error_summary_msg = self._create_error_summary(all_errors) if all_errors else None

        # Start with just last 1 interaction
        keep_count = 1

        while keep_count <= len(grouped_interactions):
            # Build candidate message list
            if len(grouped_interactions) > keep_count:
                old_interactions = grouped_interactions[:-keep_count]
                recent_interactions = grouped_interactions[-keep_count:]

                # Very aggressive summarization of old interactions
                summary_msg = self._summarize_interactions_aggressive(old_interactions)

                recent_msgs = []
                for interaction in recent_interactions:
                    recent_msgs.extend(interaction)

                # Build message list
                candidate = system_msgs + initial_msgs
                if error_summary_msg:
                    candidate.append(error_summary_msg)
                candidate.append(summary_msg)
                candidate.extend(recent_msgs)
            else:
                # Keep all interactions
                all_msgs = []
                for interaction in grouped_interactions:
                    all_msgs.extend(interaction)
                candidate = system_msgs + initial_msgs
                if error_summary_msg:
                    candidate.append(error_summary_msg)
                candidate.extend(all_msgs)

            # Truncate all tool outputs aggressively
            candidate = self._truncate_all_outputs(candidate, limit=2000)

            # Check size
            size = self._get_total_size(candidate)
            if size <= self.max_context_chars:
                print(f"   ✅ Reduced to {keep_count} recent interaction(s), size: {size:,} chars")
                return candidate

            keep_count += 1

        # Still over limit - nuclear option: keep only essentials
        print("   🔥 NUCLEAR PRUNING - Keeping only essentials!")
        return self._nuclear_prune(system_msgs, initial_msgs, grouped_interactions, error_summary_msg)

    def _nuclear_prune(self, system_msgs, initial_msgs, grouped_interactions, error_summary_msg) -> List[BaseMessage]:
        """Last resort pruning - keep absolute minimum."""
        result = system_msgs + initial_msgs

        # Add error summary if exists
        if error_summary_msg:
            result.append(error_summary_msg)

        # Add ultra-compressed summary of all work done
        if grouped_interactions:
            total_calls = sum(len(i) for i in grouped_interactions)
            summary = HumanMessage(content=f"""
[CONTEXT SEVERELY TRUNCATED - {total_calls} tool interactions removed to fit context limit]

Previous work summary:
- Total tool calls made: {total_calls}
- Interactions summarized: {len(grouped_interactions)}

Continue from where you left off. Check current state before proceeding.
""")
            result.append(summary)

            # Keep only the very last tool message (most recent state)
            if grouped_interactions:
                last_interaction = grouped_interactions[-1]
                for msg in last_interaction:
                    if isinstance(msg, ToolMessage):
                        # Truncate to 1500 chars
                        content = str(msg.content)[:1500]
                        if len(str(msg.content)) > 1500:
                            content += "\n...[truncated]"
                        result.append(ToolMessage(
                            content=content,
                            tool_call_id=getattr(msg, 'tool_call_id', '')
                        ))
                        break

        # Final size check
        size = self._get_total_size(result)
        if size > self.max_context_chars:
            # Truncate initial message if needed
            for i, msg in enumerate(result):
                if isinstance(msg, HumanMessage) and i < 3:
                    content = str(msg.content)
                    if len(content) > 5000:
                        result[i] = HumanMessage(content=content[:5000] + "\n...[truncated]")

        print(f"   🔥 Nuclear pruning complete, size: {self._get_total_size(result):,} chars")
        return result

    def _categorize_messages(self, messages: List[BaseMessage]) -> Tuple[List, List, List]:
        """Categorize messages into system, initial, and tool interactions."""
        system_msgs = []
        initial_msgs = []
        tool_interactions = []

        for i, msg in enumerate(messages):
            if isinstance(msg, SystemMessage):
                system_msgs.append(msg)
            elif i < 3:  # Keep first few messages (task setup)
                initial_msgs.append(msg)
            else:
                tool_interactions.append((i, msg))

        return system_msgs, initial_msgs, tool_interactions

    def _extract_all_errors(self, messages: List[BaseMessage]) -> List[Dict]:
        """Extract all errors from messages with context."""
        errors = []

        for msg in messages:
            content = str(getattr(msg, 'content', ''))
            content_lower = content.lower()

            # Check for error indicators
            if any(kw in content_lower for kw in ['error', 'exception', 'failed', 'traceback']):
                # Extract error type and message
                error_info = self._parse_error(content)
                if error_info and error_info['signature'] not in self.seen_errors:
                    errors.append(error_info)
                    self.seen_errors.add(error_info['signature'])

        return errors

    def _parse_error(self, content: str) -> Dict:
        """Parse error content to extract key information."""
        # Common error patterns
        patterns = [
            r'(ModuleNotFoundError|ImportError|RuntimeError|OSError|ValueError|TypeError|KeyError|AttributeError|FileNotFoundError|PermissionError|ConnectionError|TimeoutError):\s*([^\n]+)',
            r'(Error|Exception|Failed):\s*([^\n]+)',
            r'Traceback.*?(\w+Error|\w+Exception):\s*([^\n]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                error_type = match.group(1)
                error_msg = match.group(2) if len(match.groups()) > 1 else ""

                # Create signature for deduplication
                signature = f"{error_type}:{error_msg[:100]}"
                signature = re.sub(r'\d+', 'N', signature)  # Normalize numbers

                return {
                    'type': error_type,
                    'message': error_msg[:300],
                    'signature': signature,
                    'full_content': content[:1000]  # Keep some context
                }

        return None

    def _create_error_summary(self, errors: List[Dict]) -> HumanMessage:
        """Create a detailed summary of all errors encountered."""
        if not errors:
            return None

        summary_parts = [
            "=" * 60,
            "🚨 ERROR SUMMARY (All errors encountered)",
            "=" * 60,
        ]

        # Group by error type
        by_type = {}
        for error in errors:
            error_type = error['type']
            if error_type not in by_type:
                by_type[error_type] = []
            by_type[error_type].append(error)

        for error_type, type_errors in by_type.items():
            summary_parts.append(f"\n### {error_type} ({len(type_errors)} occurrence(s))")
            for error in type_errors[:3]:  # Max 3 per type
                summary_parts.append(f"  - {error['message']}")

        # Add most recent error in detail
        if errors:
            latest = errors[-1]
            summary_parts.append("\n### Latest Error (DETAILED):")
            summary_parts.append(f"Type: {latest['type']}")
            summary_parts.append(f"Message: {latest['message']}")
            if latest.get('full_content'):
                # Include relevant portion of full error
                summary_parts.append(f"Context:\n{latest['full_content'][:500]}")

        summary_parts.append("=" * 60)

        return HumanMessage(content="\n".join(summary_parts))

    def _truncate_extreme_outputs(self, messages: List[BaseMessage], limit: int = 5000) -> List[BaseMessage]:
        """Truncate extremely large outputs, keeping last 3 tool messages in full."""
        truncated = []

        # Find tool message indices
        tool_indices = [i for i, msg in enumerate(messages) if isinstance(msg, ToolMessage)]
        recent_tool_indices = set(tool_indices[-3:]) if len(tool_indices) > 3 else set(tool_indices)

        for i, msg in enumerate(messages):
            if isinstance(msg, ToolMessage) and i not in recent_tool_indices:
                content = str(getattr(msg, 'content', ''))
                if len(content) > limit:
                    # Keep first and last parts
                    first_part = content[:int(limit * 0.6)]
                    last_part = content[-int(limit * 0.3):]
                    truncated_content = (
                        first_part +
                        f"\n\n... [{len(content) - limit} chars truncated] ...\n\n" +
                        last_part
                    )
                    msg = ToolMessage(
                        content=truncated_content,
                        tool_call_id=getattr(msg, 'tool_call_id', '')
                    )
            truncated.append(msg)

        return truncated

    def _truncate_all_outputs(self, messages: List[BaseMessage], limit: int = 2000) -> List[BaseMessage]:
        """Truncate ALL tool outputs to limit (for aggressive pruning)."""
        truncated = []

        for msg in messages:
            if isinstance(msg, ToolMessage):
                content = str(getattr(msg, 'content', ''))
                if len(content) > limit:
                    # Keep first part and last part
                    first_part = content[:int(limit * 0.6)]
                    last_part = content[-int(limit * 0.3):]
                    truncated_content = (
                        first_part +
                        f"\n\n... [{len(content) - limit} chars truncated] ...\n\n" +
                        last_part
                    )
                    msg = ToolMessage(
                        content=truncated_content,
                        tool_call_id=getattr(msg, 'tool_call_id', '')
                    )
            elif isinstance(msg, AIMessage):
                content = str(getattr(msg, 'content', ''))
                if len(content) > limit:
                    truncated_content = content[:limit] + "\n...[truncated]"
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        msg = AIMessage(content=truncated_content, tool_calls=msg.tool_calls)
                    else:
                        msg = AIMessage(content=truncated_content)
            truncated.append(msg)

        return truncated

    def _group_tool_interactions(self, indexed_messages: List[tuple]) -> List[List[BaseMessage]]:
        """Group messages into tool call interactions."""
        interactions = []
        current_interaction = []

        for idx, msg in indexed_messages:
            current_interaction.append(msg)

            if isinstance(msg, ToolMessage):
                interactions.append(current_interaction)
                current_interaction = []

        if current_interaction:
            interactions.append(current_interaction)

        return interactions

    def _summarize_interactions(self, interactions: List[List[BaseMessage]]) -> HumanMessage:
        """Summarize old tool interactions into a concise summary."""
        summaries = []

        for interaction in interactions:
            for msg in interaction:
                if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get('name', 'unknown')
                        tool_args = tool_call.get('args', {})

                        if tool_name == 'read_file':
                            file_path = tool_args.get('file_path', '')
                            summaries.append(f"📄 Read: {self._shorten_path(file_path)}")
                        elif tool_name == 'execute_shell_command':
                            cmd = tool_args.get('command', '')[:40]
                            summaries.append(f"⚙️ Ran: {cmd}")
                        elif tool_name == 'list_directory':
                            summaries.append(f"📁 Listed: {tool_args.get('dir_path', '.')}")
                        elif tool_name == 'smart_install_dependencies':
                            summaries.append("📦 Installed deps")
                        else:
                            summaries.append(f"🔧 {tool_name}")

        # Create summary
        if summaries:
            # Limit to first 15 items
            display_list = summaries[:15]
            more_text = f'... +{len(summaries) - 15} more' if len(summaries) > 15 else ''

            summary_text = f"""
{'='*50}
📋 PREVIOUS ACTIONS ({len(interactions)} interactions):
{'='*50}
{chr(10).join(display_list)}
{more_text}
{'='*50}
"""
        else:
            summary_text = "[Previous actions completed]"

        return HumanMessage(content=summary_text)

    def _summarize_interactions_aggressive(self, interactions: List[List[BaseMessage]]) -> HumanMessage:
        """Ultra-concise summary for aggressive pruning."""
        tool_counts = {}

        for interaction in interactions:
            for msg in interaction:
                if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get('name', 'unknown')
                        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

        # Create ultra-compact summary
        summary_parts = [f"{name}: {count}x" for name, count in sorted(tool_counts.items(), key=lambda x: -x[1])[:8]]

        summary_text = f"[Previous: {len(interactions)} interactions - {', '.join(summary_parts)}]"

        return HumanMessage(content=summary_text)

    def _deduplicate_errors(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Remove duplicate error messages."""
        deduplicated = []
        seen_signatures = set()

        for msg in messages:
            content = str(getattr(msg, 'content', ''))

            if any(kw in content.lower() for kw in ['error', 'failed', 'exception', 'traceback']):
                signature = self._extract_error_signature(content)

                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)

                # Truncate long error messages
                if len(content) > 1500:
                    truncated_content = (
                        content[:1000] +
                        "\n... (middle truncated) ...\n" +
                        content[-400:]
                    )
                    msg = self._clone_message_with_content(msg, truncated_content)

            deduplicated.append(msg)

        return deduplicated

    def _extract_error_signature(self, error_text: str) -> str:
        """Extract a signature from error text for deduplication."""
        signature = re.sub(r'\d+', '', error_text)
        signature = re.sub(r'line \d+', 'line N', signature, flags=re.IGNORECASE)
        signature = re.sub(r'at 0x[0-9a-f]+', 'at 0xXXX', signature)
        return signature[:150]

    def _shorten_path(self, path: str) -> str:
        """Shorten file paths for display."""
        if len(path) > 40:
            parts = path.split('/')
            if len(parts) > 3:
                return f"{parts[0]}/.../{parts[-1]}"
        return path

    def _clone_message_with_content(self, msg: BaseMessage, new_content: str) -> BaseMessage:
        """Clone a message with different content."""
        if isinstance(msg, HumanMessage):
            return HumanMessage(content=new_content)
        elif isinstance(msg, AIMessage):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                return AIMessage(content=new_content, tool_calls=msg.tool_calls)
            return AIMessage(content=new_content)
        elif isinstance(msg, ToolMessage):
            return ToolMessage(content=new_content, tool_call_id=getattr(msg, 'tool_call_id', ''))
        elif isinstance(msg, SystemMessage):
            return SystemMessage(content=new_content)
        return msg

    def get_context_stats(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Get statistics about the current context."""
        total_chars = self._get_total_size(messages)
        total_messages = len(messages)

        type_counts = {}
        for msg in messages:
            msg_type = type(msg).__name__
            type_counts[msg_type] = type_counts.get(msg_type, 0) + 1

        utilization = (total_chars / self.max_context_chars) * 100

        return {
            'total_messages': total_messages,
            'total_chars': total_chars,
            'chars_remaining': self.max_context_chars - total_chars,
            'utilization_pct': utilization,
            'message_types': type_counts,
            'status': 'OK' if utilization < 80 else 'WARNING' if utilization < 100 else 'CRITICAL'
        }
