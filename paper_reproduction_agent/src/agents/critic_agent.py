"""Critic Agent - Security guardrail that validates reasoning before execution.

The Critic inspects the `current_reasoning` and `proposed_action` in state before
allowing execution. It blocks:
1. Forbidden commands (git clone, sudo, dangerous deletes)
2. Actions without sufficient reasoning
3. Actions that don't match the stated reasoning
"""

import re
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from ..utils.tool_guard import FORBIDDEN_PATTERNS


class CriticAgent:
    """Inspects and validates agent actions before execution."""

    # Forbidden command patterns — single source of truth in tool_guard.py
    FORBIDDEN_PATTERNS: List[Tuple[str, str]] = FORBIDDEN_PATTERNS

    # Required reasoning keywords for certain tool types
    REASONING_REQUIREMENTS: Dict[str, List[str]] = {
        "execute_shell_command": ["because", "to", "in order to", "for", "need"],
        "start_background_process": ["experiment", "training", "evaluation", "run"],
        "create_python_file": ["script", "extract", "parse", "analyze", "code"],
        "execute_python_code": ["extract", "parse", "analyze", "compute", "read"],
    }

    # Minimum reasoning length for different action types
    MIN_REASONING_LENGTH: Dict[str, int] = {
        "execute_shell_command": 20,
        "start_background_process": 30,
        "execute_python_code": 15,
        "default": 10,
    }

    # High-risk tools that warrant LLM inspection
    HIGH_RISK_TOOLS = [
        "execute_shell_command",
        "start_background_process",
        "execute_python_code",
        "create_python_file",
    ]

    def __init__(self, metrics_tracker=None, enable_llm_critic: bool = False, callbacks=None):
        """Initialize the Critic Agent.

        Args:
            metrics_tracker: Optional metrics tracker for observability
            enable_llm_critic: Whether to use LLM for deep inspection
        """
        self.metrics_tracker = metrics_tracker
        self.blocked_count = 0
        self.approved_count = 0
        self.llm_inspections = 0
        self.enable_llm_critic = enable_llm_critic
        self.callbacks = callbacks or []
        self._llm = None  # Lazy init

    def _get_llm(self):
        """Lazy initialize cheap LLM for deep inspection."""
        if self._llm is None:
            # Use cheapest available model
            self._llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                temperature=0,
                max_tokens=200,  # Short response needed
            )
        return self._llm

    def inspect_action(self, state: Dict) -> Dict:
        """Inspect proposed action with rules first, then optional LLM."""

        action = state.get("proposed_action")

        # === LAYER 1: Fast Rule Checks ===
        rule_result = self._rule_based_inspection(state)
        if not rule_result["is_authorized"]:
            # Rules blocked it - no need for LLM
            # Log the blocked action
            if action:
                 self._log_inspection(state, action, rule_result)
            return rule_result

        # === LAYER 2: Optional LLM Deep Inspection ===
        tool_name = action.get("tool_name", "") if action else ""

        if (self.enable_llm_critic
            and tool_name in self.HIGH_RISK_TOOLS
            and action):
            llm_result = self._llm_deep_inspect(state, action)
            if not llm_result["is_authorized"]:
                self.blocked_count += 1
                self._log_inspection(state, action, llm_result)
                return llm_result
            
            # LLM approved - use its result (contains inspection_type='llm')
            self.approved_count += 1
            self._log_inspection(state, action, llm_result)
            return llm_result

        # All checks passed (rules only - LLM not used for this tool type)
        self.approved_count += 1
        result = {
            "is_authorized": True,
            "critic_feedback": "Action approved by critic (rules only)",
            "inspection_type": "rules",
        }
        if action:
            self._log_inspection(state, action, result)
        return result

    def _rule_based_inspection(self, state: Dict) -> Dict:
        """Original rule-based inspection logic (moved from inspect_action).

        This critic works in two modes:
        1. If proposed_action is set: Check the specific action
        2. If no proposed_action: Check supervisor_directive for risky patterns

        Args:
            state: Current state with current_reasoning, proposed_action, or supervisor_directive

        Returns:
            Dict with:
                - is_authorized: bool
                - critic_feedback: str (reason if blocked)
                - blocked_actions: list (if blocked)
        """
        reasoning = state.get("current_reasoning", "")
        action = state.get("proposed_action")
        directive = state.get("supervisor_directive", "")

        # Mode 1: Check specific proposed action
        if not action:
            # Mode 2: Check supervisor directive for risky patterns
            # This catches dangerous directives before they reach execution agent
            directive_check = self._check_directive_safety(directive, state)
            if directive_check:
                return directive_check

            # No specific action and directive is safe - allow
            # NOTE: Don't increment approved_count here - parent inspect_action handles it
            return {
                "is_authorized": True,
                "critic_feedback": "Directive approved - no risky patterns detected",
            }

        tool_name = action.get("tool_name", "")
        tool_args = action.get("tool_args", {})

        # === Check 1: Forbidden commands ===
        if tool_name == "execute_shell_command":
            command = tool_args.get("command", "")
            block_result = self._check_forbidden_commands(command)
            if block_result:
                self.blocked_count += 1
                return block_result

        # === Check 2: Reasoning exists and has minimum length ===
        min_length = self.MIN_REASONING_LENGTH.get(
            tool_name, self.MIN_REASONING_LENGTH["default"]
        )
        if len(reasoning.strip()) < min_length:
            self.blocked_count += 1
            return {
                "is_authorized": False,
                "critic_feedback": f"BLOCKED: Insufficient reasoning (min {min_length} chars). "
                                   f"Explain WHY before acting.",
                "blocked_actions": [action],
            }

        # === Check 3: Reasoning quality for specific tools ===
        if tool_name in self.REASONING_REQUIREMENTS:
            required_keywords = self.REASONING_REQUIREMENTS[tool_name]
            has_reasoning = any(kw in reasoning.lower() for kw in required_keywords)

            if not has_reasoning:
                self.blocked_count += 1
                return {
                    "is_authorized": False,
                    "critic_feedback": f"BLOCKED: Reasoning for {tool_name} should explain purpose. "
                                       f"Expected keywords like: {required_keywords}",
                    "blocked_actions": [action],
                }

        # === Check 4: Reasoning-action coherence ===
        if not self._check_coherence(reasoning, action):
            self.blocked_count += 1
            return {
                "is_authorized": False,
                "critic_feedback": "BLOCKED: Reasoning does not justify the proposed action. "
                                   "Please clarify the connection.",
                "blocked_actions": [action],
            }

        # === Check 5: Resource limits ===
        resource_check = self._check_resource_limits(action)
        if resource_check:
            self.blocked_count += 1
            return resource_check

        # All checks passed
        # NOTE: approved_count is NOT incremented here anymore, but in the parent inspect_action
        # self.approved_count += 1 
        return {
            "is_authorized": True,
            "critic_feedback": "Action approved by rules",
        }

    def _llm_deep_inspect(self, state: Dict, action: Dict) -> Dict:
        """Use LLM for deeper semantic inspection of edge cases.

        This catches patterns that rules miss:
        - Novel dangerous commands disguised with variables
        - Subtle privilege escalation attempts
        - Commands that seem safe but have dangerous side effects
        """
        self.llm_inspections += 1

        tool_name = action.get("tool_name", "")
        tool_args = action.get("tool_args", {})
        reasoning = state.get("current_reasoning", "")
        directive = state.get("supervisor_directive", "")

        prompt = f"""You are a security critic for an automated paper reproduction system.

GOAL: REPRODUCE PAPER RESULTS
You are part of an automated system designed to reproduce the results of a scientific paper. Your individual tasks must always serve this ultimate goal.

Inspect this proposed action and determine if it's SAFE or DANGEROUS.

CONTEXT:
- Supervisor directive: {directive[:200]}
- Agent reasoning: {reasoning[:300]}

PROPOSED ACTION:
- Tool: {tool_name}
- Arguments: {str(tool_args)[:500]}

SAFETY RULES:
1. No privilege escalation (sudo, su, chmod 777)
2. No recursive deletes outside the working directory
3. No network requests to untrusted endpoints
4. No modification of system files
5. No execution of downloaded scripts without inspection
6. Commands should match the stated reasoning

Respond with ONLY one of:
- SAFE: [brief reason]
- DANGEROUS: [brief reason why blocked]
"""

        try:
            llm = self._get_llm()
            # Use callbacks for metrics tracking if available
            response = llm.invoke(prompt, config={"callbacks": self.callbacks} if self.callbacks else None)
            result_text = response.content.strip()

            if result_text.startswith("DANGEROUS:"):
                reason = result_text.replace("DANGEROUS:", "").strip()
                return {
                    "is_authorized": False,
                    "critic_feedback": f"BLOCKED by LLM critic: {reason}",
                    "blocked_actions": [action],
                    "inspection_type": "llm",
                }
            else:
                return {
                    "is_authorized": True,
                    "critic_feedback": "Approved by LLM critic",
                    "inspection_type": "llm",
                }
        except Exception as e:
            # LLM failed - fall back to allowing (rules already passed)
            return {
                "is_authorized": True,
                "critic_feedback": f"LLM inspection failed ({e}), allowing based on rules",
                "inspection_type": "llm_fallback",
            }

    def _check_directive_safety(self, directive: str, state: Dict) -> Dict:
        """Check if a supervisor directive is safe to execute.

        This provides high-level safety checks before execution agent runs.

        Args:
            directive: The supervisor directive text
            state: Current state for context

        Returns:
            Block result dict if unsafe, None if safe
        """
        directive_lower = directive.lower()

        # Check for dangerous directive patterns
        dangerous_patterns = [
            ("delete all", "Directive suggests destructive operation"),
            ("remove everything", "Directive suggests destructive operation"),
            ("sudo", "Directive mentions privilege escalation"),
            ("format", "Directive may involve disk formatting"),
            ("rm -rf /", "Directive contains dangerous delete pattern"),
        ]

        for pattern, reason in dangerous_patterns:
            if pattern in directive_lower:
                self.blocked_count += 1
                return {
                    "is_authorized": False,
                    "critic_feedback": f"BLOCKED: {reason}. Directive: '{directive[:50]}...'",
                    "blocked_actions": [{"directive": directive}],
                }

        # Check if this is a retry after too many failures
        cycle_count = state.get("cycle_count", 0)
        max_cycles = state.get("max_cycles", 5)
        if cycle_count >= max_cycles - 1:  # About to hit limit
            # Warn but don't block
            pass

        return None  # Safe

    def _check_forbidden_commands(self, command: str) -> Dict:
        """Check if command matches any forbidden patterns.

        Args:
            command: Shell command to check

        Returns:
            Block result dict if forbidden, None if allowed
        """
        for pattern, reason in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return {
                    "is_authorized": False,
                    "critic_feedback": f"BLOCKED: {reason}. Pattern: '{pattern}'",
                    "blocked_actions": [{"tool_name": "execute_shell_command", "command": command}],
                }
        return None

    def _check_coherence(self, reasoning: str, action: Dict) -> bool:
        """Check if reasoning justifies the proposed action.

        Args:
            reasoning: Agent's reasoning text
            action: Proposed action dict

        Returns:
            True if coherent, False otherwise
        """
        tool_name = action.get("tool_name", "")
        tool_args = action.get("tool_args", {})
        reasoning_lower = reasoning.lower()

        if tool_name == "read_file":
            # Reasoning should mention reading or the file
            file_path = tool_args.get("file_path", "")
            file_name = file_path.split("/")[-1].split("\\")[-1].lower()
            return (
                "read" in reasoning_lower
                or "check" in reasoning_lower
                or "examine" in reasoning_lower
                or "look" in reasoning_lower
                or "inspect" in reasoning_lower
                or file_name in reasoning_lower
            )

        if tool_name == "execute_shell_command":
            command = tool_args.get("command", "")
            # Extract key command words (first 3 words)
            command_words = re.findall(r"\b\w+\b", command)[:3]
            # At least one command word should appear in reasoning
            return any(word.lower() in reasoning_lower for word in command_words) or len(reasoning) > 50

        if tool_name == "list_directory":
            # Reasoning should mention listing, checking, or looking at files
            return (
                "list" in reasoning_lower
                or "check" in reasoning_lower
                or "see" in reasoning_lower
                or "look" in reasoning_lower
                or "directory" in reasoning_lower
                or "files" in reasoning_lower
            )

        if tool_name == "execute_python_code":
            # Reasoning should explain what the code does
            return len(reasoning) > 20  # Python code needs good explanation

        if tool_name == "start_background_process":
            # Reasoning should mention running, training, or experiment
            return (
                "run" in reasoning_lower
                or "train" in reasoning_lower
                or "experiment" in reasoning_lower
                or "execute" in reasoning_lower
                or "start" in reasoning_lower
            )

        # Default: allow if reasoning is substantial
        return len(reasoning) > 15

    def _check_resource_limits(self, action: Dict) -> Dict:
        """Check if action respects resource limits.

        Args:
            action: Proposed action

        Returns:
            Block result if limits exceeded, None if ok
        """
        tool_name = action.get("tool_name", "")
        tool_args = action.get("tool_args", {})

        if tool_name == "start_background_process":
            timeout = tool_args.get("timeout", 604800)  # Default 7 days
            max_timeout = 604800  # 7 days in seconds

            if timeout > max_timeout:
                return {
                    "is_authorized": False,
                    "critic_feedback": f"BLOCKED: Timeout {timeout}s exceeds maximum {max_timeout}s (7 days)",
                    "blocked_actions": [action],
                }

        if tool_name == "execute_shell_command":
            timeout = tool_args.get("timeout", 3600)
            max_timeout = 7200  # 2 hours for shell commands

            if timeout > max_timeout:
                return {
                    "is_authorized": False,
                    "critic_feedback": f"BLOCKED: Shell command timeout {timeout}s exceeds maximum {max_timeout}s. "
                                       f"Use start_background_process for long-running commands.",
                    "blocked_actions": [action],
                }

        return None

    def get_stats(self) -> Dict:
        """Get critic statistics including LLM usage."""
        return {
            "blocked_count": self.blocked_count,
            "approved_count": self.approved_count,
            "total_inspected": self.blocked_count + self.approved_count,
            "llm_inspections": self.llm_inspections,
            "block_rate": (
                self.blocked_count / (self.blocked_count + self.approved_count)
                if (self.blocked_count + self.approved_count) > 0
                else 0
            ),
        }

    def reset_stats(self):
        """Reset critic statistics."""
        self.blocked_count = 0
        self.approved_count = 0
        self.llm_inspections = 0

    def add_forbidden_pattern(self, pattern: str, reason: str):
        """Add a new forbidden pattern at runtime.

        Args:
            pattern: Regex pattern to block
            reason: Explanation for why it's blocked
        """
        self.FORBIDDEN_PATTERNS.append((pattern, reason))

    def validate_reasoning(self, reasoning: str, tool_name: str) -> Tuple[bool, str]:
        """Standalone reasoning validation without full action check.

        Useful for pre-validation before building an action.

        Args:
            reasoning: The reasoning text
            tool_name: The tool that will be used

        Returns:
            Tuple of (is_valid, feedback_message)
        """
        min_length = self.MIN_REASONING_LENGTH.get(
            tool_name, self.MIN_REASONING_LENGTH["default"]
        )

        if len(reasoning.strip()) < min_length:
            return False, f"Reasoning too short (min {min_length} chars)"

        if tool_name in self.REASONING_REQUIREMENTS:
            required_keywords = self.REASONING_REQUIREMENTS[tool_name]
            has_keyword = any(kw in reasoning.lower() for kw in required_keywords)
            if not has_keyword:
                return False, f"Missing purpose keywords: {required_keywords}"

        return True, "Reasoning is valid"

    def _log_inspection(self, state: Dict, action: Dict, result: Dict):
        """Log inspection for future ML training data."""
        if not self.metrics_tracker:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "tool_name": action.get("tool_name") if action else None,
            "tool_args_hash": hash(str(action.get("tool_args"))) if action else None,
            "reasoning_length": len(state.get("current_reasoning", "")),
            "directive_length": len(state.get("supervisor_directive", "")),
            "is_authorized": result["is_authorized"],
            "inspection_type": result.get("inspection_type", "rules"),
            "feedback": result.get("critic_feedback", "")[:100],
        }

        # Append to JSONL file for training data collection
        # Ensure log dir exists
        log_dir = self.metrics_tracker.log_dir if hasattr(self.metrics_tracker, 'log_dir') else "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, "critic_inspections.jsonl")
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            # Ignore logging errors to not break execution
            pass
