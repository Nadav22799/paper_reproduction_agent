"""Supervisor Agent - Coordinates sub-agents based on objective and state.

The Supervisor acts as the "brain" of the cyclic state machine, deciding which
sub-agent should handle the current task based on:
1. Phase completion status
2. Error types for targeted recovery routing
3. Planning update requests from sub-agents
"""

from typing import Dict, Optional
from ..utils.llm_factory import create_llm
from ..utils.hierarchical_context import HierarchicalContextManager


class SupervisorAgent:
    """Coordinates sub-agents and handles failure routing in the cyclic state machine."""

    def __init__(
        self,
        llm=None,
        metrics_tracker=None,
        hierarchical_context: HierarchicalContextManager = None,
        callbacks=None,
    ):
        """Initialize the Supervisor Agent.

        Args:
            llm: Language model for complex routing decisions
            metrics_tracker: Optional metrics tracker for observability
            hierarchical_context: Shared context manager for cross-agent knowledge
        """
        self.llm = llm or create_llm(temperature=0.1)
        self.metrics_tracker = metrics_tracker
        self.hierarchical_context = hierarchical_context
        self.callbacks = callbacks or []

    def decide_next_agent(self, state: Dict) -> Dict:
        """Determine which agent should handle the current state.

        Args:
            state: Current PaperReproductionState

        Returns:
            Dict with:
                - agent: str - Name of the agent to invoke
                - directive: str - Instruction for the agent
                - context: Optional[dict] - Additional context for recovery
        """
        # Priority 1: Check for planning update requests from sub-agents
        planning_request = state.get("planning_update_request")
        if planning_request:
            return {
                "agent": "planning_update",
                "directive": f"Update plan: {planning_request.get('reason', 'Sub-agent requested update')}",
                "context": planning_request,
            }

        # Priority 2: Handle failures with targeted routing
        failure_metadata = state.get("failure_metadata")
        if failure_metadata and failure_metadata.get("error_type") != "none":
            return self._route_failure(state)

        # Priority 3: Check phase completion order
        phase_status = state.get("phase_status", {})

        # Planning must come first
        if phase_status.get("planning") != "completed":
            return {"agent": "planning", "directive": "Create reproduction plan from README"}

        # Environment setup
        if phase_status.get("environment") != "completed":
            return {"agent": "environment", "directive": "Setup environment and verify with smoke test"}

        # Data preparation - DEFAULT: SKIP unless explicitly required
        # This makes data_prep REACTIVE (triggered by execution failure) not PROACTIVE
        if phase_status.get("data_prep") != "completed":

            # SKIP CONDITION 1: Environment smoke test passed (data already works!)
            env_results = state.get("env_setup_results", {})
            if env_results.get("success"):
                print("   ⏭️  Skipping data_prep (Environment smoke test already passed)")
                return {
                    "agent": "data_prep_skip",
                    "directive": "Data prep skipped - environment smoke test already verified functionality",
                }

            # SKIP CONDITION 2: Planning explicitly said skip
            reproduction_plan = state.get("reproduction_plan", {})
            skip_data_prep = reproduction_plan.get("skip_data_prep", False)
            if skip_data_prep:
                print("   ⏭️  Skipping data_prep (scripts auto-download data)")
                return {
                    "agent": "data_prep_skip",
                    "directive": "Data prep skipped - scripts handle data download automatically",
                }

            # ONLY RUN DATA_PREP IF: Planning found explicit data prep instructions in README
            requires_data_prep = reproduction_plan.get("requires_data_prep", False)
            if requires_data_prep:
                return {"agent": "data_prep", "directive": "Prepare datasets following checklist"}

            # DEFAULT: Skip data_prep, execution will fail-back if needed
            print("   ⏭️  Skipping data_prep (no explicit data prep required)")
            return {
                "agent": "data_prep_skip",
                "directive": "Data prep skipped - execution will route back if data issues occur",
            }

        # Execution
        if phase_status.get("execution") != "completed":
            return {"agent": "execution", "directive": "Run experiments using background process pattern"}

        # Validation
        if phase_status.get("validation") != "completed":
            return {"agent": "validation", "directive": "Verify results using code-first approach"}

        # All phases complete
        return {"agent": "complete", "directive": "All phases complete - generate final report"}

    def _get_relevant_context(self, query: str, max_entries: int = 5) -> str:
        """Query hierarchical context for relevant historical information.

        Args:
            query: Query string for semantic search
            max_entries: Maximum entries to return

        Returns:
            Formatted context string
        """
        if not self.hierarchical_context:
            return ""

        try:
            relevant = self.hierarchical_context.retrieve(
                query=query,
                max_tokens=2000,  # Small budget for supervisor decisions
                min_relevance=0.3,  # Only relevant entries
            )

            if not relevant:
                return ""

            sections = []
            for r in relevant[:max_entries]:
                source = r.get("source", "context")
                entry_type = r.get("type", "")
                content = r.get("content", "")[:200]  # Truncate for brevity
                sections.append(f"[{source}/{entry_type}] {content}")

            return "\n".join(sections)
        except Exception as e:
            print(f"   ⚠️ Context query failed: {e}")
            return ""

    def _route_failure(self, state: Dict) -> Dict:
        """Route based on error classification for targeted recovery.

        Uses hierarchical context to check past errors and avoid repeating
        failed recovery strategies.

        Args:
            state: Current state with failure_metadata

        Returns:
            Routing decision dict
        """
        failure = state["failure_metadata"]
        error_type = failure.get("error_type", "unknown")
        error_message = failure.get("error_message", "")
        retry_count = failure.get("retry_count", 0)
        attempted_fixes = failure.get("attempted_fixes", [])
        max_retries = state.get("max_recovery_attempts", 3)

        # Query context for similar past errors
        past_errors = self._get_relevant_context(f"error {error_type} {error_message[:100]}")
        if past_errors:
            print(f"   🧠 Found relevant past context for error recovery")

        # Store this error in context for future reference
        if self.hierarchical_context:
            self.hierarchical_context.add(
                content=f"[Routing] Error: {error_type} - {error_message[:300]}. Retry #{retry_count + 1}",
                source="supervisor",
                entry_type="error",
                importance=0.8,
                lazy=True,  # Defer embedding to avoid loading SentenceTransformer
            )

        # Check if we've exceeded max retries
        if retry_count >= max_retries:
            return {
                "agent": "report",
                "directive": f"Generate failure report: Max retries ({max_retries}) exceeded for {error_type} error",
                "context": failure,
            }

        # Check cycle count to prevent infinite loops
        cycle_count = state.get("cycle_count", 0)
        max_cycles = state.get("max_cycles", 5)
        if cycle_count >= max_cycles:
            return {
                "agent": "report",
                "directive": f"Generate failure report: Max cycles ({max_cycles}) exceeded",
                "context": failure,
            }

        # Build context for the target agent with historical knowledge
        context = {
            "error": error_message,
            "previous_attempts": attempted_fixes,
            "retry_number": retry_count + 1,
            "hints": failure.get("recovery_hints", []),
        }

        # Add relevant historical context if available
        if past_errors:
            context["historical_context"] = past_errors
            # Check if similar error was seen before and what worked
            if "success" in past_errors.lower():
                print("   💡 Found successful recovery pattern in history")

        # Route based on error type
        if error_type == "environment":
            return self._route_environment_error(failure, context, attempted_fixes)

        elif error_type == "data":
            return {
                "agent": "data_prep",
                "directive": f"Resolve data error: {error_message[:100]}",
                "context": context,
            }

        elif error_type == "execution":
            return self._route_execution_error(failure, context, error_message)

        elif error_type == "validation":
            return {
                "agent": "execution",
                "directive": "Re-run experiments with adjusted parameters for validation retry",
                "context": context,
            }

        elif error_type == "authorization":
            # Critic blocked an action - agent needs to try different approach
            return {
                "agent": state.get("current_agent", "execution"),
                "directive": f"Action blocked by critic: {state.get('critic_feedback', 'Unknown reason')}. Try alternative approach.",
                "context": {"blocked_action": state.get("proposed_action"), **context},
            }

        # Default: report failure
        return {
            "agent": "report",
            "directive": f"Generate failure report: Unknown error type '{error_type}'",
            "context": failure,
        }

    def _route_environment_error(self, failure: Dict, context: Dict, attempted_fixes: list) -> Dict:
        """Route environment-specific errors.

        Args:
            failure: Failure metadata
            context: Recovery context
            attempted_fixes: List of previous fix attempts

        Returns:
            Routing decision
        """
        error_message = failure.get("error_message", "").lower()

        # Check if we've already tried common fixes
        if "pip install" in " ".join(attempted_fixes):
            # Try different approach
            return {
                "agent": "environment",
                "directive": "Previous pip install failed. Try conda or version pinning.",
                "context": {**context, "try_alternative": True},
            }

        # Extract module name if it's a ModuleNotFoundError
        if "modulenotfounderror" in error_message or "no module named" in error_message:
            import re
            module_match = re.search(r"no module named ['\"]?(\w+)", error_message)
            module_name = module_match.group(1) if module_match else "unknown"
            return {
                "agent": "environment",
                "directive": f"Install missing module: {module_name}",
                "context": {**context, "missing_module": module_name},
            }

        return {
            "agent": "environment",
            "directive": f"Fix environment issue: {failure.get('error_message', '')[:100]}",
            "context": context,
        }

    def _route_execution_error(self, failure: Dict, context: Dict, error_message: str) -> Dict:
        """Route execution-specific errors.

        Args:
            failure: Failure metadata
            context: Recovery context
            error_message: The error message

        Returns:
            Routing decision
        """
        error_lower = error_message.lower()

        # OOM errors - retry with reduced batch size
        if "out of memory" in error_lower or "cuda out of memory" in error_lower:
            return {
                "agent": "execution",
                "directive": "Reduce batch size by 50% and retry experiment",
                "context": {**context, "reduce_batch": True, "oom_retry": True},
            }

        # Module not found during execution - actually an environment issue
        if "modulenotfounderror" in error_lower or "no module named" in error_lower:
            return {
                "agent": "environment",
                "directive": f"Missing module during execution: {error_message[:100]}",
                "context": context,
            }

        # File not found for data - route to data prep
        if "filenotfounderror" in error_lower and ("data" in error_lower or "dataset" in error_lower):
            return {
                "agent": "data_prep",
                "directive": f"Missing data file: {error_message[:100]}",
                "context": context,
            }

        # Timeout - retry with longer timeout or smaller dataset
        if "timeout" in error_lower:
            return {
                "agent": "execution",
                "directive": "Experiment timed out. Try with smaller dataset or longer timeout.",
                "context": {**context, "timeout_retry": True},
            }

        # Default: retry execution with generic recovery
        return {
            "agent": "execution",
            "directive": f"Retry execution: {error_message[:100]}",
            "context": context,
        }

    def should_continue(self, state: Dict) -> bool:
        """Check if the workflow should continue or terminate.

        Args:
            state: Current state

        Returns:
            True if workflow should continue, False to terminate
        """
        # Check for terminal conditions
        if state.get("final_status"):
            return False

        # Check cycle limits
        cycle_count = state.get("cycle_count", 0)
        max_cycles = state.get("max_cycles", 5)
        if cycle_count >= max_cycles:
            return False

        # Check if all phases completed
        phase_status = state.get("phase_status", {})
        all_completed = all(
            phase_status.get(phase) == "completed"
            for phase in ["planning", "environment", "data_prep", "execution", "validation"]
        )
        if all_completed:
            return False

        return True

    def update_phase_status(self, state: Dict, phase: str, status: str) -> Dict:
        """Update the status of a specific phase.

        Args:
            state: Current state
            phase: Phase name
            status: New status ("pending", "running", "completed", "failed")

        Returns:
            Updated state
        """
        if "phase_status" not in state:
            state["phase_status"] = {}

        state["phase_status"][phase] = status

        # If phase failed, set up failure routing
        if status == "failed":
            # The specific failure metadata should be set by the agent that failed
            pass

        return state

    def clear_failure_state(self, state: Dict) -> Dict:
        """Clear failure state after successful recovery.

        Args:
            state: Current state

        Returns:
            State with cleared failure fields
        """
        state["failure_metadata"] = None
        state["recovery_attempts"] = 0
        # Don't clear cycle_count - it's a global limit
        return state

    def increment_recovery_attempt(self, state: Dict) -> Dict:
        """Increment recovery attempt counter.

        Args:
            state: Current state

        Returns:
            State with incremented counter
        """
        current = state.get("recovery_attempts", 0)
        state["recovery_attempts"] = current + 1
        state["cycle_count"] = state.get("cycle_count", 0) + 1
        return state
    def classify_error(self, error_message: str) -> str:
        """Classify an error message using the LLM for semantic understanding.

        Args:
            error_message: The error message string

        Returns:
            str: One of "DATA", "ENVIRONMENT", or "CODE"
        """
        if not error_message:
            return "ENVIRONMENT"

        prompt = f"""Analyze the following error message from a machine learning experiment setup:

GOAL: REPRODUCE PAPER RESULTS
You are part of an automated system designed to reproduce the results of a scientific paper. Your individual tasks must always serve this ultimate goal.

ERROR MESSAGE:
{error_message[:2000]}

Classify the error into EXACTLY one of these categories:

1. DATA
   - Missing dataset files (e.g., "FileNotFoundError: data/cora/ind.x")
   - Wrong data path
   - Download failures

2. ENVIRONMENT
   - Missing python packages (ModuleNotFoundError)
   - Library version conflicts
   - Build/Compilation errors
   - CUDA errors

3. CODE
   - Syntax errors
   - Missing configuration files (requirements.txt, environment.yml)
   - Missing scripts (setup.py not found)
   - Logic errors

GUIDELINES:
- "No such file: data/..." -> DATA
- "No such file: requirements.txt" -> CODE
- "No module named..." -> ENVIRONMENT

Return ONLY the category name (DATA, ENVIRONMENT, or CODE)."""

        try:
            from langchain_core.messages import HumanMessage
            
            config = {}
            if self.callbacks:
                config["callbacks"] = self.callbacks
                
            response = self.llm.invoke([HumanMessage(content=prompt)], config=config)
            
            content = response.content
            if isinstance(content, list):
                 # Handle list content (e.g. from multimodal models or structured output)
                 content = " ".join([str(item) for item in content])
            
            classification = content.strip().upper()
            
            # Fallback for unexpected output
            if "DATA" in classification:
                return "DATA"
            elif "CODE" in classification:
                return "CODE"
            else:
                return "ENVIRONMENT"
                
        except Exception as e:
            print(f"⚠️  Error classification failed: {e}")
            return "ENVIRONMENT"  # Default to generic env error
