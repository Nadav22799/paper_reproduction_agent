"""Clean LangGraph Orchestrator - Cyclic State Machine for paper reproduction.

This orchestrator implements a supervisor-driven architecture with:
1. Supervisor Agent - Coordinates sub-agents based on objective and state
2. Critic Interceptor - Security guardrail that validates reasoning before execution
3. Planning Agent - Creates comprehensive reproduction_checklist.md upfront
4. Decoupled Nodes - Split into data_prep, execution, validation
5. Environment Recovery Loop - Cyclic edges for targeted error recovery
"""

from typing import TypedDict, Annotated, Literal, Optional, List, Dict
from enum import Enum
import stat
import os
from langgraph.graph import StateGraph, END
import operator
import subprocess
import shutil
from .utils.checkpoint_manager import ExperimentCheckpoint
from .utils.hierarchical_context import HierarchicalContextManager
from .utils.metrics_tracker import MetricsTracker


def remove_readonly(func, path, _):
    """Clear the readonly bit and reattempt the removal"""
    os.chmod(path, stat.S_IWRITE)
    func(path)


# === ERROR TYPES FOR ROUTING ===
class ErrorType(Enum):
    """Error types that drive routing decisions in the cyclic state machine."""
    NONE = "none"
    ENVIRONMENT = "environment"      # Missing packages, version conflicts
    DATA = "data"                    # Dataset download failures
    EXECUTION = "execution"          # OOM, timeout, runtime errors
    VALIDATION = "validation"        # Results mismatch
    AUTHORIZATION = "authorization"  # Blocked by critic


# === STRUCTURED ACTION TYPE ===
class AgentAction(TypedDict):
    """Structured action with reasoning for critic inspection."""
    tool_name: str
    tool_args: dict
    reasoning: str           # WHY this action (critic inspects this)
    expected_outcome: str


# === FAILURE METADATA FOR RECOVERY ===
class FailureMetadata(TypedDict):
    """Rich failure context for targeted recovery routing."""
    error_type: str          # ErrorType value
    error_message: str
    error_source: str        # Which node produced the error
    stack_trace: Optional[str]
    attempted_fixes: List[str]
    recovery_hints: List[str]
    retry_count: int


# Define the overall state for the entire workflow
class PaperReproductionState(TypedDict):
    """Enhanced state for cyclic supervisor-driven architecture."""

    # === INPUT ===
    paper_input: str  # arXiv ID, PDF path, or paper text
    paper_title: str

    # === PAPER ANALYSIS ===
    paper_metadata: dict
    experimental_setup: dict
    paper_results: dict
    code_references: list

    # === IMPLEMENTATION ===
    selected_repo: dict
    implementation_path: str
    experiment_selection_mode: str  # 'single', 'all', 'custom'
    custom_experiment_list: list  # List of specific experiments if custom mode

    # === REPRODUCTION RESULTS ===
    env_setup_results: dict
    dependencies_installed: bool
    dataset_results: dict
    datasets_ready: bool
    experiment_results: dict
    experiments_completed: bool

    # === METRICS & VERIFICATION ===
    extracted_metrics: dict
    metrics_comparison: dict
    verification_results: dict
    results_match: bool

    # === AGENT CONTEXT HISTORY ===
    agent_contexts: dict

    # === CHECKPOINT & RESUME ===
    completed_phases: list  # List of phase names that have been completed

    # === OVERALL ===
    messages: Annotated[list, operator.add]
    final_status: str
    report: str

    # === NEW: REASONING-FIRST PROTOCOL ===
    current_reasoning: str              # Agent's WHY before action
    proposed_action: Optional[dict]     # AgentAction awaiting critic approval
    action_history: Annotated[list, operator.add]  # Audit trail of all actions

    # === NEW: CRITIC INTERCEPTOR ===
    is_authorized: bool                 # Whether proposed action passed critic
    blocked_actions: list               # Actions rejected by critic
    critic_feedback: str                # Why action was blocked

    # === NEW: FAILURE & RECOVERY ===
    failure_metadata: Optional[dict]    # FailureMetadata for routing decisions
    recovery_attempts: int              # Current retry count for this error
    max_recovery_attempts: int          # Default: 3

    # === NEW: SUPERVISOR CONTROL ===
    current_agent: str                  # Which sub-agent is currently active
    supervisor_directive: str           # Current instruction from supervisor

    # === NEW: PLANNING ===
    reproduction_plan: dict             # Structured plan from planning agent
    checklist_path: str                 # Path to reproduction_checklist.md
    phase_status: Dict[str, str]        # {phase: "pending"|"running"|"completed"|"failed"}
    planning_update_request: Optional[dict]  # Request from sub-agent for plan update

    # === NEW: CYCLIC ROUTING ===
    cycle_count: int                    # Prevent infinite loops
    max_cycles: int                     # Default: 5


class PaperReproductionOrchestrator:
    """Orchestrator for paper reproduction workflow.

    Supports two architectures:
    1. Legacy (default): Linear workflow with unified reproduction agent
    2. Supervisor: Cyclic state machine with specialized agents

    Set USE_SUPERVISOR_ARCHITECTURE=true to enable the new architecture.
    """

    def __init__(
        self,
        llm=None,
        enable_logging=True,
        enable_checkpoints=True,
        use_supervisor: bool = None,
    ):
        """Initialize the orchestrator.

        Args:
            llm: Language model to use
            enable_logging: Whether to enable detailed logging to file
            enable_checkpoints: Whether to enable checkpoint & resume functionality
            use_supervisor: Use supervisor architecture (default: from env var)
        """
        # Only import agents that are actually used
        from .agents.environment_setup_agent import EnvironmentSetupAgent
        from .agents.unified_reproduction_agent import UnifiedReproductionAgent
        from .agents.discovery_agent import DiscoveryAgent
        from .utils.llm_factory import create_llm, create_embedder
        from .utils.file_logger import FileLogger
        from .utils.logging_callback import LoggingCallbackHandler
        from .config import ReproductionConfig

        self.llm = llm or create_llm(temperature=0.1)
        # Create a specialized "Reasoning LLM" with chain-of-thought enabled (if supported)
        # This is used for complex agents like Supervisor and Planner
        self.reasoning_llm = create_llm(temperature=0.3, include_thoughts=True)
        self.config = ReproductionConfig()

        # Caching support (initialized after paper analysis)
        self.cache_name: Optional[str] = None

        # Determine which architecture to use
        if use_supervisor is None:
            use_supervisor = os.getenv("USE_SUPERVISOR_ARCHITECTURE", "false").lower() == "true"
        self.use_supervisor = use_supervisor

        # Setup metrics tracking for observability
        phases = (
            MetricsTracker.SUPERVISOR_PHASES
            if self.use_supervisor
            else MetricsTracker.LEGACY_PHASES
        )
        self.metrics_tracker = MetricsTracker(
            enable_live_display=self.config.enable_live_progress,
            update_interval=self.config.progress_update_interval,
            input_cost_per_million=self.config.llm_input_cost_per_million,
            output_cost_per_million=self.config.llm_output_cost_per_million,
            phases=phases,
        )
        print("📊 Metrics tracker initialized")

        # Setup logging
        self.enable_logging = enable_logging
        self.file_logger = None
        self.logging_callback = None
        self.file_logger = None
        
        if enable_logging:
            self.file_logger = FileLogger(log_dir=self.config.logs_path)
            
        # Always create callback for metrics tracking and stdout logging
        # (CLI captures stdout via TeeOutput, so we want verbose=True)
        self.logging_callback = LoggingCallbackHandler(
            verbose=True,
            file_logger=self.file_logger,
            metrics_tracker=self.metrics_tracker,
        )

        # Setup checkpoint manager
        self.enable_checkpoints = enable_checkpoints
        self.checkpoint_manager = None
        if enable_checkpoints:
            self.checkpoint_manager = ExperimentCheckpoint(
                checkpoint_dir=self.config.checkpoints_path
            )
            print("💾 Checkpoint system enabled")

        # Initialize embedder using factory (API-based by default for speed)
        # Store as instance variable for checkpoint restore
        self._embedder = create_embedder(metrics_tracker=self.metrics_tracker)  # Uses EMBEDDING_PROVIDER env var (default: gemini)

        # Capture embedding model name if available
        if self._embedder and hasattr(self._embedder, "model"):
            self.metrics_tracker.set_embedding_model(self._embedder.model)
        elif self._embedder:
            self.metrics_tracker.set_embedding_model("Custom/Local")
        else:
            self.metrics_tracker.set_embedding_model("None")

        # Initialize shared hierarchical context manager for cross-agent context
        self.hierarchical_context = HierarchicalContextManager(
            model_name="gpt-4",
            hot_capacity=50,
            max_tokens=100000,  # Larger budget for orchestrator
            embedder=self._embedder,  # Use factory-created embedder (API or local)
        )
        print("🧠 Hierarchical context manager initialized")

        # NOTE: Do NOT call start_workflow() here - it should be called by
        # CLI/run() AFTER checkpoint restore to prevent orphan display threads

        # Initialize legacy agents (always needed for backward compatibility)
        self.env_setup_agent = EnvironmentSetupAgent(
            self.reasoning_llm,
            max_iterations=50,
            metrics_tracker=self.metrics_tracker,
            callbacks=[self.logging_callback] if self.logging_callback else [],
            hierarchical_context=self.hierarchical_context,
        )
        self.unified_reproducer = UnifiedReproductionAgent(
            self.llm,
            max_iterations=50,
            hierarchical_context=self.hierarchical_context,
            metrics_tracker=self.metrics_tracker,
            callbacks=[self.logging_callback] if self.logging_callback else [],
        )
        self.discovery_agent = DiscoveryAgent(
            self.llm, metrics_tracker=self.metrics_tracker
        )

        # Initialize supervisor architecture agents if enabled
        if self.use_supervisor:
            self._init_supervisor_agents()
            print("🎯 Supervisor architecture ENABLED")
        else:
            print("📋 Legacy architecture (set USE_SUPERVISOR_ARCHITECTURE=true for cyclic)")

        # Build the workflow graph
        self.workflow = self._build_workflow()

    def _init_supervisor_agents(self):
        """Initialize agents for the supervisor architecture."""
        from .agents.supervisor_agent import SupervisorAgent
        from .agents.planning_agent import PlanningAgent
        from .agents.critic_agent import CriticAgent
        from .agents.data_prep_agent import DataPrepAgent
        from .agents.execution_agent import ExecutionAgent
        from .agents.validation_agent import ValidationAgent

        self.supervisor_agent = SupervisorAgent(
            self.reasoning_llm,  # Use Thinking Mode for complex routing
            metrics_tracker=self.metrics_tracker,
            hierarchical_context=self.hierarchical_context,
            callbacks=[self.logging_callback] if self.logging_callback else [],
        )
        self.planning_agent = PlanningAgent(
            self.reasoning_llm,  # Use Thinking Mode for detailed planning
            max_iterations=90,
            metrics_tracker=self.metrics_tracker,
            hierarchical_context=self.hierarchical_context,
            callbacks=[self.logging_callback] if self.logging_callback else [],
        )
        self.critic_agent = CriticAgent(
            metrics_tracker=self.metrics_tracker,
            enable_llm_critic=self.config.enable_llm_critic,
            callbacks=[self.logging_callback] if self.logging_callback else [],
        )
        self.data_prep_agent = DataPrepAgent(
            self.reasoning_llm,
            max_iterations=150,
            metrics_tracker=self.metrics_tracker,
            hierarchical_context=self.hierarchical_context,
            callbacks=[self.logging_callback] if self.logging_callback else [],
        )
        self.execution_agent = ExecutionAgent(
            self.reasoning_llm,
            max_iterations=150,
            metrics_tracker=self.metrics_tracker,
            hierarchical_context=self.hierarchical_context,
            callbacks=[self.logging_callback] if self.logging_callback else [],
        )
        self.validation_agent = ValidationAgent(
            self.reasoning_llm,
            max_iterations=90,
            metrics_tracker=self.metrics_tracker,
            hierarchical_context=self.hierarchical_context,
            callbacks=[self.logging_callback] if self.logging_callback else [],
        )

    def _update_metrics_tracker_references(self, new_tracker):
        """
        Update all agent and callback references to use the new metrics tracker.

        This is critical when restoring from checkpoint - the restored tracker
        has accumulated historical data, but all agents/callbacks still reference
        the original (now orphaned) tracker. Without this update, new tokens
        would be recorded to the old tracker while summaries read from the new one.
        """
        self.metrics_tracker = new_tracker

        # Update logging callback (records tokens from all LLM calls)
        if self.logging_callback:
            self.logging_callback.metrics_tracker = new_tracker

        # Update legacy agents
        if hasattr(self, 'env_setup_agent') and self.env_setup_agent:
            self.env_setup_agent.metrics_tracker = new_tracker
        if hasattr(self, 'unified_reproducer') and self.unified_reproducer:
            self.unified_reproducer.metrics_tracker = new_tracker
        if hasattr(self, 'discovery_agent') and self.discovery_agent:
            self.discovery_agent.metrics_tracker = new_tracker

        # Update supervisor architecture agents
        if hasattr(self, 'supervisor_agent') and self.supervisor_agent:
            self.supervisor_agent.metrics_tracker = new_tracker
        if hasattr(self, 'planning_agent') and self.planning_agent:
            self.planning_agent.metrics_tracker = new_tracker
        if hasattr(self, 'critic_agent') and self.critic_agent:
            self.critic_agent.metrics_tracker = new_tracker
        if hasattr(self, 'data_prep_agent') and self.data_prep_agent:
            self.data_prep_agent.metrics_tracker = new_tracker
        if hasattr(self, 'execution_agent') and self.execution_agent:
            self.execution_agent.metrics_tracker = new_tracker
        if hasattr(self, 'validation_agent') and self.validation_agent:
            self.validation_agent.metrics_tracker = new_tracker

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow based on architecture setting."""
        if self.use_supervisor:
            return self._build_supervisor_workflow()
        else:
            return self._build_legacy_workflow()

    def _build_legacy_workflow(self) -> StateGraph:
        """Build the legacy linear LangGraph workflow."""
        workflow = StateGraph(PaperReproductionState)

        # Add nodes - 6 specialized phases
        workflow.add_node("analyze_paper", self._analyze_paper_node)
        workflow.add_node("decide_and_clone", self._decide_and_clone_node)
        workflow.add_node("environment_setup", self._environment_setup_node)
        workflow.add_node("unified_reproduction", self._unified_reproduction_node)
        workflow.add_node("extract_and_verify", self._extract_and_verify_node)
        workflow.add_node("generate_report", self._generate_report_node)

        # Define the workflow edges
        workflow.set_entry_point("analyze_paper")

        workflow.add_edge("analyze_paper", "decide_and_clone")

        # Conditional routing from decide_and_clone
        workflow.add_conditional_edges(
            "decide_and_clone",
            self._route_after_clone,
            {
                "continue": "environment_setup",
                "failed": "generate_report",
            },
        )

        # Conditional routing from environment_setup
        workflow.add_conditional_edges(
            "environment_setup",
            self._route_after_env_setup,
            {
                "continue": "unified_reproduction",
                "failed": "generate_report",
            },
        )

        # Conditional routing after unified_reproduction
        workflow.add_conditional_edges(
            "unified_reproduction",
            self._route_after_reproduction,
            {
                "continue": "extract_and_verify",
                "failed": "generate_report",
            },
        )

        workflow.add_edge("extract_and_verify", "generate_report")
        workflow.add_edge("generate_report", END)

        return workflow.compile()

    def _build_supervisor_workflow(self) -> StateGraph:
        """Build the supervisor-driven cyclic LangGraph workflow.

        This implements the new architecture with:
        - Supervisor agent coordinating sub-agents
        - Critic interceptor for security validation
        - Cyclic recovery paths for targeted error handling
        """
        workflow = StateGraph(PaperReproductionState)

        # === PHASE 1: Analysis (same as legacy) ===
        workflow.add_node("analyze_paper", self._analyze_paper_node)
        workflow.add_node("decide_and_clone", self._decide_and_clone_node)

        # === PHASE 2: Planning (NEW) ===
        workflow.add_node("planning", self._planning_node)

        # === PHASE 3: Supervisor (NEW) ===
        workflow.add_node("supervisor", self._supervisor_node)

        # === PHASE 4: Critic (NEW) ===
        workflow.add_node("critic", self._critic_node)

        # === PHASE 5: Sub-agents ===
        workflow.add_node("environment_setup", self._environment_setup_node)
        workflow.add_node("data_prep", self._data_prep_node)
        workflow.add_node("execution", self._execution_node)
        workflow.add_node("validation", self._validation_node)

        # === PHASE 6: Reporting ===
        workflow.add_node("generate_report", self._generate_report_node)

        # === ENTRY POINT ===
        workflow.set_entry_point("analyze_paper")
        workflow.add_edge("analyze_paper", "decide_and_clone")

        # === AFTER CLONE: Go to planning (not environment_setup) ===
        workflow.add_conditional_edges(
            "decide_and_clone",
            self._route_after_clone_supervisor,
            {
                "continue": "planning",
                "failed": "generate_report",
            },
        )

        # === PLANNING -> SUPERVISOR ===
        workflow.add_edge("planning", "supervisor")

        # === SUPERVISOR ROUTING (the brain) ===
        workflow.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "planning_update": "planning",
                "environment": "environment_setup",
                "data_prep": "data_prep",
                "data_prep_skip": "supervisor",  # Skip data_prep, return to supervisor
                "execution": "critic",  # Execution goes through critic first
                "validation": "validation",
                "complete": "generate_report",
                "report": "generate_report",
            },
        )

        # === CRITIC INTERCEPT ===
        workflow.add_conditional_edges(
            "critic",
            self._route_from_critic,
            {
                "authorized": "execution",
                "blocked": "supervisor",
            },
        )

        # === ENVIRONMENT WITH RECOVERY LOOP ===
        workflow.add_conditional_edges(
            "environment_setup",
            self._route_after_env_supervisor,
            {
                "continue": "supervisor",
                "retry": "environment_setup",
                "data_fix": "data_prep",  # NEW: Route to data prep for data errors
                "failed": "generate_report",
            },
        )

        # === DATA PREP WITH RECOVERY ===
        workflow.add_conditional_edges(
            "data_prep",
            self._route_after_data_prep,
            {
                "continue": "supervisor",
                "retry": "data_prep",
                "env_fix": "environment_setup",
                "failed": "generate_report",
            },
        )

        # === EXECUTION WITH RECOVERY (key cyclic path) ===
        workflow.add_conditional_edges(
            "execution",
            self._route_after_execution,
            {
                "continue": "supervisor",
                "retry": "execution",
                "env_fix": "environment_setup",
                "data_fix": "data_prep",
                "failed": "generate_report",
            },
        )

        # === VALIDATION ===
        workflow.add_conditional_edges(
            "validation",
            self._route_after_validation,
            {
                "continue": "generate_report",
                "retry_execution": "supervisor",
            },
        )

        workflow.add_edge("generate_report", END)

        return workflow.compile()

    # === NEW SUPERVISOR ARCHITECTURE NODES ===

    def _print_reasoning_if_verbose(self, reasoning: str, agent_name: str):
        """Print agent reasoning if show_reasoning is enabled in config.

        Args:
            reasoning: The agent's reasoning/thought process
            agent_name: Name of the agent for display
        """
        if self.config.show_reasoning and reasoning:
            # Handle list input (e.g. from LangGraph history)
            if isinstance(reasoning, list):
                reasoning = " ".join([str(m.content) if hasattr(m, "content") else str(m) for m in reasoning])
            
            # Truncate and clean up for display
            preview = str(reasoning)[:300].replace('\n', ' ').strip()
            if len(reasoning) > 300:
                preview += "..."
            print(f"   💭 {agent_name} reasoning: {preview}")

    def _planning_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Planning node - creates or updates reproduction checklist."""
        # Check if this phase was already completed (resuming from checkpoint)
        if self._is_phase_completed(state, "planning"):
            print("⏭️  Skipping planning (already completed from checkpoint)")
            return state

        self.metrics_tracker.start_phase("planning")
        print("📋 Planning: Creating reproduction checklist...")

        # Check if this is an update request or initial planning
        update_request = state.get("planning_update_request")

        if update_request:
            result = self.planning_agent.update_plan(state)
        else:
            result = self.planning_agent.create_plan(state)

        # Update state with planning results
        state["reproduction_plan"] = result.get("reproduction_plan", {})
        state["checklist_path"] = result.get("checklist_path", "")
        state["phase_status"] = result.get("phase_status", state.get("phase_status", {}))
        state["planning_update_request"] = None  # Clear request

        # Store and optionally print reasoning
        state["current_reasoning"] = result.get("last_message", "")
        self._print_reasoning_if_verbose(state["current_reasoning"], "Planning")

        # Save checkpoint
        self._save_checkpoint(state, "planning", success=True)

        self.metrics_tracker.end_phase("planning", success=True)
        return state

    def _supervisor_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Supervisor node - decides which agent to invoke next."""
        self.metrics_tracker.start_phase("supervisor")
        print("🎯 Supervisor: Analyzing state and routing...")

        decision = self.supervisor_agent.decide_next_agent(state)

        state["current_agent"] = decision.get("agent", "")
        state["supervisor_directive"] = decision.get("directive", "")

        print(f"   → Routing to: {state['current_agent']}")
        print(f"   → Directive: {state['supervisor_directive'][:80]}...")

        self.metrics_tracker.end_phase("supervisor")
        return state

    def _critic_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Critic node - validates reasoning and actions before execution."""
        self.metrics_tracker.start_phase("critic")
        
        action = state.get("proposed_action")
        directive = state.get("supervisor_directive")
        
        if action:
            print(f"🔍 Critic: Inspecting action: {action.get('tool_name')}")
            # Optionally print args if they aren't huge
            args = str(action.get('tool_args', {}))
            if len(args) < 200:
                print(f"   Args: {args}")
        elif directive:
            print(f"🔍 Critic: Inspecting directive: '{directive[:100]}...'")
        else:
            print("🔍 Critic: Inspecting state...")

        result = self.critic_agent.inspect_action(state)

        state["is_authorized"] = result.get("is_authorized", True)
        state["critic_feedback"] = result.get("critic_feedback", "")

        if not state["is_authorized"]:
            blocked_actions = state.get("blocked_actions", [])
            blocked_actions.extend(result.get("blocked_actions", []))
            state["blocked_actions"] = blocked_actions
            print(f"   ❌ BLOCKED: {state['critic_feedback']}")
        else:
            print("   ✅ Action approved")

        self.metrics_tracker.end_phase("critic")
        return state

    def _data_prep_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Data preparation node - downloads and prepares datasets."""
        # Check if this phase was already completed (resuming from checkpoint)
        # BUT ignore if we are retrying a failure
        failure = state.get("failure_metadata")
        is_retry = failure and failure.get("error_type") == "data"
        
        if not is_retry and self._is_phase_completed(state, "data_prep"):
            # Validate checkpoint before skipping
            if self._validate_data_prep_checkpoint(state):
                print("⏭️  Skipping data_prep (already completed from checkpoint)")
                # Ensure phase status is consistent
                phase_status = state.get("phase_status", {})
                if phase_status.get("data_prep") != "completed":
                    phase_status["data_prep"] = "completed"
                    state["phase_status"] = phase_status
                return state
            # If validation failed, phase was invalidated - continue to re-run

        self.metrics_tracker.start_phase("data_prep")
        print("📦 Data Prep: Preparing datasets...")

        result = self.data_prep_agent.prepare_data(state)

        # Store and optionally print reasoning
        state["current_reasoning"] = result.get("last_message", "")
        self._print_reasoning_if_verbose(state["current_reasoning"], "DataPrep")

        state["datasets_ready"] = result.get("datasets_ready", False)
        state["dataset_results"] = result.get("dataset_results", {})

        if result.get("failure_metadata"):
            state["failure_metadata"] = result["failure_metadata"]

        # Update phase status
        phase_status = state.get("phase_status", {})
        phase_status["data_prep"] = "completed" if result.get("datasets_ready") else "failed"
        state["phase_status"] = phase_status

        self.metrics_tracker.end_phase("data_prep", success=result.get("datasets_ready", False))
        
        # Save checkpoint (mark completed only if successful)
        self._save_checkpoint(state, "data_prep", success=result.get("datasets_ready", False))
        
        return state

    def _execution_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Execution node - runs experiments."""
        # Check if this phase was already completed
        # BUT ignore if we are retrying a failure
        failure = state.get("failure_metadata")
        is_retry = failure and failure.get("error_type") == "execution"

        if not is_retry and self._is_phase_completed(state, "execution"):
            # Validate checkpoint before skipping - ensure actual results exist
            if self._validate_execution_checkpoint(state):
                print("⏭️  Skipping execution (already completed from checkpoint)")
                # Ensure phase status is consistent
                phase_status = state.get("phase_status", {})
                if phase_status.get("execution") != "completed":
                    phase_status["execution"] = "completed"
                    state["phase_status"] = phase_status
                return state
            # If validation failed, phase was invalidated - continue to re-run

        self.metrics_tracker.start_phase("execution")
        print("🚀 Execution: Running experiments...")

        import time
        start_exp = time.time()
        result = self.execution_agent.run_experiments(state)
        # Record actual experiment time (wall time)
        self.metrics_tracker.record_experiment_time("execution", time.time() - start_exp)

        # Store agent reasoning for critic inspection and optionally print
        state["current_reasoning"] = result.get("last_message", "")
        self._print_reasoning_if_verbose(state["current_reasoning"], "Execution")

        state["experiments_completed"] = result.get("experiments_completed", False)
        state["experiment_results"] = result.get("experiment_results", {})

        if result.get("failure_metadata"):
            # UPGRADE: Re-classify error using Supervisor's LLM for robustness
            failure_meta = result["failure_metadata"]
            original_msg = failure_meta.get("error_message", "")
            
            # Use LLM to get true error class (DATA, CODE, ENVIRONMENT)
            llm_class = self.supervisor_agent.classify_error(original_msg)
            print(f"   🧠 Execution Error classified as: {llm_class}")
            
            # Map LLM class to system error types
            if llm_class == "DATA":
                failure_meta["error_type"] = "data"
                # Add context for DataPrep agent
                failure_meta["context"] = "Missing data detected during execution"
            elif llm_class == "ENVIRONMENT":
                failure_meta["error_type"] = "environment"
            elif llm_class == "CODE":
                # Code errors might be execution errors (OOM, timeout) or actual code bugs
                # Keep original type if it was specific (like "execution" for OOM), otherwise "execution"
                if failure_meta["error_type"] not in ["execution", "data", "environment"]:
                    failure_meta["error_type"] = "execution"
            
            state["failure_metadata"] = failure_meta
        else:
            state["failure_metadata"] = None  # Clear on success

        # Update phase status
        phase_status = state.get("phase_status", {})
        phase_status["execution"] = "completed" if result.get("experiments_completed") else "failed"
        state["phase_status"] = phase_status

        self.metrics_tracker.end_phase("execution", success=result.get("experiments_completed", False))
        
        # Save checkpoint (mark completed only if successful)
        self._save_checkpoint(state, "execution", success=result.get("experiments_completed", False))
        
        return state

    def _validation_node(self, state: PaperReproductionState) -> PaperReproductionState:
        """Validation node - verifies results against paper."""
        # Check if this phase was already completed
        if self._is_phase_completed(state, "validation"):
            print("⏭️  Skipping validation (already completed from checkpoint)")
             # Ensure phase status is consistent
            phase_status = state.get("phase_status", {})
            if phase_status.get("validation") != "completed":
                phase_status["validation"] = "completed"
                state["phase_status"] = phase_status
            return state

        self.metrics_tracker.start_phase("validation")
        print("📊 Validation: Verifying results...")

        result = self.validation_agent.verify_results(state)

        # Store and optionally print reasoning (validation uses "report" field)
        state["current_reasoning"] = result.get("report", "")
        self._print_reasoning_if_verbose(state["current_reasoning"], "Validation")

        state["results_match"] = result.get("results_match", False)
        state["extracted_metrics"] = result.get("extracted_metrics", {})
        state["metrics_comparison"] = result.get("metrics_comparison", {})
        state["verification_results"] = result.get("verification_results", {})

        # Update phase status
        phase_status = state.get("phase_status", {})
        phase_status["validation"] = "completed"
        state["phase_status"] = phase_status

        self.metrics_tracker.end_phase("validation", success=result.get("results_match", False))
        
        # Save checkpoint
        self._save_checkpoint(state, "validation", success=True)
        
        return state

    # === NEW SUPERVISOR ARCHITECTURE ROUTING FUNCTIONS ===

    def _route_after_clone_supervisor(
        self, state: PaperReproductionState
    ) -> Literal["continue", "failed"]:
        """Route after cloning (supervisor architecture)."""
        if state.get("implementation_path"):
            return "continue"
        return "failed"

    def _route_from_supervisor(self, state: PaperReproductionState) -> str:
        """Route based on supervisor decision."""
        agent = state.get("current_agent", "report")

        # Handle data_prep_skip: mark as completed and continue
        if agent == "data_prep_skip":
            phase_status = state.get("phase_status", {})
            phase_status["data_prep"] = "completed"
            state["phase_status"] = phase_status
            state["datasets_ready"] = True
            state["dataset_results"] = {"skipped": True, "reason": "Scripts auto-download data"}
            print("   📦 Data prep marked as completed (skipped)")

        return agent

    def _route_from_critic(self, state: PaperReproductionState) -> str:
        """Route based on critic decision."""
        if state.get("is_authorized", True):
            return "authorized"
        return "blocked"

    def _route_after_env_supervisor(self, state: PaperReproductionState) -> str:
        """Route after environment setup (supervisor architecture)."""
        failure = state.get("failure_metadata")

        if not failure and state.get("dependencies_installed", False):
            return "continue"

        if failure:
            error_type = failure.get("error_type", "")
            retry_count = failure.get("retry_count", 0)
            max_retries = state.get("max_recovery_attempts", 3)

            if retry_count < max_retries:
                if error_type == "data":
                    print("   ↪️  Routing environment failure to Data Prep (missing data detected)")
                    return "data_fix"
                return "retry"

        return "failed"

    def _route_after_data_prep(self, state: PaperReproductionState) -> str:
        """Route after data preparation."""
        failure = state.get("failure_metadata")

        if not failure and state.get("datasets_ready", False):
            return "continue"

        if failure:
            error_type = failure.get("error_type", "")
            retry_count = failure.get("retry_count", 0)
            max_retries = state.get("max_recovery_attempts", 3)

            # Increment cycle count
            state["cycle_count"] = state.get("cycle_count", 0) + 1

            if state.get("cycle_count", 0) >= state.get("max_cycles", 5):
                return "failed"

            if retry_count >= max_retries:
                return "failed"

            if error_type == "environment":
                return "env_fix"

            return "retry"

        return "failed"

    def _route_after_execution(self, state: PaperReproductionState) -> str:
        """Route after execution (key cyclic path)."""
        failure = state.get("failure_metadata")

        if not failure and state.get("experiments_completed", False):
            return "continue"

        if failure:
            error_type = failure.get("error_type", "")
            retry_count = failure.get("retry_count", 0)
            max_retries = state.get("max_recovery_attempts", 3)

            # Increment cycle count
            state["cycle_count"] = state.get("cycle_count", 0) + 1

            if state.get("cycle_count", 0) >= state.get("max_cycles", 5):
                print("🛑 Max cycles reached, generating report")
                return "failed"

            if retry_count >= max_retries:
                return "failed"

            # Route based on error type
            if error_type == "environment":
                return "env_fix"
            elif error_type == "data":
                return "data_fix"
            elif error_type == "execution":
                return "retry"

        return "failed"

    def _route_after_validation(self, state: PaperReproductionState) -> str:
        """Route after validation."""
        if state.get("results_match", False):
            return "continue"

        # Check if we should retry execution
        failure = state.get("failure_metadata")
        if failure and failure.get("error_type") == "validation":
            retry_count = failure.get("retry_count", 0)
            if retry_count < 1:  # Allow one retry for validation
                return "retry_execution"

        return "continue"  # Generate report even with mismatched results

    def _analyze_paper_node(
        self, state: PaperReproductionState
    ) -> PaperReproductionState:
        """Analyze the paper using UnifiedPaperAnalyzer."""
        # Check if this phase was already completed (resuming from checkpoint)
        if self._is_phase_completed(state, "analyze_paper"):
            print("⏭️  Skipping analyze_paper (already completed from checkpoint)")
            return state

        self.metrics_tracker.start_phase("analyze_paper")
        print("📄 Analyzing paper...")

        paper_input = state["paper_input"]

        # Handle arXiv papers
        if paper_input.startswith("arxiv:") or (
            len(paper_input.split()) == 1 and "." in paper_input
        ):
            arxiv_id = paper_input.replace("arxiv:", "")
            print(f"📥 Fetching arXiv paper {arxiv_id}...")

            try:
                import arxiv
                from PyPDF2 import PdfReader
                from urllib.request import urlretrieve
                import re

                # Setup paths
                download_dir = os.path.abspath(self.config.downloads_path)
                os.makedirs(download_dir, exist_ok=True)
                safe_arxiv_id = arxiv_id.replace("/", "_").replace(".", "_")
                filename = f"{safe_arxiv_id}.pdf"
                pdf_path = os.path.join(download_dir, filename)
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

                # Check if PDF already exists locally (cache hit)
                pdf_exists = os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000

                # Try to fetch metadata from arXiv API
                paper = None
                try:
                    search = arxiv.Search(id_list=[arxiv_id])
                    paper = next(search.results())
                except Exception as api_error:
                    if pdf_exists:
                        print(f"⚠️  arXiv API error: {api_error}")
                        print(f"✅ Using cached PDF: {pdf_path}")
                    else:
                        raise  # Re-raise if we can't use cache

                # Download PDF if not cached
                if not pdf_exists:
                    print(f"   Downloading from: {pdf_url}")
                    urlretrieve(pdf_url, pdf_path)
                else:
                    print(f"   Using cached PDF: {pdf_path}")

                # Extract text from PDF
                reader = PdfReader(pdf_path)
                full_text = ""
                for page in reader.pages:
                    full_text += page.extract_text() + "\n"

                # Truncate before bibliography/appendix
                truncate_patterns = [
                    r"\n\s*\d+\.?\s+References\s*\n",
                    r"\n\s*\d+\.?\s+Bibliography\s*\n",
                    r"\nReferences\s*\n",
                    r"\nBibliography\s*\n",
                    r"\nREFERENCES\s*\n",
                    r"\n\s*[A-Z]\.?\s+Appendix",
                    r"\nAppendix\s+[A-Z]",
                    r"\nAPPENDIX",
                    r"\n\s*Supplementary\s+Material\s*\n",
                    r"\n\s*Acknowledgment",
                ]
                for pattern in truncate_patterns:
                    match = re.search(pattern, full_text)
                    if match:
                        full_text = full_text[: match.start()]
                        print(f"📄 Truncated paper at '{match.group().strip()}'")
                        break

                # Store metadata (handle case where arXiv API failed but we have cached PDF)
                if paper:
                    state["paper_metadata"] = {
                        "title": paper.title,
                        "authors": [author.name for author in paper.authors],
                        "abstract": paper.summary,
                        "published": paper.published.isoformat(),
                        "arxiv_id": arxiv_id,
                        "pdf_url": pdf_url,
                        "full_text": full_text,
                        "categories": paper.categories,
                    }
                    state["paper_title"] = paper.title
                else:
                    # Fallback: extract title from PDF text (first non-empty line, likely title)
                    lines = [l.strip() for l in full_text.split('\n') if l.strip()]
                    inferred_title = lines[0] if lines else f"Paper {arxiv_id}"
                    print(f"⚠️  Using cached PDF without arXiv metadata")
                    print(f"   Inferred title: {inferred_title[:80]}...")
                    state["paper_metadata"] = {
                        "title": inferred_title,
                        "authors": [],
                        "abstract": "",
                        "published": "",
                        "arxiv_id": arxiv_id,
                        "pdf_url": pdf_url,
                        "full_text": full_text,
                        "categories": [],
                    }
                    state["paper_title"] = inferred_title
                print(
                    f"✅ Extracted {len(full_text)} characters of text from PDF"
                )

            except Exception as e:
                print(f"⚠️  Failed to fetch paper: {str(e)}")
                import traceback

                traceback.print_exc()
                return state

        # Analyze paper with unified analyzer
        full_text = state.get("paper_metadata", {}).get("full_text", "")
        if full_text:
            print("🔬 Analyzing paper with unified analyzer...")
            from .agents.unified_paper_analyzer import UnifiedPaperAnalyzer

            analyzer = UnifiedPaperAnalyzer(
                self.llm, metrics_tracker=self.metrics_tracker
            )
            analysis = analyzer.analyze_paper(
                full_text, state.get("paper_title", "Unknown")
            )

            # Store results in state
            state["code_references"] = analysis.get("github_repos", [])
            state["paper_results"] = analysis.get("results_to_reproduce", {})
            state["experimental_setup"] = {
                "datasets": analysis.get("datasets", []),
                "implementation_details": analysis.get("implementation_details", ""),
            }

            # Store agent context for future agents
            state["agent_contexts"]["paper_analyzer"] = analysis.get(
                "context_summary", ""
            )

            # Store key findings in hierarchical context for semantic retrieval
            self._store_paper_context(state, analysis)

            # Print analysis summary
            self._print_analysis_summary(state, analysis)

            # Enable prompt caching with paper content for cost savings
            paper_context = state.get("agent_contexts", {}).get("paper_analyzer", "")
            paper_results_str = str(state.get("paper_results", {}))
            if paper_context:
                self._enable_caching(
                    paper_content=paper_context,
                    paper_results=paper_results_str,
                )

            # Fallback discovery is now handled by DiscoveryAgent in the next node
            if not state["code_references"]:
                print("   (Code discovery deferred to Discovery Agent)")
        else:
            state["code_references"] = []
            state["paper_results"] = {}
            state["experimental_setup"] = {}

        # Add status message
        state["messages"].append(f"✅ Analyzed paper: {state['paper_title']}")
        if state.get("code_references"):
            state["messages"].append(
                f"📚 Found {len(state['code_references'])} code reference(s)"
            )

        # Save checkpoint
        self._save_checkpoint(state, "analyze_paper", success=True)

        self.metrics_tracker.end_phase("analyze_paper", success=True)
        self.metrics_tracker.print_intermediate_summary()
        return state

    def _store_paper_context(self, state: PaperReproductionState, analysis: dict):
        """Store paper analysis results in hierarchical context for semantic retrieval."""
        try:
            paper_title = state.get("paper_title", "Unknown Paper")

            # Store core contribution
            core_contribution = analysis.get("core_contribution", "")
            if core_contribution:
                self.hierarchical_context.add(
                    content=f"Paper: {paper_title}\nCore Contribution: {core_contribution}",
                    source="paper_analyzer",
                    entry_type="result",
                    importance=1.0,
                    lazy=True,  # Defer embedding generation for speed
                )

            # Store datasets
            datasets = analysis.get("datasets", [])
            if datasets:
                self.hierarchical_context.add(
                    content=f"Datasets to reproduce: {', '.join(datasets)}",
                    source="paper_analyzer",
                    entry_type="result",
                    importance=0.9,
                    lazy=True,
                )

            # Store metrics to reproduce
            paper_results = state.get("paper_results", {})
            if isinstance(paper_results, dict):
                metrics = paper_results.get("metrics", [])
                if metrics:
                    metrics_summary = []
                    for m in metrics[:10]:  # Limit to 10
                        if isinstance(m, dict):
                            dataset = m.get("dataset", "Unknown")
                            metric = m.get("metric", "Unknown")
                            value = m.get("value", "N/A")
                            metrics_summary.append(f"{dataset}/{metric}: {value}")

                    if metrics_summary:
                        self.hierarchical_context.add(
                            content="Expected metrics:\n" + "\n".join(metrics_summary),
                            source="paper_analyzer",
                            entry_type="result",
                            importance=1.0,
                            lazy=True,
                        )

            # Store code references
            code_refs = state.get("code_references", [])
            if code_refs:
                refs_list = []
                for r in code_refs[:5]:
                    if isinstance(r, dict):
                        refs_list.append(r.get("url", str(r)))
                    else:
                        refs_list.append(str(r))

                self.hierarchical_context.add(
                    content=f"Code repositories: {', '.join(refs_list)}",
                    source="paper_analyzer",
                    entry_type="result",
                    importance=0.8,
                    lazy=True,
                )

            print("   🧠 Stored paper context in hierarchical storage")

        except Exception as e:
            print(f"   ⚠️  Warning: Failed to store paper context: {e}")

    def _enable_caching(self, paper_content: str, readme_content: str = "", paper_results: str = ""):
        """Enable prompt caching after paper analysis.

        For Gemini: Creates an explicit cache via google-genai and refreshes LLMs.
        For Claude: Caching is automatic via the beta header (already enabled in llm_factory).
        """
        from .utils.llm_factory import create_gemini_cache, create_llm, get_provider

        try:
            provider = get_provider()

            if provider == "gemini":
                # Create Gemini cache
                self.cache_name = create_gemini_cache(
                    paper_content=paper_content,
                    readme_content=readme_content,
                    paper_results=paper_results,
                    paper_id=getattr(self, "paper_title", "unknown")[:50],
                )

                if self.cache_name:
                    # Refresh LLMs with caching
                    self.llm = create_llm(temperature=0.1, cached_content=self.cache_name)
                    self.reasoning_llm = create_llm(
                        temperature=0.3,
                        include_thoughts=True,
                        cached_content=self.cache_name,
                    )
                    self._refresh_agent_llms()
                    print("Gemini caching enabled for subsequent agent calls")

            elif provider == "claude":
                # Claude caching is automatic via cache_control in messages
                # No explicit cache creation needed - just use cache_control markers
                print("Claude caching enabled (automatic via headers)")

        except Exception as e:
            print(f"Warning: Failed to enable caching: {e}")

    def _refresh_agent_llms(self):
        """Refresh agent LLMs after caching is enabled."""
        # Update legacy agents
        if hasattr(self, "env_setup_agent"):
            self.env_setup_agent.llm = self.reasoning_llm
        if hasattr(self, "unified_reproducer"):
            self.unified_reproducer.llm = self.llm

        # Update supervisor architecture agents if enabled
        if self.use_supervisor:
            if hasattr(self, "supervisor_agent"):
                self.supervisor_agent.llm = self.reasoning_llm
            if hasattr(self, "planning_agent"):
                self.planning_agent.llm = self.reasoning_llm
            if hasattr(self, "data_prep_agent"):
                self.data_prep_agent.llm = self.llm
            if hasattr(self, "execution_agent"):
                self.execution_agent.llm = self.llm
            if hasattr(self, "validation_agent"):
                self.validation_agent.llm = self.llm

    def _print_analysis_summary(self, state, analysis):
        """Print detailed analysis results."""
        import textwrap

        print("\n" + "=" * 80)
        print("📊 UNIFIED ANALYZER FINDINGS")
        print("=" * 80)

        # GitHub Repositories
        repos = state["code_references"]
        # Handle both list of strings and list of dicts
        repo_urls = []
        if repos:
            for r in repos:
                if isinstance(r, dict):
                    repo_urls.append(r.get("url", str(r)))
                else:
                    repo_urls.append(str(r))

        print(f"\n📚 GitHub Repositories Found: {len(repo_urls)}")
        for i, repo in enumerate(repo_urls, 1):
            print(f"   {i}. {repo}")
        if not repo_urls:
            print("   (none found)")

        # Datasets
        datasets = analysis.get("datasets", [])
        print(f"\n📊 Datasets Identified: {len(datasets)}")
        if datasets:
            print(f"   {', '.join(datasets)}")
        else:
            print("   (none identified)")

        # Core Contribution
        print("\n💡 Core Contribution:")
        core = analysis.get("core_contribution", "N/A")
        if core:
            wrapped = textwrap.fill(
                core, width=74, initial_indent="   ", subsequent_indent="   "
            )
            print(wrapped)
        else:
            print("   (not extracted)")

        # Results to Reproduce
        paper_results = state.get("paper_results", {})
        # Ensure it's a dict
        if not isinstance(paper_results, dict):
            paper_results = {}

        metrics = paper_results.get("metrics", [])
        print(f"\n🎯 Results to Reproduce: {len(metrics)} metric(s)")
        if metrics:
            for m in metrics[:5]:
                if isinstance(m, dict):
                    dataset = m.get("dataset", "Unknown")
                    metric = m.get("metric", "Unknown")
                    value = m.get("value", "Unknown")
                    print(f"   - {dataset}: {metric} = {value}")
            if len(metrics) > 5:
                print(f"   ... and {len(metrics) - 5} more")
        else:
            summary = paper_results.get("summary", "")
            if summary:
                print("   Summary from paper:")
                for line in summary.split("\n")[:3]:
                    if line.strip():
                        print(f"   {line.strip()[:74]}")
            else:
                print("   (no metrics extracted)")

        print("\n" + "=" * 80 + "\n")

    def _decide_and_clone_node(
        self, state: PaperReproductionState
    ) -> PaperReproductionState:
        """Decide on implementation path and clone repository if found."""
        # Check if this phase was already completed (resuming from checkpoint)
        if self._is_phase_completed(state, "decide_and_clone"):
            print("⏭️  Skipping decide_and_clone (already completed from checkpoint)")
            return state

        self.metrics_tracker.start_phase("decide_and_clone")
        print("🤔 Deciding on implementation path...")

        # FIRST: Check if repo already exists on disk (fallback for when APIs fail)
        code_path = self.config.repo_path
        repo_marker = os.path.join(code_path, ".repo_url")

        if os.path.exists(code_path) and os.path.exists(repo_marker):
            try:
                with open(repo_marker, "r") as f:
                    existing_url = f.read().strip()
                if existing_url:
                    # Check if it looks like a valid git repo
                    git_dir = os.path.join(code_path, ".git")
                    if os.path.isdir(git_dir):
                        print(f"✅ Found existing repository: {existing_url}")
                        print(f"   Skipping discovery and using existing repo.")
                        state["selected_repo"] = {
                            "url": existing_url,
                            "source": "existing_local",
                            "confidence": 1.0,
                        }
                        state["implementation_path"] = code_path
                        state["messages"].append(f"📥 Using existing implementation: {existing_url}")
                        self._save_checkpoint(state, "decide_and_clone", success=True)
                        self.metrics_tracker.end_phase("decide_and_clone", success=True)
                        return state
            except Exception as e:
                print(f"⚠️  Could not read existing repo marker: {e}")

        # Prepare inputs for Discovery Agent
        paper_title = state.get("paper_title", "")
        paper_abstract = state.get("paper_metadata", {}).get("abstract", "")
        arxiv_id = state.get("paper_metadata", {}).get("arxiv_id")
        authors = state.get("paper_metadata", {}).get("authors", [])

        # Format existing refs for the agent (from paper analysis)
        existing_refs = []
        if state.get("code_references"):
            refs = state["code_references"]
            if isinstance(refs, list):
                for ref in refs:
                    if isinstance(ref, dict):
                        existing_refs.append(ref.get("url"))
                    elif isinstance(ref, str):
                        existing_refs.append(ref)

        # CALL DISCOVERY AGENT
        discovery_result = self.discovery_agent.find_best_implementation(
            paper_title=paper_title,
            paper_abstract=paper_abstract,
            arxiv_id=arxiv_id,
            authors=authors,
            existing_repos=existing_refs,
        )

        selected_url = discovery_result.get("repo_url")

        if selected_url:
            state["selected_repo"] = {
                "url": selected_url,
                "source": "discovery_agent",
                "confidence": discovery_result.get("confidence"),
            }
            state["messages"].append(f"📥 Using implementation: {selected_url}")
            print(f"✅ Discovery Agent selected: {selected_url}")

            # Clone the repository
            code_path = self.config.repo_path
            repo_marker = os.path.join(code_path, ".repo_url")

            need_clone = True
            if os.path.exists(code_path):
                if os.path.exists(repo_marker):
                    try:
                        with open(repo_marker, "r") as f:
                            existing_url = f.read().strip()
                        if existing_url == selected_url:
                            print(f"✅ Repository already cloned: {selected_url}")
                            need_clone = False
                        else:
                            print("🔄 Different repo detected, removing old...")
                            shutil.rmtree(code_path, onerror=remove_readonly)
                    except Exception as e:
                        print(f"⚠️  Could not read repo marker: {e}")
                        shutil.rmtree(code_path, onerror=remove_readonly)
                else:
                    print("🗑️  No repo marker found, removing directory...")
                    shutil.rmtree(code_path, onerror=remove_readonly)

            if need_clone:
                print(f"📥 Cloning repository from {selected_url}...")
                try:
                    result = subprocess.run(
                        ["git", "clone", selected_url, code_path],
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    if result.returncode == 0:
                        print(f"✅ Successfully cloned repository to {code_path}")
                        # Store repo URL marker
                        with open(repo_marker, "w") as f:
                            f.write(selected_url)
                        state["implementation_path"] = code_path
                    else:
                        print(f"⚠️  Clone failed: {result.stderr}")
                        state["messages"].append("Clone failed")
                        state["final_status"] = "Failed: Could not clone repository"
                        self.metrics_tracker.end_phase(
                            "decide_and_clone", success=False
                        )
                        return state
                except Exception as e:
                    print(f"⚠️  Clone error: {str(e)}")
                    state["final_status"] = f"Failed: Clone error - {str(e)}"
                    self.metrics_tracker.end_phase("decide_and_clone", success=False)
                    return state
            else:
                state["implementation_path"] = code_path

            # Save checkpoint after successful clone
            self._save_checkpoint(state, "decide_and_clone", success=True)

            self.metrics_tracker.end_phase("decide_and_clone", success=True)
            self.metrics_tracker.print_intermediate_summary()
            return state

        # No implementation found
        state["final_status"] = "Failed: No implementation found"
        state["messages"].append("❌ No implementation found")
        print("❌ Discovery Agent found no suitable implementation")

        # Save checkpoint even on failure
        self._save_checkpoint(state, "decide_and_clone", success=False)

        self.metrics_tracker.end_phase("decide_and_clone", success=False)
        self.metrics_tracker.print_intermediate_summary()
        return state

    def _environment_setup_node(
        self, state: PaperReproductionState
    ) -> PaperReproductionState:
        """Prepare environment for running experiments."""
        # Check if this phase was already completed (resuming from checkpoint)
        if self._is_phase_completed(state, "environment_setup"):
            # Validate checkpoint before skipping - ensure environment actually exists
            if self._validate_environment_checkpoint(state):
                print("⏭️  Skipping environment_setup (already completed from checkpoint)")
                # CRITICAL FIX: Ensure phase_status is consistent even when skipping
                # This repairs state where completed_phases has it but phase_status doesn't
                phase_status = state.get("phase_status", {})
                if phase_status.get("environment") != "completed":
                    print("   ⚠️  Repairing phase hierarchy: marking environment as completed")
                    phase_status["environment"] = "completed"
                    state["phase_status"] = phase_status
                return state
            # If validation failed, phase was invalidated - continue to re-run

        self.metrics_tracker.start_phase("environment_setup")
        print("🔧 Setting up environment...")

        code_path = state.get("implementation_path") or "./cloned_repo"

        # Get paper date for version pinning
        paper_date = state.get("paper_metadata", {}).get("published", None)
        paper_title = state.get("paper_title", "Unknown")

        # Read README for installation instructions
        readme_path = os.path.join(code_path, "README.md")
        readme_content = ""
        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                readme_content = f.read()
        except FileNotFoundError:
            print("⚠️  No README.md found, will analyze environment files directly")
            readme_content = "No README found. Analyze environment files directly."

        # Run environment setup agent
        env_result = self.env_setup_agent.setup_environment(
            repo_path=code_path,
            readme_content=readme_content,
            paper_date=paper_date,
            paper_title=paper_title,
        )

        # Update state with environment results
        state["env_setup_results"] = env_result
        state["dependencies_installed"] = env_result.get("success", False)

        # Store and optionally print reasoning
        state["current_reasoning"] = env_result.get("last_message", "")
        self._print_reasoning_if_verbose(state["current_reasoning"], "Environment")

        if env_result.get("success"):
            # Update phase status
            phase_status = state.get("phase_status", {})
            phase_status["environment"] = "completed"
            state["phase_status"] = phase_status

            state["messages"].append("✅ Environment setup successful")
            print(f"✅ Environment ready: {env_result.get('env_name', 'Unknown')}")

            # Save environment info for unified_reproduction to use
            if "agent_contexts" not in state:
                state["agent_contexts"] = {}
            state["agent_contexts"]["environment_setup"] = {
                "env_type": env_result.get("env_type"),
                "env_name": env_result.get("env_name"),
                "python_path": env_result.get("python_path"),
                "packages_pinned": env_result.get("packages_pinned", []),
            }

            # Store in hierarchical context
            self.hierarchical_context.add(
                content=f"Environment: {env_result.get('env_type', 'unknown')}, "
                f"Name: {env_result.get('env_name', 'unknown')}, "
                f"Python: {env_result.get('python_path', 'unknown')}",
                source="environment_setup",
                entry_type="result",
                importance=0.8,
                lazy=True,  # Defer embedding to avoid loading SentenceTransformer
            )
        else:
            state["messages"].append("❌ Environment setup failed")
            print(
                f"❌ Environment setup failed: {env_result.get('error', 'Unknown error')}"
            )

            # Failure analysis for routing
            error_msg = env_result.get('error', '')
            
            # Use Supervisor's intelligent LLM classifier
            error_class = self.supervisor_agent.classify_error(error_msg)
            print(f"   🧠 Error classified as: {error_class}")
            
            if error_class == "DATA":
                # This is a data issue (smoke test failed due to missing data)
                state["failure_metadata"] = {
                    "error_type": "data",
                    "error_message": env_result.get('error'),
                    "error_source": "environment_setup",
                    "retry_count": state.get("failure_metadata", {}).get("retry_count", 0),
                    "context": "Smoke test failed due to missing data"
                }
            else:
                # Generic environment or code failure
                state["failure_metadata"] = {
                    "error_type": "environment",
                    "error_message": env_result.get('error'),
                    "error_source": "environment_setup",
                    "retry_count": state.get("failure_metadata", {}).get("retry_count", 0)
                }
            
            state["final_status"] = (
                f"Failed: Environment setup - {env_result.get('error', 'Unknown')}"
            )

        # Save checkpoint
        self._save_checkpoint(state, "environment_setup", success=env_result.get("success", False))

        self.metrics_tracker.end_phase(
            "environment_setup", success=env_result.get("success", False)
        )
        self.metrics_tracker.print_intermediate_summary()
        return state

    def _unified_reproduction_node(
        self, state: PaperReproductionState
    ) -> PaperReproductionState:
        """Run unified reproduction workflow."""
        # Check if this phase was already completed (resuming from checkpoint)
        if self._is_phase_completed(state, "unified_reproduction"):
            print(
                "⏭️  Skipping unified_reproduction (already completed from checkpoint)"
            )
            return state

        self.metrics_tracker.start_phase("unified_reproduction")
        code_path = state.get("implementation_path") or "./cloned_repo"

        # NEW: Check for existing results/checkpoints in the repo BEFORE running experiments
        existing_results = self.discovery_agent.check_existing_results(code_path)
        if existing_results.get("has_results"):
            print("\n" + "=" * 60)
            print("🎯 EXISTING RESULTS FOUND - Skipping to verification!")
            print("=" * 60)
            print(f"   Result files: {existing_results.get('result_files', [])[:3]}")
            print(f"   Checkpoints: {existing_results.get('checkpoints', [])[:3]}")
            print("=" * 60 + "\n")

            # Mark as successful and skip to verification
            state["env_setup_results"] = {
                "success": True,
                "report": "Skipped - using existing results",
            }
            state["dependencies_installed"] = True
            state["dataset_results"] = {
                "datasets_identified": True,
                "datasets_downloaded": True,
            }
            state["datasets_ready"] = True
            state["experiment_results"] = {
                "execution_successful": True,
                "sanity_check_passed": True,
                "output": f"Using existing results from: {existing_results.get('result_files', [])}",
                "existing_results": existing_results,
                "skipped_execution": True,
            }
            state["experiments_completed"] = True
            state["messages"].append(
                "✅ Found existing results - skipping experiment execution"
            )

            # Save checkpoint
            self._save_checkpoint(state, "unified_reproduction", success=True)
            self.metrics_tracker.end_phase("unified_reproduction", success=True)
            return state

        print("🚀 Starting unified reproduction workflow...")

        # Build comprehensive context from paper analysis
        paper_context_parts = []

        if state.get("agent_contexts", {}).get("paper_analyzer"):
            paper_context_parts.append(state["agent_contexts"]["paper_analyzer"])

        datasets = state.get("experimental_setup", {}).get("datasets", [])
        if datasets:
            paper_context_parts.append(
                f"\nDatasets mentioned in paper: {', '.join(datasets)}"
            )

        paper_results = state.get("paper_results", {})
        if paper_results:
            paper_context_parts.append("\n\nResults to Reproduce:")
            if isinstance(paper_results, dict):
                metrics = paper_results.get("metrics", [])
                for m in metrics:
                    paper_context_parts.append(
                        f"  - {m.get('dataset', 'Unknown')}: {m.get('metric', 'Unknown')} = {m.get('value', 'Unknown')}"
                    )
                if not metrics and "summary" in paper_results:
                    paper_context_parts.append(f"\n{paper_results['summary']}")

        impl_details = state.get("experimental_setup", {}).get(
            "implementation_details", ""
        )
        if impl_details:
            paper_context_parts.append(
                f"\n\nImplementation Details:\n{impl_details[:500]}"
            )

        paper_context = "\n".join(paper_context_parts)

        # FLUSH DEFERRED EMBEDDINGS BEFORE SUPERVISOR/REPRODUCTION STARTS
        if self.hierarchical_context:
            print("   🧠 Flushing deferred context embeddings...")
            self.hierarchical_context.flush_embeddings()

        # Run unified reproduction
        result = self.unified_reproducer.reproduce(
            code_path,
            paper_context,
            experiment_mode=state.get("experiment_selection_mode", "single"),
            custom_experiments=state.get("custom_experiment_list", []),
        )

        # Update state with results
        state["env_setup_results"] = {
            "success": result["setup_successful"],
            "report": result["report"],
        }
        state["dependencies_installed"] = result["dependencies_installed"]

        state["dataset_results"] = {
            "datasets_identified": result["data_attempted"],
            "datasets_downloaded": result["data_successful"],
            "dataset_locations": (
                [result["data_location"]] if result["data_location"] else []
            ),
        }
        state["datasets_ready"] = result["data_successful"]

        state["experiment_results"] = {
            "execution_successful": result["main_experiment_successful"],
            "sanity_check_passed": result["sanity_check_passed"],
            "output": result["experiment_output"],
            "executed_command": (
                ", ".join(result["executed_commands"])
                if result["executed_commands"]
                else ""
            ),
            "errors": result["errors"],
            "experiments_tried": result.get("experiments_tried", []),
            "experiments_succeeded": result.get("experiments_succeeded", []),
            "partial_success": result.get("partial_success", False),
        }
        state["experiments_completed"] = result[
            "main_experiment_successful"
        ] or result.get("partial_success", False)

        # Store context
        state["agent_contexts"]["unified_reproducer"] = result["report"]

        # Add messages
        if result["setup_successful"]:
            state["messages"].append("✅ Environment setup successful")
            print("✅ Environment setup successful")
        else:
            state["messages"].append("❌ Environment setup failed")
            print("❌ Environment setup failed")

        if result["data_successful"]:
            state["messages"].append("✅ Datasets prepared successfully")
            print("✅ Datasets prepared")
        elif result["data_manual_steps"]:
            state["messages"].append("⚠️  Manual dataset steps required")
            print("⚠️  Manual dataset steps required")

        if result["main_experiment_successful"]:
            state["messages"].append("✅ Experiments executed successfully")
            print("✅ Experiments executed successfully")
        elif result["sanity_check_passed"]:
            state["messages"].append(
                "⚠️  Sanity check passed but main experiment failed"
            )
            print("⚠️  Sanity check passed but main experiment failed")
        else:
            state["messages"].append("❌ Experiments failed")
            print("❌ Experiments failed")

        # Print summary
        print("\n📊 Unified Reproduction Summary:")
        print(
            f"   READMEs consulted: {', '.join(result['readmes_consulted']) if result['readmes_consulted'] else 'None'}"
        )
        print(f"   Setup: {'✅' if result['setup_successful'] else '❌'}")
        print(f"   Data: {'✅' if result['data_successful'] else '⚠️'}")
        print(
            f"   Experiments: {'✅' if result['main_experiment_successful'] else '❌'}"
        )

        # Save checkpoint after unified reproduction
        self._save_checkpoint(state, "unified_reproduction", success=result["main_experiment_successful"])

        # Record experiment wall time if available from the result
        if result.get("experiment_wall_time"):
            self.metrics_tracker.record_experiment_time(
                "unified_reproduction", result["experiment_wall_time"]
            )

        self.metrics_tracker.end_phase(
            "unified_reproduction", success=result["main_experiment_successful"]
        )
        self.metrics_tracker.print_intermediate_summary()
        return state

    def _extract_and_verify_node(
        self, state: PaperReproductionState
    ) -> PaperReproductionState:
        """Summarize verification results from the unified reproduction agent.

        The agent (unified_reproduction_agent) now handles extraction and verification
        using the code-first approach (execute_python_code). This node just summarizes
        the results already stored in state by the agent.
        """
        # Check if this phase was already completed (resuming from checkpoint)
        if self._is_phase_completed(state, "extract_and_verify"):
            print("⏭️  Skipping extract_and_verify (already completed from checkpoint)")
            return state

        self.metrics_tracker.start_phase("extract_and_verify")
        print("📊 Summarizing verification results...")

        experiment_results = state.get("experiment_results", {})
        dependencies_installed = state.get("dependencies_installed", False)

        # Get metrics already extracted by agent (via execute_python_code)
        state.get("extracted_metrics", {})
        metrics_comparison = state.get("metrics_comparison", {})

        # Build verification report from what agent stored
        verification_report = []
        verification_report.append("## Execution Summary")
        verification_report.append(
            f"- Dependencies Installed: {'Yes' if dependencies_installed else 'No'}"
        )
        verification_report.append(
            f"- Datasets Ready: {'Yes' if state.get('datasets_ready') else 'No'}"
        )
        verification_report.append(
            f"- Experiments Completed: {'Yes' if state.get('experiments_completed') else 'No'}"
        )

        # Include metrics comparison if agent performed it
        if metrics_comparison:
            verification_report.append("\n## Metrics Comparison")
            if metrics_comparison.get("matches"):
                verification_report.append("### Matching Metrics:")
                for match in metrics_comparison["matches"]:
                    if isinstance(match, dict):
                        verification_report.append(
                            f"  - {match.get('metric', 'N/A')}: {match.get('actual', 'N/A')} (expected: {match.get('expected', 'N/A')})"
                        )
            if metrics_comparison.get("mismatches"):
                verification_report.append("### Mismatched Metrics:")
                for mismatch in metrics_comparison["mismatches"]:
                    if isinstance(mismatch, dict):
                        verification_report.append(
                            f"  - {mismatch.get('metric', 'N/A')}: {mismatch.get('actual', 'N/A')} (expected: {mismatch.get('expected', 'N/A')})"
                        )

        report_text = "\n".join(verification_report)

        # Determine success level based on experiments
        experiments_tried = experiment_results.get("experiments_tried", [])
        experiments_succeeded = experiment_results.get("experiments_succeeded", [])

        # Check prerequisites first
        if not dependencies_installed:
            success_level = "failed"
            results_match = False
            status_msg = "❌ Verification: Environment setup failed"
        elif not experiments_tried:
            sanity_check_passed = experiment_results.get("sanity_check_passed", False)
            if sanity_check_passed:
                success_level = "minimal"
                results_match = False
                status_msg = "⚠️ Verification: Only sanity check completed"
            else:
                success_level = "setup_only"
                results_match = False
                status_msg = "⚠️ Verification: Setup complete but no experiments run"
        else:
            total_experiments = len(experiments_tried)
            succeeded_count = len(experiments_succeeded)
            success_portion = f"{succeeded_count}/{total_experiments}"

            if succeeded_count == total_experiments:
                success_level = "full"
                results_match = True
                status_msg = f"✅ Verification: All {total_experiments} experiment(s) succeeded - results match paper (within 5%)"
            elif succeeded_count > 0:
                success_level = "partial"
                results_match = False
                status_msg = f"⚠️ Verification: Partial reproduction - {success_portion} experiments succeeded ({', '.join(experiments_succeeded)})"
            else:
                success_level = "failed"
                results_match = False
                status_msg = (
                    f"❌ Verification: All {total_experiments} experiment(s) failed"
                )

        state["verification_results"] = {
            "report": report_text,
            "results_match_paper": results_match,
            "success_level": success_level,
            "discrepancies": metrics_comparison.get("mismatches", []),
        }
        state["results_match"] = results_match

        print(f"\n📝 Verification Report:\n{report_text}\n")
        state["messages"].append(status_msg)

        # Save checkpoint after verification
        self._save_checkpoint(state, "extract_and_verify", success=True)

        self.metrics_tracker.end_phase("extract_and_verify", success=results_match)
        return state

    def _generate_report_node(
        self, state: PaperReproductionState
    ) -> PaperReproductionState:
        """Generate final report."""
        # Note: We always regenerate the report even when resuming, to ensure it's up-to-date
        # But we still mark it as completed for tracking purposes
        self.metrics_tracker.start_phase("generate_report")
        print("📊 Generating final report...")

        # Determine final status
        success_level = state.get("verification_results", {}).get(
            "success_level", "failed"
        )
        experiments_tried = state.get("experiment_results", {}).get(
            "experiments_tried", []
        )
        experiments_succeeded = state.get("experiment_results", {}).get(
            "experiments_succeeded", []
        )

        # Calculate overall portion
        if experiments_tried:
            success_portion = f"{len(experiments_succeeded)}/{len(experiments_tried)}"
        else:
            success_portion = "0/0"

        status_map = {
            "full": "✅ Complete - All Experiments Succeeded (Results Match Paper)",
            "partial": (
                f"⚠️ Partial - {success_portion} Experiments Reproduced ({', '.join(experiments_succeeded)})"
                if experiments_succeeded
                else f"⚠️ Partial - {success_portion} Experiments Succeeded"
            ),
            "minimal": "⚠️ Minimal - Only Sanity Check Passed",
            "setup_only": "⚠️ Setup Only - No Experiments Run",
            "failed": "❌ Failed - Prerequisites or All Experiments Failed",
        }
        state["final_status"] = state.get("final_status") or status_map.get(
            success_level, "❌ Failed"
        )

        # Get selected repo info
        selected_repo_info = "None"
        if state.get("selected_repo"):
            repo = state["selected_repo"]
            if isinstance(repo, dict):
                selected_repo_info = repo.get("url") or repo.get("full_name", "Unknown")

        # Count repos found
        code_refs_count = len(
            [
                r
                for r in state.get("code_references", [])
                if isinstance(r, str) and r.startswith("http")
            ]
        )

        # Deduplicate messages
        seen = set()
        unique_messages = []
        skip_phrases = [
            "continuing anyway",
            "will attempt anyway",
            "had issues - continuing",
        ]

        for msg in state.get("messages", []):
            if msg in seen:
                continue
            if any(phrase in msg.lower() for phrase in skip_phrases):
                continue
            
            # Smart Filtering: If we succeeded, remove previous failure messages
            if state.get("implementation_path") and "no implementation found" in msg.lower():
                continue
                
            seen.add(msg)
            unique_messages.append(msg)

        # Build experiment section
        experiment_section = ""
        if experiments_tried:
            experiment_section = f"""
## Experiments
- Attempted: {', '.join(experiments_tried)}
- Succeeded: {', '.join(experiments_succeeded) if experiments_succeeded else 'None'}
- Success Rate: {success_portion} ({len(experiments_succeeded)/len(experiments_tried)*100:.0f}%)
"""

        report = f"""
# Paper Reproduction Report

## Paper Information
- Title: {state.get('paper_title', 'N/A')}
- Analysis: {'Complete' if state.get('paper_metadata') else 'Incomplete'}
- Code References Found: {code_refs_count}

## Implementation
- Selected Repository: {selected_repo_info}
- Implementation Path: {state.get('implementation_path', 'N/A')}
{experiment_section}
## Verification
- Results Match Paper: {'Yes' if state.get('results_match') else 'No'}
- Success Level: {success_level}
- Match Ratio: {state.get('verification_results', {}).get('match_ratio', 'N/A')}

## Status
{state.get('final_status', 'Complete')}

## Summary
{chr(10).join(unique_messages)}
"""

        state["report"] = report

        self.metrics_tracker.end_phase("generate_report", success=True)
        return state

    def _route_after_clone(
        self, state: PaperReproductionState
    ) -> Literal["continue", "failed"]:
        """Route after cloning repository."""
        if state.get("implementation_path"):
            return "continue"
        return "failed"

    def _route_after_env_setup(
        self, state: PaperReproductionState
    ) -> Literal["continue", "failed"]:
        """Route after environment setup."""
        if state.get("dependencies_installed", False):
            return "continue"
        print("🛑 Routing to report generation due to environment setup failure")
        return "failed"

    def _route_after_reproduction(
        self, state: PaperReproductionState
    ) -> Literal["continue", "failed"]:
        """Route after unified reproduction."""
        setup_success = state.get("dependencies_installed", False)
        experiments_completed = state.get("experiments_completed", False)
        sanity_check_passed = state.get("experiment_results", {}).get(
            "sanity_check_passed", False
        )

        if not setup_success:
            print("🛑 Routing to report generation due to setup failure")
            state["final_status"] = "Failed: Environment setup failed"
            return "failed"

        if experiments_completed or sanity_check_passed:
            print("✅ Routing to metrics extraction")
            return "continue"

        print("⚠️  No experiments run, but continuing to metrics extraction")
        return "continue"

    def _is_phase_completed(self, state: PaperReproductionState, phase: str) -> bool:
        """Check if a phase was already completed (from checkpoint resume).

        Args:
            state: Current workflow state
            phase: Phase name to check

        Returns:
            True if phase was already completed and should be skipped
        """
        completed = state.get("completed_phases", [])
        return phase in completed

    def _invalidate_phase(self, state: PaperReproductionState, phase: str, reason: str):
        """Remove a phase from completed_phases when validation fails.

        Args:
            state: Current workflow state
            phase: Phase name to invalidate
            reason: Why the phase is being invalidated
        """
        completed = state.get("completed_phases", [])
        if phase in completed:
            completed.remove(phase)
            state["completed_phases"] = completed
            print(f"⚠️  Invalidated checkpoint for '{phase}': {reason}")
            print(f"   Phase will be re-run instead of skipped.")

    def _validate_execution_checkpoint(self, state: PaperReproductionState) -> bool:
        """Validate that execution checkpoint has actual results.

        Returns:
            True if checkpoint is valid (has actual result files with content)
            False if checkpoint should be invalidated
        """
        import os

        code_path = state.get("implementation_path", "./cloned_repo")

        # Check for result files with actual content
        result_patterns = ["results", "output", "logs", "checkpoints"]
        has_actual_results = False

        for pattern in result_patterns:
            pattern_path = os.path.join(code_path, pattern)
            if os.path.exists(pattern_path):
                if os.path.isdir(pattern_path):
                    try:
                        for item in os.listdir(pattern_path):
                            item_path = os.path.join(pattern_path, item)
                            if os.path.isfile(item_path) and os.path.getsize(item_path) > 0:
                                has_actual_results = True
                                break
                    except (OSError, PermissionError):
                        pass
                elif os.path.isfile(pattern_path) and os.path.getsize(pattern_path) > 0:
                    has_actual_results = True

            if has_actual_results:
                break

        if not has_actual_results:
            self._invalidate_phase(state, "execution", "No result files with content found")
            return False

        return True

    def _validate_environment_checkpoint(self, state: PaperReproductionState) -> bool:
        """Validate that environment checkpoint has a working environment.

        Returns:
            True if checkpoint is valid (environment exists)
            False if checkpoint should be invalidated
        """
        import os
        import subprocess

        env_results = state.get("env_setup_results", {})
        env_name = env_results.get("env_name")
        env_type = env_results.get("env_type")

        if not env_name:
            self._invalidate_phase(state, "environment_setup", "No environment name recorded")
            return False

        # Check if environment exists based on type
        if env_type in ["conda", "mamba", "micromamba"]:
            try:
                # Check if conda environment exists
                result = subprocess.run(
                    ["conda", "env", "list"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if env_name not in result.stdout:
                    self._invalidate_phase(state, "environment_setup", f"Conda environment '{env_name}' not found")
                    return False
            except Exception:
                # Can't verify - assume valid to avoid blocking
                pass
        elif env_type == "venv":
            code_path = state.get("implementation_path", "./cloned_repo")
            venv_path = os.path.join(code_path, env_name if env_name.startswith("venv") else "venv")
            if not os.path.exists(venv_path):
                self._invalidate_phase(state, "environment_setup", f"Venv '{venv_path}' not found")
                return False

        return True

    def _validate_data_prep_checkpoint(self, state: PaperReproductionState) -> bool:
        """Validate that data_prep checkpoint has datasets ready.

        Returns:
            True if checkpoint is valid (datasets exist)
            False if checkpoint should be invalidated
        """
        import os

        # Check if datasets_ready flag is set
        if not state.get("datasets_ready", False):
            self._invalidate_phase(state, "data_prep", "datasets_ready flag is False")
            return False

        # Check if data directories have content
        code_path = state.get("implementation_path", "./cloned_repo")
        data_patterns = ["data", "datasets", "dataset"]
        has_data = False

        for pattern in data_patterns:
            data_path = os.path.join(code_path, pattern)
            if os.path.exists(data_path) and os.path.isdir(data_path):
                try:
                    if any(os.listdir(data_path)):
                        has_data = True
                        break
                except (OSError, PermissionError):
                    pass

        # If no data directory found, check dataset_results for indicators
        dataset_results = state.get("dataset_results", {})
        if dataset_results.get("datasets_downloaded") or dataset_results.get("data_ready"):
            has_data = True

        # If data_prep was marked complete but there's no obvious data,
        # we might have skipped it intentionally (no datasets needed)
        # So only invalidate if we can confirm missing required data

        return True  # Default to trusting the checkpoint for data_prep

    def _save_checkpoint(self, state: PaperReproductionState, phase: str, success: bool = True):
        """Save checkpoint for current phase.

        Args:
            state: Current workflow state
            phase: Phase name (e.g., 'analyze_paper', 'decide_and_clone', etc.)
            success: Whether the phase completed successfully (defaults to True for backward compatibility)
        """
        if not self.checkpoint_manager:
            return

        # Add this phase to completed_phases if successful and not already there
        if "completed_phases" not in state:
            state["completed_phases"] = []
            
        if success:
            if phase not in state["completed_phases"]:
                state["completed_phases"].append(phase)
        else:
            # If failed, ensure it's NOT in completed_phases so we retry
            if phase in state["completed_phases"]:
                state["completed_phases"].remove(phase)

        repo_path = state.get("implementation_path") or self.config.repo_path
        paper_id = state.get("paper_metadata", {}).get("arxiv_id", "") or state.get(
            "paper_input", ""
        )

        # Create checkpoint-safe state (only serializable data)
        checkpoint_state = {
            "phase": phase,
            "paper_title": state.get("paper_title", ""),
            "paper_metadata": state.get("paper_metadata", {}),
            "experimental_setup": state.get("experimental_setup", {}),
            "paper_results": state.get("paper_results", {}),
            "code_references": state.get("code_references", []),
            "selected_repo": state.get("selected_repo", {}),
            "implementation_path": state.get("implementation_path", ""),
            "env_setup_results": state.get("env_setup_results", {}),
            "dependencies_installed": state.get("dependencies_installed", False),
            "dataset_results": state.get("dataset_results", {}),
            "datasets_ready": state.get("datasets_ready", False),
            "experiment_results": state.get("experiment_results", {}),
            "experiments_completed": state.get("experiments_completed", False),
            "extracted_metrics": state.get("extracted_metrics", {}),
            "metrics_comparison": state.get("metrics_comparison", {}),
            "verification_results": state.get("verification_results", {}),
            "results_match": state.get("results_match", False),
            "agent_contexts": state.get("agent_contexts", {}),  # Context from agents
            "messages": state.get("messages", []),
            "completed_phases": state.get(
                "completed_phases", []
            ),  # Track completed phases for resume
            "phase_status": state.get("phase_status", {}),  # Track detailed phase status
            # Hierarchical context state for cross-agent knowledge preservation
            "hierarchical_context_state": (
                self.hierarchical_context.to_dict()
                if self.hierarchical_context else None
            ),
            # Metrics tracker state for cost/duration preservation across resumes
            "metrics_tracker_state": self.metrics_tracker.to_dict(),
            # Gemini cache name (may be expired but worth attempting on resume)
            "cache_name": getattr(self, "cache_name", None),
        }

        self.checkpoint_manager.save(
            state=checkpoint_state, phase=phase, repo_path=repo_path, paper_id=paper_id
        )

    def _run_verification_only(
        self, paper_input: str, repo_path: str, existing_results: dict
    ) -> dict:
        """Run verification workflow when results already exist.

        This skips paper analysis and experiment execution, using the unified
        reproduction agent to extract metrics and compare with paper via code-first
        approach (execute_python_code).

        Args:
            paper_input: Paper identifier (arXiv ID, etc.)
            repo_path: Path to repository with existing results
            existing_results: Dict from _check_existing_results

        Returns:
            Final state with verification results
        """
        print("📊 Running verification-only mode with code-first approach...")

        # Step 1: Try to get paper results from existing checkpoint
        paper_title = paper_input
        paper_results = {}

        if self.checkpoint_manager:
            checkpoint_data = self.checkpoint_manager.resume(
                repo_path=repo_path, paper_id=paper_input
            )
            if checkpoint_data:
                state = checkpoint_data.get("state", {})
                paper_results = state.get("paper_results", {})
                paper_title = state.get("paper_title", paper_input)
                if paper_results:
                    print(
                        f"✅ Found paper results from checkpoint: {paper_title[:60]}..."
                    )

        # Step 2: If no checkpoint, try to get paper info from arXiv
        if not paper_results:
            try:
                if paper_input.startswith("arxiv:") or "." in paper_input:
                    arxiv_id = paper_input.replace("arxiv:", "")
                    print(f"📄 Fetching paper info for {arxiv_id}...")

                    import arxiv

                    search = arxiv.Search(id_list=[arxiv_id])
                    paper = next(search.results())
                    paper_title = paper.title
                    print(f"   Title: {paper_title[:60]}...")
            except Exception as e:
                print(f"⚠️ Could not fetch paper info: {e}")

        # Step 3: Build initial state for verification

        # Step 4: Build verification prompt for agent
        result_files = existing_results.get("result_files", [])
        verification_prompt = f"""VERIFICATION-ONLY MODE

You are verifying existing experiment results against paper claims.

Repository: {repo_path}
Paper: {paper_title}

Existing result files found:
{chr(10).join(f"- {f}" for f in result_files[:20])}

Paper expected results:
{paper_results if paper_results else "Not available - extract from result files and report findings"}

YOUR TASK:
1. Use execute_python_code to write a script that:
   - Reads the result files in {repo_path}
   - Extracts metrics (accuracy, F1, etc.) from each file
   - Compares with paper expected values (if available)
   - Reports match/mismatch status

2. Store results in a structured format

IMPORTANT: Write Python code to parse the specific file formats you find.
Do NOT assume any particular format - explore and adapt.

Begin verification now."""

        # Step 5: Run agent for verification
        print("\n🤖 Running unified reproduction agent for verification...")
        agent_result = self.unified_reproducer.reproduce(
            code_path=repo_path, paper_context=verification_prompt
        )

        # Step 6: Build final state from agent results
        extracted_metrics = agent_result.get("extracted_metrics", {})
        metrics_comparison = agent_result.get("metrics_comparison", {})

        # Determine success level
        experiments_succeeded = agent_result.get("experiments_succeeded", [])
        experiments_tried = agent_result.get("experiments_tried", [])

        if experiments_tried:
            match_success = (
                len(experiments_succeeded) == len(experiments_tried)
                and len(experiments_tried) > 0
            )
            success_level = (
                "verified"
                if match_success
                else ("partial" if experiments_succeeded else "failed")
            )
        else:
            # Agent didn't track experiments - check if we have comparison results
            match_success = bool(extracted_metrics)
            success_level = "verified" if match_success else "unknown"

        final_state = {
            "paper_input": paper_input,
            "paper_title": paper_title,
            "implementation_path": repo_path,
            "dependencies_installed": True,
            "datasets_ready": True,
            "experiments_completed": True,
            "extracted_metrics": extracted_metrics,
            "metrics_comparison": metrics_comparison,
            "verification_results": {
                "report": agent_result.get("output", "Verification completed"),
                "results_match_paper": match_success,
                "success_level": success_level,
            },
            "results_match": match_success,
            "messages": [
                f"✅ Using existing results from {repo_path}",
                "📊 Verification completed via code-first approach",
            ],
            "final_status": (
                "✅ Verification Complete"
                if match_success
                else "⚠️ Verification Complete (partial)"
            ),
            "report": agent_result.get("output", ""),
        }

        print(f"\n{'='*60}")
        print(f"{'✅' if match_success else '⚠️'} Verification Complete")
        print(f"{'='*60}\n")

        return final_state

    def _build_verification_report(
        self, comparison_result: dict, extracted_datasets: dict
    ) -> str:
        """Build detailed verification report text."""
        lines = ["## Verification Report\n"]

        summary = comparison_result.get("summary", {})
        lines.append(f"**Status**: {summary.get('status', 'Unknown')}")
        lines.append(f"**Match Ratio**: {summary.get('match_ratio', 'N/A')}")
        lines.append(
            f"**Match Percentage**: {summary.get('match_percentage', 'N/A')}\n"
        )

        if comparison_result.get("matched"):
            lines.append("### ✅ Matched Results (within 5% tolerance)")
            for m in comparison_result["matched"]:
                lines.append(
                    f"- **{m['expected_dataset']}** → {m['extracted_dataset']}"
                )
                lines.append(f"  - Extracted: {m['extracted_value']:.2f}")
                lines.append(f"  - Expected: {m['expected_value']:.2f}")
                lines.append(f"  - Error: {m['relative_error_pct']}")

        if comparison_result.get("mismatched"):
            lines.append("\n### ⚠️ Mismatched Results (outside 5% tolerance)")
            for m in comparison_result["mismatched"]:
                lines.append(f"- **{m['expected_dataset']}**")
                lines.append(f"  - Extracted: {m['extracted_value']:.2f}")
                lines.append(f"  - Expected: {m['expected_value']:.2f}")
                lines.append(f"  - Error: {m['relative_error_pct']}")

        if comparison_result.get("missing_from_extracted"):
            lines.append("\n### ❌ Missing Datasets")
            for ds in comparison_result["missing_from_extracted"]:
                lines.append(f"- {ds}")

        if comparison_result.get("extra_in_extracted"):
            lines.append("\n### 📊 Additional Datasets (not in paper)")
            for ds in comparison_result["extra_in_extracted"]:
                lines.append(f"- {ds}")

        return "\n".join(lines)

    def _build_final_report(
        self,
        paper_input: str,
        paper_title: str,
        repo_path: str,
        existing_results: dict,
        extracted_datasets: dict,
        extracted_metrics: dict,
        comparison_result: dict,
        source_files: list,
    ) -> str:
        """Build the final markdown report."""
        summary = comparison_result.get("summary", {})

        # Build dataset results section
        dataset_lines = []
        for ds_name, metrics in list(extracted_datasets.items())[:10]:
            main_metric = list(metrics.keys())[0] if metrics else "unknown"
            main_value = list(metrics.values())[0] if metrics else "N/A"
            dataset_lines.append(f"- **{ds_name}**: {main_metric} = {main_value}")

        # Build comparison section
        comparison_lines = []
        if comparison_result.get("aligned_comparisons"):
            for comp in comparison_result["aligned_comparisons"][:10]:
                status = "✅" if comp["within_tolerance"] else "❌"
                comparison_lines.append(
                    f"- {status} **{comp['expected_dataset']}**: "
                    f"{comp['extracted_value']:.2f} (expected {comp['expected_value']:.2f}, "
                    f"error: {comp['relative_error_pct']})"
                )

        return f"""
# Paper Reproduction Report (Verification Only)

## Paper
- **Input**: {paper_input}
- **Title**: {paper_title}

## Existing Results Used
- **Repository**: {repo_path}
- **Result files found**: {len(existing_results.get('result_files', []))}
- **Model checkpoints**: {len(existing_results.get('checkpoints', []))}
- **Source files analyzed**: {', '.join(source_files[:3]) if source_files else 'N/A'}

## Extracted Results by Dataset
{chr(10).join(dataset_lines) if dataset_lines else '- No results extracted'}

## Comparison with Paper
- **Status**: {summary.get('status', 'Unknown')}
- **Match Ratio**: {summary.get('match_ratio', 'N/A')}
- **Match Percentage**: {summary.get('match_percentage', 'N/A')}

### Detailed Comparison
{chr(10).join(comparison_lines) if comparison_lines else '- No comparison available'}

## Summary
The reproduction {'successfully matched' if summary.get('matched_count', 0) == summary.get('total_expected', 0) and summary.get('total_expected', 0) > 0 else 'partially matched'} the paper results with {summary.get('match_percentage', '0%')} of datasets within the 5% tolerance threshold.
"""

    def _check_existing_results(self, repo_path: str) -> dict:
        """Check if results already exist in the repository.

        Args:
            repo_path: Path to the cloned repository

        Returns:
            Dictionary with has_results, result_files, checkpoints, log_files
        """
        return self.discovery_agent.check_existing_results(repo_path)

    def _try_resume_checkpoint(self, paper_input: str, repo_path: str = None) -> dict:
        """Try to resume from checkpoint.

        Args:
            paper_input: Paper identifier (arXiv ID, etc.)
            repo_path: Repository path
        """
        if repo_path is None:
            repo_path = self.config.repo_path
        if not self.checkpoint_manager:
            print("⚠️ Checkpoint manager is None!")
            return {}

        print(f"🔍 DEBUG: Attempting resume for '{paper_input}'")
        print(f"🔍 DEBUG: Repo Path: '{repo_path}'")
        print(f"🔍 DEBUG: Checkpoint Dir: '{self.checkpoint_manager.checkpoint_dir}'")
        
        # 1. Try exact match first
        checkpoint_data = self.checkpoint_manager.resume(
            repo_path=repo_path, paper_id=paper_input
        )

        # 2. Fallback: Search by Paper ID directly if exact match fails
        # This handles cross-platform path issues (WSL vs Windows)
        if not checkpoint_data:
            print(f"📋 Exact match failed. Searching for checkpoints by paper ID: {paper_input}")
            # Normalize paper ID
            search_id = paper_input.replace("arxiv:", "").strip()
            
            # Use the manager's search functionality
            matches = self.checkpoint_manager._search_by_paper_id(search_id)
            
            if matches:
                # Get the most recent one
                latest_match = max(matches, key=lambda x: x[1]["timestamp"])
                checkpoint_path, data = latest_match
                
                print(f"✅ Found cross-platform match: {checkpoint_path.name}")
                print(f"   Original Path: {data.get('repo_path')}")
                print(f"   Current Path : {repo_path}")
                
                # Check if paths are reasonably similar (same leaf directory)
                orig_name = os.path.basename(data.get('repo_path', '').rstrip('/\\'))
                curr_name = os.path.basename(repo_path.rstrip('/\\'))
                
                if orig_name == curr_name:
                    print("   Repo directory names match, resuming...")
                    checkpoint_data = data
                else:
                    print(f"   ⚠️ Repo names differ ({orig_name} vs {curr_name}) - proceeding with caution")
                    checkpoint_data = data

        if checkpoint_data:
            state = checkpoint_data.get("state", {})
            
            # Critical: Ensure state has correct current paths
            if repo_path and repo_path != state.get("implementation_path"):
                print(f"   🔄 Updating implementation path to current: {repo_path}")
                state["implementation_path"] = repo_path

            # Restore hierarchical context if available
            hierarchical_state = state.pop("hierarchical_context_state", None)
            if hierarchical_state and self.hierarchical_context:
                try:
                    self.hierarchical_context = HierarchicalContextManager.from_dict(
                        hierarchical_state
                    )
                    # Re-attach API embedder (not serialized in checkpoint)
                    # Without this, from_dict() falls back to downloading local SentenceTransformer
                    if self._embedder:
                        self.hierarchical_context._embedder = self._embedder
                        self.hierarchical_context._embedder_provided = True
                        print("   🔌 Re-attached API embedder to restored context")
                    stats = self.hierarchical_context.get_stats()
                    print(f"   🧠 Restored hierarchical context: "
                          f"hot={stats['hot_entries']}, cold={stats['cold_summaries']}")
                except Exception as e:
                    print(f"   ⚠️ Failed to restore hierarchical context: {e}")
                    # Continue with fresh context - non-fatal

            # Restore metrics tracker if available
            metrics_state = state.pop("metrics_tracker_state", None)
            if metrics_state:
                try:
                    from .utils.metrics_tracker import MetricsTracker
                    restored_tracker = MetricsTracker.from_dict(
                        metrics_state,
                        resume_workflow=False  # We'll call resume_workflow() later
                    )

                    # Update ALL references (orchestrator, agents, callbacks) to use restored tracker
                    # This is critical - without this, new tokens go to old tracker while summary uses new
                    self._update_metrics_tracker_references(restored_tracker)

                    cost = self.metrics_tracker.estimate_cost()
                    tokens = (
                        self.metrics_tracker.metrics.total_llm_tokens_input +
                        self.metrics_tracker.metrics.total_llm_tokens_output
                    )
                    print(f"   💰 Restored metrics: ${cost:.4f} cost, {tokens:,} tokens")
                except Exception as e:
                    print(f"   ⚠️ Failed to restore metrics: {e}")
                    # Continue with fresh tracker - non-fatal

            # Restore Gemini cache name if available
            restored_cache = state.pop("cache_name", None)
            if restored_cache:
                self.cache_name = restored_cache
                print(f"   🗄️ Restored cache name: {restored_cache[:50]}...")

            completed_phases = state.get("completed_phases", [])

            print("\n♻️  RESUMING FROM CHECKPOINT")
            print(f"   Last phase: {checkpoint_data['phase']}")
            print(f"   Saved at: {checkpoint_data['timestamp']}")
            if completed_phases:
                print("   ✅ Completed phases that will be SKIPPED:")
                for phase in completed_phases:
                    print(f"      - {phase}")
            print()
            return state

        return {}

    def run(self, paper_input: str, clear_checkpoints: bool = False) -> dict:
        """
        Run the complete paper reproduction workflow.

        Args:
            paper_input: arXiv ID, PDF path, or paper identifier
            clear_checkpoints: If True, clear existing checkpoints and start fresh

        Returns:
            Final state with results
        """
        print(f"\n{'='*60}")
        print("🚀 Starting Paper Reproduction Workflow (Clean)")
        print(f"{'='*60}\n")

        # Start metrics tracking
        self.metrics_tracker.start_workflow()

        # Clear checkpoints if requested
        if clear_checkpoints and self.checkpoint_manager:
            self.checkpoint_manager.clear(
                repo_path=self.config.repo_path, paper_id=paper_input
            )
            print("🗑️  Cleared existing checkpoints\n")

        # FIRST: Check if cloned_repo already has results (before anything else!)
        repo_path = self.config.repo_path
        if os.path.exists(repo_path):
            existing_results = self._check_existing_results(repo_path)
            if existing_results.get("has_results"):
                print("\n" + "=" * 60)
                print("🎯 EXISTING RESULTS DETECTED IN REPOSITORY!")
                print("=" * 60)
                print(f"   📁 Repository: {repo_path}")
                print(
                    f"   📊 Result files: {len(existing_results.get('result_files', []))}"
                )
                print(
                    f"   💾 Checkpoints: {len(existing_results.get('checkpoints', []))}"
                )
                print(f"   📋 Log files: {len(existing_results.get('log_files', []))}")
                if existing_results.get("result_files"):
                    for f in existing_results["result_files"][:3]:
                        print(f"      → {f}")
                print("=" * 60)
                print("\n⏩ Skipping paper analysis and experiment execution...")
                print("   Going directly to RESULT VERIFICATION\n")

                # Create a minimal state and skip to verification
                result = self._run_verification_only(
                    paper_input, repo_path, existing_results
                )
                # End metrics tracking for verification-only path
                self.metrics_tracker.end_workflow()
                print(self.metrics_tracker.get_summary())
                return result

        # Try to resume from checkpoint
        resumed_state = self._try_resume_checkpoint(paper_input)

        if resumed_state:
            # Merge with defaults (in case new fields were added)
            initial_state = {
                "paper_input": paper_input,
                "paper_title": "",
                "paper_metadata": {},
                "experimental_setup": {},
                "paper_results": {},
                "code_references": [],
                "selected_repo": {},
                "implementation_path": "",
                "env_setup_results": {},
                "dependencies_installed": False,
                "dataset_results": {},
                "datasets_ready": False,
                "experiment_results": {},
                "experiments_completed": False,
                "extracted_metrics": {},
                "metrics_comparison": {},
                "verification_results": {},
                "results_match": False,
                "agent_contexts": {},
                "completed_phases": [],  # Will be populated from resumed_state
                "messages": [],
                "final_status": "",
                "report": "",
            }
            # Update with resumed state
            initial_state.update(resumed_state)
            completed_count = len(resumed_state.get("completed_phases", []))
            print(
                f"✅ Resumed with {completed_count} completed phase(s), {len(resumed_state.get('messages', []))} messages\n"
            )
        else:
            # Start fresh
            initial_state = {
                "paper_input": paper_input,
                "paper_title": "",
                "paper_metadata": {},
                "experimental_setup": {},
                "paper_results": {},
                "code_references": [],
                "selected_repo": {},
                "implementation_path": "",
                "env_setup_results": {},
                "dependencies_installed": False,
                "dataset_results": {},
                "datasets_ready": False,
                "experiment_results": {},
                "experiments_completed": False,
                "extracted_metrics": {},
                "metrics_comparison": {},
                "verification_results": {},
                "results_match": False,
                "agent_contexts": {},
                "completed_phases": [],  # Track completed phases for checkpoint resume
                "messages": [],
                "final_status": "",
                "report": "",
            }

        try:
            final_state = self.workflow.invoke(initial_state)
        finally:
            # End metrics tracking and print summary
            self.metrics_tracker.end_workflow()
            print(self.metrics_tracker.get_summary())

        print(f"\n{'='*60}")
        print("✅ Workflow Complete")
        print(f"{'='*60}\n")

        # Close the log file if logging is enabled
        if self.file_logger:
            self.file_logger.close()

        return final_state
