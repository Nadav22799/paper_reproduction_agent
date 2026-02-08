# 🧠 Paper Reproduction Agent — Technical Deep Dive

> Architecture details and design decisions for the multi-agent paper reproduction system.

## 🏗️ Architecture: Cyclic Supervisor Multi-Agent System

The system uses a **cyclic state graph** (LangGraph) where a Supervisor agent dynamically routes tasks to specialized sub-agents. Unlike linear pipelines, the Supervisor can route *backwards* — e.g., from Execution back to Environment Setup — to recover from errors without restarting the entire workflow.

### Agent Roles

#### 1. Paper Analyzer (`src/agents/unified_paper_analyzer.py`)
*   **Entry Point**: Turns an unstructured PDF/arXiv ID into structured metadata.
*   Uses Regex + LLM to extract code repository links, hardware requirements, datasets, and expected results.

#### 2. Supervisor Agent (`src/agents/supervisor_agent.py`)
*   **Router**: Inspects the full state (phase status, error history, past actions) and decides which agent runs next.
*   **Error Classification**: Classifies errors semantically into types (environment, data, execution, validation) and routes to the appropriate recovery agent.
*   **Loop Protection**: Uses hierarchical context to detect repeated failures and switch strategies (e.g., "Try Conda instead of Pip").

#### 3. Planning Agent (`src/agents/planning_agent.py`)
*   Reads the repository README and code structure.
*   Creates a `reproduction_checklist.md` with environment steps, experiment commands, and expected metrics.
*   Supports three modes: single experiment, full reproduction, or custom selection.

#### 4. Critic Agent (`src/agents/critic_agent.py`)
*   **Safety Guardrail**: Intercepts every Execution action (e.g., `rm -rf`, destructive `sed`).
*   Forces the agent to justify risky actions via Chain of Thought before approval.

#### 5. Environment Setup Agent (`src/agents/environment_setup_agent.py`)
*   Resolves dependencies, creates virtual environments, handles legacy compatibility (e.g., TF 1.x on modern Python).
*   Reports failures back to the Supervisor for targeted recovery.

#### 6. Data Prep Agent (`src/agents/data_prep_agent.py`)
*   Downloads and verifies datasets required by the paper.

#### 7. Execution Agent (`src/agents/execution_agent.py`)
*   Runs experiments with resource awareness.
*   Checks `nvidia-smi` and adjusts batch sizes to prevent OOM errors.

#### 8. Validation Agent (`src/agents/validation_agent.py`)
*   Extracts output metrics from logs and compares against paper-reported values.
*   Reports match/mismatch with explicit error margins.

---

## 📚 Hierarchical Context Memory

**File:** `src/utils/hierarchical_context.py`

Solves the limited context window problem with a 3-tier memory system:

### Storage Tiers

| Tier | Storage | Access | Capacity |
|------|---------|--------|----------|
| **Hot** | `OrderedDict` (RAM) | Direct lookup, FIFO | 30 entries (default) |
| **Warm** | ChromaDB (HNSW index, cosine space) | Semantic vector search | Unbounded |
| **Cold** | LLM-generated summaries | Compressed historical context | Planned for V2 |

### Relevance Scoring

When an agent requests context, entries are scored by a multi-factor formula:

```
Score = 0.4 × Semantic_Similarity + 0.3 × Recency_Decay + 0.2 × Importance + 0.1 × Source_Authority
```

**Source Authority Weights:**
| Source | Weight | Rationale |
|--------|--------|-----------|
| `paper_analyzer` | 1.0 | Ground truth from the paper |
| `user` | 1.0 | Explicit user instructions |
| `reproduction` | 0.9 | Core execution results |
| `discovery` | 0.8 | Repository analysis |
| `environment_setup` | 0.7 | Environment state |
| `system` | 0.5 | Internal bookkeeping |
| `debug` | 0.3 | Noisy debug output |

### Per-Agent Context Retrieval

Each agent queries the hierarchical context with a task-specific semantic query and injects results into its `HumanMessage` (not the system prompt — keeping system prompts static and cacheable).

| Agent | Semantic Query | Token Budget | Excludes Self |
|-------|---------------|-------------|---------------|
| **Planning** | `"paper analysis datasets experiments requirements"` | 2,000 | `["planning"]` |
| **Environment** | `"environment setup python conda pip error"` | 1,500 | `["environment_setup"]` |
| **Data Prep** | `"data download dataset path environment setup"` | 1,500 | `["data_prep"]` |
| **Execution** | `"environment setup experiment execution python command"` | 3,000 | `["execution"]` |
| **Validation** | `"experiment results metrics accuracy expected values"` | 2,000 | `["validation"]` |
| **Supervisor** | Dynamic: `f"error {error_type} {message[:100]}"` | 2,000 | None |

**Design decisions:**
*   `exclude_sources` prevents feedback loops — agents don't see their own stale output.
*   Execution gets the largest budget (3,000 tokens) because it needs cross-agent context: env setup results, tools, and smoke tests.
*   All dynamic context goes into `HumanMessage`, keeping system prompts static and cacheable across ReAct iterations.

### System Prompt Token Counts

| Agent | Tokens |
|-------|--------|
| Environment Setup | 2,705 |
| Planning | 1,581 |
| Execution | 1,071 |
| Validation | 514 |
| Data Prep | 459 |

---

## 🔄 Error Recovery Flow

When an error occurs during execution:

1. The failing agent returns an error state with metadata (error type, message, retry count).
2. The **Supervisor** classifies the error type:
   - `ENVIRONMENT` → routes to Environment Setup Agent
   - `DATA` → routes to Data Prep Agent
   - `EXECUTION` → retries with adjusted parameters (e.g., reduced batch size for OOM)
   - `VALIDATION` → re-examines results or re-runs specific experiments
3. Before routing, the Supervisor queries hierarchical context: `"error {type} {message}"`.
4. If a similar error was resolved before in this session, the Supervisor retrieves that solution and applies it directly.
5. Safety limits prevent infinite loops (max iterations per agent configurable in orchestrator).

### Max Iterations per Agent

| Agent | Max Iterations |
|-------|---------------|
| Planning | 90 |
| Execution | 150 |
| Data Prep | 150 |
| Validation | 90 |
| Environment Setup | 150 |

---

## 📂 Source Layout

```
src/
├── agents/              # Specialized Agents
│   ├── supervisor_agent.py       # Router & error recovery
│   ├── planning_agent.py         # Strategy & checklist creation
│   ├── critic_agent.py           # Safety guardrail
│   ├── environment_setup_agent.py
│   ├── data_prep_agent.py
│   ├── execution_agent.py
│   ├── validation_agent.py
│   └── unified_reproduction_agent.py  # Legacy linear mode
├── tools/               # Sandboxed Execution
│   ├── code_execution_tools.py   # Shell, Python, file operations
│   ├── code_search_tools.py      # Repository search via GitHub API
│   └── file_utils.py             # Intelligent file search
├── utils/               # Engineering Utilities
│   ├── hierarchical_context.py   # 3-tier context memory
│   ├── context_manager.py        # LLM context window management
│   ├── metrics_tracker.py        # Token/cost/time tracking
│   ├── resource_detector.py      # Hardware introspection
│   ├── dependency_resolver.py    # Package version compatibility
│   ├── oom_handler.py            # OOM detection & batch size adjustment
│   └── llm_factory.py            # LLM provider factory (Gemini/Claude)
├── config/
│   ├── __init__.py               # ReproductionConfig (Pydantic)
│   └── prompts.py                # Static system prompts per agent
└── orchestrator.py               # LangGraph state machine
```

### Legacy Mode Support
Setting `USE_SUPERVISOR_ARCHITECTURE=false` reverts to the linear `UnifiedReproductionAgent` flow. This is preserved for benchmarking comparisons.

---

## ⚙️ LLM Configuration

**File:** `src/utils/llm_factory.py`

Supports two providers with automatic detection:
- **Gemini** (default): Uses implicit caching (automatic on Gemini 3/2.5 models, no explicit cache management needed)
- **Claude**: Uses automatic prompt caching via `anthropic-beta` header

Environment variables: `LLM_PROVIDER`, `GOOGLE_API_KEY` / `ANTHROPIC_API_KEY`, `EMBEDDING_PROVIDER`

---

## 🚀 Extending the System

To add a new capability (e.g., multi-node training support):
1. Add resource detection logic to `src/utils/resource_detector.py`.
2. Update `code_execution_tools.py` with the new command runner (e.g., `srun`, `ray submit`).
3. Update the relevant agent's system prompt in `src/config/prompts.py`.
4. The Supervisor will automatically route to it based on error classification.
