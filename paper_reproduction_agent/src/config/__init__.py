from pathlib import Path
import os
from pydantic import BaseModel, Field


class ReproductionConfig(BaseModel):
    """Configuration for the Paper Reproduction Agent.

    Centralizes all file paths and execution parameters.
    Defaults are set to match the original hardcoded values.
    """

    # Base working directory (defaults to current working directory)
    base_dir: Path = Field(default_factory=Path.cwd)

    # Subdirectories (relative to base_dir if not absolute)
    repo_dir_name: str = "cloned_repo"
    downloads_dir_name: str = "downloads"
    logs_dir_name: str = "logs"
    checkpoints_dir_name: str = "checkpoints"

    # Execution Limits
    max_steps: int = 50
    execution_timeout: int = 3600  # 1 hour

    # Observability settings
    enable_live_progress: bool = Field(
        default_factory=lambda: os.getenv("enable_live_progress", "true").lower()
        == "true"
    )
    progress_update_interval: int = Field(
        default_factory=lambda: int(os.getenv("progress_update_interval", "15"))
    )

    # Cost per 1M input tokens (configurable per model)
    llm_input_cost_per_million: float = Field(
        default_factory=lambda: float(os.getenv("llm_input_cost_per_million", "3.00"))
    )
    # Cost per 1M output tokens
    llm_output_cost_per_million: float = Field(
        default_factory=lambda: float(os.getenv("llm_output_cost_per_million", "15.00"))
    )

    # Verbose reasoning output - shows agent reasoning during execution
    show_reasoning: bool = Field(
        default_factory=lambda: os.getenv("SHOW_REASONING", "false").lower() == "true"
    )

    # LLM Critic - deeper inspection for edge cases
    enable_llm_critic: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_LLM_CRITIC", "false").lower() == "true"
    )

    # Critic mode: "auto" (fully autonomous, blocks silently) or
    # "critic" (asks user approval for potentially dangerous actions)
    critic_mode: str = Field(
        default_factory=lambda: os.getenv("CRITIC_MODE", "auto")
    )

    # Embedding provider for hierarchical context: "gemini", "openai", "local", or "none"
    # "gemini" uses API (fast, no local model loading)
    # "local" uses SentenceTransformer (slower, 100MB RAM)
    # "none" disables embeddings (hot storage only)
    embedding_provider: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "gemini")
    )

    # Embedding model (for API providers)
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    )

    def _resolve_path(self, path_name: str) -> str:
        """Helper to resolve paths relative to base_dir."""
        path = Path(path_name)
        if path.is_absolute():
            return str(path)
        return str(self.base_dir / path)

    @property
    def repo_path(self) -> str:
        """Full path to the cloned repository."""
        return self._resolve_path(self.repo_dir_name)

    @property
    def downloads_path(self) -> str:
        """Full path to the downloads directory."""
        return self._resolve_path(self.downloads_dir_name)

    @property
    def logs_path(self) -> str:
        """Full path to the logs directory."""
        return self._resolve_path(self.logs_dir_name)

    @property
    def checkpoints_path(self) -> str:
        """Full path to the checkpoints directory."""
        return self._resolve_path(self.checkpoints_dir_name)

    class Config:
        arbitrary_types_allowed = True
