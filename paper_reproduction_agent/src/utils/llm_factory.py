"""LLM Factory - Gemini and Claude support.

Simplified version supporting only two providers:
- Gemini (primary) with implicit caching (automatic on Gemini 3/2.5)
- Claude (alternative) with automatic prompt caching via beta header
"""

import os


def get_provider() -> str:
    """Get configured LLM provider.

    Returns:
        Provider name: "gemini" or "claude"

    Raises:
        ValueError: If no provider configured
    """
    explicit = os.getenv("LLM_PROVIDER", "").lower()
    if explicit in ["gemini", "claude", "anthropic"]:
        return "claude" if explicit == "anthropic" else explicit

    # Auto-detect based on available keys
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude"

    raise ValueError(
        "No LLM configured! Set one of:\n"
        "  - GOOGLE_API_KEY or GEMINI_API_KEY (for Gemini)\n"
        "  - ANTHROPIC_API_KEY (for Claude)\n"
        "Optionally set LLM_PROVIDER=gemini or LLM_PROVIDER=claude"
    )


def create_llm(
    temperature: float = 0.3,
    include_thoughts: bool = True,
):
    """
    Create LLM instance based on configured provider.

    Args:
        temperature: Generation temperature (default from env or 0.3)
        include_thoughts: Enable chain-of-thought (Gemini only)

    Returns:
        Configured LLM instance
    """
    if temperature == 0.3:
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))

    provider = get_provider()

    if provider == "gemini":
        return _create_gemini_llm(temperature, include_thoughts)
    else:
        return _create_claude_llm(temperature)


def _create_gemini_llm(
    temperature: float,
    include_thoughts: bool,
):
    """Create Gemini LLM (implicit caching is automatic on Gemini 3/2.5)."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

    print(f"Using Gemini: {model}")
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        include_thoughts=include_thoughts,
    )


def _create_claude_llm(temperature: float):
    """Create Claude LLM with prompt caching enabled via beta header."""
    from langchain_anthropic import ChatAnthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    # Enable prompt caching via beta header
    extra_headers = {"anthropic-beta": "prompt-caching-2024-07-31"}

    print(f"Using Claude: {model} (caching enabled)")
    return ChatAnthropic(
        anthropic_api_key=api_key,
        model=model,
        temperature=temperature,
        extra_headers=extra_headers,
    )



# === EMBEDDINGS ===


class GeminiEmbedder:
    """Gemini API-based embedder matching SentenceTransformer interface.

    Uses Google's gemini-embedding-001 model for fast, API-based embeddings.
    No local model loading required - saves ~100MB RAM.
    """

    def __init__(self, model: str = None, metrics_tracker=None):
        """Initialize Gemini embedder.

        Args:
            model: Embedding model name (default: from EMBEDDING_MODEL env var or gemini-embedding-001)
            metrics_tracker: Optional MetricsTracker for token tracking
        """
        self.model = model or os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        self._client = None
        self.metrics_tracker = metrics_tracker

    def _ensure_client(self):
        """Lazy initialize the Gemini client."""
        if self._client is None:
            try:
                from google import genai

                api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set")
                self._client = genai.Client(api_key=api_key)
                print(f"Gemini embedder initialized: {self.model}")
            except ImportError:
                raise ImportError(
                    "google-genai not installed. Install with: pip install google-genai"
                )
        return self._client

    def encode(self, text: str):
        """Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            np.ndarray: Embedding vector (768 dimensions for gemini-embedding-001)
        """
        import numpy as np

        client = self._ensure_client()

        try:
            result = client.models.embed_content(
                model=self.model,
                contents=text,
            )
            # Track tokens (Gemini API doesn't return token count, estimate from text)
            if self.metrics_tracker and hasattr(
                self.metrics_tracker, "record_embedding_tokens"
            ):
                estimated_tokens = len(text) // 4  # Rough estimate: ~4 chars per token
                self.metrics_tracker.record_embedding_tokens(estimated_tokens)
            # Return as numpy array to match SentenceTransformer interface
            return np.array(result.embeddings[0].values)
        except Exception as e:
            print(f"Gemini embedding failed: {e}")
            raise


def create_embedder(provider: str = None, metrics_tracker=None):
    """Create embedder (Gemini by default, works with any LLM provider).

    Args:
        provider: "gemini" or "none" (default from EMBEDDING_PROVIDER env var or "gemini")
        metrics_tracker: Optional MetricsTracker for token tracking

    Returns:
        Embedder with .encode(text) -> np.array method, or None if provider is "none"
    """
    provider = (provider or os.getenv("EMBEDDING_PROVIDER", "gemini")).lower()

    if provider == "none":
        print("Embeddings disabled (EMBEDDING_PROVIDER=none)")
        return None

    if provider == "gemini":
        return GeminiEmbedder(metrics_tracker=metrics_tracker)

    print(f"Unknown embedding provider '{provider}', using Gemini")
    return GeminiEmbedder(metrics_tracker=metrics_tracker)
