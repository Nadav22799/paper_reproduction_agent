"""LLM Factory - Automatically select and configure LLM based on available API keys.

Simplified version that treats all providers (including local vLLM) the same way.
No special wrappers needed - just configure vLLM server with tool calling support.

To use local vLLM:
1. Start vLLM: vllm serve MODEL --enable-auto-tool-choice --tool-call-parser hermes
2. Set in .env:
   - OPENAI_API_KEY=not-needed
   - OPENAI_API_BASE=http://localhost:8000/v1
   - LLM_MODEL=your-model-name
"""

import os


def create_llm(temperature: float = 0.3, include_thoughts: bool = True):
    """
    Create LLM instance based on available API keys or local model.

    Priority order:
    1. Local model (if USE_LOCAL_LLM=true) - No API key needed, runs on your machine
    2. Google Gemini (if GOOGLE_API_KEY is provided) - Fast and supports function calling
    3. OpenAI (if OPENAI_API_KEY is provided) - Most capable
    4. Groq (if GROQ_API_KEY is provided) - Fast and free
    5. Anthropic (if ANTHROPIC_API_KEY is provided)

    Args:
        temperature: Temperature for LLM generation (defaults to env var or 0.3)

    Returns:
        Configured LLM instance

    Raises:
        ValueError: If no API keys are available and local model not configured
    """
    # Load temperature from env if not explicitly provided (or if default)
    if temperature == 0.3:
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    # Check for local LLM first (highest priority - no rate limits!)
    use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
    if use_local:
        local_backend = os.getenv("LOCAL_LLM_BACKEND", "ollama").lower()

        if local_backend == "ollama":
            try:
                from langchain_ollama import ChatOllama

                model = os.getenv("LOCAL_LLM_MODEL", "gemma2:2b")
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

                print(f"🖥️  Using Local Ollama with {model}")
                print(f"   Base URL: {base_url}")

                return ChatOllama(
                    model=model,
                    temperature=temperature,
                    base_url=base_url,
                )
            except ImportError:
                print(
                    "⚠️  langchain-ollama not installed. Install with: pip install langchain-ollama"
                )
                print("⚠️  Also ensure Ollama is installed: https://ollama.com/download")
                print("Falling back to API providers...")

        elif local_backend == "huggingface":
            try:
                from langchain_community.llms import VLLM

                model_name = os.getenv("LOCAL_LLM_MODEL", "google/gemma-3-270m")

                print(f"🖥️  Using Local HuggingFace model via vLLM: {model_name}")
                print("   (This may take a while to download on first run)")
                print("   vLLM provides better performance and tool calling support")

                # Check for HuggingFace token in .env
                hf_token = os.getenv("HUGGINGFACE_TOKEN")
                if hf_token:
                    print("   Using HuggingFace token from .env")
                    # Set token as environment variable for vLLM
                    os.environ["HF_TOKEN"] = hf_token
                else:
                    print("   No HUGGINGFACE_TOKEN in .env - using cached credentials")

                # vLLM parameters
                vllm_kwargs = {
                    "model": model_name,
                    "trust_remote_code": True,
                    "max_model_len": 2048,  # Context length
                    "temperature": temperature,
                    "gpu_memory_utilization": 0.8,  # Use 80% of GPU memory
                    "dtype": "half",  # Use float16 for better performance
                }

                # If no GPU or incompatible GPU, use CPU
                import torch

                if not torch.cuda.is_available():
                    print("   No compatible GPU detected, using CPU (will be slower)")
                    vllm_kwargs["device"] = "cpu"

                return VLLM(**vllm_kwargs)
            except ImportError as e:
                print(f"⚠️  HuggingFace dependencies not installed: {e}")
                print("Install with: pip install transformers torch accelerate")
                print("Falling back to API providers...")
            except Exception as e:
                print(f"⚠️  Error loading local model: {e}")
                print("Falling back to API providers...")

    # Check for Google Gemini (native API with proper function calling support)
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if google_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

            print(f"🤖 Using Google Gemini with {model}")
            print("   Native API (supports function calling)")

            return ChatGoogleGenerativeAI(
                model=model,
                include_thoughts=include_thoughts,
                google_api_key=google_key,
                temperature=temperature,
            )
        except ImportError:
            print(
                "⚠️  langchain-google-genai not installed. Install with: pip install langchain-google-genai"
            )
            print("Falling back to other providers...")
        except Exception as e:
            print(f"⚠️  Failed to initialize Gemini: {e}")
            print(f"   Model: {os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')}")
            print("   Falling back to other providers...")

    # Check for OpenAI or local vLLM (OpenAI-compatible)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        from langchain_openai import ChatOpenAI

        model = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
        base_url = os.getenv("OPENAI_API_BASE")  # Support local vLLM server

        # Make max_tokens configurable
        max_tokens_config = os.getenv("LLM_MAX_TOKENS")
        max_tokens = int(max_tokens_config) if max_tokens_config else None

        llm_kwargs = {
            "openai_api_key": openai_key,
            "model": model,
            "temperature": temperature,
        }

        if base_url:
            llm_kwargs["base_url"] = base_url

        if max_tokens is not None:
            llm_kwargs["max_tokens"] = max_tokens

        llm = ChatOpenAI(**llm_kwargs)

        # Log which provider we're using
        if base_url:
            print(f"🖥️  Using Local vLLM Server with {model}")
            print(f"   Base URL: {base_url}")
            print("   Works like OpenAI API - no wrapper needed!")
            print(
                "   Make sure vLLM is started with: --enable-auto-tool-choice --tool-call-parser hermes"
            )
        else:
            print(f"🤖 Using OpenAI with {model}")

        return llm

    # Check for Groq second (fast and free alternative)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            from langchain_groq import ChatGroq

            model = os.getenv("GROQ_MODEL", "llama3-70b-8192")

            print(f"🚀 Using Groq with {model}")

            return ChatGroq(
                groq_api_key=groq_key,
                model_name=model,
                temperature=temperature,
            )
        except ImportError:
            print(
                "⚠️  langchain-groq not installed. Install with: pip install langchain-groq"
            )
            print("Falling back to other providers...")

    # Check for Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        from langchain_anthropic import ChatAnthropic

        model = os.getenv("LLM_MODEL", "claude-3-sonnet-20240229")

        print(f"🧠 Using Anthropic with {model}")

        return ChatAnthropic(
            anthropic_api_key=anthropic_key,
            model=model,
            temperature=temperature,
        )

    # No API keys found
    raise ValueError(
        "No LLM configured! Please choose one of:\n"
        "\n"
        "Option 1 - Use Local LLM (No API key needed, no rate limits!):\n"
        "  Set in .env: USE_LOCAL_LLM=true\n"
        "  Backend options:\n"
        "    - LOCAL_LLM_BACKEND=ollama (easiest, recommended)\n"
        "    - LOCAL_LLM_BACKEND=huggingface (raw transformers)\n"
        "  Model: LOCAL_LLM_MODEL=gemma2:2b (or google/gemma-2-2b-it for HF)\n"
        "\n"
        "Option 2 - Use API Provider (needs API key):\n"
        "  - GOOGLE_API_KEY or GEMINI_API_KEY (Gemini API, supports function calling)\n"
        "  - OPENAI_API_KEY (most capable)\n"
        "  - GROQ_API_KEY (fast and free, but has rate limits)\n"
        "  - ANTHROPIC_API_KEY\n"
        "\n"
        "Add your choice to the .env file."
    )


def get_available_providers() -> list:
    """
    Get list of available LLM providers based on API keys.

    Returns:
        List of available provider names
    """
    providers = []

    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        providers.append("gemini")
    if os.getenv("GROQ_API_KEY"):
        providers.append("groq")
    if os.getenv("OPENAI_API_KEY"):
        providers.append("openai")
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append("anthropic")

    return providers


def create_specific_llm(provider: str, temperature: float = 0.1, include_thoughts: bool = True):
    """
    Create LLM for a specific provider.

    Args:
        provider: Provider name ("gemini", "groq", "openai", or "anthropic")
        temperature: Temperature for generation

    Returns:
        Configured LLM instance

    Raises:
        ValueError: If provider not available or unknown
    """
    provider = provider.lower()

    if provider == "gemini":
        google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not google_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set")

        from langchain_google_genai import ChatGoogleGenerativeAI

        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=google_key,
            temperature=temperature,
            include_thoughts=include_thoughts,
        )

    elif provider == "groq":
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY not set")

        from langchain_groq import ChatGroq

        model = os.getenv("GROQ_MODEL", "llama3-70b-8192")

        return ChatGroq(
            groq_api_key=groq_key,
            model_name=model,
            temperature=temperature,
        )

    elif provider == "openai":
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not set")

        from langchain_openai import ChatOpenAI

        model = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")

        return ChatOpenAI(
            openai_api_key=openai_key,
            model=model,
            temperature=temperature,
        )

    elif provider == "anthropic":
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        from langchain_anthropic import ChatAnthropic

        model = os.getenv("LLM_MODEL", "claude-3-sonnet-20240229")

        return ChatAnthropic(
            anthropic_api_key=anthropic_key,
            model=model,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unknown provider: {provider}. Use 'gemini', 'groq', 'openai', or 'anthropic'"
        )


# === EMBEDDING PROVIDERS ===

class GeminiEmbedder:
    """Gemini API-based embedder matching SentenceTransformer interface.

    Uses Google's text-embedding-004 model for fast, API-based embeddings.
    No local model loading required - saves ~100MB RAM.
    """

    def __init__(self, model: str = None, metrics_tracker=None):
        """Initialize Gemini embedder.

        Args:
            model: Embedding model name (default: from EMBEDDING_MODEL env var or text-embedding-004)
            metrics_tracker: Optional MetricsTracker for token tracking
        """
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-004")
        self._client = None
        self.metrics_tracker = metrics_tracker

    def _ensure_client(self):
        """Lazy initialize the Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set")
                genai.configure(api_key=api_key)
                self._client = genai
                print(f"🔗 Gemini embedder initialized with {self.model}")
            except ImportError:
                raise ImportError("google-generativeai not installed. Install with: pip install google-generativeai")
        return self._client

    def encode(self, text: str):
        """Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            List[float]: Embedding vector (768 dimensions for text-embedding-004)
        """
        import numpy as np
        genai = self._ensure_client()

        try:
            result = genai.embed_content(
                model=f"models/{self.model}",
                content=text,
                task_type="retrieval_document"
            )
            # Track tokens (Gemini API doesn't return token count, estimate from text)
            if self.metrics_tracker and hasattr(self.metrics_tracker, "record_embedding_tokens"):
                estimated_tokens = len(text) // 4  # Rough estimate: ~4 chars per token
                self.metrics_tracker.record_embedding_tokens(estimated_tokens)
            # Return as numpy array to match SentenceTransformer interface
            return np.array(result['embedding'])
        except Exception as e:
            print(f"⚠️  Gemini embedding failed: {e}")
            raise


class OpenAIEmbedder:
    """OpenAI API-based embedder matching SentenceTransformer interface."""

    def __init__(self, model: str = None, metrics_tracker=None):
        """Initialize OpenAI embedder.

        Args:
            model: Embedding model name (default: text-embedding-3-small)
            metrics_tracker: Optional MetricsTracker for token tracking
        """
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self._client = None
        self.metrics_tracker = metrics_tracker

    def _ensure_client(self):
        """Lazy initialize the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not set")
                self._client = OpenAI(api_key=api_key)
                print(f"🔗 OpenAI embedder initialized with {self.model}")
            except ImportError:
                raise ImportError("openai not installed. Install with: pip install openai")
        return self._client

    def encode(self, text: str):
        """Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            List[float]: Embedding vector
        """
        import numpy as np
        client = self._ensure_client()

        try:
            response = client.embeddings.create(
                model=self.model,
                input=text
            )
            # Track tokens using actual usage from response
            if self.metrics_tracker and hasattr(self.metrics_tracker, "record_embedding_tokens"):
                if hasattr(response, "usage") and response.usage:
                    self.metrics_tracker.record_embedding_tokens(response.usage.total_tokens)
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"⚠️  OpenAI embedding failed: {e}")
            raise


def create_embedder(provider: str = None, metrics_tracker=None):
    """Create embedder based on provider setting.

    Args:
        provider: "gemini", "openai", "local", or "none"
                  Defaults to EMBEDDING_PROVIDER env var or "gemini"
        metrics_tracker: Optional MetricsTracker for token tracking

    Returns:
        Embedder with .encode(text) -> np.array method, or None if provider is "none"
    """
    provider = (provider or os.getenv("EMBEDDING_PROVIDER", "gemini")).lower()

    if provider == "none":
        print("📭 Embeddings disabled (EMBEDDING_PROVIDER=none)")
        return None

    if provider == "gemini":
        return GeminiEmbedder(metrics_tracker=metrics_tracker)

    if provider == "openai":
        return OpenAIEmbedder(metrics_tracker=metrics_tracker)

    if provider == "local":
        try:
            from sentence_transformers import SentenceTransformer
            model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            print(f"🔗 Loading local embedding model: {model}")
            # Note: SentenceTransformer doesn't support metrics_tracker natively
            return SentenceTransformer(model)
        except ImportError:
            print("⚠️  sentence-transformers not installed, falling back to Gemini")
            return GeminiEmbedder(metrics_tracker=metrics_tracker)

    print(f"⚠️  Unknown embedding provider '{provider}', using Gemini")
    return GeminiEmbedder(metrics_tracker=metrics_tracker)
