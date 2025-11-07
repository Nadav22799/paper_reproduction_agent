"""LLM Factory - Automatically select and configure LLM based on available API keys."""

import os
import re
from typing import Optional, Any, List
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig


def clean_special_tokens(content: str) -> str:
    """Remove special tokens from content."""
    if not content:
        return content
    # Remove <think>...</think> blocks
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    # Remove <tool_call>...</tool_call> blocks
    content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL)
    # Clean up extra whitespace
    content = re.sub(r'\n\s*\n', '\n\n', content).strip()
    return content


class CleanContentChatOpenAI:
    """Wrapper for ChatOpenAI that cleans special tokens from content."""

    def __init__(self, base_llm):
        self.base_llm = base_llm
        # Expose all base_llm methods
        for attr in dir(base_llm):
            if not attr.startswith('_') and attr not in ['invoke', 'generate', '_generate']:
                setattr(self, attr, getattr(base_llm, attr))

    def invoke(self, input, config=None, **kwargs):
        """Invoke with content cleaning."""
        result = self.base_llm.invoke(input, config, **kwargs)
        if hasattr(result, 'content') and result.content:
            result.content = clean_special_tokens(result.content)
        return result

    def generate(self, messages, stop=None, callbacks=None, **kwargs):
        """Generate with content cleaning."""
        result = self.base_llm.generate(messages, stop, callbacks, **kwargs)
        for generations in result.generations:
            for gen in generations:
                if hasattr(gen, 'message') and hasattr(gen.message, 'content'):
                    if gen.message.content:
                        gen.message.content = clean_special_tokens(gen.message.content)
        return result

    def stream(self, input, config=None, **kwargs):
        """Stream with content cleaning."""
        for chunk in self.base_llm.stream(input, config, **kwargs):
            if hasattr(chunk, 'content') and chunk.content:
                chunk.content = clean_special_tokens(chunk.content)
            yield chunk

    def bind_tools(self, tools, **kwargs):
        """Bind tools - delegate to base LLM and wrap the result."""
        bound_llm = self.base_llm.bind_tools(tools, **kwargs)
        # Return a wrapped version so cleaning continues to work
        return CleanContentChatOpenAI(bound_llm)

    def with_structured_output(self, schema, **kwargs):
        """Support structured output - delegate to base LLM."""
        structured_llm = self.base_llm.with_structured_output(schema, **kwargs)
        return CleanContentChatOpenAI(structured_llm)

    def __getattr__(self, name):
        """Delegate unknown attributes to base_llm."""
        return getattr(self.base_llm, name)


def create_llm(temperature: float = 0.1):
    """
    Create LLM instance based on available API keys or local model.

    Priority order:
    1. Local model (if USE_LOCAL_LLM=true) - No API key needed, runs on your machine
    2. Google Gemini (if GOOGLE_API_KEY is provided) - Fast and supports function calling
    3. OpenAI (if OPENAI_API_KEY is provided) - Most capable
    4. Groq (if GROQ_API_KEY is provided) - Fast and free
    5. Anthropic (if ANTHROPIC_API_KEY is provided)

    Args:
        temperature: Temperature for LLM generation

    Returns:
        Configured LLM instance

    Raises:
        ValueError: If no API keys are available and local model not configured
    """
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
                print("⚠️  langchain-ollama not installed. Install with: pip install langchain-ollama")
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
                google_api_key=google_key,
                temperature=temperature,
            )
        except ImportError:
            print("⚠️  langchain-google-genai not installed. Install with: pip install langchain-google-genai")
            print("Falling back to other providers...")
        except Exception as e:
            print(f"⚠️  Failed to initialize Gemini: {e}")
            print(f"   Model: {os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')}")
            print("   Falling back to other providers...")

    # Check for OpenAI (prioritize if available)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        from langchain_openai import ChatOpenAI

        model = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
        base_url = os.getenv("OPENAI_API_BASE")  # Support local vLLM server

        # Make max_tokens configurable - GPT-OSS harmony format may need different limits
        max_tokens_config = os.getenv("LLM_MAX_TOKENS")
        max_tokens = int(max_tokens_config) if max_tokens_config else None

        llm_kwargs = {
            "openai_api_key": openai_key,
            "model": model,
            "temperature": temperature,
            "base_url": base_url,
        }

        if max_tokens is not None:
            llm_kwargs["max_tokens"] = max_tokens

        base_llm = ChatOpenAI(**llm_kwargs)

        if base_url:
            print(f"🖥️  Using Local vLLM Server (OpenAI-compatible) with {model}")
            print(f"   Base URL: {base_url}")
            if max_tokens is not None:
                print(f"   Max tokens: {max_tokens}")
            else:
                print(f"   Max tokens: None (using vLLM server defaults)")

            # Check if wrapping should be disabled
            disable_wrapper = os.getenv("DISABLE_VLLM_WRAPPER", "false").lower() == "true"

            if disable_wrapper:
                print(f"   VLLMMultiTurnWrapper DISABLED (DISABLE_VLLM_WRAPPER=true)")
                print(f"   Using base ChatOpenAI directly")
                return base_llm

            # Wrap with multi-turn wrapper for vLLM
            from .vllm_wrapper import VLLMMultiTurnWrapper
            debug_mode = os.getenv("VLLM_DEBUG", "false").lower() == "true"
            enable_recovery = os.getenv("VLLM_ENABLE_RECOVERY", "true").lower() == "true"

            if debug_mode:
                print(f"   Debug mode enabled (VLLM_DEBUG=true)")

            if enable_recovery:
                print(f"   Empty response recovery enabled (will attempt to fix empty responses after tool calls)")
            else:
                print(f"   Empty response recovery disabled (faster but may fail on empty responses)")

            print(f"   Wrapping with VLLMMultiTurnWrapper")
            return VLLMMultiTurnWrapper(base_llm, debug=debug_mode, enable_recovery=enable_recovery)
        else:
            print(f"🤖 Using OpenAI with {model}")
            return base_llm

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
            print("⚠️  langchain-groq not installed. Install with: pip install langchain-groq")
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


def create_specific_llm(provider: str, temperature: float = 0.1):
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
        raise ValueError(f"Unknown provider: {provider}. Use 'gemini', 'groq', 'openai', or 'anthropic'")
