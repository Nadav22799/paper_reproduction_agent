# Installation Guide

## Quick Install

```bash
# From the Agents directory
pip install -e paper_reproduction_agent
```

## Dependencies

### Core Dependencies
- LangGraph & LangChain ecosystem
- PDF processing (PyPDF2)
- Paper APIs (arxiv)
- ML libraries (torch, numpy, pandas, scikit-learn)

### Google Gemini Integration (New!)
Required for the `search_error_solution` tool that helps debug errors:
- `google-generativeai>=0.8.0` - Gemini API client
- `google-ai-generativelanguage>=0.7.0` - Core language service
- `langchain-google-genai>=3.0.0` - LangChain integration

### Environment Variables

Set one or more LLM provider API keys:

```bash
# Google Gemini (for search_error_solution tool)
export GOOGLE_API_KEY="your-key"
export GEMINI_API_KEY="your-key"  # Alternative

# OpenAI
export OPENAI_API_KEY="your-key"

# Anthropic
export ANTHROPIC_API_KEY="your-key"

# Groq
export GROQ_API_KEY="your-key"
```

## Troubleshooting

### Dependency Conflicts

If you see:
```
langchain-google-genai 3.0.1 requires google-ai-generativelanguage>=0.7.0,
but you have google-ai-generativelanguage 0.6.15
```

Fix with:
```bash
pip install --upgrade google-ai-generativelanguage
```

### Missing google-generativeai

If you get:
```
ModuleNotFoundError: No module named 'google.generativeai'
```

Install:
```bash
pip install google-generativeai
```

## Verify Installation

```python
# Test Google Gemini integration
import google.generativeai as genai
print("✅ Google Generative AI installed")

# Test paper reproduction tools
from paper_reproduction_agent.src.orchestrator import PaperReproductionOrchestrator
print("✅ Paper Reproduction Agent ready")
```

## Development Setup

```bash
pip install -e "paper_reproduction_agent[dev]"
```

This includes:
- pytest (testing)
- black (code formatting)
- ruff (linting)
