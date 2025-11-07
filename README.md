# Paper Reproduction Agent System

An intelligent multi-agent system for automatically reproducing and verifying academic paper implementations using LangGraph and LangChain.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-green.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

This system automates the challenging task of reproducing machine learning research papers by:

1. **Analyzing Papers** - Extracting algorithms, experimental setups, and results from academic papers
2. **Finding Implementations** - Searching GitHub and Papers with Code for existing implementations
3. **Setting Up Environments** - Smart dependency installation with automatic fallback strategies
4. **Preparing Datasets** - Autonomous dataset discovery and download
5. **Running Experiments** - Finding and executing the quickest experiments
6. **Verifying Results** - Comparing experimental outputs with paper claims
7. **Debugging Issues** - Self-healing code repair when experiments fail

---

## 🏗️ Project Structure

```
Agents/
├── paper_reproduction_agent/    # Main agent system
│   ├── src/
│   │   ├── agents/             # Specialized agents
│   │   │   ├── paper_analyzer.py
│   │   │   ├── code_searcher.py
│   │   │   ├── environment_setup.py
│   │   │   ├── dataset_manager.py
│   │   │   ├── experiment_runner.py
│   │   │   ├── metrics_extractor.py
│   │   │   └── code_debugger.py
│   │   ├── tools/              # Agent tools
│   │   │   ├── paper_tools.py
│   │   │   ├── code_search_tools.py
│   │   │   └── code_execution_tools.py
│   │   ├── utils/              # Utilities
│   │   └── orchestrator.py     # LangGraph workflow
│   ├── run.py                  # Quick run script
│   └── README.md               # Detailed documentation
│
├── docs/                        # Documentation
│   ├── QUICKSTART.md           # Getting started guide
│   ├── API_KEYS.md             # API setup instructions
│   ├── GEMINI_SETUP.md         # Google Gemini configuration
│   ├── LOCAL_LLM_SETUP.md      # Local LLM setup (vLLM, Ollama)
│   ├── codebase_analysis.md    # Architecture deep-dive
│   ├── issues_and_details.md   # Known issues and improvements
│   └── INDEX.md                # Documentation index
│
├── tests/                       # Test suite
├── cloned_repo/                # Runtime directory (git-ignored)
├── downloads/                  # Downloaded papers (git-ignored)
└── logs/                       # Execution logs (git-ignored)
```

---

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Agents
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys:**

   Create a `.env` file in the `paper_reproduction_agent/` directory:
   ```bash
   # Choose one or more LLM providers
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   GOOGLE_API_KEY=your_key_here

   # Optional: GitHub token for API access
   GITHUB_TOKEN=your_token_here
   ```

   See [docs/API_KEYS.md](docs/API_KEYS.md) for detailed setup instructions.

### Basic Usage

**Reproduce a paper from arXiv:**

```bash
cd paper_reproduction_agent
python run.py 2106.09685
```

**Using Python API:**

```python
from paper_reproduction_agent.src.orchestrator import PaperReproductionOrchestrator

orchestrator = PaperReproductionOrchestrator()
result = orchestrator.run("2106.09685")  # arXiv ID

print(result['report'])
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](docs/QUICKSTART.md) | Step-by-step getting started guide |
| [paper_reproduction_agent/README.md](paper_reproduction_agent/README.md) | Complete agent system documentation |
| [API_KEYS.md](docs/API_KEYS.md) | API key setup for different LLM providers |
| [GEMINI_SETUP.md](docs/GEMINI_SETUP.md) | Google Gemini configuration |
| [LOCAL_LLM_SETUP.md](docs/LOCAL_LLM_SETUP.md) | Setup for vLLM and Ollama |
| [codebase_analysis.md](docs/codebase_analysis.md) | Detailed architecture analysis |
| [issues_and_details.md](docs/issues_and_details.md) | Known issues and improvement areas |

---

## 🤖 Multi-Agent Architecture

The system uses 7 specialized agents orchestrated by LangGraph:

```
┌─────────────────┐
│ Paper Analyzer  │ - Extracts paper information
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Code Searcher   │ - Finds implementations (GitHub, Papers with Code)
└────────┬────────┘
         │
    ┌────┴────┐
    │ Router  │ - Decides: use existing or create new
    └────┬────┘
         │
         ▼
┌─────────────────┐
│ Environment     │ - Smart dependency installation
│ Setup Agent     │ - Automatic version fallback
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Dataset Manager │ - Finds and downloads datasets
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Experiment      │ - Runs quickest experiments
│ Runner          │ - Captures results
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Metrics         │ - Compares with paper results
│ Extractor       │ - Validates claims
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Code Debugger   │ - Fixes issues (max 3 attempts)
└─────────────────┘
```

---

## ✨ Key Features

### 🎯 Smart Environment Setup
- Automatic Python version compatibility checking
- Intelligent dependency resolution with fallback strategies
- Virtual environment isolation
- Support for requirements.txt, setup.py, and pyproject.toml

### 🔍 Autonomous Dataset Management
- Searches README for download instructions
- Discovers download scripts automatically
- Executes dataset preparation commands

### ⚡ Quick Experiment Execution
- Prioritizes demo scripts over full training
- Avoids complex distributed training commands
- Finds minimal working examples
- Captures all output for analysis

### 🛠️ Self-Healing Debugging
- Analyzes error messages
- Applies systematic fixes
- Iterative debugging (up to 3 attempts)
- Verification after each fix

### 🔌 Multiple LLM Support
- **OpenAI** (GPT-4, GPT-3.5)
- **Anthropic** (Claude 3.5 Sonnet, Opus, Haiku)
- **Google** (Gemini 2.5 Flash, Pro)
- **Local** (vLLM, Ollama)

---

## 📊 Example: Reproducing LoRA

```bash
python run.py 2106.09685  # LoRA: Low-Rank Adaptation of Large Language Models
```

**What happens:**
1. ✅ Fetches paper from arXiv
2. ✅ Finds official implementation on GitHub
3. ✅ Clones repository to `cloned_repo/`
4. ✅ Creates virtual environment with Python 3.12
5. ✅ Installs dependencies (loralib)
6. ✅ Searches for datasets
7. ✅ Runs experiments
8. 📊 Generates comprehensive report

---

## 🔧 Configuration

### Choosing LLM Provider

Edit `paper_reproduction_agent/.env`:

```bash
# Use OpenAI GPT-4
DEFAULT_LLM=openai
OPENAI_MODEL=gpt-4-turbo-preview

# Use Anthropic Claude
DEFAULT_LLM=anthropic
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Use Google Gemini (faster, cheaper)
DEFAULT_LLM=google
GOOGLE_MODEL=gemini-2.5-flash
```

### Runtime Directories

| Directory | Purpose | Git Status |
|-----------|---------|------------|
| `cloned_repo/` | Cloned paper repositories | ❌ Ignored (temporary) |
| `downloads/` | Downloaded papers/datasets | ❌ Ignored (temporary) |
| `logs/` | Execution logs | ❌ Ignored (temporary) |
| `paper_reproduction_agent/` | Main codebase | ✅ Tracked |
| `docs/` | Documentation | ✅ Tracked |
| `tests/` | Test suite | ✅ Tracked |

**Note:** `.gitkeep` files preserve directory structure while keeping contents ignored.

---

## 🐛 Known Issues & Improvements

See [docs/issues_and_details.md](docs/issues_and_details.md) for:

- **DatasetManagerAgent** - Missing error handling (line 71)
- **ExperimentRunnerAgent** - Better filtering for complex commands
- **Result Parsing** - Reduce reliance on keyword matching
- **Python Version Management** - Simplify complex parsing logic

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add more robust error handling in agents
- [ ] Improve experiment selection heuristics
- [ ] Better metric extraction from paper tables
- [ ] Support for more ML frameworks (JAX, TensorFlow)
- [ ] Enhanced debugging strategies
- [ ] Unit tests for all agents

---

## 📝 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- Built following the [Hugging Face Agents Course](https://huggingface.co/learn/agents-course)
- Powered by [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain)
- Inspired by the need for reproducible research in ML

---

## 📞 Support

- **Documentation:** See [docs/INDEX.md](docs/INDEX.md) for full documentation index
- **Issues:** Open an issue on GitHub
- **Quick Help:** Check [docs/QUICKSTART.md](docs/QUICKSTART.md)

---

**Happy Paper Reproduction! 🎉**
