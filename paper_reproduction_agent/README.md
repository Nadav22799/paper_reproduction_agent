# Paper Reproduction Agent

A multi-agent system built with LangGraph for automatically reproducing and verifying academic paper implementations.

## Overview

This system uses specialized AI agents to:
1. **Analyze academic papers** - Extract algorithms, experimental setups, and results
2. **Search for existing code** - Find implementations on GitHub and Papers with Code
3. **Reproduce code from scratch** - Implement algorithms based on paper descriptions
4. **Verify results** - Run experiments and compare with paper's reported results
5. **Debug and fix issues** - Automatically fix broken implementations

## Architecture

The system consists of 5 specialized agents orchestrated by LangGraph:

```
┌─────────────────────┐
│  Paper Analyzer     │ - Extracts info from papers
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Code Searcher      │ - Finds existing implementations
└──────────┬──────────┘
           │
      ┌────┴────┐
      │ Router  │
      └────┬────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌─────────┐  ┌─────────────────┐
│ Use     │  │ Code Reproducer │ - Writes code from scratch
│ Existing│  └────────┬────────┘
└────┬────┘           │
     │                │
     └────────┬───────┘
              ▼
     ┌─────────────────┐
     │ Code Verifier   │ - Tests and validates
     └────────┬────────┘
              │
              ▼
     ┌─────────────────┐
     │ Code Debugger   │ - Fixes issues
     └─────────────────┘
```

## Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd paper_reproduction_agent
```

2. Install dependencies:
```bash
pip install -e .
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required API keys:
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` - For LLM access
- `GITHUB_TOKEN` - For GitHub API access (optional but recommended)

## Quick Start

### Basic Usage

```python
from src.orchestrator import PaperReproductionOrchestrator

# Initialize orchestrator
orchestrator = PaperReproductionOrchestrator()

# Reproduce a paper from arXiv
result = orchestrator.run("arxiv:1706.03762")

# Check results
print(result['report'])
```

### Using Individual Agents

```python
from src.agents.paper_analyzer import PaperAnalyzerAgent
from src.agents.code_searcher import CodeSearcherAgent

# Analyze a paper
analyzer = PaperAnalyzerAgent()
analysis = analyzer.analyze_paper("arxiv:2010.11929")

# Search for implementations
searcher = CodeSearcherAgent()
results = searcher.search_implementations(
    paper_title="Vision Transformer",
    paper_keywords=["transformer", "vision", "image classification"]
)
```

## Agents

### 1. Paper Analyzer Agent

Extracts information from academic papers:
- Paper metadata (title, authors, abstract)
- Algorithm descriptions and pseudocode
- Experimental setup (datasets, metrics, hyperparameters)
- Reported results
- Code repository references

**Tools:**
- `fetch_arxiv_paper` - Download papers from arXiv
- `extract_algorithm_pseudocode` - Find algorithm descriptions
- `extract_experimental_setup` - Extract experiment details
- `extract_results_tables` - Parse results tables
- `extract_code_references` - Find repository URLs

### 2. Code Searcher Agent

Finds existing implementations:
- Searches GitHub repositories
- Searches Papers with Code database
- Evaluates repository quality
- Identifies official implementations

**Tools:**
- `search_github_repos` - Search GitHub
- `search_papers_with_code` - Query Papers with Code
- `get_repo_contents` - Browse repository files
- `clone_repository` - Clone repos locally

### 3. Code Reproducer Agent

Writes code from scratch:
- Creates implementation plan
- Implements algorithms from descriptions
- Writes training scripts
- Creates evaluation code
- Generates documentation

**Tools:**
- `create_python_file` - Write code files
- `check_python_syntax` - Validate syntax
- `run_linter` - Check code quality

### 4. Code Verifier Agent

Tests and validates implementations:
- Sets up execution environment
- Installs dependencies
- Runs code and experiments
- Compares results with paper
- Creates verification reports

**Tools:**
- `execute_python_script` - Run code
- `install_dependencies` - Set up environment
- `run_pytest` - Execute tests
- `capture_execution_metrics` - Measure performance
- `compare_outputs` - Validate results

### 5. Code Debugger Agent

Fixes broken implementations:
- Analyzes error messages
- Identifies root causes
- Applies fixes systematically
- Verifies fixes work
- Iterative debugging

**Tools:**
- All code execution tools
- Error analysis capabilities
- Automated fixing

## Examples

See `example.py` for detailed examples:

```bash
python example.py
```

### Example 1: Reproduce from arXiv

```python
orchestrator = PaperReproductionOrchestrator()
result = orchestrator.run("arxiv:1706.03762")  # Attention Is All You Need
```

### Example 2: Verify Existing Code

```python
from src.agents.code_verifier import CodeVerifierAgent

verifier = CodeVerifierAgent()
verification = verifier.verify_implementation(
    code_path="./my_implementation",
    paper_results={"accuracy": 0.95}
)
```

### Example 3: Search for Implementations

```python
from src.agents.code_searcher import CodeSearcherAgent

searcher = CodeSearcherAgent()
results = searcher.search_implementations("BERT: Pre-training of Deep Bidirectional Transformers")
```

## Workflow

The complete workflow:

1. **Paper Analysis**
   - Fetches paper from arXiv or reads local PDF
   - Extracts algorithms, methods, and experimental setup
   - Identifies reported results

2. **Code Search**
   - Searches for existing implementations
   - Evaluates repository quality
   - Decides: use existing or create new

3. **Implementation** (if needed)
   - Creates implementation plan
   - Writes code from scratch
   - Generates training/evaluation scripts

4. **Verification**
   - Sets up environment
   - Runs experiments
   - Compares results with paper

5. **Debugging** (if needed)
   - Identifies errors
   - Applies fixes
   - Re-runs verification
   - Iterates until success or max attempts

6. **Report Generation**
   - Comprehensive reproduction report
   - Status of each step
   - Final results

## Configuration

### Custom LLM

```python
from langchain_openai import ChatOpenAI

custom_llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0.0
)

orchestrator = PaperReproductionOrchestrator(llm=custom_llm)
```

### Using Anthropic Claude

```python
from langchain_anthropic import ChatAnthropic

claude_llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0.1
)

orchestrator = PaperReproductionOrchestrator(llm=claude_llm)
```

## Project Structure

```
paper_reproduction_agent/
├── src/
│   ├── agents/
│   │   ├── paper_analyzer.py      # Paper analysis agent
│   │   ├── code_searcher.py       # Code search agent
│   │   ├── code_reproducer.py     # Code reproduction agent
│   │   ├── code_verifier.py       # Verification agent
│   │   └── code_debugger.py       # Debugging agent
│   ├── tools/
│   │   ├── paper_tools.py         # Paper analysis tools
│   │   ├── code_search_tools.py   # Code search tools
│   │   └── code_execution_tools.py # Execution tools
│   ├── orchestrator.py            # LangGraph workflow
│   └── __init__.py
├── example.py                      # Usage examples
├── pyproject.toml                  # Dependencies
├── .env.example                    # Environment template
└── README.md                       # This file
```

## Features

✅ **Multi-agent architecture** - Specialized agents for each task
✅ **LangGraph orchestration** - Intelligent workflow routing
✅ **Comprehensive tools** - 20+ tools for paper analysis and code execution
✅ **Automatic debugging** - Self-healing code repair
✅ **Result verification** - Validates against paper claims
✅ **Flexible inputs** - arXiv, PDF, or paper text
✅ **Multiple LLM support** - OpenAI, Anthropic, or custom

## Limitations

- Requires high-quality paper descriptions for reproduction
- Complex papers may need manual intervention
- Some experiments require specific hardware (GPUs, datasets)
- API rate limits may affect performance

## Contributing

Contributions welcome! Areas for improvement:
- Additional tool implementations
- Support for more paper formats
- Enhanced debugging strategies
- Better result comparison metrics
- Support for more ML frameworks

## License

MIT License

## Acknowledgments

Built following the [Hugging Face Agents Course](https://huggingface.co/learn/agents-course) using LangGraph.

## Support

For issues and questions:
- Open an issue on GitHub
- Check the examples in `example.py`
- Review the agent documentation

---

**Happy Paper Reproduction! 🎉**
