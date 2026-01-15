# 🤖 AI Agents Portfolio

An experimental laboratory for building autonomous, self-healing, and scientifically rigorous AI agents.

## 🚀 Featured Agent: Paper Reproduction Agent

**[ View Full Documentation ](paper_reproduction_agent/README.md)**

A specialized specific agent designed to read research papers, verify their claims, and write code to reproduce them from scratch.

### 🌟 Why this matters for AI Engineering
Most "Agents" are just chatbots. This is a **State Machine** that handles the "messiness" of the real world:
*   **Self-Healing**: Automatically detects and fixes broken dependencies (e.g. `numpy` version conflicts).
*   **Hardware Aware**: Detects GPUs (e.g. L40S clusters) and scales training strategies accordingly.
*   **Scientific Audit**: Parses the PDF to find the "Gold Standard" result and verifies reproduction within significant margins.

### 📊 Benchmark (NeuralWalker Paper)
| Metric | Result | Analysis |
| :--- | :--- | :--- |
| **Stability** | $\pm$ 0.0004 | High precision across 5 seeds. |
| **Setup** | Autonomous | Self-corrected `sed` syntax errors. |

---

## 🛠️ Repository Structure

```
Agents/
├── paper_reproduction_agent/    # 🏆 Flagship Project
│   ├── src/                     # Core Agent Logic (LangGraph)
│   ├── examples/                # Run scripts (basic_run.py)
│   ├── docs/                    # Architecture & Marketing Kits
│   └── BENCHMARK.md             # Scientific Verification Log
│
└── ... (Future Agents)
```

## ⚡ Quick Start

To run the Paper Reproduction Agent:

```bash
cd paper_reproduction_agent
pip install -e .
python examples/basic_run.py "2406.03386"
```

---

## 👨‍💻 About
Built by Nadav Cohen. Focused on Agentic Infrastructure and Reproducible ML.
