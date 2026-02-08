# 🤖 AI Agents Portfolio

A collection of autonomous AI agent systems focused on scientific rigor and robust engineering.

## 🚀 Featured Agent: Paper Reproduction Agent

**[View Full Documentation](paper_reproduction_agent/README.md)**

A specialized agent that reads research papers, clones their code, sets up environments, and scientifically verifies their results — autonomously.

### Why This Matters for AI Engineering

Unlike prompt-based wrappers, this is a **stateful multi-agent system** that handles real-world complexity:
*   **Self-Healing**: Detects and resolves broken dependencies (e.g., TF 1.x on modern Python, `numpy` version conflicts).
*   **Hardware Aware**: Detects GPUs (e.g., L40S clusters, laptop MX250) and scales training strategies accordingly.
*   **Scientific Verification**: Extracts expected metrics from the paper PDF and compares reproduced results within tolerance margins.

### 📊 Benchmark Results

| Paper | Challenge | Status | Key Result | Cost |
| :--- | :--- | :--- | :--- | :--- |
| **GCN** (2016) | Legacy TF 1.x | ✅ Full | 6/7 metrics matched | $0.95 |
| **GAT** (2017) | CPU-only | ✅ Full | 82.7% (paper: 83.0 ± 0.7%) | $0.89 |
| **NeuralWalker** (2024) | Multi-GPU scale | ⚠️ Partial | ZINC: 0.0636 ± 0.0004 | — |

*See [BENCHMARK.md](paper_reproduction_agent/BENCHMARK.md) for detailed analysis.*

---

## 🛠️ Repository Structure

```
Agents/
├── paper_reproduction_agent/    # Flagship Project
│   ├── src/                     # Core Agent Logic (LangGraph)
│   ├── BENCHMARK.md             # Scientific verification log
│   └── ARCHITECTURE.md          # Technical deep-dive
│
└── ... (Future Agents)
```

## ⚡ Quick Start

```bash
cd paper_reproduction_agent
pip install -e .
python src/cli.py reproduce 1609.02907
```

---

## 👨‍💻 About
Built by Nadav Cohen. Focused on agentic infrastructure and reproducible ML.
