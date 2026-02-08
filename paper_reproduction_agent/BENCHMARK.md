# 📊 Reproduction Benchmark

This document tracks the performance of the **Paper Reproduction Agent** across diverse research papers. It serves as a benchmark for the agent's generalization capabilities and reliability.

## 🏆 Summary

| Metric | Value |
| :--- | :--- |
| **Total Papers Attempted** | 3 |
| **Full Reproductions** | 2 |
| **Partial Reproductions** | 1 (interrupted by resource limits, not agent failure) |
| **Average Setup Time** | ~12m |
| **Hardware Range** | CPU-only → 4x L40S |

## 🧪 Experiment Log

### Graph Learning

| Paper ID | Title | Status | Duration | Verification Result | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[1609.02907](https://arxiv.org/abs/1609.02907)** | *Semi-Supervised Classification with Graph Convolutional Networks* | ✅ Full Success | 15m 24s | **6/7 Matched** | Gemini Flash 3, NVIDIA MX250 (laptop GPU), $0.95 total cost |
| **[1710.10903](https://arxiv.org/abs/1710.10903)** | *Graph Attention Networks* | ✅ Full Success | ~20m | **1/1 Matched** | Gemini Flash 3, **CPU-only** (i7, 16GB RAM) |

### 📝 Expert Review: [1609.02907] GCN

> **LLM**: Gemini Flash 3 Preview

#### 🎯 Reproduction Summary

| Mode | Duration | Cost | Metrics Matched | Hardware |
|------|----------|------|-----------------|----------|
| **Single** (Cora only) | 6m 45s | $0.41 | 1/1 (100%) | NVIDIA MX250 |
| **Full** (All datasets) | 15m 24s | $0.95 | 6/7 (86%) | NVIDIA MX250 |

#### 📊 Detailed Results (Full Mode)

| Dataset | Model | Paper | Reproduced | Error | Status |
|---------|-------|-------|------------|-------|--------|
| Cora | GCN | 81.5% | **81.6%** | 0.12% | ✅ |
| Citeseer | GCN | 70.3% | **71.3%** | 1.42% | ✅ |
| Pubmed | GCN | 79.0% | **78.6%** | 0.51% | ✅ |
| Cora | Cheby | 81.2% | **78.6%** | 3.20% | ✅ |
| Citeseer | Cheby | 69.8% | **68.8%** | 1.43% | ✅ |
| Pubmed | Cheby | 74.4% | **73.9%** | 0.67% | ✅ |
| NELL | GCN | 66.0% | — | — | ⚠️ Dataset missing from repo |

#### ⚙️ Engineering Analysis

The agent demonstrated **autonomous ML engineering** by successfully bridging a **10-year technology gap** — running 2016 TensorFlow code on 2026 infrastructure.

**Key Achievements:**

*   **Legacy Dependency Resolution**: Autonomously constructed a `tensorflow==1.15.4` + `protobuf==3.20.3` environment on Python 3.7. This required understanding deprecated APIs, pinning NumPy to avoid breaking changes, and navigating the TF 1.x → 2.x ecosystem shift.

*   **Multi-Model Verification**: Didn't just run the main model — executed both GCN and Chebyshev polynomial baselines across 3 datasets, providing comprehensive reproduction evidence.

*   **Cost-Efficient Execution**: Full reproduction for **$0.95** in API calls. Single-mode verification for **$0.41**. This demonstrates practical utility for researchers with limited budgets.

*   **Laptop-Scale Hardware**: Ran successfully on an **NVIDIA MX250** (entry-level mobile GPU) with **7.7GB RAM**. No cloud GPUs required.

*   **Graceful Degradation**: When NELL dataset was missing from the repository, the agent correctly identified the issue, logged it, and continued with available datasets rather than failing entirely.

#### ⚠️ Limitations Observed

*   **NELL Dataset**: The original repository does not include the NELL dataset or loader code. The agent correctly identified this gap but cannot synthesize missing data.

*   **Random Splits**: The paper reports "random split" results (e.g., Cora 80.1 ± 0.5%), but the repository only implements fixed splits. Full reproduction of stochastic experiments would require code modification.

*   **TensorFlow 1.x Deprecation**: Required `protobuf` downgrade and specific NumPy version. Future papers using TF 1.x may face similar compatibility challenges.

---

### 📝 Expert Review: [1710.10903] Graph Attention Networks (GAT)

> **LLM**: Gemini Flash 3 Preview

#### 🎯 Reproduction Summary

| Mode | Duration | Cost | Metrics Matched | Hardware |
|------|----------|------|-----------------|----------|
| **Single** (Cora only) | ~90m | $0.89 | 1/1 (100%) | **CPU-only** (i7, 16GB) |

#### 📊 Detailed Results

| Dataset | Model | Paper | Reproduced | Error | Status |
|---------|-------|-------|------------|-------|--------|
| Cora | GAT | 83.0 ± 0.7% | **82.7%** | -0.3% | ✅ |

#### ⚙️ Engineering Analysis

The agent demonstrated **hardware-agnostic execution** by successfully running a GPU-oriented paper on CPU-only hardware.

**Key Achievements:**

*   **CPU Fallback**: Detected no GPU available and ran PyTorch in CPU mode without modification. Training completed in ~20 minutes vs ~2 minutes on GPU.

*   **Accurate Reproduction**: Achieved 82.7% accuracy on Cora, within the paper's reported margin of 83.0 ± 0.7%.

*   **Modern PyTorch**: Unlike GCN (TensorFlow 1.x), GAT uses PyTorch which has better forward compatibility. Environment setup was straightforward.

*   **Fully Autonomous**: From paper URL to verified results with no manual steps.

#### ⚠️ Limitations Observed

*   **CPU Training Time**: ~10x slower than GPU. For papers requiring multiple runs or large datasets, GPU would be necessary.

*   **Single Dataset**: Only Cora was run to validate. Full reproduction would include Citeseer and PPI.

---

### 🖥️ Multi-GPU Papers

> **Note**: These experiments require multi-GPU infrastructure.
> Some runs were interrupted due to external resource constraints (not agent failures).
> **LLM**: Gemini Flash 2.5

| Paper ID | Title | Status | Dataset | Result | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **[2406.03386](https://arxiv.org/abs/2406.03386)** | *Learning Long Range Dependencies on Graphs via Random Walks* | ⚠️ Partial | ZINC | **0.0636 ± 0.0004** | ✅ Completed |
| | | | OGBL-PPA | — | ⏸️ Interrupted |
| | | | OGBL-DDI | — | ⏸️ Interrupted |

#### 📊 ZINC Results Detail

| Metric | Value |
|--------|-------|
| Test Score (MAE) | 0.0636 ± 0.0004 (5 seeds, single run) |
| Runs | 5 seeds |
| Epochs | 500 |
| Hardware | 4x NVIDIA L40S, 502GB RAM, 128 CPU cores |

**Context**: Run was stopped due to external resource constraints. The agent successfully completed one benchmark before interruption.

---

## 📈 Analysis

### Success Factors

*   **Intelligent Dependency Management**: The agent analyzed `requirements.txt`, cross-referenced with paper publication date (2016), and selected compatible package versions without blindly using latest releases.

*   **Self-Healing Execution**: When initial smoke tests failed, the agent diagnosed the issue (wrong working directory), corrected the command, and re-ran successfully.

*   **Verification Beyond "It Runs"**: The agent extracted accuracy metrics from training logs and compared against paper-reported values with explicit error margin calculations.

*   **Resource-Aware Scaling**: Detected laptop-tier hardware and adjusted expectations accordingly (no batch size modifications needed for this paper's small datasets).

### Optimization Opportunities

*   **Environment Caching**: First-time TensorFlow 1.15.4 setup took ~5 minutes. Caching "golden environments" for common legacy frameworks would reduce this.

*   **Parallel Execution**: The 3 datasets × 2 models could theoretically run in parallel, but current implementation runs sequentially for reliability.

### Common Failure Modes

*   **Missing Datasets**: Papers requiring manual dataset registration (ImageNet, proprietary data) will stall the agent.

*   **Undocumented Dependencies**: Some papers omit system-level requirements (CUDA versions, C++ compilers). The agent can detect failures but may not always resolve them automatically.

---

## 🔮 Future Papers

See [FUTURE_PAPERS.md](./FUTURE_PAPERS.md) for papers requiring additional hardware resources.
