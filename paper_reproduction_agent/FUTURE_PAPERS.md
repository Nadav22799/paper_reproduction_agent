# 🔮 Future Papers (Hardware Required)

Papers I plan to reproduce once I have access to appropriate hardware.

## 🖥️ Requires Multi-GPU Setup

### Graph Learning

| Paper ID | Title | Hardware Needed | Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **[2406.03386](https://arxiv.org/abs/2406.03386)** | *Learning Long Range Dependencies on Graphs via Random Walks* | 4x NVIDIA L40S (or equivalent) | ⚠️ **Partial** | ZINC ✅ (0.0636 ± 0.0004), OGBL datasets interrupted |

### NLP / Transformers

| Paper ID | Title | Hardware Needed | Notes |
| :--- | :--- | :--- | :--- |
| *TBD* | Large Language Model papers | Multi-GPU cluster | Most LLM papers require A100/H100 class hardware |

### Computer Vision

| Paper ID | Title | Hardware Needed | Notes |
| :--- | :--- | :--- | :--- |
| *TBD* | Vision Transformer variants | 1-2x A100 | ImageNet training typically requires high-end GPUs |

---

## ✅ Completed (Laptop Hardware)

| Paper ID | Title | Result | Hardware |
| :--- | :--- | :--- | :--- |
| **[1609.02907](https://arxiv.org/abs/1609.02907)** | *Semi-Supervised Classification with GCN* | 81.6% Cora (paper: 81.5%) | MX250 GPU |
| **[1710.10903](https://arxiv.org/abs/1710.10903)** | *Graph Attention Networks (GAT)* | 82.7% Cora (paper: 83.0 ± 0.7%) | **CPU-only** |

## 📋 Laptop-Friendly Papers (Queued)

Papers I can run on my current hardware (i7 + 16GB RAM + MX250):

| Paper ID | Title | Why Laptop-Friendly | Priority |
| :--- | :--- | :--- | :--- |
| **[1706.02216](https://arxiv.org/abs/1706.02216)** | *Inductive Representation Learning on Large Graphs (GraphSAGE)* | Scalable algorithm, works on CPU | 🔴 High |
| **[1301.3781](https://arxiv.org/abs/1301.3781)** | *Word2Vec* | CPU-optimized, small text corpora available | 🟡 Medium |
| **[1607.06450](https://arxiv.org/abs/1607.06450)** | *Layer Normalization* | Simple experiments, quick to run | 🟢 Low |

---

## 📝 Notes

- **Hardware estimation** is based on paper-reported training configurations
- Some papers may work on smaller hardware with reduced batch sizes or fewer epochs
- Cloud GPU options (Lambda Labs, RunPod, Colab Pro) can be used for one-off reproductions
