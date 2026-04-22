# 🌌 NANO: Non-Abelian Network Optimization
### *Topological Density over Numerical Precision*

[![Version](https://img.shields.io/badge/Version-3.1--STABLE-blueviolet?style=for-the-badge)](https://huggingface.co/RthItalia)
[![License](https://img.shields.io/badge/License-Proprietary-red?style=for-the-badge)](LICENSE)
[![NANO-Native](https://img.shields.io/badge/Inference-NANO--Native-teal?style=for-the-badge)](E:\NANO\nano_native_inference_v4_golden.py)

**NANO** is a unified framework for extreme LLM compression (75%+) that rejects the "Precision Illusion" of standard 16-bit deployment. By treating neural networks as topologically dense manifolds rather than absolute numerical sets, NANO enables billion-parameter models to run on commodity hardware with minimal RAM signatures.

---

## 🚀 Key Innovations

### 🧠 CDR: Conceptual Density Reduction
NANO identifies the "Informational Ballast" within weights using **Norm Divergence**. Instead of uniform quantization, we vary bit-depth (8, 6, 4, 2) dynamically, physically pruning zero-entropy parameters.

### 🌐 Radial-Former: Geometric Encoding
We replace massive (1GB+) categorical embedding tables with **Phi-based Sinusoidal Geometry**. Token IDs are decomposed into 18-bit descriptors, mapped into a 12-dimensional continuous hidden space.

### ⚡ Native Bit-Logic Shell
The **NANO Direct Shell** executes matmul operations directly on bit-packed `uint8` buffers.
- **JIT Unpacking**: Weights stay packed until the moment of computation in L3 cache.
- **Thermal Shields**: High-energy normalization stabilized via FP32 RMSNorm patches.

---

## 📊 Performance Benchmarks (v3.1)

| Model Target | Baseline Size | NANO Size (Zip) | RAM Signature | Fidelity (Cosine) |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen-2.5-3B** | 12.0 GB | **799 MB** | 2.4 GB | 0.9906 |
| **Qwen-2.5-7B** | 15.0 GB | **891 MB** | 4.1 GB | 0.9906 |
| **Qwen-2.5-14B** | 28.0 GB | **1.48 GB** | 7.5 GB | 0.9884 |

---

## 🛠️ Quick Start

### 1. Requirements
```bash
pip install torch transformers accelerate bitsandbytes safetensors
```

### 2. Native Inference Entrypoint
Use the **Golden Shell** for the most efficient execution:

```python
from nano_native_inference_v4_golden import patch_nano_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load the lightweight base shell
tokenizer = AutoTokenizer.from_pretrained("RthItalia/NanoLLM-Qwen2.5-7B-v3.1")
model = AutoModelForCausalLM.from_pretrained("RthItalia/NanoLLM-Qwen2.5-7B-v3.1")

# 2. Inject NANO Topological Intelligence
model = patch_nano_model(model, "path/to/nano_topology.json", "path/to/radial_projection.pt")

# 3. Generate with Native Bit-Logic
inputs = tokenizer("The history of computers started with", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(out[0]))
```

---

## 📜 Scientific Paper
Read the full technical breakdown: [NANO: Topological Density and Emergent Weight Geometry](NANO_SCIENTIFIC_PAPER.md).

---

## 🤝 Project Status
NANO is currently at **RC1** (Release Candidate 1).  
Scaling to 70B parameter models is currently in testing.

---
© 2026 RthItalia — *Intelligence doesn't require precision. It requires structure.*
