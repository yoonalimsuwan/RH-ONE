``
# RH ONE — Fully Differentiable Riemann Hypothesis & SSC Research Platform

**A unified computational framework for studying the Riemann Hypothesis through the lenses of Self‑Organised Criticality, Quantum Chaos, and Random Matrix Theory.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXX-blue)]() <!-- replace later -->

RH ONE provides an **end‑to‑end differentiable pipeline** that:

1. Computes high‑precision Riemann zeta zeros via the Riemann–Siegel formula.
2. Unfolds the zeros to unit mean spacing.
3. Initialises a **Semantic‑State Contraction (SSC)** particle system from real zero data.
4. Evolves the particles with fully differentiable dynamics (learnable SOC kernel, Hilbert transform, soft histogram).
5. Compares the emergent statistics — spacings, pair correlation, spectral form factor, number variance — against GUE predictions using differentiable losses.
6. Trains the SSC parameters to replicate the statistical universality of the Riemann zeros.

The result is a powerful tool for exploring the **Hilbert–Pólya conjecture**, testing whether a dynamical system can naturally produce the spectral statistics of the zeta zeros, and investigating connections between number theory, quantum chaos, and self‑organised criticality.

---

## ✨ Key Features

- **Riemann Zeros Engine**  
  Riemann–Siegel Z(t) with asymptotic theta, Gram‑point finding with robust fallback (subdivision + local extremum detection), and Φ‑operator refinement. Handles Gram’s law failures gracefully.

- **Fully Differentiable SSC Simulator**  
  All components are built in PyTorch and support automatic differentiation:
  - Soft Gaussian kernel density estimator (KDE) for particle density.
  - Hilbert transform via FFT.
  - Learnable SOC kernel (amplitude, exponent, decay).
  - Differentiable RG low‑pass filter.
  - Trainable drift and noise parameters.

- **Differentiable GUE Statistics**  
  Four loss functions implemented with soft binning / smooth counts, enabling gradient flow:
  - Nearest‑neighbour spacing (Wigner surmise)
  - Pair correlation (R₂(r) – GUE)
  - Spectral form factor (connected part)
  - Number variance (Σ²(L) – soft count approximation)

- **Full Research Pipeline**  
  `real zeros → unfolding → SSC initialisation → simulation → statistical comparison → training`  
  All steps run in a single script; no external data preparation needed.

- **Multi‑Backend & Distributed**  
  Runs on CUDA, Apple MPS, Huawei Ascend NPU, and CPU. Supports multi‑GPU training via PyTorch DDP with custom histogram all‑reduce.

- **Lightweight**  
  Works on consumer hardware (3 GB RAM, Google Colab T4). No heavy dependencies beyond PyTorch, NumPy, SciPy.

---

## ⚡ “O(1)” Computing – Why It Matters

The cost of computing actual Riemann zeta zeros grows rapidly with height $t$; traditional numerical methods scale as $\mathcal{O}(t^{1/2+\epsilon})$.  
**RH ONE takes a different approach**: instead of computing new zeros directly, it **trains a dynamical system (SSC) to capture the universal statistical behaviour** of the zeros.  

Once trained, the SSC simulator can:

- **Generate surrogate statistical ensembles in constant time per sample**, regardless of the height of the zero sequence.  
- Produce particle configurations whose spacing, pair‑correlation, and number variance match GUE predictions **without any further zero‑finding**.  

This means:
- **Exploration at scale**: you can sweep parameter spaces, test hypotheses, and generate millions of statistically correct samples in seconds — a task that would be computationally prohibitive if each sample required recomputing zeros.
- **Constant‑time inference**: after training, generating a new set of $N$ “effective zeros” is $\mathcal{O}(N)$, but **the underlying statistical laws are captured in the model weights, which are independent of the number of original zeros used for training**. In that sense, the **statistical prediction is $\mathcal{O}(1)$ with respect to the height of the Riemann zeros**.

This paradigm makes RH ONE an ideal test‑bed for theoretical physicists and mathematicians studying the origin of GUE universality.

---

## 📦 Installation

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, NumPy, SciPy, Matplotlib (optional).

```bash
git clone https://github.com/yourusername/rh-one.git
cd rh-one
pip install -r requirements.txt
```

Optional: For distributed training, install PyTorch with CUDA support.
Ascend NPU: Install torch-npu according to Huawei’s instructions.

---

🚀 Quick Start

```bash
# 1. Compute first 100 Riemann zeros above index 100
python rh_one.py --mode zeros --start-index 100 --num-zeros 100

# 2. Run a short demo of the SSC simulator (no training)
python rh_one.py --mode demo --N 10000 --steps 200 --device cuda

# 3. Train the SSC model to match GUE statistics
python rh_one.py --mode train --N 10000 --epochs 200 --device cuda

# 4. Run the full pipeline: zeros → SSC → statistical comparison
python rh_one.py --mode full_pipeline --start-index 100 --num-zeros 200 --N 10000 --steps 300
```

All outputs (plots, logs) are saved in the working directory.

---

📖 Usage Modes

Mode Description
zeros Compute real Riemann zeros using the Riemann–Siegel formula.
train Train the SSC dynamics (learnable SOC kernel, drift, noise) to minimise statistical discrepancy with GUE.
demo Run one SSC simulation without training and plot the spacing histogram.
full_pipeline End‑to‑end run: real zeros → unfolding → SSC → statistical plots.
(default) stats Placeholder for additional statistical analysis tools.

Training supports all standard PyTorch features: checkpoints, learning rate scheduling, tensorboard, etc. (easily extendable).

---

🧠 Architecture

```
 Riemann Zeros (NumPy/SciPy)
        │
        ▼
 Unfolding & Cumulative Positions
        │
        ▼
 SSC Initialisation (from real zeros or uniform)
        │
        ▼
 ┌─────────────────────────────────────────┐
 │  SSC Simulator (PyTorch, differentiable) │
 │  ┌───────────┐   ┌──────────────────┐   │
 │  │ Soft KDE  │→→→│ Hilbert Transform │   │
 │  └───────────┘   └──────────────────┘   │
 │  ┌───────────┐   ┌──────────────────┐   │
 │  │ Learnable │   │   Diff RG Filter │   │
 │  │ SOC Kernel│   │                  │   │
 │  └───────────┘   └──────────────────┘   │
 │  ┌──────────────────────────────────┐   │
 │  │   Drift: -α H[ρ]K - β ∇ρ + γ x  │   │
 │  │   Noise: σ √dt dW              │   │
 │  └──────────────────────────────────┘   │
 └─────────────────────────────────────────┘
        │
        ▼
 Differentiable Statistics → Loss ← GUE Targets
        │
        ▼
 Backprop → Update SSC parameters
```

---

📊 Statistical Measures (GUE)

The platform implements the standard diagnostics for quantum chaos:

· Spacing Distribution $P(s)$ — compared with the Wigner surmise.
· Pair Correlation $R_2(r)$ — GUE prediction $1 - (\sin(\pi r)/\pi r)^2$.
· Spectral Form Factor (connected part) — Gaussian‑windowed.
· Number Variance $\Sigma^2(L)$ — soft‑count differentiable version.

All metrics are computed directly on the unfolded positions (unit mean spacing), following the conventions of random matrix theory.

---

🔬 References & Theoretical Background

· Riemann–Siegel asymptotic formula (Siegel 1932, Edwards 1974)
· Gram points and zero finding (Edwards 1974, standard analytic number theory)
· Hilbert–Pólya conjecture (quantum mechanical operator for zeta zeros)
· Random Matrix Theory (Mehta, Gaudin, Dyson) and GUE universality
· Self‑Organised Criticality (Bak, Tang, Wiesenfeld)
· SSC framework: “A Rigorous Applied Framework for Semantic‑State Contraction” (2025)
· GPU Hilbert transform via torch.fft

For a complete literature overview, see the references in the source code header.

---

📜 License

This project is distributed under the MIT License.
Copyright (c) 2026 Yoon A Limsuwan.
See the LICENSE file for full details.

⚠️ This software is intended exclusively for peaceful civilian applications.

---

🤝 Contributing

Contributions are welcome!
Areas for improvement:

· Additional statistical measures (e.g., Dyson–Mehta Δ₃)
· More complex SSC dynamics (higher‑order interactions)
· Integration with other zeta‑function libraries (e.g., mpmath)
· Hyperparameter optimisation, logging, and visualisation dashboards

Please open an issue or pull request on GitHub.

---

📬 Contact

Author: Yoon A Limsuwan
Repository: https://github.com/yourusername/rh-one
Email: your.email@example.com

---

If you use RH ONE in your research, please cite the corresponding paper/DOI (to be assigned).

```

---

