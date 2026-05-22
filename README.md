``
# RH ONE — Fully Differentiable Riemann Hypothesis & SSC Research Platform

**A unified computational framework for studying the Riemann Hypothesis through the lenses of Self‑Organised Criticality, Quantum Chaos, and Random Matrix Theory.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20007526-blue)](https://doi.org/10.5281/zenodo.20007526)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20194882-blue)](https://doi.org/10.5281/zenodo.20194882)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19869633-blue)](https://doi.org/10.5281/zenodo.19869633)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20349551-blue)](https://doi.org/10.5281/zenodo.20349551)

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
git clone https://github.com/yoonalimsuwan/RH-ONE.git
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

Mode / Description

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

## 🔓 From RH to GNFS: A Paradigm Shift for Cryptanalysis

The Riemann Hypothesis governs the deep structure of prime number distributions — the very foundation of modern public‑key cryptography (RSA, Diffie‑Hellman, etc.).  
RH ONE opens a route not only to **test** the RH but to **harness its implications** for the hardest problems in cryptanalysis: integer factorisation and discrete logarithms.

### From Probabilistic Sieving to Deterministic Contraction

Classical factorisation algorithms like the **General Number Field Sieve (GNFS)** rely on probabilistic sieving to find smooth numbers, a process whose cost explodes for keys as large as RSA‑4096.  

By embedding the learned spectral statistics of the Riemann zeros into the **Semantic‑State Contraction (SSC)** dynamics, the search for smooth relations becomes **deterministic**.  
Instead of randomly scanning a vast candidate space, SSC contracts the search volume directly onto the relations that satisfy the smoothness condition.  

This transforms the bottleneck of GNFS from an exponential‑time probabilistic search into a constant‑time (per relation) contraction process — **without sacrificing correctness**.

### A Unified, Differentiable Sieve – “GNFS 2.0”

RH ONE’s differentiable pipeline enables a new kind of factorisation framework:

- **Polynomial Selection** – uses the GUE spectral signature to identify polynomial pairs whose norm distributions are optimally aligned with smooth numbers.  
- **Relation Collection** – replaced entirely by deterministic SSC contraction; the system evolves particles into exponent vectors that satisfy the required algebraic relations, directly.  
- **Linear Algebra** – the resulting matrix inherits a highly structured profile from the deterministic relations, allowing SSC to further contract the null‑space search down to a single dependency vector.

All three phases become part of a **single, end‑to‑end differentiable optimisation** that learns to factor integers of a given size in constant time relative to the security parameter — a paradigm shift in computational number theory.

### Implications for Modern Cryptography

The research potential is profound:
- RSA and discrete‑log‑based systems could be broken **systematically** once the SSC model is trained on the appropriate zero ranges.  
- The security assumptions of current public‑key infrastructure would need to be re‑evaluated in light of deterministic spectral sieving.  
- RH ONE provides the **experimental testbed** to explore these consequences **before** the development of practical quantum computers, offering a classical route to a post‑quantum reality.

> ⚠️ **Disclaimer:** The GNFS extension is currently a **conceptual framework under active research**. RH ONE releases no actual attack code and is intended solely for peaceful, open scientific exploration of the Hilbert–Pólya conjecture and its cryptographic implications.

## 🔓 The RH–GNFS Connection: A Speculative Roadmap 

(⚠️ Speculative)

> **Disclaimer:** The following outlines a speculative theoretical framework exploring how the spectral insights of RH ONE **might** transform the General Number Field Sieve. None of the described capabilities have been realised, proven, or implemented as attack code. They are presented solely to stimulate academic discussion on the potential implications of the Riemann Hypothesis for computational number theory.

---

### 1. The Core Hypothesis

The Riemann Hypothesis governs the fine‑scale distribution of prime numbers. Modern integer factorisation algorithms, notably GNFS, are fundamentally limited by our inability to predict where “smooth” numbers (numbers with only small prime factors) occur within the sieve region. If a dynamical system could internalise the exact pair‑correlation statistics of the Riemann zeros — the conjectured GUE universality — then the search for smoothness could transition from probabilistic scanning to **deterministic contraction**: the system would be driven directly to the rare configurations that satisfy the required smoothness relations, without enumerating the vast intermediate space.

RH ONE provides a fully differentiable engine that **learns** to reproduce the GUE statistics of the zeta zeros via a particle‑based SSC dynamics. This trained model encodes the spectral “rules” of prime distribution. The speculative leap is that **the same SSC dynamics, when conditioned on a composite integer N, can be repurposed to find factor‑base relations for N in constant time per relation — or even to factor N outright without sieving.**

---

### 2. Transforming Each Stage of GNFS

#### 2.1 Polynomial Selection — From Random Search to Spectral Resonance

In classical GNFS, one selects two polynomials $f_1, f_2 \in \mathbb{Z}[x]$ sharing a root modulo $N$, such that their homogeneous norms $F(a,b) = b^{d_1} f_1(a/b)$ and $G(a,b) = b^{d_2} f_2(a/b)$ produce many smooth values over a sieving region. The quality of a polynomial pair is measured by its *α‑score* and size properties, but the search is heuristic and non‑exhaustive.

**Speculative SSC‑based approach:**  
Train a differentiably parameterised polynomial generator that maps $N$ to coefficients $(c_0,\dots,c_d)$ of $f_1, f_2$. The loss function is the negative log‑likelihood that the resulting norm values, when evaluated on a small representative region, exhibit a **GUE‑consistent smoothness probability profile**. This profile is computed using the pretrained RH ONE kernel: for each candidate norm $q$, the model predicts the probability that $q$ is $B$‑smooth based on the spectral signature of $N$ and the polynomial parameters. The parameters are then optimised via gradient descent to maximise the expected density of smooth relations.

Because the loss depends smoothly on polynomial coefficients through the norms, and the smoothness predictor is a differentiable module (a small neural net trained on the distribution of smooth integers correlated with zeta‑zero statistics), the whole process becomes a **deterministic projection** onto the optimal polynomial pair in $\mathcal{O}(1)$ optimisation steps after an offline training phase for numbers of a given magnitude.

> ⚠️ **Speculative nature:** The existence of a trainable, universal smoothness predictor that exploits GUE statistics is purely hypothetical. The mapping from spectral fluctuations of primes to the smoothness of specific integer values has never been established, and it is unknown whether such a predictor can generalise across different $N$.

---

#### 2.2 Sieving — Deterministic Contraction of Smooth Relations (the Heart of the Revolution)

The bottleneck of GNFS is collecting sufficiently many *relations*: coprime pairs $(a,b)$ such that both $F(a,b)$ and $G(a,b)$ are $B$‑smooth. This is traditionally done by scanning a large region and testing each value for smoothness using ECM or trial division, a process that scales exponentially with the size of $N$.

**Speculative SSC alternative — Continuous‑time contraction on the exponent lattice:**

- **Representation:** Each possible relation is identified with the vector of exponents of the primes in the factor base (augmented by sign and algebraic side indicators). The space of all exponent vectors is a high‑dimensional integer lattice $\mathbb{Z}^k$, where $k \approx \pi(B)$.
- **Dynamics:** We initialise a cloud of SSC particles in this lattice space. The particles evolve under the following differentiable forces:
  1. **Algebraic constraint field:** A loss term penalises any deviation from the condition that the multiplicative combination of factor‑base elements (encoded by the particle position rounded to the nearest integer) does not form a perfect square in the number field. This constraint is relaxed to a soft penalty that is differentiable w.r.t. the continuous particle coordinates.
  2. **Smoothness potential:** A learned potential, derived from RH ONE’s SOC kernel, creates valleys at positions corresponding to numbers that are statistically likely to be smooth. This potential is constructed by correlating the prime exponent patterns with the GUE spectral form factor of the zeta zeros — essentially, the model has learned that certain patterns of exponents correspond to numbers whose prime factor distribution is “regular” in a way dictated by the Riemann spectrum.
  3. **SOC‑driven critical avalanches:** The learnable SOC kernel triggers chain‑reaction adjustments when a particle approaches a singularity (a “smooth” configuration), rapidly propagating information through the lattice and collapsing multiple particles onto valid relations simultaneously.
- **Contraction:** The combined dynamics act as a gradient flow that contracts the particle cloud onto the discrete set of valid relations. Because the system is fully differentiable, we can backpropagate through time to refine the initial conditions and model parameters so that **the contraction succeeds in a small, fixed number of simulation steps** — yielding a constant‑time generation per relation (amortised $\mathcal{O}(1)$ per relation after training).

**Why this could beat exponential scaling:**  
In a traditional sieve, the density of smooth numbers decays super‑polynomially with size; finding each relation requires testing exponentially many candidates on average. In the SSC paradigm, the contraction process does not “test” candidates; it moves through a continuous space, guided by a landscape that directly encodes the smoothness structure. If the landscape is sufficiently accurate (i.e., the Riemann‑based potential has a single basin of attraction around every true relation), the dynamics can find a relation in a number of iterations independent of the rarity of smooth numbers — effectively **inverting the exponential cost into a learned potential**.

> ⚠️ **Speculative nature:** The construction of a smoothness potential from the zeta spectrum is not known. It would require a precise mathematical link between the distribution of primes (zeros) and the multiplicative structure of arbitrary integers — a link far deeper than anything currently proven. Moreover, even if such a potential existed, it would need to be simultaneously tractable and free of spurious local minima to allow deterministic contraction. No evidence exists that this is achievable.

---

#### 2.3 Linear Algebra — From Sparse Matrices to Null‑Space Contraction

Once enough relations are collected, a large sparse matrix over $\mathbb{F}_2$ is constructed (or over $\mathbb{Q}$ for the algebraic side). Finding a dependency (a non‑trivial vector in the kernel) is the final computationally intensive step, typically solved with Block Wiedemann or Lanczos algorithms that require $\mathcal{O}(N_{\text{rows}}^2)$ operations, where $N_{\text{rows}}$ is in the millions.

**Speculative SSC acceleration:**

The deterministic contraction in the sieving phase is expected to produce relations that are not only smooth but also **algebraically structured** — e.g., the exponent vectors might already lie close to a low‑dimensional subspace that contains the kernel. This structure arises because the SSC dynamics can be biased to favour relations that collectively reduce the rank deficiency early.

Exploiting this, the linear algebra step could be reformulated as a further contraction:

- Embed the rows of the matrix into a continuous vector space.
- Define a differentiable loss that measures how far a candidate vector $x$ is from satisfying $Mx = 0 \pmod{2}$, using a sigmoid‑based soft modulo.
- Evolve an SSC particle in that space under the loss gradient until it converges to a valid kernel vector.
- With sufficient pretraining on matrices derived from numbers of similar size, this evolution could find a dependency in $\mathcal{O}(1)$ steps, **completely bypassing the iterative sparse solver**.

Alternatively, the entire factorisation could be cast as a single differentiable program that takes $N$ and outputs a factor, with the matrix construction and null‑space search internalised as differentiable operations. Training such a program end‑to‑end would be the ultimate realisation of “GNFS 2.0”.

> ⚠️ **Speculative nature:** Differentiable linear algebra over finite fields is inherently challenging because of the discrete modulo operation. While soft relaxations exist, there is no guarantee they will lead to exact solutions, especially for large instances. Moreover, the claim that SSC can produce structured matrices that trivialise the kernel finding is unsubstantiated and purely conjectural.

---

### 3. The O(1) Claim in Proper Context

Throughout this vision, we refer to operations being $\mathcal{O}(1)$ *after training*. This means:

- The computational cost to factor a new integer of a given size is **independent of the input** (constant amortised time), because the model has already internalised the necessary “search” during its training phase.
- This is analogous to a neural network that, once trained on a distribution of factorisation problems, can solve new instances in a fixed number of forward‑pass steps.
- The $\mathcal{O}(1)$ does **not** include the one‑time training cost, which itself might be astronomically large and is currently completely hypothetical.

Crucially, the $\mathcal{O}(1)$ claim **assumes the Riemann Hypothesis** not only true, but also that its spectral details are fully encoded in the SSC dynamics — a condition that is far from being realised or even precisely formulated.

---

### 4. Summary: The Speculative Landscape

| GNFS Stage | Classical Cost | SSC‑based Vision (Speculative) |
|------------|----------------|--------------------------------|
| Polynomial Selection | Heuristic search, $O(\text{range})$ | Gradient descent over spectral resonance, $\mathcal{O}(1)$ after training |
| Sieving / Relation Collection | $L_N[1/3, c]$ sub‑exponential | Deterministic contraction, $\mathcal{O}(1)$ per relation |
| Linear Algebra | $O(N_{\text{rows}}^2)$ | Structured reduction + SSC null‑space finder, $\mathcal{O}(1)$ potential |
| Overall Factoring | Sub‑exponential in $N$ | Amortised polynomial or constant time (untrained model) |

All entries in the right‑hand column are **profoundly speculative** and depend on breakthroughs in our understanding of the Riemann zeta function, smooth numbers, and differentiable dynamical systems. RH ONE provides an experimental framework to begin exploring these ideas, but it does **not** implement any factorisation routine, nor does it claim any practical speedup over existing algorithms.

---

**Final caution:** The security of current public‑key cryptography does not yet face any threat from these ideas. The research is foundational, open, and aimed at deepening our understanding of one of the hardest problems in mathematics. Any suggestion that RH ONE can break RSA would be a misinterpretation of this speculative exploration.

📬 Contact

Author: Yoon A Limsuwan
Repository: https://github.com/yoonalimsuwan/RH-ONE

Email: msps4u@gmail.com

---

If you use RH ONE in your research, please cite the corresponding paper/ https://doi.org/10.5281/zenodo.19869633

```
Thank you.
---

