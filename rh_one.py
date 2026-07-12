"""
=======================================================================
RH ONE — Fully Differentiable Riemann Hypothesis & SSC Research Platform
=======================================================================
Author : PAI , Yoon A Limsuwan
License: MIT
Year   : 2026

A unified, fully differentiable computational framework for studying the
Riemann Hypothesis, Self‑Organised Criticality (SOC), Quantum Chaos,
and Random Matrix Theory. Integrates high‑precision Riemann zero computation,
differentiable particle simulations, and a comprehensive statistical toolkit.

This platform provides:
  • Riemann–Siegel Z(t) and robust Gram‑based zero finding
  • Semantic‑State Contraction (SSC) particle dynamics with soft, differentiable
    density estimation (Gaussian kernel density estimator)
  • Learnable SOC kernel and RG filter (fully differentiable)
  • Differentiable statistical losses:
      - nearest‑neighbour spacing (Wigner surmise)
      - pair correlation (GUE)
      - spectral form factor (connected part)
      - number variance (soft‑count differentiable)
  • Full pipeline: real Riemann zeros → unfolding → SSC initialisation from
    zero data → simulation → statistical comparison with GUE predictions
  • Multi‑GPU distributed training (DDP) with all‑reduce utilities
  • Lightweight: runs on 3 GB RAM, Colab T4, Apple MPS, Huawei Ascend NPU,
    NVIDIA CUDA, and CPU‑only systems

The code covers all essential statistical measures of the Riemann zeta zeros
and their conjectured GUE universality, including spacings, two‑point
correlations, spectral rigidity and number variance.  It enables end‑to‑end
differentiable training of a dynamical system to reproduce these statistics,
offering a new route to explore the Hilbert–Pólya idea, quantum chaos, and
self‑organised criticality in the context of the Riemann Hypothesis.

Open‑source foundations (with licences):
  • PyTorch (BSD‑3‑Clause)        — automatic differentiation & GPU/NPU
  • NumPy (BSD‑3‑Clause)          — array operations (non‑differentiable helpers)
  • SciPy (BSD‑3‑Clause)          — root‑finding & special functions
  • Matplotlib (PSF)              — optional visualisation

References & Credits:
  - Riemann–Siegel asymptotic methods (Siegel, 1932; Edwards, 1974)
  - SSC framework (A Rigorous Applied Framework..., 2025)
  - GPU acceleration via torch.fft and distributed reduction
  - Gram points & zero finding: standard techniques from analytic number theory

MIT License:
Copyright (c) 2026 Yoon A Limsuwan
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

This software is intended exclusively for peaceful civilian applications.
"""

import math, os, sys, argparse, logging, warnings, hashlib, json, time
from typing import Tuple, List, Optional, Dict, Any, Union, Callable
import numpy as np
from scipy.special import loggamma
from scipy.optimize import brentq

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Optional plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger("RH_ONE")

# ============================= Device & Backend ================================
def get_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "ascend" and hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def setup_distributed():
    """Initialise torch.distributed if launched with torchrun."""
    if dist.is_available() and not dist.is_initialized():
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
            return dist.get_rank(), dist.get_world_size()
    return 0, 1

def all_reduce_histogram(hist_local: torch.Tensor) -> torch.Tensor:
    """Sum histograms across all processes and normalise to probability density."""
    dist.all_reduce(hist_local, op=dist.ReduceOp.SUM)
    return hist_local / hist_local.sum()

# ====================== Riemann Zeta Zeros (NumPy/SciPy) ========================
def theta_riemann_siegel(t: float) -> float:
    """Riemann–Siegel theta function with asymptotic expansion."""
    if t < 10.0:
        return np.imag(loggamma(0.25 + 0.5j * t)) - 0.5 * t * np.log(np.pi)
    t2 = t / 2.0
    main = t2 * np.log(t2 / (np.pi * np.e)) - np.pi / 8.0
    corr = 1.0/(48.0*t) + 7.0/(5760.0*t**3) - 31.0/(80640.0*t**5)
    return main + corr

def Z_riemann_siegel(t: float, n_terms: int = 2) -> float:
    """Hardy Z‑function via Riemann–Siegel formula with up to C1 correction."""
    N = int(np.sqrt(t / (2 * np.pi)))
    th = theta_riemann_siegel(t)
    n = np.arange(1, N + 1, dtype=np.float64)
    main = 2.0 * np.sum(np.cos(th - t * np.log(n)) / np.sqrt(n))
    if n_terms >= 1:
        tau = np.sqrt(t / (2 * np.pi))
        p = tau - N
        C0 = ((-1)**(N-1)) * (t/(2*np.pi))**(-0.25) * \
             np.cos(2*np.pi*(p*p - p - 1.0/16.0)) / np.cos(2*np.pi*p)
        main += C0
    if n_terms >= 2:
        C1 = ((-1)**(N-1)) * (t/(2*np.pi))**(-0.75) * \
             (1.0/(12.0*np.pi)) * np.sin(2*np.pi*(p*p - p - 1.0/16.0)) / \
             np.cos(2*np.pi*p)**3
        main += C1
    return main

def Z_prime(t: float, h: float = 1e-6) -> float:
    return (Z_riemann_siegel(t+h) - Z_riemann_siegel(t-h)) / (2.0*h)

def gram_point_fast(n: int) -> float:
    """Asymptotic Gram point g_n with Newton refinement."""
    t0 = 2.0 * np.pi * np.exp(1.0 + (n*np.pi + np.pi/8.0) * 2.0 / (2.0*np.pi*n + 1e-10))
    for _ in range(3):
        th = theta_riemann_siegel(t0)
        dth = 0.5 * np.log(t0 / (2.0*np.pi))
        t0 = t0 - (th - n*np.pi) / dth
    return t0

def phi_refine(t0: float, eta0: float = 0.7, max_iter: int = 50, tol: float = 1e-12) -> float:
    """Φ‑operator refinement for zero ordinate."""
    t = float(t0)
    for _ in range(max_iter):
        logt = np.log(t) if t > 1.0 else 1.0
        eta = eta0 / logt
        Z = Z_riemann_siegel(t, n_terms=2)
        Zp = Z_prime(t)
        if abs(Zp) < 1e-12:
            break
        t_next = t - eta * Z / Zp
        if abs(t_next - t) < tol:
            break
        t = t_next
    return t

def find_zeros_gram(start_index: int, num_zeros: int,
                    t_start: Optional[float] = None) -> np.ndarray:
    """
    Robust zero finding using Gram intervals with systematic handling
    of Gram’s law failures (multiple zeros per interval, missing sign changes).
    """
    zeros = []
    n = start_index
    g1 = t_start if t_start is not None else gram_point_fast(n)
    max_iters = num_zeros * 5   # safety margin
    subdiv_fine = 100           # very fine subdivision for difficult blocks

    for _ in range(max_iters):
        g2 = gram_point_fast(n+1)
        z1 = Z_riemann_siegel(g1, n_terms=2)
        z2 = Z_riemann_siegel(g2, n_terms=2)

        found = False
        # first try direct sign change
        if z1 * z2 < 0:
            try:
                t0 = brentq(lambda t: Z_riemann_siegel(t, n_terms=2), g1, g2)
                zero = phi_refine(t0)
                zeros.append(zero)
                found = True
            except Exception:
                pass

        # fine subdivision for missed or multiple zeros
        if not found or (z1 * z2 >= 0):
            ts = np.linspace(g1, g2, subdiv_fine+1)
            zs = np.array([Z_riemann_siegel(t, n_terms=2) for t in ts])
            for i in range(subdiv_fine):
                if zs[i] * zs[i+1] < 0:
                    try:
                        t0 = brentq(lambda t: Z_riemann_siegel(t, n_terms=2), ts[i], ts[i+1])
                        zero = phi_refine(t0)
                        # avoid duplicates
                        if not zeros or abs(zero - zeros[-1]) > 1e-8:
                            zeros.append(zero)
                            found = True
                    except Exception:
                        pass

        # If still nothing found but Z is very small near midpoint, try finding local extremum
        if not found:
            mid = (g1 + g2) / 2.0
            if abs(Z_riemann_siegel(mid, n_terms=2)) < 1e-6:
                try:
                    zero = phi_refine(mid)
                    if not zeros or abs(zero - zeros[-1]) > 1e-8:
                        zeros.append(zero)
                        found = True
                except Exception:
                    pass

        if len(zeros) >= num_zeros:
            break

        n += 1
        g1 = g2

    return np.array(zeros[:num_zeros])

# ================= Differentiable SSC Components (PyTorch) =====================
class LearnableSOCKernel(nn.Module):
    """Learnable Self‑Organised Criticality kernel with trainable parameters.

    BUGFIX (2026-06): `alpha` (the power-law exponent) was previously stored
    as `exp(log_alpha)`, unconstrained above 0. Since the kernel value at
    small r scales like `r_eps ** (-alpha)`, an unconstrained alpha can drift
    upward during optimization and make the kernel diverge near r=0 (same
    failure family as SUPER DNS ONE's CSOCKernel power-law instability).
    Fixed by bounding alpha to [alpha_min, alpha_max] via a sigmoid
    reparameterization, and by tying the small-r regularizer to the kernel's
    own length scale (lambd) instead of a fixed 1e-6, so the floor stays
    meaningful regardless of the units/scale r is expressed in.
    """
    def __init__(self, init_Cs: float = 0.18, init_lambda: float = 12.0,
                 init_alpha: float = 0.5, init_tau: float = 10.0,
                 alpha_min: float = 0.05, alpha_max: float = 2.0,
                 device: str = 'cpu'):
        super().__init__()
        self.log_Cs = nn.Parameter(torch.tensor(math.log(init_Cs), device=device))
        self.log_lambda = nn.Parameter(torch.tensor(math.log(init_lambda), device=device))
        self.log_tau = nn.Parameter(torch.tensor(math.log(init_tau), device=device))

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        init_alpha = min(max(init_alpha, alpha_min + 1e-6), alpha_max - 1e-6)
        # Inverse-sigmoid init so sigmoid(raw_alpha) recovers init_alpha exactly.
        frac = (init_alpha - alpha_min) / (alpha_max - alpha_min)
        raw_init = math.log(frac / (1.0 - frac))
        self.raw_alpha = nn.Parameter(torch.tensor(raw_init, device=device))

    @property
    def Cs(self): return torch.exp(self.log_Cs)
    @property
    def lambd(self): return torch.exp(self.log_lambda)
    @property
    def alpha(self):
        return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.raw_alpha)
    @property
    def tau(self): return torch.exp(self.log_tau)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        # Floor tied to the kernel's own length scale rather than a fixed
        # constant, so the regularizer stays a sensible fraction of lambd
        # regardless of the units that r is expressed in.
        r_eps = 1e-3 * self.lambd
        return self.Cs * torch.pow(r + r_eps, -self.alpha) * torch.exp(-r / self.lambd)

class DiffRGRefiner(nn.Module):
    """Differentiable Renormalisation Group filter (Fourier low‑pass)."""
    def __init__(self, keep_fraction: float = 0.5):
        super().__init__()
        self.keep_fraction = keep_fraction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("DiffRGRefiner expects (batch, length) input")
        x_hat = torch.fft.rfft(x, dim=1)
        freqs = torch.fft.rfftfreq(x.size(1), device=x.device)
        mask = freqs <= (self.keep_fraction * freqs.max())
        mask = mask.to(x.dtype).unsqueeze(0)
        return torch.fft.irfft(x_hat * mask, n=x.size(1), dim=1)

def soft_histogram(x: torch.Tensor, grid: torch.Tensor,
                   sigma: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Differentiable density estimator using a Gaussian kernel on a regular grid.
    Input:
        x    : (batch, N) particle positions
        grid : (NGRID,)  evaluation points
        sigma: kernel bandwidth
    Returns:
        density : (batch, NGRID)  probability density estimate
    """
    dx = grid[1] - grid[0]
    # shape: (batch, N, 1) - (1, 1, NGRID) -> (batch, N, NGRID)
    diff = x.unsqueeze(2) - grid.unsqueeze(0).unsqueeze(0)
    weights = torch.exp(-0.5 * (diff / sigma) ** 2)
    # normalise by sum over grid to obtain density (integral over domain = 1)
    density = weights.sum(dim=1) / (weights.sum(dim=1).sum(dim=1, keepdim=True) * dx + 1e-12)
    return density

class SSCSimulator(nn.Module):
    """
    Semantic‑State Contraction particle dynamics (fully differentiable).

    Drift   : -α H[ρ] * K(r)  - β ∇ρ  + γ x
    Noise   : σ √dt  * dW

    Density ρ is obtained by a soft (Gaussian) histogram, ensuring gradients
    flow through the entire pipeline.
    """
    def __init__(self, N_particles: int, XMIN: float, XMAX: float, NGRID: int,
                 alpha: float = 0.8, beta: float = 0.05, gamma: float = 0.0,
                 sigma: float = 0.3, dt: float = 0.01,
                 soc_kernel: Optional[LearnableSOCKernel] = None,
                 rg_filter: Optional[DiffRGRefiner] = None,
                 device: str = 'cpu'):
        super().__init__()
        self.N = N_particles
        self.XMIN = XMIN
        self.XMAX = XMAX
        self.NGRID = NGRID
        self.dt = dt
        self.soc = soc_kernel if soc_kernel is not None else LearnableSOCKernel(device=device)
        self.rg = rg_filter
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha), device=device))
        self.log_beta = nn.Parameter(torch.tensor(math.log(beta), device=device))
        # BUGFIX (2026-06): previously stored as log(|gamma|+1e-6) and exposed
        # via `self.gamma = exp(log_gamma)`, which silently discarded the sign
        # of the constructor argument -- self.gamma was always positive (with
        # a ~1e-6 floor) no matter what `gamma` was passed in, including
        # negative values. The sign is now stored separately as a fixed
        # (non-trainable) buffer, and only the magnitude is optimized in log
        # space, so self.gamma faithfully reproduces the requested sign while
        # keeping the magnitude positive and learnable.
        self.register_buffer('gamma_sign',
                              torch.tensor(1.0 if gamma >= 0 else -1.0, device=device))
        self.log_gamma = nn.Parameter(torch.tensor(math.log(abs(gamma) + 1e-6), device=device))
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma), device=device))
        self.grid = nn.Parameter(torch.linspace(XMIN, XMAX, NGRID, device=device),
                                 requires_grad=False)
        # kernel bandwidth: fraction of grid spacing
        dx = (XMAX - XMIN) / (NGRID - 1)
        self.sigma_kde = dx * 0.5

    @property
    def alpha(self): return torch.exp(self.log_alpha)
    @property
    def beta(self): return torch.exp(self.log_beta)
    @property
    def gamma(self): return self.gamma_sign * torch.exp(self.log_gamma)
    @property
    def sigma_val(self): return torch.exp(self.log_sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enable DDP calling via forward."""
        return self.step(x)

    def initial_uniform(self, batch_size: int = 1) -> torch.Tensor:
        return torch.rand(batch_size, self.N, device=self.grid.device) * \
               (self.XMAX - self.XMIN) + self.XMIN

    def initial_zeros(self, zeros_unfolded_cum: torch.Tensor) -> torch.Tensor:
        """
        Initialize particle positions from unfolded cumulative positions of
        real Riemann zeros.  `zeros_unfolded_cum` is a 1D tensor of sorted
        unfolded ordinates (or cumulative spacings).
        """
        sorted_z = zeros_unfolded_cum.to(self.grid.device)
        z_min, z_max = sorted_z[0], sorted_z[-1]
        if z_max == z_min:
            return self.initial_uniform()
        r = torch.rand(self.N, device=self.grid.device)
        idx = (r * (len(sorted_z) - 1)).long().clamp(0, len(sorted_z)-2)
        frac = r * (len(sorted_z) - 1) - idx.float()
        x0 = sorted_z[idx]
        x1 = sorted_z[idx+1]
        x = x0 + frac * (x1 - x0)
        return (x - z_min) / (z_max - z_min) * (self.XMAX - self.XMIN) + self.XMIN

    def hilbert_fft(self, density: torch.Tensor) -> torch.Tensor:
        """Hilbert transform of a periodic density via FFT (differentiable)."""
        F = torch.fft.fft(density, dim=1)
        k = torch.fft.fftfreq(density.size(1), device=density.device)
        # Differentiable sign: -i for k>0, +i for k<0, 0 for k=0
        mult = -1j * torch.where(k > 0, torch.ones_like(k),
                                 torch.where(k < 0, -torch.ones_like(k), torch.zeros_like(k)))
        H = torch.real(torch.fft.ifft(F * mult.unsqueeze(0), dim=1))
        return H

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Single simulation step using fully differentiable density estimation.
        """
        # differentiable density via soft histogram
        density = soft_histogram(x, self.grid, self.sigma_kde)

        if self.rg is not None:
            density = self.rg(density)

        H = self.hilbert_fft(density)
        dx = self.grid[1] - self.grid[0]

        # Gradient of density (finite difference on grid, fully differentiable)
        grad = torch.zeros_like(density)
        grad[:, 1:-1] = (density[:, 2:] - density[:, :-2]) / (2*dx)

        # Interpolate H and grad at particle positions
        idx = ((x - self.XMIN) / dx).long().clamp(0, self.NGRID-2)
        x0 = self.grid[idx]
        x1 = self.grid[idx+1]
        w1 = (x - x0) / (x1 - x0)
        w0 = 1.0 - w1

        batch_idx = torch.arange(x.size(0), device=x.device).unsqueeze(1)
        Hp = w0 * H[batch_idx, idx] + w1 * H[batch_idx, idx+1]
        Gp = w0 * grad[batch_idx, idx] + w1 * grad[batch_idx, idx+1]

        soc_scale = self.soc(torch.abs(x - self.XMIN) / (self.XMAX - self.XMIN))
        drift = -self.alpha * Hp * soc_scale - self.beta * Gp + self.gamma * x
        noise = self.sigma_val * math.sqrt(self.dt) * torch.randn_like(x)
        return x + drift * self.dt + noise

    def simulate(self, num_steps: int, initial_x: Optional[torch.Tensor] = None,
                 batch_size: int = 1) -> torch.Tensor:
        if initial_x is None:
            x = self.initial_uniform(batch_size)
        else:
            x = initial_x
        for _ in range(num_steps):
            x = self.step(x)
        return x

# ================= Differentiable Statistical Metrics =========================
def unfold_positions_torch(zeros: torch.Tensor) -> torch.Tensor:
    """
    Unfold zero ordinates to unit mean spacing and return cumulative unfolded positions.
    Input: (batch, N) raw ordinates.
    Output: (batch, N) cumulative unfolded positions.
    """
    sorted_z, _ = torch.sort(zeros, dim=1)
    spacings = sorted_z[:, 1:] - sorted_z[:, :-1]
    midpoints = (sorted_z[:, :-1] + sorted_z[:, 1:]) / 2.0
    density = torch.log(midpoints / (2*math.pi)) / (2*math.pi)
    unfolded_spacings = spacings * density
    # prepend zero to obtain cumulative positions
    unfolded_positions = torch.cat([
        torch.zeros(zeros.shape[0], 1, device=zeros.device),
        torch.cumsum(unfolded_spacings, dim=1)
    ], dim=1)
    return unfolded_positions

def wigner_surmise_torch(s: torch.Tensor) -> torch.Tensor:
    return (32.0 / math.pi**2) * s**2 * torch.exp(-4.0 * s**2 / math.pi)

def spacing_loss(spacings: torch.Tensor, bins: int = 50) -> torch.Tensor:
    s_max = 3.0
    bin_edges = torch.linspace(0, s_max, bins+1, device=spacings.device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    s_flat = spacings.flatten().unsqueeze(1)
    w = bin_centers[1] - bin_centers[0]
    weights = torch.clamp(1.0 - torch.abs(s_flat - bin_centers) / w, min=0.0)
    empirical = weights.mean(dim=0) + 1e-12
    empirical = empirical / empirical.sum()
    theoretical = wigner_surmise_torch(bin_centers)
    theoretical = theoretical / theoretical.sum()
    return (empirical * (torch.log(empirical) - torch.log(theoretical))).sum()

def pair_correlation_loss(positions: torch.Tensor, rmax: float = 5.0,
                          bins: int = 200) -> torch.Tensor:
    """
    Differentiable pair‑correlation loss (shape comparison only).
    Empirical and GUE curves are each normalised to unit area.
    """
    N = positions.size(1)
    diff = positions.unsqueeze(2) - positions.unsqueeze(1)
    triu_idx = torch.triu_indices(N, N, offset=1)
    diffs = diff[:, triu_idx[0], triu_idx[1]].abs().flatten()
    bin_edges = torch.linspace(0, rmax, bins+1, device=positions.device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    w = bin_centers[1] - bin_centers[0]
    weights = torch.clamp(1.0 - torch.abs(diffs.unsqueeze(1) - bin_centers) / w, min=0.0)
    empirical = weights.mean(dim=0) + 1e-12
    empirical = empirical / (empirical.sum() * w)
    r = bin_centers
    gue = 1.0 - (torch.sin(math.pi * r) / (math.pi * r + 1e-6))**2
    gue = gue / (gue.sum() * w)
    return ((empirical - gue)**2).sum()

def spectral_form_factor_loss(positions: torch.Tensor, tau_max: float = 20.0,
                              n_tau: int = 100, Lambda: float = 80.0) -> torch.Tensor:
    u = positions - positions.mean(dim=1, keepdim=True)
    tau = torch.linspace(0, tau_max, n_tau, device=positions.device)
    w = torch.exp(-u**2 / (2*Lambda**2))
    Z = torch.exp(1j * torch.einsum('bt,bu->btu', tau, u)) * w.unsqueeze(1)
    Zw = Z.sum(dim=2)
    denom = (w**2).sum(dim=1, keepdim=True)
    sff = (Zw.abs()**2) / denom
    trivial = (w.sum(dim=1, keepdim=True)**2) / denom
    sff_conn = sff - trivial
    ideal = torch.minimum(tau, torch.ones_like(tau))
    return ((sff_conn.mean(dim=0) - ideal)**2).sum()

def number_variance_loss(positions: torch.Tensor, L_max: float = 5.0,
                         n_L: int = 50, sigma_soft: float = 0.1) -> torch.Tensor:
    """
    Differentiable number variance loss.
    Compares Σ²(L) with the logarithmic GUE asymptotic:
    Σ²_GUE(L) ≈ (1/π²) log(2πL) + const.
    The constant is fitted from the first half of the data.
    Uses soft counts (sigmoid) to maintain differentiability.
    """
    batch, N = positions.shape
    L_vals = torch.linspace(0.1, L_max, n_L, device=positions.device)
    # Use a deterministic subset for efficiency
    N_sample = min(N, 500)
    pos_sample = positions[:, :N_sample]  # (batch, N_sample)
    diff_full = torch.abs(pos_sample.unsqueeze(2) - positions.unsqueeze(1))  # (batch, N_sample, N)
    sigma2_list = []
    for L in L_vals:
        soft_n = torch.sigmoid((L - diff_full) / sigma_soft).sum(dim=2)  # (batch, N_sample)
        mean_n = soft_n.mean(dim=1, keepdim=True)
        var = ((soft_n - mean_n)**2).mean(dim=1)  # (batch,)
        sigma2_list.append(var)
    sigma2 = torch.stack(sigma2_list, dim=1)  # (batch, n_L)

    log_term = (1.0 / math.pi**2) * torch.log(2 * math.pi * L_vals)
    diff_val = sigma2 - log_term.unsqueeze(0)  # (batch, n_L)
    const = diff_val[:, :n_L//2].mean(dim=1, keepdim=True)  # (batch, 1)
    target = log_term.unsqueeze(0) + const
    return ((sigma2 - target)**2).mean()

# ================= Non‑Differentiable Analysis Helpers =========================
def unfold_numpy(zeros: np.ndarray) -> np.ndarray:
    """Unfold zero ordinates to unit mean spacing (NumPy)."""
    sorted_z = np.sort(zeros)
    spacings = np.diff(sorted_z)
    midpoints = (sorted_z[:-1] + sorted_z[1:]) / 2.0
    density = np.log(midpoints / (2 * np.pi)) / (2 * np.pi)
    return spacings * density

def wigner_surmise_np(s: np.ndarray) -> np.ndarray:
    return (32.0 / np.pi**2) * s**2 * np.exp(-4.0 * s**2 / np.pi)

def pair_correlation_np(positions: np.ndarray, rmax: float = 5.0,
                        bins: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard pair correlation R₂(r) from unfolded positions.
    Normalisation follows the convention:
        R₂(r) = (number of pairs in bin) / [ (N choose 2) * dr / L ].
    This yields R₂(r) → 2 for a Poisson process (uncorrelated limit),
    and allows direct overlay with 2·[1 - (sin πr/πr)²] for GUE.
    """
    N = len(positions)
    diffs = []
    for i in range(N):
        j = np.searchsorted(positions, positions[i] + rmax, side='right')
        window = positions[i+1:j] - positions[i]
        diffs.extend(window.tolist())
    diffs = np.array(diffs)
    hist, edges = np.histogram(diffs, bins=bins, range=(0, rmax))
    dr = edges[1] - edges[0]
    L = positions[-1] - positions[0]
    norm = (N * (N - 1) / 2.0) * dr / L
    R2 = hist / norm
    r_centers = (edges[:-1] + edges[1:]) / 2.0
    return r_centers, R2

def spectral_form_factor_np(positions: np.ndarray, tau_max: float = 20.0,
                            n_tau: int = 100, Lambda: float = 80.0) -> Tuple[np.ndarray, np.ndarray]:
    """SFF from unfolded positions (Gaussian window)."""
    u = positions - np.mean(positions)
    tau = np.linspace(0, tau_max, n_tau)
    w = np.exp(-u**2 / (2 * Lambda**2))
    Z = np.exp(1j * np.outer(tau, u)) * w
    Zw = Z.sum(axis=1)
    denom = (w**2).sum()
    sff = (np.abs(Zw)**2) / denom
    trivial = (w.sum()**2) / denom
    return tau, sff - trivial

def number_variance_np(positions: np.ndarray, L_max: float = 5.0,
                       n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Number variance Σ²(L) for unfolded positions."""
    L_vals = np.linspace(0.1, L_max, n_points)
    sigma2 = []
    for L in L_vals:
        counts = [np.sum((positions >= p) & (positions <= p + L)) for p in positions]
        sigma2.append(np.var(counts))
    return L_vals, np.array(sigma2)

# ================== Training Manager =========================
class RHTrainer:
    """Orchestrates training of SSC parameters and SOC kernel to match GUE statistics."""
    def __init__(self, simulator: SSCSimulator, device: str = 'cpu',
                 use_ddp: bool = False):
        self.device = device
        self.use_ddp = use_ddp
        if use_ddp:
            rank, world_size = setup_distributed()
            self.rank = rank
            self.world_size = world_size
            self.sim = DDP(simulator, device_ids=[rank] if torch.cuda.is_available() else None)
        else:
            self.rank = 0
            self.world_size = 1
            self.sim = simulator
        self.optimizer = Adam(simulator.parameters(), lr=0.01)

    def compute_loss(self, positions: torch.Tensor) -> torch.Tensor:
        unfolded_pos = unfold_positions_torch(positions)   # (batch, N) cumulative unfolded
        spacings = unfolded_pos[:, 1:] - unfolded_pos[:, :-1]
        loss_s = spacing_loss(spacings)
        loss_pc = pair_correlation_loss(unfolded_pos)
        loss_sff = spectral_form_factor_loss(unfolded_pos)
        loss_nv = number_variance_loss(unfolded_pos)
        return loss_s + 0.5 * loss_pc + 0.1 * loss_sff + 0.2 * loss_nv

    def train_step(self, num_sim_steps: int = 100, batch_size: int = 1):
        self.sim.train()
        self.optimizer.zero_grad()
        if self.use_ddp:
            x = self.sim.module.initial_uniform(batch_size)
            for _ in range(num_sim_steps):
                x = self.sim(x)   # now uses forward -> step
        else:
            x = self.sim.initial_uniform(batch_size)
            for _ in range(num_sim_steps):
                x = self.sim.step(x)
        loss = self.compute_loss(x)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, epochs: int = 100, num_sim_steps: int = 100, batch_size: int = 1):
        for epoch in range(epochs):
            loss = self.train_step(num_sim_steps, batch_size)
            if self.rank == 0 and epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: loss={loss:.6f}")

# ================== Main CLI =========================
def parse_args():
    p = argparse.ArgumentParser(description="RH ONE — Riemann Hypothesis & SSC Research Platform")
    p.add_argument('--mode', default='train', choices=['train','zeros','stats','demo','full_pipeline'])
    p.add_argument('--device', default='cpu', choices=['cpu','cuda','mps','ascend'])
    p.add_argument('--N', type=int, default=10000)
    p.add_argument('--XMIN', type=float, default=-5.0)
    p.add_argument('--XMAX', type=float, default=5.0)
    p.add_argument('--NGRID', type=int, default=512)
    p.add_argument('--alpha', type=float, default=0.8)
    p.add_argument('--beta', type=float, default=0.05)
    p.add_argument('--gamma', type=float, default=0.0)
    p.add_argument('--sigma', type=float, default=0.3)
    p.add_argument('--dt', type=float, default=0.01)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--use-ddp', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--start-index', type=int, default=100)
    p.add_argument('--num-zeros', type=int, default=100)
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.use_ddp and 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                                rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        sim = SSCSimulator(N_particles=args.N, XMIN=args.XMIN, XMAX=args.XMAX,
                           NGRID=args.NGRID, alpha=args.alpha, beta=args.beta,
                           gamma=args.gamma, sigma=args.sigma, dt=args.dt,
                           device=device)
        trainer = RHTrainer(sim, device=device, use_ddp=True)
        trainer.train(epochs=args.epochs, num_sim_steps=args.steps,
                     batch_size=args.batch_size)
        dist.destroy_process_group()
        return

    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    if args.mode == 'zeros':
        t_start = gram_point_fast(args.start_index)
        zeros = find_zeros_gram(args.start_index, args.num_zeros, t_start=t_start)
        print(f"First 5 zeros above index {args.start_index}:")
        for i, z in enumerate(zeros[:5]):
            print(f"  {args.start_index + i}: {z:.8f}")
        s = unfold_numpy(zeros)
        print(f"Mean spacing: {s.mean():.4f} (should be ~1)")

    elif args.mode == 'train':
        sim = SSCSimulator(N_particles=args.N, XMIN=args.XMIN, XMAX=args.XMAX,
                           NGRID=args.NGRID, alpha=args.alpha, beta=args.beta,
                           gamma=args.gamma, sigma=args.sigma, dt=args.dt,
                           device=device)
        trainer = RHTrainer(sim, device=device)
        trainer.train(epochs=args.epochs, num_sim_steps=args.steps,
                     batch_size=args.batch_size)

    elif args.mode == 'demo':
        sim = SSCSimulator(N_particles=args.N, XMIN=args.XMIN, XMAX=args.XMAX,
                           NGRID=args.NGRID, device=device)
        x = sim.initial_uniform(batch_size=1)
        for _ in range(args.steps):
            x = sim.step(x)
        unfolded_pos = unfold_positions_torch(x)
        spacings = (unfolded_pos[:, 1:] - unfolded_pos[:, :-1]).flatten().cpu().numpy()
        if HAS_MPL:
            plt.hist(spacings, bins=50, density=True, alpha=0.5, label='SSC')
            s = np.linspace(0, 3, 200)
            plt.plot(s, wigner_surmise_np(s), 'r-', label='GUE')
            plt.legend()
            plt.title('RH ONE - Spacing Distribution')
            plt.savefig('rh_one_demo.png')
            logger.info("Demo plot saved to rh_one_demo.png")
        else:
            logger.info("Matplotlib not available; no plot.")
        logger.info(f"Mean spacing: {spacings.mean():.4f}, std: {spacings.std():.4f}")

    elif args.mode == 'full_pipeline':
        # 1. Riemann zeros
        logger.info("Computing Riemann zeros...")
        t_start = gram_point_fast(args.start_index)
        zeros = find_zeros_gram(args.start_index, args.num_zeros, t_start=t_start)
        logger.info(f"Computed {len(zeros)} zeros.")
        # 2. Unfold and cumulative positions
        s_np = unfold_numpy(zeros)
        unfolded_positions = np.cumsum(np.insert(s_np, 0, 0))
        unfolded_tensor = torch.tensor(unfolded_positions, dtype=torch.float32, device=device)

        # 3. SSC simulator initialised from zeros
        sim = SSCSimulator(N_particles=args.N, XMIN=args.XMIN, XMAX=args.XMAX,
                           NGRID=args.NGRID, alpha=args.alpha, beta=args.beta,
                           gamma=args.gamma, sigma=args.sigma, dt=args.dt,
                           device=device)
        x0 = sim.initial_zeros(unfolded_tensor).unsqueeze(0)

        # 4. SSC simulation
        logger.info(f"Running SSC simulation for {args.steps} steps...")
        x_final = sim.simulate(num_steps=args.steps, initial_x=x0)
        pos_ssc = unfold_positions_torch(x_final)  # (1, N) cumulative unfolded
        spacings_ssc = (pos_ssc[:, 1:] - pos_ssc[:, :-1]).flatten().cpu().numpy()

        # 5. Statistical comparison
        logger.info("Statistical comparison:")
        print(f"Zeros mean spacing: {s_np.mean():.4f}")
        print(f"SSC   mean spacing: {spacings_ssc.mean():.4f}")

        # Pair correlation for zeros
        r_z, R2_z = pair_correlation_np(unfolded_positions, rmax=5.0)
        # Pair correlation for SSC
        ssc_unfolded_pos = pos_ssc[0].cpu().numpy()
        r_ssc, R2_ssc = pair_correlation_np(ssc_unfolded_pos, rmax=5.0)

        if HAS_MPL:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            # Spacing histogram
            axes[0,0].hist(s_np, bins=40, density=True, alpha=0.5, label='Riemann zeros')
            axes[0,0].hist(spacings_ssc, bins=40, density=True, alpha=0.5, label='SSC')
            r = np.linspace(0, 3, 200)
            axes[0,0].plot(r, wigner_surmise_np(r), 'r-', label='GUE')
            axes[0,0].set_title('Spacing Distribution')
            axes[0,0].legend()
            # Pair correlation (with normalisation yielding R2→2 for Poisson)
            axes[0,1].plot(r_z, R2_z, label='Zeros')
            axes[0,1].plot(r_ssc, R2_ssc, label='SSC')
            r_g = np.linspace(0.01, 5, 200)
            axes[0,1].plot(r_g, 2.0 * (1.0 - (np.sin(np.pi*r_g)/(np.pi*r_g))**2),
                           'r--', label='GUE (×2)')
            axes[0,1].set_title('Pair Correlation R₂(r)')
            axes[0,1].legend()
            # Number variance
            L_z, nv_z = number_variance_np(unfolded_positions, L_max=5.0)
            L_ssc, nv_ssc = number_variance_np(ssc_unfolded_pos, L_max=5.0)
            axes[1,0].plot(L_z, nv_z, label='Zeros')
            axes[1,0].plot(L_ssc, nv_ssc, label='SSC')
            axes[1,0].set_title('Number Variance Σ²(L)')
            axes[1,0].legend()
            # SFF (connected)
            tau_z, sff_z = spectral_form_factor_np(unfolded_positions)
            axes[1,1].plot(tau_z, sff_z, label='Zeros')
            axes[1,1].axhline(y=0, color='k', ls='--')
            axes[1,1].set_title('Spectral Form Factor (connected)')
            axes[1,1].legend()
            plt.suptitle('RH ONE – Full Pipeline: Riemann Zeros ↔ SSC')
            plt.tight_layout()
            plt.savefig('rh_one_full_pipeline.png')
            logger.info("Full pipeline plot saved to rh_one_full_pipeline.png")
        else:
            logger.info("Install matplotlib for graphical output.")

    else:
        logger.error("Unknown mode.")

if __name__ == "__main__":
    main()
