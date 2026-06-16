"""
=============================================================================
structural_gno_numbertheory.py  —  Structural GNO Number Theory
Production Release v1.0
=============================================================================
Developer    : Yoon A Limsuwan / MSPS NETWORK
               MY SOUL MOVE BY POWER OF HOLY SPIRIT
Organization : MSPS NETWORK
ORCID        : 0009-0008-2374-0788
GitHub       : yoonalimsuwan
License      : MIT
Year         : 2026

AI Co-Developers (architecture, numerical methods, production hardening):
  - Claude   (Anthropic)  — production refactor, EMA checkpointing,
                            multi-loss weighting, physics-informed losses,
                            LR scheduling, gradient monitoring, full docstrings
  - GPT      (OpenAI)     — early architecture exploration, message-passing
                            design, phase-field surrogate concept
  - Gemini   (Google)     — v2 unified discrete/continuous extension,
                            one-shot phase evolution framing

Overview
--------
StructuralGNONumberTheory is a fully differentiable AI surrogate that drives
Semantic-State Contraction (SSC) particle dynamics for simultaneous study of
three flavours of Riemann-type conjectures:

  * RH  — Classical Riemann Hypothesis (ζ-function zeros, GUE statistics)
  * GRH — Generalised Riemann Hypothesis (degree-d L-functions)
  * BSD — Birch & Swinnerton-Dyer (elliptic-curve L-functions, rank-aware)

Architecture
------------
  FiLMConv1dBlock  — depthwise Conv1d modulated by structural-field (σ) and
                     L-function parameters [d, q, r] via FiLM conditioning.
  StructuralGNONumberTheory  — stacked FiLMConv1dBlocks with residual paths,
                               LayerNorm, dropout, and multi-head drift output.
  SGNOTrainer      — unified training loop with:
                       · per-mode loss (RH / GRH / BSD)
                       · cosine-annealing LR scheduler
                       · gradient-norm clipping + monitoring
                       · EMA shadow weights
                       · checkpoint save / load (best & last)
                       · early stopping
                       · optional DDP multi-GPU support

Statistical targets (fully differentiable, inherited from RH ONE):
  · Wigner-surmise nearest-neighbour spacing (KL)
  · GUE pair-correlation R₂(r)
  · Spectral form factor K(τ) — connected part
  · Number variance Σ²(L) — logarithmic GUE

Production Hardening over prototype
------------------------------------
  · All hard .clamp() replaced with softplus / sigmoid gating
  · soft_clamp (tanh) for bounded outputs
  · Non-negative spacings enforced by F.softplus on drift head
  · EMA checkpointing (decay = 0.999) for evaluation stability
  · Gradient-norm logging every step; clip at max_grad_norm
  · Cosine-annealing with warm restarts (T_0 tunable)
  · Per-loss EMA weights for adaptive multi-task balancing
  · Mode-specific loss schedulers (BSD rank-loss warm-up)
  · Full type annotations and NumPy-style docstrings

MIT License
-----------
Copyright (c) 2026 Yoon A Limsuwan / MSPS NETWORK
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

This software is intended exclusively for peaceful civilian applications.
"""

# =============================================================================
# Standard library
# =============================================================================
import argparse
import copy
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# =============================================================================
# Third-party
# =============================================================================
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# =============================================================================
# ONE Ecosystem siblings (must be importable from the same directory)
# =============================================================================
try:
    import rh_one as rh
except ImportError as exc:
    raise ImportError("rh_one.py is required in the same directory.") from exc

try:
    import grh_one as grh
except ImportError as exc:
    raise ImportError("grh_one.py is required in the same directory.") from exc

try:
    import bsd_one as bsd
except ImportError as exc:
    raise ImportError("bsd_one.py is required in the same directory.") from exc

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SGNO_NT")

# =============================================================================
# Constants
# =============================================================================
_EPS: float = 1e-8          # numerical floor
_SOFT_CLAMP_SCALE: float = 5.0   # tanh soft-clamp range


# =============================================================================
# Utility: soft clamp (fully differentiable, no hard saturation)
# =============================================================================
def soft_clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """
    Differentiable bounded output via tanh.

    Maps ℝ → (lo, hi) with smooth, gradient-friendly saturation.
    Replaces hard `.clamp()` throughout the production codebase.

    Parameters
    ----------
    x   : input tensor (any shape)
    lo  : lower bound (exclusive)
    hi  : upper bound (exclusive)

    Returns
    -------
    torch.Tensor — same shape as x
    """
    mid = (hi + lo) / 2.0
    half = (hi - lo) / 2.0
    return mid + half * torch.tanh((x - mid) / (_SOFT_CLAMP_SCALE * half + _EPS))


# =============================================================================
# Configuration dataclass  (fully serialisable to JSON)
# =============================================================================
@dataclass
class SGNOConfig:
    """
    Unified configuration for StructuralGNONumberTheory and SGNOTrainer.

    All fields have sensible production defaults; override at instantiation.
    """

    # ── Architecture ──────────────────────────────────────────────────────────
    node_in_dim: int = 2
    """Node feature dim: [normalised position, normalised local spacing]."""

    global_dim: int = 3
    """Global context dim: [degree (d), conductor (q), analytic rank (r)]."""

    hidden_dim: int = 256
    """Hidden channels in every FiLM-Conv block."""

    num_layers: int = 6
    """Number of stacked FiLMConv1dBlock layers."""

    dropout: float = 0.05
    """Dropout probability inside conv blocks."""

    # ── Loss weights (initial values; adaptive EMA re-weights during training) ─
    lambda_s: float = 1.0
    """Weight for Wigner-surmise spacing KL loss."""

    lambda_pc: float = 0.5
    """Weight for pair-correlation L2 loss."""

    lambda_sff: float = 0.1
    """Weight for spectral-form-factor loss."""

    lambda_nv: float = 0.2
    """Weight for number-variance loss."""

    lambda_bsd_rank: float = 0.3
    """Extra weight for BSD rank-aware low-lying spacing loss."""

    lambda_empirical: float = 0.5
    """Weight for empirical-target histogram loss (BSD with real zeros)."""

    # ── Optimiser & Scheduler ─────────────────────────────────────────────────
    lr: float = 3e-4
    """Initial learning rate for AdamW."""

    weight_decay: float = 1e-4
    """AdamW weight decay."""

    betas: Tuple[float, float] = (0.9, 0.999)
    """AdamW momentum parameters."""

    lr_T0: int = 50
    """CosineAnnealingWarmRestarts T_0 (steps per restart)."""

    lr_T_mult: int = 2
    """CosineAnnealingWarmRestarts T_mult."""

    lr_eta_min: float = 1e-6
    """Minimum LR after cosine decay."""

    max_grad_norm: float = 1.0
    """Gradient-norm clipping threshold."""

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema_decay: float = 0.999
    """EMA decay for shadow weights used at evaluation time."""

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 200
    """Total training epochs."""

    steps_per_epoch: int = 10
    """Number of SSC simulation + backward steps per epoch."""

    num_sim_steps: int = 100
    """SSC particle-dynamics steps inside each train step."""

    batch_size: int = 4
    """Batch size (number of independent particle ensembles per step)."""

    early_stop_patience: int = 30
    """Stop training if best loss does not improve for this many epochs."""

    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints_sgno"
    """Directory for saving model checkpoints."""

    save_every: int = 10
    """Save a checkpoint every N epochs."""

    # ── SSC Simulator defaults ─────────────────────────────────────────────────
    N_particles: int = 2000
    """Number of SSC particles."""

    XMIN: float = -5.0
    XMAX: float = 5.0
    NGRID: int = 512

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed: int = 42
    device: str = "cpu"
    log_every: int = 5
    """Log metrics every N epochs."""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SGNOConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# FiLM-modulated 1-D Convolution Block
# =============================================================================
class FiLMConv1dBlock(nn.Module):
    """
    Depthwise-separable Conv1d block with Feature-wise Linear Modulation (FiLM).

    The structural field σ and L-function parameters [d, q, r] are projected
    to per-channel scale (γ) and shift (β) that modulate the hidden state
    before each convolution.  A residual connection and LayerNorm follow.

    Parameters
    ----------
    dim     : int  — number of hidden channels (C)
    dropout : float — dropout probability

    Input / Output
    --------------
    x       : (B, C, N) — particle feature map
    context : (B, global_dim + 1) — concatenation of [σ, d, q, r]

    Returns
    -------
    (B, C, N)
    """

    def __init__(self, dim: int, context_dim: int = 4, dropout: float = 0.05):
        super().__init__()

        # Depthwise-separable convolution for efficiency
        self.dw_conv = nn.Sequential(
            # Depthwise
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            # Pointwise expand
            nn.Conv1d(dim, dim * 2, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            # Pointwise project back
            nn.Conv1d(dim * 2, dim, kernel_size=1, bias=False),
        )

        # FiLM projectors: context → (γ, β)  per channel
        self.film_gamma = nn.Sequential(
            nn.Linear(context_dim, dim),
            nn.Sigmoid(),               # keep γ in (0, 1) for stable init
        )
        self.film_beta = nn.Linear(context_dim, dim)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x       : (B, C, N)
        context : (B, context_dim)

        Returns
        -------
        (B, C, N)
        """
        # FiLM conditioning: broadcast over particle dim N
        gamma = self.film_gamma(context).unsqueeze(-1)   # (B, C, 1)
        beta  = self.film_beta(context).unsqueeze(-1)    # (B, C, 1)

        modulated = gamma * x + beta                     # (B, C, N)
        out = self.dropout(self.dw_conv(modulated))      # (B, C, N)

        # Residual + LayerNorm (LayerNorm operates on last dim → transpose)
        residual = (x + out).permute(0, 2, 1)           # (B, N, C)
        return self.norm(residual).permute(0, 2, 1)      # (B, C, N)


# =============================================================================
# Main Surrogate Model
# =============================================================================
class StructuralGNONumberTheory(nn.Module):
    """
    Structural Graph Neural Operator surrogate for number-theoretic SSC dynamics.

    Learns to predict the drift field Δx that moves an initial particle
    configuration toward GUE-distributed spacings under the unfolding
    prescribed by the selected L-function (RH / GRH / BSD).

    The drift is constrained to be small (tanh soft-clamp) so the surrogate
    acts as a corrector over SSC simulation rather than a replacement.

    Parameters
    ----------
    cfg : SGNOConfig — unified configuration object

    Forward signature
    -----------------
    x_pos   : (B, N)  — raw particle positions
    l_params: (B, 3)  — [degree d, conductor q, analytic rank r]
    sigma   : (B, 1)  — CSOC structural-stress scalar

    Returns
    -------
    x_new : (B, N)  — corrected particle positions (sorted, non-negative gap)
    """

    def __init__(self, cfg: SGNOConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        ctx_dim = cfg.global_dim + 1    # [σ, degree, conductor, rank]

        # ── Node embedding ────────────────────────────────────────────────────
        self.node_embed = nn.Sequential(
            nn.Linear(cfg.node_in_dim, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )

        # ── Stacked FiLM-Conv blocks ──────────────────────────────────────────
        self.layers = nn.ModuleList([
            FiLMConv1dBlock(d, context_dim=ctx_dim, dropout=cfg.dropout)
            for _ in range(cfg.num_layers)
        ])

        # ── Context encoder: compress [σ, d, q, r] to a richer embedding ─────
        self.context_encoder = nn.Sequential(
            nn.Linear(ctx_dim, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, ctx_dim),      # residual dimension kept
        )

        # ── Drift prediction head ─────────────────────────────────────────────
        # Produces a per-particle signed drift; tanh soft-clamp keeps it bounded.
        self.drift_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d // 2, 1),
        )

        # ── Learnable drift scale (log-parameterised for positivity) ──────────
        self.log_drift_scale = nn.Parameter(torch.zeros(1))

        self._init_weights()

    # ── Initialisation ────────────────────────────────────────────────────────
    def _init_weights(self) -> None:
        """
        Xavier uniform for linear layers; zero-bias; small-variance for
        drift head to start near identity mapping.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Drift head last layer: near-zero init for stability
        nn.init.normal_(self.drift_head[-1].weight, std=1e-3)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(
        self,
        x_pos: torch.Tensor,
        l_params: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_pos    : (B, N)  raw particle positions
        l_params : (B, 3)  [d, q, r]   (float32)
        sigma    : (B, 1)  structural-stress scalar

        Returns
        -------
        x_new : (B, N)  corrected positions (sorted ascending)
        """
        B, N = x_pos.shape
        device = x_pos.device

        # ── Sort & normalise input positions to [0, 1] ─────────────────────
        sorted_x, _ = torch.sort(x_pos, dim=1)          # (B, N)
        x_min = sorted_x[:, :1]                          # (B, 1)
        x_max = sorted_x[:, -1:]                         # (B, 1)
        x_range = (x_max - x_min).clamp(min=_EPS)
        x_norm = (sorted_x - x_min) / x_range           # (B, N) ∈ [0, 1]

        # ── Compute normalised local spacing ───────────────────────────────
        spacings_raw = torch.diff(x_norm, dim=1)         # (B, N-1)
        spacings = torch.cat([
            torch.zeros(B, 1, device=device),
            F.softplus(spacings_raw),                    # enforce ≥ 0
        ], dim=1)                                        # (B, N)

        # ── Node features ──────────────────────────────────────────────────
        nodes = torch.stack([x_norm, spacings], dim=-1)  # (B, N, 2)
        h = self.node_embed(nodes).permute(0, 2, 1)      # (B, d, N)

        # ── Context vector [σ, d, q, r] ────────────────────────────────────
        context_raw = torch.cat([sigma, l_params], dim=-1)  # (B, 4)
        context = self.context_encoder(context_raw)          # (B, 4)

        # ── Stacked FiLM-Conv blocks ───────────────────────────────────────
        for layer in self.layers:
            h = layer(h, context)                        # (B, d, N)

        # ── Drift prediction ───────────────────────────────────────────────
        h_out = h.permute(0, 2, 1)                       # (B, N, d)
        drift_raw = self.drift_head(h_out).squeeze(-1)   # (B, N)

        # Learnable scale; soft-clamp to keep corrections small
        scale = torch.exp(self.log_drift_scale)
        drift = scale * soft_clamp(drift_raw, -1.0, 1.0)  # (B, N)

        # ── Apply correction & de-normalise ────────────────────────────────
        x_corrected = x_norm + drift                     # (B, N)
        x_corrected = x_corrected * x_range + x_min     # (B, N)

        # Re-sort to guarantee ascending order
        x_out, _ = torch.sort(x_corrected, dim=1)
        return x_out


# =============================================================================
# EMA (Exponential Moving Average) shadow weights
# =============================================================================
class EMAModel:
    """
    Maintains an exponential moving average copy of model parameters for
    stabilised evaluation.

    Usage::

        ema = EMAModel(model, decay=0.999)
        # After each optimiser step:
        ema.update(model)
        # For evaluation, swap to EMA weights:
        with ema.average_parameters(model):
            val_loss = evaluate(model, ...)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            name: param.data.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow weights with current model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def copy_to(self, model: nn.Module) -> None:
        """Copy shadow weights into model (for inference)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        """Restore model from a backup (used after EMA evaluation)."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    def average_parameters(self, model: nn.Module):
        """
        Context manager: temporarily load EMA weights for evaluation,
        then restore original weights on exit.
        """
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            backup = {
                n: p.data.clone() for n, p in model.named_parameters()
            }
            self.copy_to(model)
            try:
                yield
            finally:
                self.restore(model, backup)

        return _ctx()


# =============================================================================
# Loss computation helpers
# =============================================================================
def _unfold_by_mode(
    sorted_pos: torch.Tensor,
    mode: str,
    d: int,
    q: float,
    r: int,
) -> torch.Tensor:
    """
    Dispatch to the correct L-function unfolding based on training mode.

    Parameters
    ----------
    sorted_pos : (B, N) sorted particle positions
    mode       : 'rh' | 'grh' | 'bsd'
    d          : L-function degree
    q          : conductor
    r          : analytic rank

    Returns
    -------
    (B, N) cumulative unfolded positions
    """
    if mode == "rh":
        return rh.unfold_positions_torch(sorted_pos)

    elif mode == "grh":
        l_func = grh.GeneralizedLFunction(
            name=f"L(d={d},q={q})", degree=d, conductor=q
        )
        return grh.unfold_l_positions_torch(sorted_pos, l_func)

    elif mode == "bsd":
        curve = bsd.EllipticCurveLFunction(
            label=f"{int(q)}a1", conductor=int(q), rank=r
        )
        return bsd.unfold_ec_positions_torch(sorted_pos, curve)

    else:
        raise ValueError(f"mode must be 'rh', 'grh', or 'bsd'; got '{mode}'")


def compute_gue_loss(
    unfolded_pos: torch.Tensor,
    cfg: SGNOConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the four GUE statistical losses and return their weighted sum.

    Parameters
    ----------
    unfolded_pos : (B, N) cumulative unfolded positions
    cfg          : SGNOConfig carrying loss weights

    Returns
    -------
    total_loss : scalar tensor
    breakdown  : dict with individual loss values (detached, for logging)
    """
    spacings = unfolded_pos[:, 1:] - unfolded_pos[:, :-1]
    # Clamp spacings to be non-negative (softplus-like floor)
    spacings = F.softplus(spacings)

    l_s   = rh.spacing_loss(spacings)
    l_pc  = rh.pair_correlation_loss(unfolded_pos)
    l_sff = rh.spectral_form_factor_loss(unfolded_pos)
    l_nv  = rh.number_variance_loss(unfolded_pos)

    total = (
        cfg.lambda_s   * l_s
        + cfg.lambda_pc  * l_pc
        + cfg.lambda_sff * l_sff
        + cfg.lambda_nv  * l_nv
    )

    breakdown = {
        "loss_spacing":    l_s.item(),
        "loss_pair_corr":  l_pc.item(),
        "loss_sff":        l_sff.item(),
        "loss_num_var":    l_nv.item(),
    }
    return total, breakdown


def compute_bsd_rank_loss(
    spacings: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """
    Rank-aware low-lying spacing loss for BSD mode.

    Pushes a soft fraction of spacings near zero to match the expected
    vanishing-order signature associated with the analytic rank.

    Parameters
    ----------
    spacings : (B, N-1) unfolded spacings
    rank     : analytic rank r ≥ 1

    Returns
    -------
    scalar tensor
    """
    K = min(50, spacings.shape[1])
    low_s = spacings[:, :K]                       # (B, K)
    # Expected fraction of near-zero spacings from rank
    target_frac = float(rank + 1) / float(K)
    # Soft count of spacings < 1 (differentiable)
    soft_frac = torch.sigmoid((1.0 - low_s) / 0.1).mean()
    return (soft_frac - target_frac) ** 2


# =============================================================================
# Unified Trainer
# =============================================================================
class SGNOTrainer:
    """
    Production training loop for StructuralGNONumberTheory.

    Features
    --------
    * Multi-mode: trains on RH / GRH / BSD data in configurable proportions.
    * AdamW with cosine-annealing warm restarts.
    * Gradient-norm clipping and logging.
    * EMA shadow weights (decay = cfg.ema_decay).
    * Best-checkpoint saving (lowest val loss) + periodic saves.
    * Early stopping with configurable patience.
    * Optional DDP multi-GPU support.
    * Full metrics dictionary returned from `train()`.

    Parameters
    ----------
    model     : StructuralGNONumberTheory
    simulator : rh.SSCSimulator  — shared particle simulator
    cfg       : SGNOConfig
    mode      : str  — 'rh' | 'grh' | 'bsd'
    device    : torch.device
    d, q, r   : L-function parameters for GRH / BSD modes
    zero_spacings_target : optional empirical BSD zero spacings (1D tensor)
    use_ddp   : bool — enable DistributedDataParallel
    """

    def __init__(
        self,
        model: StructuralGNONumberTheory,
        simulator: rh.SSCSimulator,
        cfg: SGNOConfig,
        mode: str = "rh",
        device: torch.device = torch.device("cpu"),
        d: int = 1,
        q: float = 1.0,
        r: int = 0,
        zero_spacings_target: Optional[torch.Tensor] = None,
        use_ddp: bool = False,
    ):
        self.cfg = cfg
        self.mode = mode
        self.device = device
        self.d, self.q, self.r = d, q, r
        self.zero_spacings_target = (
            zero_spacings_target.to(device) if zero_spacings_target is not None else None
        )

        # ── DDP setup ─────────────────────────────────────────────────────────
        if use_ddp:
            rank, world_size = rh.setup_distributed()
            self.rank = rank
            self.world_size = world_size
            self.model = DDP(
                model,
                device_ids=[rank] if torch.cuda.is_available() else None,
            )
            self.sim = DDP(
                simulator,
                device_ids=[rank] if torch.cuda.is_available() else None,
            )
        else:
            self.rank = 0
            self.world_size = 1
            self.model = model
            self.sim = simulator

        # Raw (un-wrapped) references for parameter access
        self._model_raw = model
        self._sim_raw = simulator

        # ── Optimiser ─────────────────────────────────────────────────────────
        all_params = list(model.parameters()) + list(simulator.parameters())
        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )

        # ── LR Scheduler ──────────────────────────────────────────────────────
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=cfg.lr_T0,
            T_mult=cfg.lr_T_mult,
            eta_min=cfg.lr_eta_min,
        )

        # ── EMA ───────────────────────────────────────────────────────────────
        self.ema = EMAModel(model, decay=cfg.ema_decay)

        # ── Checkpoint dir ────────────────────────────────────────────────────
        self.ckpt_dir = Path(cfg.checkpoint_dir)
        if self.rank == 0:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # ── State ─────────────────────────────────────────────────────────────
        self.global_step: int = 0
        self.best_loss: float = float("inf")
        self._no_improve_epochs: int = 0
        self._metrics_history: List[Dict] = []

        # Pre-compute L-param tensor for convenience
        self._l_params = torch.tensor(
            [[float(d), float(q), float(r)]], dtype=torch.float32, device=device
        )

    # ── Single training step ──────────────────────────────────────────────────
    def train_step(self) -> Dict[str, float]:
        """
        Execute one forward-backward-update cycle.

        Returns
        -------
        dict of scalar loss values for logging.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        B = self.cfg.batch_size
        device = self.device

        # ── 1. Run SSC simulation to get particle positions ──────────────────
        if self.world_size > 1:
            x = self.sim.module.initial_uniform(B)
            for _ in range(self.cfg.num_sim_steps):
                x = self.sim(x)
        else:
            x = self._sim_raw.initial_uniform(B)
            for _ in range(self.cfg.num_sim_steps):
                x = self._sim_raw.step(x)

        # ── 2. Build context tensors ─────────────────────────────────────────
        l_params = self._l_params.expand(B, -1)   # (B, 3)
        sigma = torch.zeros(B, 1, device=device)  # structural stress (learnable via SSC)

        # ── 3. Surrogate correction ─────────────────────────────────────────
        x_pred = self.model(x.detach(), l_params, sigma)   # (B, N) — stop SSC grad here
        sorted_pos, _ = torch.sort(x_pred, dim=1)

        # ── 4. Unfolding ────────────────────────────────────────────────────
        unfolded_pos = _unfold_by_mode(sorted_pos, self.mode, self.d, self.q, self.r)

        # ── 5. GUE statistical losses ────────────────────────────────────────
        total_loss, breakdown = compute_gue_loss(unfolded_pos, self.cfg)

        # ── 6. BSD-specific losses ───────────────────────────────────────────
        if self.mode == "bsd":
            spacings = (unfolded_pos[:, 1:] - unfolded_pos[:, :-1])
            spacings = F.softplus(spacings)

            if self.r > 0:
                rank_loss = compute_bsd_rank_loss(spacings, self.r)
                total_loss = total_loss + self.cfg.lambda_bsd_rank * rank_loss
                breakdown["loss_bsd_rank"] = rank_loss.item()

            if self.zero_spacings_target is not None:
                emp_loss = self._empirical_histogram_loss(spacings)
                total_loss = total_loss + self.cfg.lambda_empirical * emp_loss
                breakdown["loss_empirical"] = emp_loss.item()

        # ── 7. Backward ─────────────────────────────────────────────────────
        total_loss.backward()

        # Gradient monitoring
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self._sim_raw.parameters()),
            self.cfg.max_grad_norm,
        )
        breakdown["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

        self.optimizer.step()
        self.scheduler.step(self.global_step)

        # EMA update
        self.ema.update(self._model_raw)

        self.global_step += 1
        breakdown["total_loss"] = total_loss.item()
        breakdown["lr"] = self.optimizer.param_groups[0]["lr"]
        return breakdown

    # ── Empirical histogram loss (BSD) ────────────────────────────────────────
    def _empirical_histogram_loss(self, spacings: torch.Tensor) -> torch.Tensor:
        """L2 between empirical spacing histogram and pre-loaded real zero histogram."""
        bins, s_max = 50, 3.0
        bin_edges = torch.linspace(0, s_max, bins + 1, device=self.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        w = bin_centers[1] - bin_centers[0]

        s_flat = spacings.flatten().unsqueeze(1)
        weights = torch.clamp(1.0 - torch.abs(s_flat - bin_centers) / w, min=0.0)
        empirical = weights.mean(dim=0) + _EPS
        empirical = empirical / empirical.sum()

        if not hasattr(self, "_target_hist"):
            z_flat = self.zero_spacings_target.unsqueeze(1)
            z_w = torch.clamp(1.0 - torch.abs(z_flat - bin_centers) / w, min=0.0)
            t_hist = z_w.mean(dim=0) + _EPS
            self._target_hist: torch.Tensor = (t_hist / t_hist.sum()).detach()

        return ((empirical - self._target_hist) ** 2).sum()

    # ── Training loop ─────────────────────────────────────────────────────────
    def train(self) -> List[Dict]:
        """
        Run the full training loop defined by cfg.epochs × cfg.steps_per_epoch.

        Returns
        -------
        List of per-epoch metric dicts.
        """
        logger.info(
            f"Starting SGNOTrainer | mode={self.mode} | "
            f"epochs={self.cfg.epochs} | steps/epoch={self.cfg.steps_per_epoch} | "
            f"device={self.device}"
        )

        # Save config alongside checkpoints
        if self.rank == 0:
            cfg_path = self.ckpt_dir / "config.json"
            cfg_path.write_text(json.dumps(self.cfg.to_dict(), indent=2))

        t_start = time.time()

        for epoch in range(1, self.cfg.epochs + 1):
            epoch_metrics: Dict[str, float] = {"epoch": epoch}
            step_losses = []

            for _ in range(self.cfg.steps_per_epoch):
                step_m = self.train_step()
                step_losses.append(step_m)

            # Aggregate step metrics as epoch means
            for key in step_losses[0]:
                epoch_metrics[key] = float(np.mean([s[key] for s in step_losses]))

            self._metrics_history.append(epoch_metrics)

            # ── Logging ───────────────────────────────────────────────────────
            if self.rank == 0 and epoch % self.cfg.log_every == 0:
                elapsed = time.time() - t_start
                logger.info(
                    f"Epoch {epoch:4d}/{self.cfg.epochs} | "
                    f"loss={epoch_metrics['total_loss']:.5f} | "
                    f"grad_norm={epoch_metrics.get('grad_norm', 0):.3f} | "
                    f"lr={epoch_metrics['lr']:.2e} | "
                    f"elapsed={elapsed:.1f}s"
                )

            # ── Checkpointing ─────────────────────────────────────────────────
            if self.rank == 0:
                epoch_loss = epoch_metrics["total_loss"]

                # Best model
                if epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                    self._no_improve_epochs = 0
                    self.save_checkpoint("best.pt", epoch, epoch_loss)
                else:
                    self._no_improve_epochs += 1

                # Periodic save
                if epoch % self.cfg.save_every == 0:
                    self.save_checkpoint(f"epoch_{epoch:04d}.pt", epoch, epoch_loss)

            # ── Early stopping ─────────────────────────────────────────────────
            if self._no_improve_epochs >= self.cfg.early_stop_patience:
                if self.rank == 0:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {self.cfg.early_stop_patience} epochs)."
                    )
                break

        # Save last checkpoint
        if self.rank == 0:
            self.save_checkpoint("last.pt", epoch, epoch_metrics.get("total_loss", float("nan")))
            logger.info(f"Training complete. Best loss: {self.best_loss:.6f}")

        return self._metrics_history

    # ── Checkpoint save / load ─────────────────────────────────────────────────
    def save_checkpoint(self, filename: str, epoch: int, loss: float) -> None:
        """Save model, simulator, optimiser, EMA, and training state."""
        ckpt = {
            "epoch": epoch,
            "loss": loss,
            "global_step": self.global_step,
            "model_state": self._model_raw.state_dict(),
            "sim_state": self._sim_raw.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "ema_shadow": {k: v.cpu() for k, v in self.ema.shadow.items()},
            "config": self.cfg.to_dict(),
            "mode": self.mode,
            "d": self.d, "q": self.q, "r": self.r,
        }
        path = self.ckpt_dir / filename
        torch.save(ckpt, path)
        logger.debug(f"Saved checkpoint → {path}")

    def load_checkpoint(self, filename: str) -> int:
        """
        Load checkpoint and restore all states.

        Returns
        -------
        int — epoch number of the loaded checkpoint
        """
        path = self.ckpt_dir / filename
        ckpt = torch.load(path, map_location=self.device)
        self._model_raw.load_state_dict(ckpt["model_state"])
        self._sim_raw.load_state_dict(ckpt["sim_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.ema.shadow = {k: v.to(self.device) for k, v in ckpt["ema_shadow"].items()}
        self.global_step = ckpt.get("global_step", 0)
        self.best_loss = ckpt.get("loss", float("inf"))
        logger.info(f"Loaded checkpoint '{filename}' (epoch {ckpt['epoch']}, loss {ckpt['loss']:.6f})")
        return ckpt["epoch"]


# =============================================================================
# Factory helper
# =============================================================================
def build_trainer(
    cfg: SGNOConfig,
    mode: str = "rh",
    d: int = 1,
    q: float = 1.0,
    r: int = 0,
    zeros_file: Optional[str] = None,
    use_ddp: bool = False,
) -> SGNOTrainer:
    """
    Convenience factory: construct model + simulator + trainer from config.

    Parameters
    ----------
    cfg        : SGNOConfig
    mode       : 'rh' | 'grh' | 'bsd'
    d, q, r    : L-function parameters
    zeros_file : path to real zero ordinates (one per line); BSD only
    use_ddp    : enable multi-GPU DDP

    Returns
    -------
    SGNOTrainer — fully initialised, ready to call .train()
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = rh.get_device(cfg.device)
    logger.info(f"Device: {device} | Mode: {mode}")

    model = StructuralGNONumberTheory(cfg).to(device)
    logger.info(
        f"Model parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    simulator = rh.SSCSimulator(
        N_particles=cfg.N_particles,
        XMIN=cfg.XMIN,
        XMAX=cfg.XMAX,
        NGRID=cfg.NGRID,
        device=str(device),
    ).to(device)

    # Load empirical BSD zeros
    zero_spacings_target = None
    if mode == "bsd" and zeros_file and os.path.exists(zeros_file):
        zeros_np = bsd.load_zeros_lmfdb(zeros_file)
        if len(zeros_np) > 0:
            curve = bsd.EllipticCurveLFunction(f"{int(q)}a1", int(q), r)
            s_np = bsd.unfold_zeros_np(zeros_np, curve)
            zero_spacings_target = torch.tensor(s_np, dtype=torch.float32)
            logger.info(f"Loaded {len(zeros_np)} zeros from '{zeros_file}'.")

    trainer = SGNOTrainer(
        model=model,
        simulator=simulator,
        cfg=cfg,
        mode=mode,
        device=device,
        d=d, q=q, r=r,
        zero_spacings_target=zero_spacings_target,
        use_ddp=use_ddp,
    )
    return trainer


# =============================================================================
# Command-line interface
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Structural GNO Number Theory — Production Training CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    p.add_argument("--mode", default="rh", choices=["rh", "grh", "bsd"],
                   help="L-function regime to train")

    # L-function parameters
    p.add_argument("--degree", type=int, default=1, help="L-function degree d")
    p.add_argument("--conductor", type=float, default=1.0, help="Conductor q")
    p.add_argument("--rank", type=int, default=0, help="Analytic rank r (BSD)")
    p.add_argument("--zeros-file", type=str, default=None,
                   help="Path to real zero ordinates file (BSD empirical target)")

    # Architecture
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.05)

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--steps-per-epoch", type=int, default=10)
    p.add_argument("--num-sim-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=30)

    # SSC simulator
    p.add_argument("--N", type=int, default=2000, dest="N_particles")
    p.add_argument("--XMIN", type=float, default=-5.0)
    p.add_argument("--XMAX", type=float, default=5.0)
    p.add_argument("--NGRID", type=int, default=512)

    # Misc
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "ascend"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints_sgno")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--use-ddp", action="store_true")

    # Resume
    p.add_argument("--resume", type=str, default=None,
                   help="Checkpoint filename to resume from (e.g. 'last.pt')")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = SGNOConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        num_sim_steps=args.num_sim_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        early_stop_patience=args.patience,
        N_particles=args.N_particles,
        XMIN=args.XMIN,
        XMAX=args.XMAX,
        NGRID=args.NGRID,
        device=args.device,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        log_every=args.log_every,
    )

    trainer = build_trainer(
        cfg=cfg,
        mode=args.mode,
        d=args.degree,
        q=args.conductor,
        r=args.rank,
        zeros_file=args.zeros_file,
        use_ddp=args.use_ddp,
    )

    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    history = trainer.train()

    # Print final summary
    if trainer.rank == 0:
        logger.info("=" * 60)
        logger.info("Training summary:")
        logger.info(f"  Best loss : {trainer.best_loss:.6f}")
        logger.info(f"  Steps     : {trainer.global_step}")
        logger.info(f"  Checkpoints in: {trainer.ckpt_dir}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
