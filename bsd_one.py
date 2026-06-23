"""
=================================================================
BSD ONE — Fully Differentiable Birch & Swinnerton-Dyer Extension
=================================================================
Author : Yoon A Limsuwan
License: MIT
Year   : 2026

Fully differentiable training of SSC dynamics to emulate GUE statistics
of elliptic curve L‑function zeros. Supports loading real zero data
and optional rank‑aware loss.

Usage:
  python bsd_one.py --label 11a1 --conductor 11 --rank 0 --epochs 200 --N 2000
  python bsd_one.py --label 37a1 --conductor 37 --rank 1 --zeros-file zeros_37a1.txt
"""

import math
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import os
from typing import Optional

try:
    import rh_one as rh
except ImportError:
    raise ImportError("rh_one.py is required.")

# =====================================================================
# 1. Elliptic Curve L‑Function Definition
# =====================================================================
class EllipticCurveLFunction:
    """L‑function of an elliptic curve E/Q."""
    def __init__(self, label: str, conductor: int, rank: int = 0):
        self.label = label
        self.conductor = float(conductor)
        self.rank = rank
        self.degree = 2

    @property
    def name(self):
        return f"EC {self.label} (q={int(self.conductor)}, r={self.rank})"

    def density_torch(self, t: torch.Tensor) -> torch.Tensor:
        """Differentiable asymptotic zero density for a degree‑2 L‑function."""
        return (self.degree / (2 * math.pi)) * torch.log(t / (2 * math.pi)) \
               + (1 / (2 * math.pi)) * math.log(self.conductor)

# =====================================================================
# 2. Differentiable Unfolding for EC L‑functions
# =====================================================================
def unfold_ec_positions_torch(sorted_positions: torch.Tensor,
                              curve: EllipticCurveLFunction) -> torch.Tensor:
    """Convert sorted particle positions to cumulative unfolded positions."""
    spacings = sorted_positions[:, 1:] - sorted_positions[:, :-1]
    midpoints = (sorted_positions[:, :-1] + sorted_positions[:, 1:]) / 2.0
    density = curve.density_torch(midpoints)
    unfolded_spacings = spacings * density
    return torch.cat([
        torch.zeros(sorted_positions.shape[0], 1, device=sorted_positions.device),
        torch.cumsum(unfolded_spacings, dim=1)
    ], dim=1)

# =====================================================================
# 3. Data Loading Utilities (non‑differentiable, for target creation)
# =====================================================================
def load_zeros_lmfdb(filepath: str) -> np.ndarray:
    """Load zero ordinates (imaginary parts) from a text file."""
    zeros = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    zeros.append(float(line))
                except ValueError:
                    continue
    return np.sort(np.array(zeros))

def unfold_zeros_np(zeros: np.ndarray, curve: EllipticCurveLFunction) -> np.ndarray:
    """Unfold zero ordinates using EC density (NumPy)."""
    sorted_z = np.sort(zeros)
    spacings = np.diff(sorted_z)
    midpoints = (sorted_z[:-1] + sorted_z[1:]) / 2.0
    density = curve.density_torch(torch.tensor(midpoints)).numpy()
    return spacings * density

# =====================================================================
# 4. BSD Trainer – Fully Differentiable
# =====================================================================
class BSDTrainer:
    """
    Train SSC to match GUE statistics of an elliptic curve L‑function.
    Optionally uses real zero spacings as an empirical target,
    and adds a rank‑aware low‑lying spacing loss.
    """
    def __init__(self,
                 simulator: rh.SSCSimulator,
                 curve: EllipticCurveLFunction,
                 device: str = 'cpu',
                 use_ddp: bool = False,
                 zero_spacings_target: Optional[torch.Tensor] = None):
        self.device = device
        self.curve = curve
        self.use_ddp = use_ddp
        self.zero_spacings_target = zero_spacings_target

        if use_ddp:
            rank, world_size = rh.setup_distributed()
            self.rank = rank
            self.world_size = world_size
            self.sim = nn.parallel.DistributedDataParallel(
                simulator,
                device_ids=[rank] if torch.cuda.is_available() else None
            )
        else:
            self.rank = 0
            self.world_size = 1
            self.sim = simulator

        self.optimizer = Adam(simulator.parameters(), lr=0.01)

    def compute_loss(self, positions: torch.Tensor) -> torch.Tensor:
        """Full loss: GUE statistics + empirical target (if given) + rank hint."""
        sorted_pos, _ = torch.sort(positions, dim=1)
        unfolded_pos = unfold_ec_positions_torch(sorted_pos, self.curve)
        spacings = unfolded_pos[:, 1:] - unfolded_pos[:, :-1]

        # 1. Standard GUE losses (from rh_one)
        loss_s   = rh.spacing_loss(spacings)
        loss_pc  = rh.pair_correlation_loss(unfolded_pos)
        loss_sff = rh.spectral_form_factor_loss(unfolded_pos)
        loss_nv  = rh.number_variance_loss(unfolded_pos)
        loss = loss_s + 0.5 * loss_pc + 0.1 * loss_sff + 0.2 * loss_nv

        # 2. Empirical target loss (if real zero spacings are provided)
        if self.zero_spacings_target is not None:
            bins = 50
            s_max = 3.0
            bin_edges = torch.linspace(0, s_max, bins+1, device=spacings.device)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            w = bin_centers[1] - bin_centers[0]
            s_flat = spacings.flatten().unsqueeze(1)
            weights = torch.clamp(1.0 - torch.abs(s_flat - bin_centers) / w, min=0.0)
            empirical = weights.mean(dim=0) + 1e-12
            empirical = empirical / empirical.sum()

            if not hasattr(self, 'target_hist'):
                z_flat = self.zero_spacings_target.to(spacings.device).unsqueeze(1)
                z_weights = torch.clamp(1.0 - torch.abs(z_flat - bin_centers) / w, min=0.0)
                target_hist = z_weights.mean(dim=0) + 1e-12
                target_hist = target_hist / target_hist.sum()
                self.target_hist = target_hist.detach()
            empirical_loss = ((empirical - self.target_hist) ** 2).sum()
            loss = loss + 0.5 * empirical_loss

        # 3. Rank‑aware low‑lying spacing loss (heuristic)
        if self.curve.rank > 0:
            K = min(50, spacings.shape[1])
            low_spacings = spacings[:, :K]
            target_frac = (self.curve.rank + 1) / K  # crude approximation
            soft_frac = torch.sigmoid((1.0 - low_spacings) / 0.1).mean()
            rank_loss = (soft_frac - target_frac) ** 2
            loss = loss + 0.1 * rank_loss

        return loss

    def train_step(self, num_sim_steps: int = 100, batch_size: int = 1) -> float:
        self.sim.train()
        self.optimizer.zero_grad()
        if self.use_ddp:
            x = self.sim.module.initial_uniform(batch_size)
            for _ in range(num_sim_steps):
                x = self.sim(x)
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
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

# =====================================================================
# 5. Command‑Line Interface
# =====================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BSD ONE — Fully Differentiable Training for Elliptic Curves"
    )
    parser.add_argument('--label', type=str, default="11a1")
    parser.add_argument('--conductor', type=int, default=11)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--zeros-file', type=str, default=None,
                        help="Path to file with zero ordinates (one per line)")
    parser.add_argument('--N', type=int, default=2000, help="Number of SSC particles")
    parser.add_argument('--XMIN', type=float, default=-5.0)
    parser.add_argument('--XMAX', type=float, default=5.0)
    parser.add_argument('--NGRID', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--use-ddp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = rh.get_device(args.device)

    # 1. Define the elliptic curve
    curve = EllipticCurveLFunction(args.label, args.conductor, args.rank)

    # 2. Optionally load real zeros and compute target spacings
    zero_spacings_target = None
    if args.zeros_file and os.path.exists(args.zeros_file):
        print(f"Loading zeros from {args.zeros_file}")
        zeros = load_zeros_lmfdb(args.zeros_file)
        if len(zeros) > 0:
            s_np = unfold_zeros_np(zeros, curve)
            zero_spacings_target = torch.tensor(s_np, dtype=torch.float32)
            print(f"Loaded {len(zeros)} zeros. Mean unfolded spacing = {s_np.mean():.4f}")
        else:
            print("Zero file empty – training without empirical target.")
    else:
        print("No zero file specified – using theoretical GUE losses only.")

    # 3. Build SSC simulator
    sim = rh.SSCSimulator(
        N_particles=args.N,
        XMIN=args.XMIN, XMAX=args.XMAX,
        NGRID=args.NGRID,
        alpha=0.8, beta=0.05, gamma=0.0, sigma=0.3, dt=0.01,
        device=device
    )

    # 4. Create trainer and start
    trainer = BSDTrainer(sim, curve, device=device, use_ddp=args.use_ddp,
                         zero_spacings_target=zero_spacings_target)
    print(f"Training SSC for {curve.name} ...")
    trainer.train(epochs=args.epochs, num_sim_steps=args.steps,
                  batch_size=args.batch_size)
