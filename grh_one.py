"""
=======================================================================
GRH ONE — Generalized Riemann Hypothesis Fully Differentiable Training
=======================================================================
Author : PAI , Yoon A Limsuwan
License: MIT
Year   : 2026

Fully differentiable extension of RH ONE for training SSC dynamics
to match GUE statistics under the asymptotic zero density of an
arbitrary L-function.

Usage:
  python grh_one.py --degree 1 --conductor 5 --epochs 200 --N 2000
"""

import math
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

# Import RH ONE as the core engine
try:
    import rh_one as rh
except ImportError:
    raise ImportError("rh_one.py must be in the same directory.")

# =====================================================================
# 1. L‑Function Definition
# =====================================================================
class GeneralizedLFunction:
    """Container for L‑function metadata (degree, conductor)."""
    def __init__(self, name: str = "L-function", degree: int = 1, conductor: float = 1.0):
        self.name = name
        self.d = degree
        self.q = conductor

    def density_torch(self, t: torch.Tensor) -> torch.Tensor:
        """Asymptotic zero density (PyTorch, differentiable)."""
        return (self.d / (2 * math.pi)) * torch.log(t / (2 * math.pi)) + (1 / (2 * math.pi)) * math.log(self.q)


# =====================================================================
# 2. Differentiable Unfolding for L‑Functions
# =====================================================================
def unfold_l_positions_torch(sorted_positions: torch.Tensor,
                             l_func: GeneralizedLFunction) -> torch.Tensor:
    """
    Convert sorted particle positions into cumulative unfolded positions
    using the L‑function asymptotic density.

    Args:
        sorted_positions: (batch, N) sorted raw positions (e.g. SSC output).
        l_func: L‑function density object.

    Returns:
        (batch, N) cumulative unfolded positions (first column = 0).
    """
    spacings = sorted_positions[:, 1:] - sorted_positions[:, :-1]
    midpoints = (sorted_positions[:, :-1] + sorted_positions[:, 1:]) / 2.0
    density = l_func.density_torch(midpoints)
    unfolded_spacings = spacings * density
    return torch.cat([
        torch.zeros(sorted_positions.shape[0], 1, device=sorted_positions.device),
        torch.cumsum(unfolded_spacings, dim=1)
    ], dim=1)


# =====================================================================
# 3. Fully Differentiable Trainer for GRH
# =====================================================================
class GRHTrainer:
    """
    Train SSC parameters to produce GUE statistics under the unfolding
    prescribed by a given L‑function.
    """
    def __init__(self,
                 simulator: rh.SSCSimulator,
                 l_func: GeneralizedLFunction,
                 device: str = 'cpu',
                 use_ddp: bool = False):
        self.device = device
        self.l_func = l_func
        self.use_ddp = use_ddp

        if use_ddp:
            rank, world_size = rh.setup_distributed()
            self.rank = rank
            self.world_size = world_size
            self.sim = nn.parallel.DistributedDataParallel(simulator,
                                                          device_ids=[rank] if torch.cuda.is_available() else None)
        else:
            self.rank = 0
            self.world_size = 1
            self.sim = simulator

        self.optimizer = Adam(simulator.parameters(), lr=0.01)

    def compute_loss(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute GUE losses using L‑function unfolding."""
        # Sort positions (differentiable sort via torch.sort)
        sorted_pos, _ = torch.sort(positions, dim=1)
        # Unfold with L‑function density
        unfolded_pos = unfold_l_positions_torch(sorted_pos, self.l_func)

        # Now compute the same statistical losses as in RH ONE
        spacings = unfolded_pos[:, 1:] - unfolded_pos[:, :-1]
        loss_s = rh.spacing_loss(spacings)
        loss_pc = rh.pair_correlation_loss(unfolded_pos)
        loss_sff = rh.spectral_form_factor_loss(unfolded_pos)
        loss_nv = rh.number_variance_loss(unfolded_pos)

        return loss_s + 0.5 * loss_pc + 0.1 * loss_sff + 0.2 * loss_nv

    def train_step(self, num_sim_steps: int = 100, batch_size: int = 1) -> float:
        self.sim.train()
        self.optimizer.zero_grad()

        # Initialise particles uniformly (differentiable sampling)
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
# 4. Example Usage & CLI
# =====================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GRH ONE — Fully Differentiable Training")
    parser.add_argument('--degree', type=int, default=1, help="L‑function degree")
    parser.add_argument('--conductor', type=float, default=1.0, help="L‑function conductor")
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

    # Define L‑function
    l_func = GeneralizedLFunction(
        name=f"L(d={args.degree}, q={args.conductor})",
        degree=args.degree,
        conductor=args.conductor
    )

    # Build SSC simulator (same as in RH ONE)
    sim = rh.SSCSimulator(
        N_particles=args.N,
        XMIN=args.XMIN, XMAX=args.XMAX,
        NGRID=args.NGRID,
        alpha=0.8, beta=0.05, gamma=0.0, sigma=0.3, dt=0.01,
        device=device
    )

    # Create trainer and run
    trainer = GRHTrainer(sim, l_func, device=device, use_ddp=args.use_ddp)
    print(f"Training SSC for {l_func.name} ...")
    trainer.train(epochs=args.epochs, num_sim_steps=args.steps, batch_size=args.batch_size)
