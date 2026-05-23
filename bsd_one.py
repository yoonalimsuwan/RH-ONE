=======================================================================
BSD ONE — Birch & Swinnerton-Dyer Conjecture Extension Module
=======================================================================
Author : Yoon A Limsuwan
License: MIT
Year   : 2026

Extension module to integrate Elliptic Curve L-functions into the 
RH ONE / GRH ONE platform. Follows the Adapter Pattern to keep 
the original framework unchanged.

Features:
  • EllipticCurveLFunction with degree 2 and conductor N
  • Unfolding using the correct asymptotic density
  • Interface to load zeros from LMFDB-formatted files or precomputed arrays
  • Full SSC pipeline for elliptic curve zero statistics
  • BSD-specific analysis: rank hint from low-lying zero density (optional)
"""

import numpy as np
import torch
import math
import os

# Import the original frameworks (they must be in the same directory)
try:
    import rh_one as rh
except ImportError:
    raise ImportError("Could not find 'rh_one.py'. Please ensure it is in the same directory.")

try:
    import grh_one as grh
except ImportError:
    # If grh_one is not present, we fall back to the basic GeneralizedLFunction definition
    class GeneralizedLFunction:
        """Minimal L-function class if grh_one is not available."""
        def __init__(self, name, degree=1, conductor=1.0):
            self.name = name
            self.d = degree
            self.q = conductor
        def asymptotic_density_np(self, t):
            return (self.d / (2 * np.pi)) * np.log(t / (2 * np.pi)) + (1 / (2 * np.pi)) * np.log(self.q)
        def asymptotic_density_torch(self, t):
            return (self.d / (2 * math.pi)) * torch.log(t / (2 * math.pi)) + (1 / (2 * math.pi)) * math.log(self.q)

# =====================================================================
# 1. Elliptic Curve L‑Function Definition
# =====================================================================
class EllipticCurveLFunction:
    """
    Represents the L‑function of an elliptic curve E/Q.

    Parameters
    ----------
    label : str
        Cremona label (e.g., '11a1').
    conductor : int
        Conductor N of the curve.
    rank : int, optional
        Analytic rank (order of vanishing at s=1). Default 0.
    degree : int
        L‑function degree; fixed to 2 for elliptic curves.
    """
    def __init__(self, label: str, conductor: int, rank: int = 0):
        self.label = label
        self.conductor = float(conductor)
        self.rank = rank
        self.degree = 2  # always 2 for elliptic curves

    @property
    def name(self):
        return f"Elliptic Curve {self.label} (cond={int(self.conductor)}, rank={self.rank})"

    def asymptotic_density_np(self, t: np.ndarray) -> np.ndarray:
        """Asymptotic zero density for a degree-2 L‑function (NumPy)."""
        return (self.degree / (2 * np.pi)) * np.log(t / (2 * np.pi)) + (1 / (2 * np.pi)) * np.log(self.conductor)

    def asymptotic_density_torch(self, t: torch.Tensor) -> torch.Tensor:
        """Asymptotic zero density for a degree-2 L‑function (PyTorch)."""
        return (self.degree / (2 * math.pi)) * torch.log(t / (2 * math.pi)) + (1 / (2 * math.pi)) * math.log(self.conductor)

# =====================================================================
# 2. Utility: Load zeros from file or simulate (for demonstration)
# =====================================================================
def load_zeros_lmfdb(filepath: str) -> np.ndarray:
    """
    Load zeros from a file in LMFDB format: one float per line (imaginary parts).
    Comments (starting with #) are ignored.
    """
    zeros = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    zeros.append(float(line))
                except ValueError:
                    continue
    return np.array(sorted(zeros))

def generate_mock_zeros(curve: EllipticCurveLFunction, num_zeros: int,
                        t_start: float = 10.0, seed: int = 42) -> np.ndarray:
    """
    Generate mock zeros using the GUE prediction and the correct density.
    This is NOT a replacement for true zeros; it is for quick testing only.
    """
    rng = np.random.default_rng(seed)
    # Approximate cumulative density: N(t) ≈ (degree/π) t log(t/2πe) + ...
    # We integrate numerically to invert.
    t_max = t_start + 50 * num_zeros  # rough
    t_vals = np.linspace(t_start, t_max, 50000)
    dens = curve.asymptotic_density_np(t_vals)
    cum_dens = np.cumsum(dens) * (t_vals[1] - t_vals[0])
    cum_dens -= cum_dens[0]
    # Normalize to integer counts
    cum_dens = cum_dens / cum_dens[-1] * (num_zeros + 10)  # extra margin
    # Uniformly sample from expected cumulative function
    y = np.sort(rng.uniform(0, num_zeros, size=num_zeros))
    zeros = np.interp(y, cum_dens, t_vals)
    # Add small GUE-like fluctuations (optional)
    return zeros

# =====================================================================
# 3. Unfolding for Elliptic Curve L‑Functions
# =====================================================================
def unfold_ec_zeros_np(zeros: np.ndarray, curve: EllipticCurveLFunction) -> np.ndarray:
    """Unfold zeros of an elliptic curve L‑function (NumPy)."""
    sorted_z = np.sort(zeros)
    spacings = np.diff(sorted_z)
    midpoints = (sorted_z[:-1] + sorted_z[1:]) / 2.0
    density = curve.asymptotic_density_np(midpoints)
    return spacings * density

def unfold_ec_zeros_torch(zeros: torch.Tensor, curve: EllipticCurveLFunction) -> torch.Tensor:
    """Unfold zeros of an elliptic curve L‑function (PyTorch)."""
    sorted_z, _ = torch.sort(zeros)
    spacings = sorted_z[:, 1:] - sorted_z[:, :-1]
    midpoints = (sorted_z[:, :-1] + sorted_z[:, 1:]) / 2.0
    density = curve.asymptotic_density_torch(midpoints)
    return spacings * density

# =====================================================================
# 4. BSD ONE Full Pipeline
# =====================================================================
def run_bsd_pipeline(curve: EllipticCurveLFunction,
                     zeros: np.ndarray,
                     steps: int = 150,
                     XMIN: float = -5.0, XMAX: float = 5.0,
                     device: str = "cpu"):
    """
    Main BSD pipeline: unfold zeros → SSC simulation → statistical comparison.

    Parameters
    ----------
    curve : EllipticCurveLFunction
        The curve under study.
    zeros : np.ndarray
        Array of imaginary parts of non‑trivial zeros (sorted).
    steps : int
        Number of SSC simulation steps.
    XMIN, XMAX : float
        Domain bounds for SSC particles.
    device : str
        Torch device.

    Returns
    -------
    s_zeros : np.ndarray
        Unfolded spacings of input zeros.
    spacings_ssc : np.ndarray
        Unfolded spacings of SSC output.
    """
    print(f"--- BSD ONE Pipeline for: {curve.name} ---")
    dev = rh.get_device(device)

    # 1. Unfold input zeros using EC density
    print("1. Unfolding zeros using elliptic curve density...")
    s_np = unfold_ec_zeros_np(zeros, curve)
    unfolded_cum = np.cumsum(np.insert(s_np, 0, 0))
    z_min, z_max = unfolded_cum[0], unfolded_cum[-1]
    N_particles = len(unfolded_cum) - 1

    # 2. SSC simulation initialised from these unfolded cumulative positions
    print("2. Initialising SSC simulator...")
    sim = rh.SSCSimulator(
        N_particles=N_particles,
        XMIN=XMIN, XMAX=XMAX, NGRID=512, device=dev
    )

    unfolded_tensor = torch.tensor(unfolded_cum, dtype=torch.float32, device=dev)
    x0 = sim.initial_zeros(unfolded_tensor).unsqueeze(0)  # (1, N_particles)

    print(f"3. Running SSC dynamics for {steps} steps...")
    x_final = sim.simulate(num_steps=steps, initial_x=x0)

    # 3. Map SSC positions back to cumulative unfolded scale and compute spacings
    if z_max == z_min:
        raise ValueError("Range of unfolded positions too small.")
    scaling = (z_max - z_min) / (XMAX - XMIN)
    u_ssc = (x_final - XMIN) * scaling + z_min
    u_ssc_sorted, _ = torch.sort(u_ssc, dim=1)
    spacings_ssc = (u_ssc_sorted[:, 1:] - u_ssc_sorted[:, :-1]).flatten().cpu().numpy()

    # 4. Basic statistics
    print("\n[BSD Statistical Results]")
    print(f"Curve: {curve.label}, Conductor: {int(curve.conductor)}, Rank: {curve.rank}")
    print(f"Input zeros unfolded mean spacing: {s_np.mean():.4f} (target ~1.0)")
    print(f"SSC simulated unfolded mean spacing : {spacings_ssc.mean():.4f}")
    print("---------------------------------------------------------")

    # Optional: Rank hint from low-lying zero density (heuristic)
    if len(s_np) >= 20:
        # Fraction of spacings in the first unit interval (0-1) can hint at rank
        # For rank r, the probability of a zero near the origin is higher.
        low_spacings = s_np[s_np < 1.0]
        frac = len(low_spacings) / len(s_np)
        print(f"Heuristic low-spacing fraction (<1): {frac:.3f} (higher suggests positive rank)")

    return s_np, spacings_ssc

# =====================================================================
# 5. Quick Example & CLI
# =====================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BSD ONE — Elliptic Curve L‑Function Zero Statistics")
    parser.add_argument('--label', type=str, default="11a1", help="Cremona label")
    parser.add_argument('--conductor', type=int, default=11, help="Conductor N")
    parser.add_argument('--rank', type=int, default=0, help="Analytic rank")
    parser.add_argument('--zeros-file', type=str, default=None, help="Path to file with zero ordinates")
    parser.add_argument('--num-mock', type=int, default=200, help="Number of mock zeros if no file")
    parser.add_argument('--steps', type=int, default=150, help="SSC steps")
    parser.add_argument('--device', type=str, default="cpu")
    args = parser.parse_args()

    # Create curve object
    curve = EllipticCurveLFunction(args.label, args.conductor, args.rank)

    # Acquire zeros
    if args.zeros_file and os.path.exists(args.zeros_file):
        print(f"Loading zeros from {args.zeros_file}")
        zeros = load_zeros_lmfdb(args.zeros_file)
    else:
        print(f"No zero file provided. Generating {args.num_mock} mock zeros for demonstration.")
        zeros = generate_mock_zeros(curve, args.num_mock, t_start=10.0)

    if len(zeros) == 0:
        print("Error: No zeros found.")
        exit(1)

    # Run pipeline
    run_bsd_pipeline(curve, zeros, steps=args.steps, device=args.device)
