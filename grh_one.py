=======================================================================
GRH ONE — Generalized Riemann Hypothesis Extension Module (Corrected)
=======================================================================
Author : Yoon A Limsuwan
License: MIT
Year   : 2026

Extension module to integrate L-functions into the RH ONE platform.
Uses the Adapter Pattern to maintain the integrity of the original 
'rh_one.py' framework without requiring modifications.
"""

import numpy as np
import torch
import math

# Import the original RH ONE framework
try:
    import rh_one as rh
except ImportError:
    raise ImportError("Could not find 'rh_one.py'. Please ensure it is in the same directory.")

# =====================================================================
# 1. Mathematical Core for L-Functions
# =====================================================================
class GeneralizedLFunction:
    """Representational class for any L-function (e.g., Dirichlet L-functions)."""
    def __init__(self, name: str, degree: int = 1, conductor: float = 1.0):
        self.name = name
        self.d = degree        # L-function degree
        self.q = conductor     # L-function conductor

    def asymptotic_density_np(self, t: np.ndarray) -> np.ndarray:
        """Calculate the smooth asymptotic counting density (NumPy)."""
        return (self.d / (2 * np.pi)) * np.log(t / (2 * np.pi)) + (1 / (2 * np.pi)) * np.log(self.q)

    def asymptotic_density_torch(self, t: torch.Tensor) -> torch.Tensor:
        """Calculate the smooth asymptotic counting density (PyTorch)."""
        return (self.d / (2 * math.pi)) * torch.log(t / (2 * math.pi)) + (1 / (2 * math.pi)) * math.log(self.q)

# =====================================================================
# 2. GRH-Adjusted Unfolding Functions
# =====================================================================
def unfold_l_zeros_numpy(zeros: np.ndarray, l_func: GeneralizedLFunction) -> np.ndarray:
    """Unfold zeros using the specific density of the given L-Function (NumPy)."""
    sorted_z = np.sort(zeros)
    spacings = np.diff(sorted_z)
    midpoints = (sorted_z[:-1] + sorted_z[1:]) / 2.0
    density = l_func.asymptotic_density_np(midpoints)
    return spacings * density

def unfold_l_zeros_torch(zeros: torch.Tensor, l_func: GeneralizedLFunction) -> torch.Tensor:
    """Unfold zeros using the specific density of the given L-Function (PyTorch)."""
    sorted_z, _ = torch.sort(zeros)
    spacings = sorted_z[:, 1:] - sorted_z[:, :-1]
    midpoints = (sorted_z[:, :-1] + sorted_z[:, 1:]) / 2.0
    density = l_func.asymptotic_density_torch(midpoints)
    return spacings * density

# =====================================================================
# 3. The Corrected GRH Adapter Pipeline
# =====================================================================
def run_grh_pipeline(l_func: GeneralizedLFunction, start_index: int = 100, num_zeros: int = 200, steps: int = 150):
    """
    Runs the full SSC simulation pipeline by processing data according to 
    the properties of the chosen L-Function, then passing it into the 
    original SSCSimulator.

    Important: The SSC output positions are mapped back to the original 
    unfolded cumulative scale to ensure consistent unfolding.
    """
    print(f"--- Starting GRH ONE Pipeline for: {l_func.name} (d={l_func.d}, q={l_func.q}) ---")
    device = rh.get_device("cpu")

    # Step 1: Retrieve zeros from the original Riemann zeta code (we'll treat them as placeholder
    #          for L-function zeros for demonstration; in a real scenario you'd supply actual L-function zeros)
    print("1. Processing zeros (using Riemann zeros as surrogate)...")
    t_start = rh.gram_point_fast(start_index)
    zeros_raw = rh.find_zeros_gram(start_index, num_zeros, t_start=t_start)

    # Step 2: Unfold zeros using the L-function specific formula
    print("2. Unfolding zeros using L-function parameters...")
    s_np = unfold_l_zeros_numpy(zeros_raw, l_func)
    # Cumulative unfolded positions (integrated density)
    unfolded_positions = np.cumsum(np.insert(s_np, 0, 0))
    z_min, z_max = unfolded_positions[0], unfolded_positions[-1]

    # Step 3: Prepare for SSC simulation.
    # We'll use the original simulator with a fixed domain [XMIN, XMAX].
    XMIN, XMAX = -5.0, 5.0
    N_particles = len(unfolded_positions) - 1  # number of spacings = number of zeros
    print("3. Loading into SSCSimulator core...")
    sim = rh.SSCSimulator(
        N_particles=N_particles,
        XMIN=XMIN, XMAX=XMAX, NGRID=512, device=device
    )

    # Convert to tensor and map to SSC domain using initial_zeros
    unfolded_tensor = torch.tensor(unfolded_positions, dtype=torch.float32, device=device)
    x0 = sim.initial_zeros(unfolded_tensor).unsqueeze(0)  # shape (1, N_particles)

    # Step 4: Run SSC dynamics
    print(f"4. Running SSC simulation for {steps} steps...")
    x_final = sim.simulate(num_steps=steps, initial_x=x0)  # shape (1, N_particles)

    # Step 5: Map SSC output back to cumulative unfolded positions
    # The mapping in initial_zeros is: x_scaled = (u - z_min)/(z_max - z_min) * (XMAX - XMIN) + XMIN
    # So we invert: u = (x_scaled - XMIN) * (z_max - z_min)/(XMAX - XMIN) + z_min
    if z_max == z_min:
        raise ValueError("Zero range too small; cannot invert scaling.")
    scaling = (z_max - z_min) / (XMAX - XMIN)
    u_ssc = (x_final - XMIN) * scaling + z_min
    # Sort for unfolding (simulator does not guarantee sorted output)
    u_ssc_sorted, _ = torch.sort(u_ssc, dim=1)
    # Unfolded spacings are just the differences of cumulative positions
    spacings_ssc = (u_ssc_sorted[:, 1:] - u_ssc_sorted[:, :-1]).flatten().cpu().numpy()

    # Step 6: Compare statistics
    print("\n[Statistical Results]")
    print(f"L-Function Zeros Unfolded Mean Spacing (Target: ~1.0): {s_np.mean():.4f}")
    print(f"SSC Simulated Unfolded Mean Spacing:                  {spacings_ssc.mean():.4f}")
    print("------------------------------------------------------------------")

    return s_np, spacings_ssc

if __name__ == "__main__":
    # Example: Dirichlet L-function with conductor q=5
    dirichlet_l = GeneralizedLFunction(name="Dirichlet L-Function (q=5)", degree=1, conductor=5.0)

    # Execute pipeline (using Riemann zeros as proxy; replace with true L-function zeros for real research)
    run_grh_pipeline(l_func=dirichlet_l, start_index=1000, num_zeros=300, steps=100)
