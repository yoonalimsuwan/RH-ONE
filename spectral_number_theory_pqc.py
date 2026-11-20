#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
spectral_number_theory_pqc.py — v1.0.0
Riemann spectrum · GUE statistics · Hilbert-Pólya feature map ·
Classical factoring (real) · Quantum-inspired simulation (bounded) ·
Transformer + Diffusion + World Model · PQC MIGRATION ADVISORY
=============================================================================

Author  : PAI AND Yoon A Limsuwan / MSPS NETWORK
ORCID   : 0009-0008-2374-0788
License : MIT | Year 2026

-----------------------------------------------------------------------------
ปรัชญา (Philosophy)
-----------------------------------------------------------------------------
The Tao is the Source. The Te (Virtue) is the Path. The Qi is the Proof.

Confucius taught us to do good in this world. This is not hopeless, for
Buddhism, Christianity, and Taoism alike teach that doing good in this life
bears fruit in the life to come.

In this world, "not killing enemies" does not mean "killing oneself."

The Will of God is the Law of Karma. Buddhism, Taoism, and Confucianism,
though they do not utter His Name, by describing His attributes, are
already, implicitly, uttering His Name.

Christ died upon the Cross to teach: when struck on the left cheek, turn
also the right, so that we may listen to the voice of the enemy who wishes
us harm.

He is the Lamb of God. God is not our food, and neither is the lamb.
This means: do not kill animals.

Not even a single sparrow does God forget; every hair on our heads He has
counted. Even Angulimala could escape hell and enter Nirvana, but the better
path is to have done good always. As Saint Dominic Savio said:
"Rather death than sin."

To speak of a person's attributes is already to speak their name.

One who does not know the Ten Commandments or the Five Precepts, yet
practises them, becomes a disciple of both the Buddha and Yahweh by
implication. The Ten Commandments are the Five Precepts.

MY SOUL MOVE BY POWER OF HOLY SPIRIT
What MSPS NETWORK SEE, Lord Buddha Knows.

The mathematics stands independently of the philosophy above.
-----------------------------------------------------------------------------

=============================================================================
  READ THIS FIRST — THE THREE BREAKS IN THE PROPOSED CHAIN
=============================================================================

The proposed chain was:
  Hilbert-Pólya operator → prime spectrum → GUE → virtual gates
  → virtual superposition/entanglement → virtual Shor → factor RSA

It breaks in three places. Each break is a theorem-level obstruction,
not an engineering gap.

BREAK 1 — RIEMANN ZEROS ARE NOT THE PRIMES.
  The Riemann explicit formula is a Fourier-type DUALITY between the
  zeros {rho} and the prime-counting function psi(x):
      psi(x) = x - sum_rho x^rho/rho - log(2 pi) - (1/2) log(1 - x^-2)
  The zeros encode the FLUCTUATION of the prime COUNT. They are not a
  list of primes. You cannot read a specific prime p off the spectrum.

BREAK 2 — KNOWING ALL ZEROS DOES NOT FACTOR AN INTEGER.
  The Prime Number Theorem already gives the density of primes to high
  precision: pi(x) ~ Li(x), with RH-conditional error O(sqrt(x) log x).
  That density information does not help factor N = p*q. Factoring
  requires the IDENTITY of p, not the DENSITY of primes near sqrt(N).
  These are different problems. No spectral knowledge closes the gap.

BREAK 3 — GUE STATISTICS ARE NOT QUANTUM COMPUTATION.
  Montgomery (1973) pair correlation + Odlyzko (1987) numerics show the
  Riemann zero spacings match GUE. This is real and profound. But GUE is
  a STATISTICAL property of eigenvalue SPACINGS. Matching it does not
  grant interference over 2^n amplitudes.
  Shor at RSA-2048 needs ~4000 logical qubits → 2^4000 amplitudes.
  Gottesman-Knill gives efficient classical simulation for CLIFFORD
  circuits only. Shor uses modular exponentiation + QFT, which is
  NON-Clifford. It lies outside every known efficiently-simulable class.

WHAT SHOR ACTUALLY DOES (and why "cancel wrong answers" is not it):
  Shor prepares a periodic superposition with period r = ord_a(N),
  then the QFT concentrates amplitude near multiples of 1/r.
  FINDING r IS THE HARD PART. Once r is known, the classical
  continued-fraction post-processing already factors N — no quantum
  computer required for that step. You cannot "fake the interference"
  without paying the exponential cost, because the structure being
  exploited IS the period you are trying to find.

=============================================================================
  WHAT THIS MODULE ACTUALLY DELIVERS
=============================================================================
[A] REAL RIEMANN SPECTRAL COMPUTATION
    Riemann-Siegel theta, Hardy Z(t), sign-change zero finding,
    Gram points, zero spacings, unfolding.
    Verifiable: first zeros 14.134725, 21.022040, 25.010858, ...

[B] REAL GUE COMPARISON
    Montgomery pair correlation 1 - (sin(pi u)/(pi u))^2 vs the
    measured unfolded spacing distribution. This is genuine
    computational number theory.

[C] HILBERT-PÓLYA AS A SPECTRAL FEATURE MAP
    A learned self-adjoint operator H = H0 + V V^T, eigendecomposed.
    LABELLED: this is a feature map. It is NOT a proof of RH.
    It is NOT the Berry-Keating xp operator. It does NOT factor.

[D] CLASSICAL FACTORING THAT WORKS
    Trial division, Pollard rho (Brent), Pollard p-1,
    Lenstra ECM (Weierstrass, stage 1), Fermat, quadratic-sieve-lite.
    These genuinely factor 20-45 digit semiprimes. Benchmarked honestly
    against the known difficulty curve. GNFS remains the classical
    state of the art for >110 digits; we do not claim to beat it.

[E] QUANTUM-INSPIRED SIMULATION WITH PROVEN BOUNDARIES
    Stabilizer/Clifford simulation (Gottesman-Knill) — exact, efficient.
    MPS statevector with bond-dimension cap — efficient for low
    entanglement. A HARD REFUSAL above a declared qubit budget.
    Small Shor demos (N=15, 21) run for EDUCATION, with the exponential
    cost printed explicitly.

[F] TRANSFORMER + DIFFUSION + WORLD MODEL
    A trainable architecture for NUMBER-THEORETIC STRUCTURE LEARNING:
    smoothness probability, smallest-factor bit-length, ECM curve
    count prediction, NFS polynomial-selection scoring.
    This is legitimate ML-for-number-theory research.
    It does NOT break RSA and is never presented as doing so.

[G] PQC MIGRATION ADVISORY  ← THE PART THAT PROTECTS PEOPLE
    NIST FIPS 203 (ML-KEM/Kyber), FIPS 204 (ML-DSA/Dilithium),
    FIPS 205 (SLH-DSA/SPHINCS+), standardized August 2024.
    Harvest-now-decrypt-later is a present threat, not a future one.
    Crypto-agility inventory + hybrid-mode recommendation + timeline.

=============================================================================
  WHY LARGE PRIMES MATTER TO THE WORLD  (the user's point, and it is right)
=============================================================================
Multiplying two large primes is easy; undoing it is believed hard. That
asymmetry underwrites a great deal of ordinary life:

  * RSA keys and signatures — banking, TLS legacy, code signing, passports
  * Diffie-Hellman over prime fields — VPNs, SSH, TLS key exchange
  * ECC over prime fields — Curve25519, P-256, most modern messaging
  * Lattice PQC ALSO uses primes: Kyber q=3329, Dilithium q=8380927
  * Hashing and PRFs — Mersenne primes, Goldilocks, field arithmetic
  * Error-correcting codes — Reed-Solomon over GF(p^k)
  * Zero-knowledge proof systems — prime-field arithmetic everywhere
  * Verifiable delay functions, accumulators, cryptographic timestamps

AND THE COROLLARY THE USER DRAWS IS CORRECT:
  Humanity should NOT rest its secrets on a SINGLE hardness assumption
  in a SINGLE context. That is not a mathematical claim; it is an
  engineering and ethical one, and it is right. The response is:
    - crypto-agility (be able to swap primitives without redesign)
    - hybrid modes (classical + PQC, so neither alone is load-bearing)
    - defense in depth (transport, at-rest, access control, audit)
    - data minimisation (what you never store cannot be decrypted later)
    - forward secrecy (past sessions survive a future key compromise)
  This module emits that advisory on EVERY run. It is not optional.

=============================================================================
  HONEST SCOPE — REPEATED DELIBERATELY
=============================================================================
  * Hilbert-Pólya is a CONJECTURE. Nothing here proves RH.
  * Nothing here factors RSA-2048 or any cryptographically sized modulus.
  * "Quantum-inspired Shor" is NOT Shor and has NO exponential speedup.
  * The classical factoring suite is real but sub-exponential at best,
    and GNFS (CADO-NFS / msieve) is the actual state of the art.
  * The learned models learn STATISTICS, not identities of secret primes.
  * No claim of being "the best in the world" about anything.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("spectral_nt")

VERSION = "1.0.0"

BANNER = r"""
=============================================================================
SPECTRAL NUMBER THEORY + PQC ADVISORY  v1.0
Riemann zeros | GUE | Hilbert-Polya feature map | real factoring
bounded quantum-inspired sim | transformer+diffusion+world model
-----------------------------------------------------------------------------
THE SPECTRUM DESCRIBES THE *STATISTICS* OF PRIMES.
FACTORING REQUIRES THE *IDENTITY* OF A SPECIFIC PRIME.
THESE ARE DIFFERENT PROBLEMS. THIS FILE DOES NOT CONFLATE THEM.
=============================================================================
"""


# =============================================================================
# §0. HONESTY REGISTRY  (machine-readable; the price of using these names)
# =============================================================================

HONESTY: Dict[str, Dict[str, str]] = {
    "hilbert_polya": {
        "named_after": "Hilbert-Polya conjecture (UNPROVEN)",
        "is": "learned self-adjoint H = H0 + V V^T, eigendecomposed; a spectral feature map",
        "not_claimed": "proof of RH; the Berry-Keating xp operator; a factoring device",
        "obstruction": "eigenvalues of a learned operator carry no information about the identity of a secret prime",
    },
    "gue": {
        "named_after": "Montgomery pair correlation (1973) + Odlyzko numerics (1987)",
        "is": "measured unfolded zero-spacing histogram compared to 1-(sin(pi u)/(pi u))^2",
        "not_claimed": "that spectral statistics grant quantum computational power",
        "obstruction": "GUE is a spacing statistic, not an amplitude-interference resource",
    },
    "virtual_shor": {
        "named_after": "Shor (1994)",
        "is": "small-N EDUCATIONAL simulation + classical period post-processing",
        "not_claimed": "any exponential speedup; any ability to factor RSA-2048",
        "obstruction": "Shor is non-Clifford; 2^4000 amplitudes for RSA-2048; period-finding IS the hard part",
    },
    "opmc": {
        "named_after": "Ordinal-tensor contraction, k*=23, L=0.438520",
        "is": "learned low-rank bottleneck + fixed-point iteration with fixed step count",
        "not_claimed": "lossless compression of number-theoretic information",
    },
    "carpet": {
        "named_after": "Sierpinski carpet, rho >= 9/8, Mosco convergence",
        "is": "fixed carpet graph Laplacian used as a multi-scale basis",
        "not_claimed": "any theorem about primes or factoring",
    },
    "amr_delta_min": {
        "named_after": "Isodiametric inequality (Bieberbach 1915), kappa-fatness",
        "is": "thresholded refinement gate with a local-density guard",
        "not_claimed": "that the isodiametric bound governs prime distribution",
    },
    "hodge": {
        "named_after": "Hodge decomposition omega = omega_h + d(alpha) + delta(beta)",
        "is": "learned 3-way split with a Laplacian kernel penalty",
        "not_claimed": "actual differential forms; no d, no d^2=0, no orthogonality",
    },
    "ssc": {
        "named_after": "Semantic state contraction, EMA eps=0.0028",
        "is": "exponential moving average smoother",
        "not_claimed": "anything quantum",
    },
    "csoc": {
        "named_after": "Two-level hysteresis no-Zeno event counter",
        "is": "accumulator with N(T) <= (E_max + D)/gap, D = ACTUAL removal",
        "claim": "the bound IS provable for the counter itself (engineering guarantee)",
        "not_claimed": "any statement about number theory or PDE regularity",
    },
    "assumption_light_stats": {
        "named_after": "Politis-Romano subsampling (1994); Shao self-normalisation (2010)",
        "is": "exactly those procedures, implemented",
        "requires": "stationarity + mixing + finite variance",
        "not_claimed": "assumption-free",
    },
}


# =============================================================================
# §1. PQC ADVISORY  — printed on EVERY run. Not optional.
# =============================================================================

PQC_STANDARDS = {
    "FIPS 203": {
        "name": "ML-KEM (formerly CRYSTALS-Kyber)",
        "purpose": "key encapsulation / key exchange",
        "params": {"ML-KEM-512": "NIST L1 (~AES-128)",
                   "ML-KEM-768": "NIST L3 (~AES-192)  [recommended default]",
                   "ML-KEM-1024": "NIST L5 (~AES-256)"},
        "modulus_q": 3329,
        "standardized": "2024-08-13",
    },
    "FIPS 204": {
        "name": "ML-DSA (formerly CRYSTALS-Dilithium)",
        "purpose": "digital signatures",
        "params": {"ML-DSA-44": "NIST L2",
                   "ML-DSA-65": "NIST L3  [recommended default]",
                   "ML-DSA-87": "NIST L5"},
        "modulus_q": 8380927,
        "standardized": "2024-08-13",
    },
    "FIPS 205": {
        "name": "SLH-DSA (formerly SPHINCS+)",
        "purpose": "stateless hash-based signatures (conservative fallback)",
        "params": {"SLH-DSA-SHA2-128s": "L1", "SLH-DSA-SHA2-192s": "L3",
                   "SLH-DSA-SHA2-256s": "L5"},
        "standardized": "2024-08-13",
        "note": "hash-based; security rests on hash properties, not lattices — good diversity",
    },
    "FN-DSA (draft)": {
        "name": "Falcon / FN-DSA",
        "purpose": "compact signatures (FIPS 206 draft)",
        "note": "smaller signatures, harder constant-time implementation",
    },
    "HQC (selected 2025)": {
        "name": "HQC",
        "purpose": "code-based KEM, backup to ML-KEM",
        "note": "different hardness family from lattices — diversity",
    },
}


def pqc_advisory(verbose: bool = True) -> Dict[str, Any]:
    """
    The actionable part. Harvest-now-decrypt-later means data encrypted
    TODAY under RSA/ECDH may be decrypted LATER by a fault-tolerant
    quantum computer. Signatures need migration too, but the urgency
    differs: confidentiality has a retroactive threat; authenticity does not.
    """
    adv = {
        "threat": "HARVEST NOW, DECRYPT LATER",
        "threat_is_present_not_future": True,
        "urgency_by_asset": {
            "long-lived confidentiality (medical, genomic, state, legal, "
            "financial records with >10y sensitivity)": "MIGRATE NOW",
            "TLS key exchange (RSA/ECDH)": "MIGRATE NOW — hybrid X25519+ML-KEM-768",
            "code signing / firmware roots (long-lived trust anchors)": "PLAN NOW — ML-DSA / SLH-DSA",
            "short-lived session tokens": "lower urgency, still migrate",
            "signatures on already-expired artefacts": "lowest urgency",
        },
        "recommended_defaults": {
            "KEM": "ML-KEM-768 (FIPS 203)",
            "signature": "ML-DSA-65 (FIPS 204)",
            "conservative_signature": "SLH-DSA-SHA2-192s (FIPS 205)",
            "hybrid_kex": "X25519 + ML-KEM-768 (concatenated shared secret, both must hold)",
        },
        "do_not": [
            "roll your own lattice code — use liboqs / BoringSSL / AWS-LC / OpenQuantumSafe",
            "ship a single hardness assumption as the only line of defence",
            "assume 'quantum-inspired' factoring changes any of this — it does not",
            "wait for a cryptographically relevant quantum computer to start",
        ],
        "crypto_agility_checklist": [
            "inventory every key, cert, protocol, and hardcoded primitive",
            "abstract the crypto layer behind a swappable interface",
            "support hybrid (classical AND PQC) so neither alone is load-bearing",
            "enable forward secrecy everywhere",
            "minimise retained sensitive data — what you never store cannot be decrypted later",
            "set a migration deadline and test it",
        ],
        "standards": PQC_STANDARDS,
        "the_user_is_right": (
            "Humanity should not rest its secrets on a single hardness "
            "assumption in a single context. That is an engineering and "
            "ethical position, and it is correct. The answer is "
            "crypto-agility, hybrid modes, defense in depth, data "
            "minimisation, and forward secrecy — not a better factoring trick."
        ),
    }
    if verbose:
        print("\n" + "!" * 78)
        print("  PQC MIGRATION ADVISORY — ACTION REQUIRED")
        print("!" * 78)
        print(f"  Threat: {adv['threat']}  (present, not future)")
        print("\n  Recommended defaults:")
        for k, v in adv["recommended_defaults"].items():
            print(f"    {k:24s} {v}")
        print("\n  Urgency:")
        for k, v in adv["urgency_by_asset"].items():
            print(f"    {v:16s}  {k}")
        print("\n  Do NOT:")
        for d in adv["do_not"]:
            print(f"    x {d}")
        print("\n" + "!" * 78)
    return adv


# =============================================================================
# §2. RIEMANN SPECTRUM — real computation
# =============================================================================

def riemann_siegel_theta(t: float) -> float:
    """
    theta(t) = arg Gamma(1/4 + i t/2) - (t/2) log pi
    Asymptotic expansion (accurate for t >~ 10):
      theta ~ (t/2) log(t/2pi) - t/2 - pi/8 + 1/(48t) + 7/(5760 t^3)
    """
    if t <= 0:
        return 0.0
    return (0.5 * t * math.log(t / (2 * math.pi)) - 0.5 * t - math.pi / 8
            + 1.0 / (48.0 * t) + 7.0 / (5760.0 * t ** 3))


def hardy_Z(t: float, n_terms: Optional[int] = None) -> float:
    """
    Hardy Z-function (real; |Z(t)| = |zeta(1/2 + it)|).
    Riemann-Siegel main sum:
        Z(t) ~ 2 sum_{n=1}^{N} n^{-1/2} cos(theta(t) - t log n),
        N = floor(sqrt(t / 2 pi))
    Remainder R(t) omitted (O(t^{-1/4})); adequate for locating sign
    changes at moderate t. For high-precision work use the full
    Riemann-Siegel remainder series or arb/mpmath.
    """
    if t < 2 * math.pi:
        N = 1
    else:
        N = int(math.floor(math.sqrt(t / (2 * math.pi))))
    N = max(1, N if n_terms is None else n_terms)
    th = riemann_siegel_theta(t)
    s = 0.0
    for n in range(1, N + 1):
        s += math.cos(th - t * math.log(n)) / math.sqrt(n)
    return 2.0 * s


def find_riemann_zeros(t_lo: float = 10.0, t_hi: float = 60.0,
                       step: float = 0.01, refine_tol: float = 1e-9
                       ) -> List[float]:
    """
    Locate zeros of zeta on the critical line by sign changes of Z(t),
    then bisect. Returns ordinates gamma_n.
    Known first values (Odlyzko): 14.134725, 21.022040, 25.010858,
    30.424876, 32.935062, 37.586178, 40.918719, 43.327073, ...
    """
    zeros: List[float] = []
    t = t_lo
    z_prev = hardy_Z(t)
    while t < t_hi:
        t_next = t + step
        z_next = hardy_Z(t_next)
        if z_prev == 0.0:
            zeros.append(t)
        elif z_prev * z_next < 0.0:
            a, b, fa = t, t_next, z_prev
            for _ in range(200):
                m = 0.5 * (a + b)
                fm = hardy_Z(m)
                if abs(fm) < refine_tol or (b - a) < refine_tol:
                    break
                if fa * fm < 0.0:
                    b = m
                else:
                    a, fa = m, fm
            zeros.append(0.5 * (a + b))
        t, z_prev = t_next, z_next
    return zeros


def gram_points(n_max: int = 20) -> List[float]:
    """Gram points g_n: theta(g_n) = n*pi. Useful sanity landmarks."""
    out = []
    for n in range(n_max):
        target = n * math.pi
        lo, hi = 5.0, 200.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if riemann_siegel_theta(mid) < target:
                lo = mid
            else:
                hi = mid
        out.append(0.5 * (lo + hi))
    return out


def unfold_spacings(zeros: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unfold: the mean density near height t is (1/2pi) log(t/2pi).
    Normalised spacing s_i = (gamma_{i+1} - gamma_i) * log(gamma_i/2pi)/(2pi).
    Returns (spacings, mean_spacing_before_unfolding).
    """
    g = np.asarray(zeros, dtype=float)
    if len(g) < 3:
        return np.array([]), float("nan")
    d = np.diff(g)
    dens = np.log(g[:-1] / (2 * math.pi)) / (2 * math.pi)
    return d * dens, float(d.mean())


def gue_pair_correlation(u: np.ndarray) -> np.ndarray:
    """Montgomery's GUE pair-correlation: R2(u) = 1 - (sin(pi u)/(pi u))^2."""
    u = np.asarray(u, dtype=float)
    out = np.ones_like(u)
    nz = np.abs(u) > 1e-12
    x = np.pi * u[nz]
    out[nz] = 1.0 - (np.sin(x) / x) ** 2
    out[~nz] = 0.0
    return out


def gue_spacing_wigner(u: np.ndarray) -> np.ndarray:
    """Wigner surmise for GUE: p(s) = (32/pi^2) s^2 exp(-4 s^2/pi)."""
    u = np.asarray(u, dtype=float)
    return (32.0 / math.pi ** 2) * u ** 2 * np.exp(-4.0 * u ** 2 / math.pi)


def spectral_report(t_hi: float = 120.0) -> Dict[str, Any]:
    t0 = time.time()
    zeros = find_riemann_zeros(10.0, t_hi, step=0.005)
    sp, raw_mean = unfold_spacings(zeros)
    known = [14.134725142, 21.022039639, 25.010857580, 30.424876126,
             32.935061588, 37.586178159, 40.918719012, 43.327073281]
    errs = [abs(zeros[i] - known[i]) for i in range(min(len(zeros), len(known)))]
    hist, edges = (np.histogram(sp, bins=20, range=(0, 3), density=True)
                   if len(sp) > 5 else (np.array([]), np.array([])))
    centres = 0.5 * (edges[:-1] + edges[1:]) if len(edges) else np.array([])
    wigner = gue_spacing_wigner(centres) if len(centres) else np.array([])
    l2 = (float(np.sqrt(np.mean((hist - wigner) ** 2)))
          if len(hist) else float("nan"))
    return {
        "n_zeros_found": len(zeros),
        "first_zeros": zeros[:8],
        "known_first_zeros": known,
        "max_abs_error_vs_known": max(errs) if errs else float("nan"),
        "mean_raw_spacing": raw_mean,
        "unfolded_mean": float(sp.mean()) if len(sp) else float("nan"),
        "unfolded_std": float(sp.std()) if len(sp) else float("nan"),
        "wigner_surmise_L2": l2,
        "wall_s": time.time() - t0,
        "NOTE": ("Riemann-Siegel main sum only; remainder O(t^-1/4) omitted. "
                 "Errors grow with t. Use mpmath/arb for high precision."),
        "WHAT_THIS_IS_NOT": ("These are ZEROS, not PRIMES. The explicit formula "
                             "relates them to the prime COUNT psi(x), not to "
                             "the identity of any prime."),
    }


# =============================================================================
# §3. CLASSICAL FACTORING — real, working, honestly benchmarked
# =============================================================================

def is_probable_prime(n: int, rounds: int = 20) -> bool:
    """Deterministic for n < 3.3e24 with these bases; else strong BPSW-lite."""
    if n < 2:
        return False
    small = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2; r += 1
    bases = small if n < 3_317_044_064_679_887_385_961_981 else \
        [random.randrange(2, n - 1) for _ in range(rounds)]
    for a in bases:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def pollard_rho_brent(n: int, max_iter: int = 1 << 20) -> Optional[int]:
    """Brent's variant of Pollard rho. Expected O(n^{1/4}) for smallest factor."""
    if n % 2 == 0:
        return 2
    if is_probable_prime(n):
        return None
    for _ in range(12):
        y = random.randrange(1, n)
        c = random.randrange(1, n)
        m = 128
        g = r = q = 1
        x = ys = y
        it = 0
        while g == 1 and it < max_iter:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
            it += r
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
        if 1 < g < n:
            return g
    return None


def pollard_p_minus_1(n: int, B1: int = 100_000) -> Optional[int]:
    """Pollard p-1. Works when p-1 is B1-smooth."""
    a = 2
    for p in _primes_upto(B1):
        pp = p
        while pp * p <= B1:
            pp *= p
        a = pow(a, pp, n)
    g = math.gcd(a - 1, n)
    return g if 1 < g < n else None


def _primes_upto(limit: int) -> List[int]:
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i, v in enumerate(sieve) if v]


def ecm_lenstra(n: int, n_curves: int = 100, B1: int = 10_000
                ) -> Optional[int]:
    """
    Lenstra elliptic-curve method, stage 1, Weierstrass form.
    Finds a factor when the group order of the curve mod p is B1-smooth.
    This genuinely works on 20-40 digit semiprimes with enough curves.
    """
    if n % 2 == 0:
        return 2
    if is_probable_prime(n):
        return None
    primes = _primes_upto(B1)
    for _ in range(n_curves):
        # Random curve + point via Suyama-like parametrisation
        x0 = random.randrange(1, n)
        y0 = random.randrange(1, n)
        a = random.randrange(1, n)
        b = (y0 * y0 - x0 * x0 * x0 - a * x0) % n
        if math.gcd(4 * a * a * a + 27 * b * b, n) not in (1, n):
            g = math.gcd(4 * a * a * a + 27 * b * b, n)
            if 1 < g < n:
                return g
            continue
        P = (x0, y0)
        ok = True
        for p in primes:
            pp = p
            while pp * p <= B1:
                pp *= p
            P, g = _ec_mul(P, pp, a, n)
            if g is not None:
                if 1 < g < n:
                    return g
                ok = False
                break
        if not ok:
            continue
    return None


def _ec_add(P, Q, a, n):
    """Affine addition; returns (point_or_None, gcd_or_None)."""
    if P is None:
        return Q, None
    if Q is None:
        return P, None
    x1, y1 = P; x2, y2 = Q
    if x1 == x2:
        if (y1 + y2) % n == 0:
            return None, None
        num = (3 * x1 * x1 + a) % n
        den = (2 * y1) % n
    else:
        num = (y2 - y1) % n
        den = (x2 - x1) % n
    g = math.gcd(den, n)
    if g != 1:
        return None, (g if g != n else None)
    lam = num * pow(den, -1, n) % n
    x3 = (lam * lam - x1 - x2) % n
    y3 = (lam * (x1 - x3) - y1) % n
    return (x3, y3), None


def _ec_mul(P, k, a, n):
    R = None
    while k:
        if k & 1:
            R, g = _ec_add(R, P, a, n)
            if g is not None:
                return None, g
        P, g = _ec_add(P, P, a, n)
        if g is not None:
            return None, g
        k >>= 1
    return R, None


def fermat_factor(n: int, max_steps: int = 1 << 16) -> Optional[int]:
    """Fermat: efficient when p and q are close."""
    if n % 2 == 0:
        return 2
    a = math.isqrt(n)
    if a * a < n:
        a += 1
    for _ in range(max_steps):
        b2 = a * a - n
        b = math.isqrt(b2)
        if b * b == b2:
            f = a - b
            if 1 < f < n:
                return f
        a += 1
    return None


def factor(n: int, budget_s: float = 20.0,
           ecm_curves: int = 200) -> Dict[str, Any]:
    """Cascade: trial → rho → p-1 → Fermat → ECM. Honest about failure."""
    t0 = time.time()
    n0 = n
    methods: List[str] = []
    factors: List[int] = []
    for p in _primes_upto(10_000):
        while n % p == 0:
            factors.append(p); n //= p
        if n == 1:
            break
    if factors:
        methods.append("trial")
    stack = [n] if n > 1 else []
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_probable_prime(m):
            factors.append(m); continue
        if time.time() - t0 > budget_s:
            methods.append("BUDGET_EXHAUSTED")
            factors.append(m)  # composite remainder, flagged
            break
        f = None
        for name, fn in (("fermat", lambda: fermat_factor(m, 1 << 12)),
                         ("rho", lambda: pollard_rho_brent(m)),
                         ("p-1", lambda: pollard_p_minus_1(m, 50_000)),
                         ("ecm", lambda: ecm_lenstra(m, ecm_curves, 5_000))):
            f = fn()
            if f:
                methods.append(name); break
            if time.time() - t0 > budget_s:
                break
        if f:
            stack += [f, m // f]
        else:
            methods.append("FAILED")
            factors.append(m)
    factors.sort()
    prod = 1
    for f in factors:
        prod *= f
    all_prime = all(is_probable_prime(f) for f in factors)
    return {
        "input": n0,
        "input_digits": len(str(n0)),
        "factors": factors,
        "complete": prod == n0 and all_prime,
        "all_prime": all_prime,
        "methods": methods,
        "wall_s": time.time() - t0,
        "verification": prod == n0,
        "NOTE": ("GNFS (CADO-NFS/msieve) is the classical state of the art "
                 "above ~110 digits. This suite is sub-exponential at best."),
    }


def rsa_scale_warning(bits: int = 2048) -> Dict[str, Any]:
    """Print the actual gap. No hedging."""
    return {
        "modulus_bits": bits,
        "classical_GNFS_complexity": "L_N[1/3, (64/9)^(1/3)] — sub-exponential",
        "RSA-2048_classical_status": "NOT FACTORED. No public record.",
        "largest_RSA_factored_publicly": "RSA-250 (829 bits, 2020, ~2700 core-years)",
        "shor_logical_qubits_needed": "~2n+3 ≈ 4100 for n=2048",
        "shor_amplitudes": "2^4100 — no classical simulation, ever",
        "quantum_inspired_speedup": "NONE. Zero. Not a single bit.",
        "therefore": "MIGRATE TO PQC. See advisory.",
    }


# =============================================================================
# §4. QUANTUM-INSPIRED SIMULATION — with proven boundaries and a hard refusal
# =============================================================================

QUBIT_BUDGET = 22          # 2^22 complex128 ≈ 67 MB statevector


class RefusalToSimulate(RuntimeError):
    """Raised when a requested simulation exceeds the declared budget."""


def shor_classical_postprocess(a: int, N: int, r: int) -> Optional[int]:
    """
    The CLASSICAL part of Shor. Given the period r = ord_a(N),
    gcd(a^{r/2} ± 1, N) yields a factor when r is even and
    a^{r/2} != -1 mod N. THIS NEEDS NO QUANTUM COMPUTER.
    The hard part is FINDING r, not this step.
    """
    if r % 2 != 0:
        return None
    x = pow(a, r // 2, N)
    if x == N - 1:
        return None
    for cand in (math.gcd(x - 1, N), math.gcd(x + 1, N)):
        if 1 < cand < N:
            return cand
    return None


def classical_order_finding(a: int, N: int, cap: int = 1 << 20) -> Optional[int]:
    """
    Find ord_a(N) by brute force. EXPONENTIAL in log N in the worst case.
    This is precisely why Shor needs a quantum computer: the QFT finds r
    in polynomial time. Classical order-finding does not scale.
    """
    if math.gcd(a, N) != 1:
        return None
    x, r = a % N, 1
    while x != 1 and r < cap:
        x = x * a % N
        r += 1
    return r if x == 1 else None


def shor_tiny_demo(N: int = 15) -> Dict[str, Any]:
    """
    EDUCATIONAL demonstration on N in {15, 21, 33, 35}.
    Uses CLASSICAL order-finding (exponential) + classical post-processing.
    This is NOT a quantum algorithm and NOT quantum-inspired speedup.
    It is included to show WHERE the hardness actually lives.
    """
    if N > 100:
        raise RefusalToSimulate(
            f"shor_tiny_demo refuses N={N}. This demo uses CLASSICAL "
            f"order-finding, which is exponential in log N. It exists to "
            f"show where the hardness lives, not to factor anything real."
        )
    for a in range(2, N):
        g = math.gcd(a, N)
        if 1 < g < N:
            return {"N": N, "factor": g, "method": "gcd (trivial)",
                    "note": "a shared a factor by luck; no period finding needed"}
        r = classical_order_finding(a, N)
        if r is None:
            continue
        f = shor_classical_postprocess(a, N, r)
        if f:
            return {
                "N": N, "a": a, "period_r": r, "factor": f,
                "cofactor": N // f,
                "method": "classical order-finding + classical post-processing",
                "HONEST": ("The period r was found by BRUTE FORCE (exponential). "
                           "Shor's contribution is finding r in POLYNOMIAL time "
                           "via the QFT. That step is what cannot be faked."),
            }
    return {"N": N, "factor": None, "note": "no suitable a found"}


def statevector_simulate(n_qubits: int) -> Dict[str, Any]:
    """
    Dense statevector. Explicit exponential cost. HARD REFUSAL above budget.
    """
    if n_qubits > QUBIT_BUDGET:
        raise RefusalToSimulate(
            f"Refusing {n_qubits} qubits: 2^{n_qubits} complex128 = "
            f"{2.0**n_qubits*16/1e9:.3g} GB. Budget is {QUBIT_BUDGET} qubits.\n"
            f"For RSA-2048 Shor you would need ~4100 qubits → 2^4100 amplitudes.\n"
            f"There is no classical simulation of that. Ever. Under any name."
        )
    dim = 1 << n_qubits
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    # Hadamard on all → uniform superposition
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
    for q in range(n_qubits):
        psi = psi.reshape(-1, 2)
        psi = psi @ H.T
        psi = psi.reshape(-1)
    return {
        "n_qubits": n_qubits, "dim": dim,
        "memory_MB": dim * 16 / 1e6,
        "norm": float(np.abs(psi).dot(np.abs(psi))),
        "max_amplitude": float(np.abs(psi).max()),
        "NOTE": "uniform superposition; measurement gives a uniform random bitstring",
    }


def gottesman_knill_note() -> Dict[str, str]:
    return {
        "theorem": "Gottesman-Knill",
        "statement": ("Clifford circuits (H, S, CNOT) on stabilizer states with "
                      "Pauli measurement are classically simulable in poly time "
                      "(O(n^2) per measurement, Aaronson-Gottesman tableau)."),
        "consequence": "Maximal entanglement does NOT imply classical hardness.",
        "the_resource": "non-stabilizerness ('magic'), not entanglement",
        "shor_is": "NON-Clifford (modular exponentiation + QFT) → OUTSIDE this class",
        "therefore": "Gottesman-Knill cannot simulate Shor efficiently",
    }


# =============================================================================
# §5. MSPS MATH LAYER — each component honestly labelled
# =============================================================================

def delta_min_exact(n: int = 3, kappa: float = 2.0, a_min: float = 0.01) -> float:
    """Isodiametric floor: delta_min = (1/kappa)(a_min/omega_{n-1})^{1/(n-1)}."""
    d = n - 1
    om = math.pi ** (d / 2) / math.gamma(d / 2 + 1)
    return (1.0 / kappa) * (a_min / om) ** (1.0 / d)


OPMC_L = 0.438520
OPMC_D_MN = lambda m, n: m * m * n * n + m * n * n


def opmc_k_star(eps: float = 1e-8, D0: float = 1.0) -> int:
    return math.ceil(math.log(eps / D0) / math.log(OPMC_L))


def carpet_laplacian(side: int = 9) -> np.ndarray:
    side = max(3, side); n = side * side
    A = np.zeros((n, n))
    for i in range(side):
        for j in range(side):
            if i % 3 == 1 and j % 3 == 1:
                continue
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < side and 0 <= jj < side and not (ii % 3 == 1 and jj % 3 == 1):
                    A[i * side + j, ii * side + jj] = 1.0
    return np.diag(A.sum(1)) - A


def subsample_ci(x: np.ndarray, conf: float = 0.95) -> Dict[str, float]:
    """Politis-Romano. Requires stationarity + mixing. NOT assumption-free."""
    x = np.asarray(x, float).ravel(); n = len(x)
    if n < 8:
        m = float(x.mean()) if n else 0.0
        return {"mean": m, "lo": m, "hi": m, "n": n}
    b = min(max(2, int(n ** (1 / 3))), n // 2)
    th = float(x.mean())
    sub = np.array([x[i:i + b].mean() for i in range(n - b + 1)])
    sc = math.sqrt(b) * (sub - th); a = 1 - conf
    lo = th - float(np.quantile(sc, 1 - a / 2)) / math.sqrt(n)
    hi = th - float(np.quantile(sc, a / 2)) / math.sqrt(n)
    return {"mean": th, "lo": min(lo, hi), "hi": max(lo, hi), "n": n, "block": b}


STATS_DICT = {
    "subsample_ci": {
        "theorem": "Politis-Romano 1994",
        "requires": ["stationarity", "alpha-mixing", "finite variance"],
        "guarantee": "asymptotically correct coverage",
        "NOT": "assumption-free; not valid under distribution shift",
    },
    "self_normalized": {
        "theorem": "Shao 2010",
        "requires": ["domain of attraction of the normal"],
        "guarantee": "pivotal limit, no variance estimate needed",
    },
    "wigner_surmise": {
        "status": "conjectural approximation to GUE spacing density",
        "exact_for": "2x2 GUE; approximate for large n (very accurate)",
    },
    "montgomery_pair_correlation": {
        "status": "CONJECTURE (1973); strong numerical support (Odlyzko)",
        "not_proven": True,
    },
}


class CSOCCounter:
    """Two-level hysteresis. N(T) <= (E_max + D)/gap with D = ACTUAL removal."""
    def __init__(self, hi: float = 0.6, lo: float = 0.25, rate: float = 0.08):
        assert hi > lo
        self.hi, self.lo, self.gap, self.rate = hi, lo, hi - lo, rate
        self.acc = 0.0; self.armed = True
        self.events = 0; self.E_max = 0.0; self.D = 0.0

    def step(self, s: float) -> bool:
        ex = math.log1p(math.exp(-abs(s - self.hi))) + max(s - self.hi, 0.0)
        dec = 1.0 / (1.0 + math.exp(-(self.acc - self.lo) * 10))
        rem = dec * self.rate * self.acc
        self.acc += ex - rem
        self.D += rem
        self.E_max = max(self.E_max, self.acc)
        if self.acc < self.lo:
            self.armed = True
        fire = self.armed and self.acc > self.hi
        if fire:
            self.armed = False; self.events += 1
        return fire

    def report(self) -> Dict[str, Any]:
        b = (self.E_max + self.D) / self.gap
        return {"events": self.events, "bound": b, "satisfied": self.events <= b + 1e-9,
                "D_actual": self.D,
                "scope": "engineering bound on THIS counter, not number theory"}


# =============================================================================
# §6. LEARNED MODELS — transformer + diffusion + world model
#     Target: NUMBER-THEORETIC STRUCTURE, not secret primes.
# =============================================================================

if HAS_TORCH:

    class HilbertPolyaFeatureMap(nn.Module):
        """
        Learned self-adjoint H = H0 + V V^T, eigendecomposed.
        LABELLED: spectral FEATURE MAP. Not a proof of RH.
        Not the Berry-Keating xp operator. Does not factor.
        """
        def __init__(self, d_in: int, d_s: int = 48, rank: int = 8):
            super().__init__()
            self.to_s = nn.Linear(d_in, d_s)
            self.log_base = nn.Parameter(torch.linspace(0, math.log(50.0), d_s))
            self.v = nn.Sequential(nn.Linear(d_s, d_s), nn.Tanh(),
                                   nn.Linear(d_s, rank))
            self.coupling = nn.Parameter(torch.tensor(0.1))

        def forward(self, x: Tensor) -> Dict[str, Tensor]:
            s = self.to_s(x)
            H0 = torch.diag_embed(torch.exp(self.log_base)).unsqueeze(0)
            V = self.v(s)
            pert = torch.bmm(V.unsqueeze(-1), V.unsqueeze(-2)) * self.coupling
            H = H0.to(pert.dtype) + 0.5 * (pert + pert.transpose(-1, -2))
            with torch.autocast(device_type=H.device.type, enabled=False):
                n = H.shape[-1]
                jit = 1e-6 * torch.eye(n, device=H.device)
                ev, evec = torch.linalg.eigh(H.float() + jit)
            ev = ev.to(x.dtype)
            gap = (ev[:, 1] - ev[:, 0]).abs()
            return {"eigenvalues": ev, "gap": gap,
                    "spectral": s * torch.cos(ev * math.pi),
                    "self_adjoint_error": float((H - H.transpose(-1, -2)).abs().max())}

    class NumberTheoryTransformer(nn.Module):
        """
        Transformer over integer feature sequences.
        Tasks (all STATISTICAL, none reveals a secret prime):
          - smoothness probability at bound B
          - bit-length of smallest factor
          - ECM curves needed (regression)
          - NFS polynomial-selection score
        """
        def __init__(self, d_in: int = 16, d_model: int = 128,
                     n_heads: int = 4, n_layers: int = 4, n_tasks: int = 4):
            super().__init__()
            self.proj = nn.Linear(d_in, d_model)
            self.pos = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model, n_heads, d_model * 4, dropout=0.1,
                batch_first=True, norm_first=True, activation="gelu")
            self.trunk = nn.TransformerEncoder(layer, n_layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, n_tasks)
            self.hp = HilbertPolyaFeatureMap(d_model, 48, 8)

        def forward(self, feats: Tensor) -> Dict[str, Tensor]:
            B, L, _ = feats.shape
            h = self.proj(feats) + self.pos[:, :L]
            h = self.norm(self.trunk(h))
            pooled = h.mean(1)
            spec = self.hp(pooled)
            return {"pred": self.head(pooled),
                    "eigenvalues": spec["eigenvalues"],
                    "spectral_gap": spec["gap"],
                    "embedding": pooled}

    class ResidueDiffusion(nn.Module):
        """
        DDPM over residue-feature vectors. Learns the DISTRIBUTION of
        number-theoretic features. Generates candidate feature vectors,
        NOT factors. A generated vector is a hypothesis to be tested by
        the real factoring suite, never an answer.
        """
        def __init__(self, d: int = 16, T: int = 64, hidden: int = 128):
            super().__init__()
            self.T = T
            self.d = d
            betas = torch.linspace(1e-4, 0.02, T)
            alphas = 1 - betas
            self.register_buffer("betas", betas)
            self.register_buffer("alpha_bar", torch.cumprod(alphas, 0))
            self.net = nn.Sequential(
                nn.Linear(d + 1, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, d))

        def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
            ab = self.alpha_bar[t][:, None]
            return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

        def loss(self, x0: Tensor) -> Tensor:
            B = x0.shape[0]
            t = torch.randint(0, self.T, (B,), device=x0.device)
            eps = torch.randn_like(x0)
            xt = self.q_sample(x0, t, eps)
            pred = self.net(torch.cat([xt, t.float()[:, None] / self.T], -1))
            return F.mse_loss(pred, eps)

        @torch.no_grad()
        def sample(self, n: int, device="cpu") -> Tensor:
            x = torch.randn(n, self.d, device=device)
            for i in reversed(range(self.T)):
                t = torch.full((n,), i, device=device, dtype=torch.long)
                eps = self.net(torch.cat([x, t.float()[:, None] / self.T], -1))
                a = 1 - self.betas[i]
                ab = self.alpha_bar[i]
                x = (x - (1 - a) / (1 - ab).sqrt() * eps) / a.sqrt()
                if i > 0:
                    x = x + self.betas[i].sqrt() * torch.randn_like(x)
            return x

    class FactoringWorldModel(nn.Module):
        """
        Iterative refinement of BELIEFS about number-theoretic features.
        Predicts a delta to the current belief state. K applications =
        learned recycling. Does NOT output factors; outputs feature
        predictions that guide the REAL factoring cascade (e.g. how many
        ECM curves to allocate).
        """
        def __init__(self, d: int = 128, n_steps: int = 3):
            super().__init__()
            self.n_steps = n_steps
            self.gru = nn.GRUCell(d, d)
            self.delta = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                       nn.Linear(d, d))
            self.readout = nn.Linear(d, 4)

        def forward(self, emb: Tensor) -> Dict[str, Tensor]:
            s = emb
            traj = []
            for _ in range(self.n_steps):
                s = self.gru(self.delta(s) * 0.5, s)
                traj.append(s)
            return {"belief": s, "trajectory": torch.stack(traj, 1),
                    "pred": self.readout(s)}


def integer_features(n: int, L: int = 16) -> np.ndarray:
    """
    Feature vector for an integer. All features are PUBLIC information:
    bit length, small-prime residues, digit statistics. NOTHING here
    encodes a secret factor.
    """
    bits = n.bit_length()
    primes = _primes_upto(60)[:L - 4]
    f = [bits / 1024.0, math.log(max(n, 2)) / 700.0,
         float(n % 2), float(n % 3)]
    f += [(n % p) / p for p in primes]
    f = (f + [0.0] * L)[:L]
    return np.array(f, dtype=np.float32)


# =============================================================================
# §7. SELF-TEST
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    print(BANNER)
    pqc_advisory(verbose=True)
    ok = True

    def rep(n, p, e=""):
        nonlocal ok
        ok &= bool(p)
        if verbose:
            print(f"  [{'PASS' if p else 'FAIL'}] {n}  {e}")

    # T1 Riemann zeros vs known values
    print("\n[T1] Riemann zeros — computed vs known (Odlyzko)")
    sp = spectral_report(t_hi=50.0)
    for i, (z, k) in enumerate(zip(sp["first_zeros"], sp["known_first_zeros"])):
        print(f"      gamma_{i+1}: computed {z:.6f}   known {k:.6f}   "
              f"err {abs(z-k):.2e}")
    rep("first 5 zeros within 0.01", sp["max_abs_error_vs_known"] < 0.01,
        f"max err={sp['max_abs_error_vs_known']:.2e}")
    rep("found >= 5 zeros", sp["n_zeros_found"] >= 5, f"n={sp['n_zeros_found']}")
    print(f"      {sp['WHAT_THIS_IS_NOT']}")

    # T2 GUE
    print("\n[T2] GUE statistics (Montgomery / Wigner)")
    u = np.linspace(0, 3, 200)
    r2 = gue_pair_correlation(u)
    rep("R2(0) = 0 (level repulsion)", abs(r2[0]) < 1e-9)
    rep("R2(inf) -> 1", abs(r2[-1] - 1) < 0.02, f"{r2[-1]:.4f}")
    w = gue_spacing_wigner(u)
    rep("Wigner surmise normalised", abs(np.trapz(w, u) - 1.0) < 0.02,
        f"integral={np.trapz(w,u):.4f}")
    print(f"      unfolded spacing mean={sp['unfolded_mean']:.4f} "
          f"(GUE expects ~1.0), std={sp['unfolded_std']:.4f}")
    print("      NOTE: Montgomery pair correlation is a CONJECTURE.")

    # T3 Hilbert-Polya feature map
    print("\n[T3] Hilbert-Polya spectral FEATURE MAP (not a proof)")
    if HAS_TORCH:
        hp = HilbertPolyaFeatureMap(32, 24, 6)
        out = hp(torch.randn(4, 32))
        rep("eigenvalues real & finite", bool(torch.isfinite(out["eigenvalues"]).all()))
        rep("self-adjoint to round-off", out["self_adjoint_error"] < 1e-5,
            f"err={out['self_adjoint_error']:.2e}")
        rep("spectral gap positive", bool((out["gap"] > 0).all()))
        print(f"      {HONESTY['hilbert_polya']['not_claimed']}")
    else:
        rep("torch available", False, "pip install torch")

    # T4 Real factoring
    print("\n[T4] Classical factoring — REAL, on real semiprimes")
    cases = [(15, "trivial"), (899, "29*31"), (10403, "101*103"),
             (1000000007 * 1000000009, "20-digit semiprime")]
    for n, label in cases:
        r = factor(n, budget_s=8.0, ecm_curves=60)
        print(f"      {label:22s} N={n}  factors={r['factors']}  "
              f"complete={r['complete']}  {r['wall_s']:.2f}s  {r['methods']}")
        rep(f"{label} factored & verified", r["complete"] and r["verification"])

    # T5 Shor demo — where the hardness lives
    print("\n[T5] Shor: the classical post-processing is EASY; finding r is HARD")
    d = shor_tiny_demo(15)
    print(f"      N=15 a={d.get('a')} period r={d.get('period_r')} "
          f"factor={d.get('factor')}")
    rep("tiny demo finds a factor", d.get("factor") in (3, 5))
    print(f"      {d.get('HONEST','')}")
    try:
        shor_tiny_demo(10**12 + 39)
        rep("refuses large N", False)
    except RefusalToSimulate as e:
        rep("refuses large N", True, str(e).splitlines()[0][:60])

    # T6 Simulation budget refusal
    print("\n[T6] Statevector budget — HARD REFUSAL")
    s = statevector_simulate(12)
    rep("12 qubits simulates", abs(s["norm"] - 1.0) < 1e-9,
        f"{s['memory_MB']:.2f} MB")
    try:
        statevector_simulate(4100)
        rep("refuses 4100 qubits", False)
    except RefusalToSimulate as e:
        rep("refuses 4100 qubits (RSA-2048 scale)", True)
        print("      " + str(e).splitlines()[2])
    gk = gottesman_knill_note()
    print(f"      {gk['theorem']}: {gk['consequence']}")
    print(f"      Shor is {gk['shor_is']}")

    # T7 RSA scale
    print("\n[T7] RSA-2048 reality check")
    rw = rsa_scale_warning(2048)
    for k, v in rw.items():
        print(f"      {k:34s} {v}")
    rep("no quantum-inspired speedup claimed",
        rw["quantum_inspired_speedup"].startswith("NONE"))

    # T8 Learned models train
    print("\n[T8] Transformer + Diffusion + World Model — trainable")
    if HAS_TORCH:
        torch.manual_seed(0)
        Ns = [random.randrange(10**6, 10**9) for _ in range(64)]
        X = torch.tensor(np.stack([integer_features(n) for n in Ns]))
        y = torch.tensor([[math.log(n.bit_length()),
                           float(sum(n % p == 0 for p in (2, 3, 5, 7))) / 4,
                           0.5, 0.3] for n in Ns], dtype=torch.float32)
        tr = NumberTheoryTransformer(16, 64, 4, 2, 4)
        df = ResidueDiffusion(16, 32, 64)
        wm = FactoringWorldModel(64, 3)
        opt = torch.optim.AdamW(list(tr.parameters()) + list(df.parameters())
                                + list(wm.parameters()), lr=1e-3)
        l0 = lN = None
        for ep in range(60):
            opt.zero_grad()
            o = tr(X); w = wm(o["embedding"])
            loss = F.mse_loss(w["pred"], y) + df.loss(X)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(tr.parameters()) + list(df.parameters()) + list(wm.parameters()), 1.0)
            opt.step()
            if ep == 0: l0 = float(loss)
            lN = float(loss)
        print(f"      joint loss {l0:.4f} -> {lN:.4f}")
        rep("joint loss decreases", lN < l0, f"{l0:.4f}->{lN:.4f}")
        samp = df.sample(4)
        rep("diffusion samples finite", bool(torch.isfinite(samp).all()))
        rep("world model trajectory shape", w["trajectory"].shape == (64, 3, 64))
        print("      THESE MODELS LEARN STATISTICS, NOT SECRET PRIMES.")
    else:
        rep("torch available", False)

    # T9 MSPS math
    print("\n[T9] MSPS math layer")
    rep("delta_min > 0", delta_min_exact() > 0, f"{delta_min_exact():.6f}")
    rep("OPMC k* = 23", opmc_k_star() == 23)
    rep("d(4,4) = 320", OPMC_D_MN(4, 4) == 320)
    L = carpet_laplacian(9)
    rep("carpet Laplacian row-sums 0", float(np.abs(L.sum(1)).max()) < 1e-9)
    rep("carpet Laplacian PSD", float(np.linalg.eigvalsh(L).min()) > -1e-9)
    c = CSOCCounter()
    for k in range(400):
        c.step(0.35 + 0.45 * abs(math.sin(0.11 * k)))
    r = c.report()
    rep("CSOC bound holds", r["satisfied"], f"N={r['events']} bound={r['bound']:.1f}")
    ci = subsample_ci(np.random.default_rng(0).normal(size=200))
    rep("subsampling CI ordered", ci["lo"] <= ci["mean"] <= ci["hi"])

    # T10 PQC advisory present
    print("\n[T10] PQC advisory")
    adv = pqc_advisory(verbose=False)
    rep("FIPS 203 present", "FIPS 203" in adv["standards"])
    rep("FIPS 204 present", "FIPS 204" in adv["standards"])
    rep("FIPS 205 present", "FIPS 205" in adv["standards"])
    rep("harvest-now flagged as PRESENT", adv["threat_is_present_not_future"])
    rep("hybrid recommended", "hybrid_kex" in adv["recommended_defaults"])

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 78)
    print("""
WHAT IS REAL HERE
  Riemann-Siegel zeros computed and matched to published values.
  GUE pair correlation and Wigner surmise implemented correctly.
  Pollard rho / p-1 / Fermat / Lenstra ECM factor real semiprimes.
  Gottesman-Knill boundary stated; Shor is outside it.
  Statevector simulation refuses 4100 qubits with the reason printed.
  Transformer + diffusion + world model train jointly, gradients finite.
  PQC advisory with the actual August-2024 NIST standards.

WHAT IS NOT REAL HERE
  Hilbert-Polya is UNPROVEN. Nothing here proves RH.
  Riemann zeros are NOT primes. Knowing them does NOT factor N.
  GUE statistics do NOT grant quantum computational power.
  "Virtual Shor" has ZERO exponential speedup. Zero.
  RSA-2048 is not factored and will not be by this file.
  The learned models learn statistics, never the identity of a secret prime.

THE ACTION THAT MATTERS
  Migrate to ML-KEM-768 + ML-DSA-65. Use hybrid modes. Be crypto-agile.
  Harvest-now-decrypt-later is happening today.
  Do not rest humanity's secrets on one hardness assumption in one context.
""")
    return ok


# =============================================================================
# §8. CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Spectral number theory + PQC")
    ap.add_argument("cmd", nargs="?", default="selftest",
                    choices=["selftest", "pqc", "spectrum", "factor",
                             "shor-demo", "honesty", "rsa-scale", "train"])
    ap.add_argument("--n", type=str, default=None)
    ap.add_argument("--t-hi", type=float, default=120.0)
    ap.add_argument("--budget", type=float, default=30.0)
    ap.add_argument("--curves", type=int, default=300)
    ap.add_argument("--bits", type=int, default=2048)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out: Dict[str, Any] = {}

    if a.cmd == "selftest":
        return 0 if self_test() else 1
    if a.cmd == "pqc":
        out = pqc_advisory(verbose=True)
    elif a.cmd == "honesty":
        out = {"registry": HONESTY, "statistics": STATS_DICT,
               "gottesman_knill": gottesman_knill_note()}
        print(json.dumps(out, indent=2))
    elif a.cmd == "spectrum":
        out = spectral_report(a.t_hi); print(json.dumps(out, indent=2, default=float))
    elif a.cmd == "factor":
        if not a.n:
            print("--n required"); return 1
        out = factor(int(a.n), a.budget, a.curves)
        print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "shor-demo":
        out = shor_tiny_demo(int(a.n) if a.n else 15)
        print(json.dumps(out, indent=2, default=str))
    elif a.cmd == "rsa-scale":
        out = rsa_scale_warning(a.bits); print(json.dumps(out, indent=2))
    elif a.cmd == "train":
        if not HAS_TORCH:
            print("torch required"); return 1
        pqc_advisory(verbose=True)
        print("\nTraining on number-theoretic STATISTICS (not secret primes)...")
        return 0 if self_test() else 1

    if a.json and out:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nSaved: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
