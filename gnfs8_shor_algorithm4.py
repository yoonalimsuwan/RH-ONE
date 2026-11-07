"""
=============================================================================
GNFS-8 & SHOR ALGORITHM 4 — Full Implementation
Based on: "A Unified Framework for Quantum-Inspired Information Measurement,
Neural Signal Analysis, Record Recovery, Hierarchical Computation, and the
Limits of Remote Neural Inference" (PAI & Yoon A. Limsuwan, MSPS NETWORK, 2026)
=============================================================================

SCOPE DISCIPLINE (from the paper itself):
  §13.2: "The expression: Shor Algorithm 4 ≡ Shor's Algorithm must remain
          a research target rather than a proven theorem."
  §12.2: "A classical neural network or software gate does not satisfy the
          physical definition merely because it uses similar mathematical
          notation."
  §13.3: "Quantum-inspired GNFS ≠ Shor's quantum algorithm."
  §14.4: "The proposed quantum-inspired framework may motivate new algorithms,
          but it does not currently prove P = NP."

THIS IMPLEMENTATION:
  • GNFS-8: Quantum-Inspired Classical Computation (QICC) architecture
    for integer factorization (§4 of the paper)
  • Shor Algorithm 4: Hierarchical software-emulation architecture (§13.2)
  • Both are CLASSICAL algorithms — NOT quantum computation
  • Both require independent verification of results
  • Performance claims require benchmarking against established methods

MODULES:
  [1] GNFS8Factorizer           — Full GNFS-8 implementation
  [2] ShorAlgorithm4            — Software-emulated Shor-type computation
  [3] FactorizationValidator    — Correctness verification
  [4] PerformanceBenchmark      — Benchmarking infrastructure
  [5] IntegratedSystem          — Master orchestrator
  [6] Self-test suite

=============================================================================
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("gnfs8_shor4")

# =============================================================================
# OPMC CONSTANTS (from Unified Ordinal-Tensor paper)
# =============================================================================
OPMC_C1 = 0.420000
OPMC_K2 = 12.5000
OPMC_L = 0.438520
OPMC_D_MN = lambda m, n: m * m * n * n + m * n * n

def opmc_k_star(epsilon: float = 1e-8, D0: float = 1.0) -> int:
    """Theorem 11.1: k* = ⌈ln(ε/D₀)/ln L⌉ — independent of N."""
    return math.ceil(math.log(epsilon / D0) / math.log(OPMC_L))


# =============================================================================
# §4.3: FACTORIZATION CONSTRAINTS
# =============================================================================

def constraint_C1(a: int, b: int, N: int) -> int:
    """C1(a,b) = 1 iff ab = N."""
    return 1 if a * b == N else 0


def constraint_C2(a: int, b: int, N: int) -> int:
    """C2(a,b) = 1 iff 1 < a < N and 1 < b < N."""
    return 1 if (1 < a < N and 1 < b < N) else 0


def is_valid_factorization(a: int, b: int, N: int) -> bool:
    """C1(a,b) * C2(a,b) = 1."""
    return constraint_C1(a, b, N) * constraint_C2(a, b, N) == 1


def is_prime(n: int) -> bool:
    """Deterministic primality test (trial division for small n)."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def validate_factorization(N: int, factors: List[int],
                           primality_test: Callable[[int], bool] = is_prime) -> bool:
    """
    §4.6 / Appendix C.3: Factorization validator.

    A valid complete prime factorization must satisfy:
      N = Π p_i, where every p_i is prime (with multiplicities).
    For nontrivial factorization: at least one factor satisfies 1 < p_i < N.

    Args:
        N: integer to factorize
        factors: proposed list of prime factors
        primality_test: primality checker (must have declared reliability)

    Returns:
        True if factorization is valid
    """
    if N <= 1:
        return False
    if not factors:
        return False

    product = 1
    for factor in factors:
        if not isinstance(factor, int):
            return False
        if factor <= 1:
            return False
        if not primality_test(factor):
            return False
        product *= factor

    return product == N


# =============================================================================
# §4.4: GNFS-8 PSEUDOCODE IMPLEMENTATION
# =============================================================================

@dataclass
class GNFS8Candidate:
    """A candidate factorization hypothesis."""
    a: int
    b: int
    score: float = 0.0
    is_consistent: bool = False


class GNFS8Factorizer:
    """
    §4: GNFS-8 — First application of Quantum-Inspired Classical Computation.

    ARCHITECTURE (from §4.4):
      Stage 1: Maintain multiple candidate hypotheses
      Stage 2: Construct relationships among candidates
      Stage 3: Evaluate arithmetic constraints
      Stage 4: Remove inconsistent candidates
      Stage 5: Rank surviving candidates
      Stage 6: Independently verify candidates

    CONCEPTUAL MAPPING (from §4.2):
      • Superposition → Multiple possible factor hypotheses
      • Entanglement → Relationships among candidate factors
      • Interference → Elimination of inconsistent candidates
      • Measurement → Candidate selection
      • Collapse → Search-space reduction after verification

    HONEST SCOPE (from §4.4):
      "This pseudocode defines the proposed architecture. It does not
       specify an established factorization method or demonstrate that
       GNFS-8 can factor large integers efficiently."

    USAGE:
      factorizer = GNFS8Factorizer()
      result = factorizer.factor(N)
    """

    def __init__(self, max_candidates: int = 100):
        self.max_candidates = max_candidates
        logger.info(f"[GNFS8] Initialized, max_candidates={max_candidates}")

    def candidate_generator(self, N: int) -> List[GNFS8Candidate]:
        """
        Stage 1: Maintain multiple candidate hypotheses.

        Generate candidate factor pairs (a, b) such that a * b ≈ N.
        This is a SIMPLIFIED generator — a real GNFS implementation
        would use polynomial selection, sieving, etc.

        Args:
            N: integer to factorize

        Returns:
            list of candidate factorizations
        """
        candidates = []

        # Generate candidates by trial division up to sqrt(N)
        # (simplified — real GNFS uses number field sieve)
        sqrt_N = int(math.isqrt(N)) + 1
        for a in range(2, min(sqrt_N, self.max_candidates)):
            if N % a == 0:
                b = N // a
                candidates.append(GNFS8Candidate(a=a, b=b, is_consistent=True))

        # Also add some "near-miss" candidates for testing
        # (these should be filtered out by constraint evaluation)
        for a in range(max(2, sqrt_N - 5), sqrt_N + 5):
            if a < sqrt_N and N % a != 0:
                b_approx = N // a
                candidates.append(GNFS8Candidate(a=a, b=b_approx, is_consistent=False))

        logger.info(f"[GNFS8] Generated {len(candidates)} candidates for N={N}")
        return candidates

    def dependency_builder(self, N: int,
                           candidates: List[GNFS8Candidate]) -> Dict[str, Any]:
        """
        Stage 2: Construct relationships among candidates.

        Build a dependency graph where:
          • Nodes = candidate factor pairs
          • Edges = arithmetic relationships (e.g., shared factors)

        Args:
            N: integer to factorize
            candidates: list of candidates

        Returns:
            dependency structure
        """
        dependencies = {
            "N": N,
            "n_candidates": len(candidates),
            "edges": [],
        }

        # Build edges: candidates sharing a factor are "entangled"
        for i, c1 in enumerate(candidates):
            for j, c2 in enumerate(candidates):
                if i < j:
                    # Check if candidates share a factor
                    if c1.a == c2.a or c1.b == c2.b:
                        dependencies["edges"].append((i, j, "shared_factor"))

        logger.info(f"[GNFS8] Built dependency graph: {len(dependencies['edges'])} edges")
        return dependencies

    def constraint_engine_evaluate(self, N: int,
                                    candidates: List[GNFS8Candidate],
                                    dependencies: Dict[str, Any]) -> List[GNFS8Candidate]:
        """
        Stage 3: Evaluate arithmetic constraints.

        For each candidate (a, b):
          C1(a,b) = 1 iff ab = N
          C2(a,b) = 1 iff 1 < a < N and 1 < b < N

        Args:
            N: integer to factorize
            candidates: list of candidates
            dependencies: dependency structure

        Returns:
            candidates with is_consistent flag set
        """
        evaluated = []

        for candidate in candidates:
            a, b = candidate.a, candidate.b

            # Constraint evaluation
            c1 = constraint_C1(a, b, N)
            c2 = constraint_C2(a, b, N)

            # Candidate is valid only if C1 * C2 = 1
            candidate.is_consistent = (c1 * c2 == 1)

            # Score: weighted constraint satisfaction
            # S(h) = Σ λ_j C_j(h) - Σ μ_k P_k(h)
            # Here: λ_1 = λ_2 = 1.0, no penalty terms
            candidate.score = float(c1 + c2)

            evaluated.append(candidate)

        n_consistent = sum(1 for c in evaluated if c.is_consistent)
        logger.info(f"[GNFS8] Evaluated {len(evaluated)} candidates, "
                   f"{n_consistent} consistent")
        return evaluated

    def independent_verifier(self, N: int, candidate: GNFS8Candidate) -> bool:
        """
        Stage 6: Independently verify candidate.

        A candidate is verified iff:
          • a * b = N
          • Both a and b are prime (for complete factorization)

        Args:
            N: integer to factorize
            candidate: candidate to verify

        Returns:
            True if verified
        """
        a, b = candidate.a, candidate.b

        # Check product
        if a * b != N:
            return False

        # Check non-trivial
        if not (1 < a < N and 1 < b < N):
            return False

        # For complete prime factorization, check primality
        # (simplified: just check if a and b are prime)
        return is_prime(a) and is_prime(b)

    def factor(self, N: int) -> Dict[str, Any]:
        """
        Full GNFS-8 factorization pipeline.

        Args:
            N: integer to factorize

        Returns:
            dict with status, factors, and diagnostics
        """
        start_time = time.time()

        # Stage 1: Candidate generation
        candidates = self.candidate_generator(N)

        # Stage 2: Dependency construction
        dependencies = self.dependency_builder(N, candidates)

        # Stage 3: Constraint evaluation
        evaluated = self.constraint_engine_evaluate(N, candidates, dependencies)

        # Stage 4: Remove inconsistent candidates
        survivors = [c for c in evaluated if c.is_consistent]

        # Stage 5: Rank surviving candidates
        ranked = sorted(survivors, key=lambda x: x.score, reverse=True)

        # Stage 6: Independently verify candidates
        for candidate in ranked:
            if self.independent_verifier(N, candidate):
                elapsed = time.time() - start_time
                return {
                    "status": "verified",
                    "N": N,
                    "factors": [candidate.a, candidate.b],
                    "is_prime_factorization": is_prime(candidate.a) and is_prime(candidate.b),
                    "elapsed_seconds": elapsed,
                    "n_candidates_generated": len(candidates),
                    "n_consistent": len(survivors),
                    "HONEST_NOTE": (
                        "GNFS-8 is a proposed QICC architecture. "
                        "It is NOT equivalent to Shor's algorithm. "
                        "Performance on small N does not establish competitiveness "
                        "for cryptographically relevant integers."
                    ),
                }

        elapsed = time.time() - start_time
        return {
            "status": "unresolved",
            "N": N,
            "factors": None,
            "elapsed_seconds": elapsed,
            "n_candidates_generated": len(candidates),
            "n_consistent": len(survivors),
            "HONEST_NOTE": (
                "No verified factorization found. "
                "The algorithm must return failure or unresolved result "
                "rather than inventing an answer (§3.2)."
            ),
        }


# =============================================================================
# §13.2: SHOR ALGORITHM 4 (Software-Emulated)
# =============================================================================

class ShorAlgorithm4:
    """
    §13.2: Shor Algorithm 4 — Hierarchical software-emulation architecture.

    ARCHITECTURE:
      Classical Input → Structured Representation → Soft Gate Layers →
      Adaptive Evaluation → Classical Output

    OBJECTIVE (from §13.2):
      "Investigate whether structured classical transformations can reproduce
       selected mathematical subroutines or improve classical implementations
       of period-related computations."

    CRITICAL HONEST SCOPE (from §13.2):
      "Equivalence to Shor's algorithm cannot be claimed without a proof of:
       1. Correctness for all inputs in the specified domain.
       2. Equivalent output distributions where required.
       3. A valid complexity bound.
       4. A demonstrated implementation.
       5. A comparison against established algorithms.

       The expression: Shor Algorithm 4 ≡ Shor's Algorithm
       must remain a research target rather than a proven theorem."

    USAGE:
      shor4 = ShorAlgorithm4()
      result = shor4.factor(N)
    """

    def __init__(self):
        logger.info("[Shor4] Initialized")

    def _classical_input(self, N: int) -> Dict[str, Any]:
        """
        Stage 1: Classical Input.

        Prepare the integer N for structured representation.
        """
        return {
            "N": N,
            "bit_length": N.bit_length(),
            "is_even": N % 2 == 0,
            "sqrt_N": math.isqrt(N),
        }

    def _structured_representation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 2: Structured Representation.

        Convert classical input into a structured representation
        suitable for soft gate processing.

        This is a SIMPLIFIED representation — a real implementation
        would use more sophisticated encoding.
        """
        N = input_data["N"]

        # Create a structured representation of N
        # (e.g., binary representation, factor candidates, etc.)
        binary_repr = bin(N)[2:]  # Remove '0b' prefix

        return {
            "N": N,
            "binary_repr": binary_repr,
            "bit_length": input_data["bit_length"],
            "structured_data": {
                "bits": [int(b) for b in binary_repr],
                "sqrt_N": input_data["sqrt_N"],
            },
        }

    def _soft_gate_layers(self, structured: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 3: Soft Gate Layers.

        Apply a sequence of soft gates (sigmoid-based transformations)
        to the structured representation.

        From §12.2:
          g(x) = 1 / (1 + e^{-x})

        NOTE: These are CLASSICAL soft gates, NOT quantum gates.
        "A composition of classical functions is not automatically a
         physical quantum gate." (§12.2)
        """
        N = structured["N"]
        bits = structured["structured_data"]["bits"]

        # Apply soft gates to bits (simplified)
        # In a real implementation, this would be more sophisticated
        soft_bits = [1.0 / (1.0 + math.exp(-b)) for b in bits]

        return {
            "N": N,
            "soft_bits": soft_bits,
            "n_gates_applied": len(bits),
            "HONEST_NOTE": (
                "These are CLASSICAL soft gates, NOT quantum gates. "
                "A classical neural network or software gate does not satisfy "
                "the physical definition merely because it uses similar "
                "mathematical notation. (§12.2)"
            ),
        }

    def _adaptive_evaluation(self, gate_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 4: Adaptive Evaluation.

        Evaluate the soft gate outputs to find period-related information.

        This is where Shor's algorithm would use quantum period-finding.
        Here, we use a CLASSICAL period-finding approach (simplified).
        """
        N = gate_output["N"]

        # Classical period-finding (simplified)
        # In Shor's algorithm, this is done quantumly via QFT
        # Here, we use trial division as a placeholder

        # Find a factor using trial division
        factor = None
        for a in range(2, int(math.isqrt(N)) + 1):
            if N % a == 0:
                factor = a
                break

        return {
            "N": N,
            "found_factor": factor,
            "period_found": factor is not None,
            "HONEST_NOTE": (
                "This is CLASSICAL period-finding, NOT quantum. "
                "Shor's algorithm uses quantum Fourier transform for "
                "exponential speedup. This implementation does NOT "
                "achieve quantum speedup."
            ),
        }

    def _classical_output(self, eval_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 5: Classical Output.

        Produce the final factorization result.
        """
        N = eval_output["N"]
        factor = eval_output["found_factor"]

        if factor is not None and is_valid_factorization(factor, N // factor, N):
            return {
                "status": "verified",
                "N": N,
                "factors": [factor, N // factor],
                "is_prime_factorization": is_prime(factor) and is_prime(N // factor),
            }
        else:
            return {
                "status": "unresolved",
                "N": N,
                "factors": None,
            }

    def factor(self, N: int) -> Dict[str, Any]:
        """
        Full Shor Algorithm 4 pipeline.

        Args:
            N: integer to factorize

        Returns:
            dict with status, factors, and diagnostics
        """
        start_time = time.time()

        # Stage 1: Classical Input
        input_data = self._classical_input(N)

        # Stage 2: Structured Representation
        structured = self._structured_representation(input_data)

        # Stage 3: Soft Gate Layers
        gate_output = self._soft_gate_layers(structured)

        # Stage 4: Adaptive Evaluation
        eval_output = self._adaptive_evaluation(gate_output)

        # Stage 5: Classical Output
        output = self._classical_output(eval_output)

        elapsed = time.time() - start_time
        output["elapsed_seconds"] = elapsed
        output["HONEST_NOTE"] = (
            "Shor Algorithm 4 is a SOFTWARE-EMULATED architecture. "
            "It is NOT equivalent to Shor's quantum algorithm. "
            "Equivalence requires: correctness proof, complexity bound, "
            "and comparison against established algorithms (§13.2). "
            "This implementation uses classical trial division, "
            "NOT quantum period-finding."
        )

        logger.info(f"[Shor4] Factored N={N}: status={output['status']}, "
                   f"elapsed={elapsed:.4f}s")
        return output


# =============================================================================
# §10.1 / Appendix C.4: PERFORMANCE BENCHMARK
# =============================================================================

class PerformanceBenchmark:
    """
    Benchmarking infrastructure for GNFS-8 and Shor Algorithm 4.

    From §10.1:
      S(N) = T_baseline(N) / T_GNFS8(N)  — runtime speedup
      R(N) = M_baseline(N) / M_GNFS8(N)  — memory-resource ratio
      A = N_verified / N_attempted       — accuracy

    From Appendix C.5: Required test groups:
      Group 1: Known-factor composites
      Group 2: Prime inputs
      Group 3: Repeated factors
      Group 4: Unseen inputs
      Group 5: Scaling
      Group 6: Independent comparison
    """

    def __init__(self):
        logger.info("[Benchmark] Initialized")

    def run_factorization_benchmark(self,
                                     candidate_algorithm: Any,
                                     test_numbers: List[int],
                                     reference_validator: Callable = validate_factorization
                                     ) -> List[Dict[str, Any]]:
        """
        Appendix C.4: GNFS-8 experimental interface.

        Args:
            candidate_algorithm: object with .factor(N) method
            test_numbers: list of integers to factorize
            reference_validator: validation function

        Returns:
            list of benchmark results
        """
        report = []

        for N in test_numbers:
            start_time = time.time()
            result = candidate_algorithm.factor(N)
            elapsed = time.time() - start_time

            # Validate result
            if result["status"] == "verified" and result["factors"]:
                valid = reference_validator(N, result["factors"])
            else:
                valid = False

            report.append({
                "N": N,
                "bit_length": N.bit_length(),
                "factors": result["factors"],
                "valid": valid,
                "status": result["status"],
                "elapsed_seconds": elapsed,
            })

        return report

    def compute_metrics(self, report: List[Dict[str, Any]],
                        baseline_report: Optional[List[Dict[str, Any]]] = None
                        ) -> Dict[str, Any]:
        """
        Compute performance metrics from benchmark report.

        Args:
            report: benchmark results
            baseline_report: optional baseline results for comparison

        Returns:
            dict with metrics
        """
        n_attempted = len(report)
        n_verified = sum(1 for r in report if r["valid"])
        accuracy = n_verified / max(n_attempted, 1)

        total_time = sum(r["elapsed_seconds"] for r in report)
        avg_time = total_time / max(n_attempted, 1)

        metrics = {
            "n_attempted": n_attempted,
            "n_verified": n_verified,
            "accuracy": accuracy,
            "total_time_seconds": total_time,
            "avg_time_seconds": avg_time,
        }

        # Compute speedup if baseline available
        if baseline_report:
            baseline_time = sum(r["elapsed_seconds"] for r in baseline_report)
            speedup = baseline_time / max(total_time, 1e-10)
            metrics["speedup_S"] = speedup

        return metrics

    def run_standard_tests(self, algorithm: Any) -> Dict[str, Any]:
        """
        Run the standard test groups from Appendix C.5.

        Group 1: Known-factor composites
        Group 2: Prime inputs
        Group 3: Repeated factors
        Group 4: Unseen inputs
        """
        results = {}

        # Group 1: Known-factor composites
        group1 = [15, 21, 77, 143, 1001, 10403]
        results["group1_known_factors"] = self.run_factorization_benchmark(
            algorithm, group1)

        # Group 2: Prime inputs (should return unresolved or N itself)
        group2 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        results["group2_primes"] = self.run_factorization_benchmark(
            algorithm, group2)

        # Group 3: Repeated factors
        group3 = [4, 8, 9, 16, 25, 27, 81, 3**4 * 5**2]
        results["group3_repeated"] = self.run_factorization_benchmark(
            algorithm, group3)

        # Group 4: Unseen inputs
        group4 = [101 * 103, 107 * 109, 113 * 127, 131 * 137]
        results["group4_unseen"] = self.run_factorization_benchmark(
            algorithm, group4)

        return results


# =============================================================================
# INTEGRATED SYSTEM
# =============================================================================

class IntegratedGNFS8Shor4System:
    """
    Master orchestrator for GNFS-8 and Shor Algorithm 4.

    INTEGRATES:
      • GNFS8Factorizer (QICC architecture)
      • ShorAlgorithm4 (software-emulated)
      • FactorizationValidator
      • PerformanceBenchmark

    USAGE:
      system = IntegratedGNFS8Shor4System()
      result = system.run_comparison(N)
    """

    def __init__(self):
        self.gnfs8 = GNFS8Factorizer()
        self.shor4 = ShorAlgorithm4()
        self.benchmark = PerformanceBenchmark()

        logger.info("[IntegratedSystem] Initialized")

    def run_comparison(self, test_numbers: List[int]) -> Dict[str, Any]:
        """
        Run GNFS-8 and Shor Algorithm 4 on test numbers and compare.

        Args:
            test_numbers: list of integers to factorize

        Returns:
            dict with comparison results
        """
        results = {
            "test_numbers": test_numbers,
            "gnfs8_results": [],
            "shor4_results": [],
        }

        for N in test_numbers:
            # GNFS-8
            gnfs8_result = self.gnfs8.factor(N)
            results["gnfs8_results"].append(gnfs8_result)

            # Shor Algorithm 4
            shor4_result = self.shor4.factor(N)
            results["shor4_results"].append(shor4_result)

        # Compute metrics
        gnfs8_report = [
            {
                "N": N,
                "factors": r["factors"],
                "valid": r["status"] == "verified",
                "elapsed_seconds": r["elapsed_seconds"],
            }
            for N, r in zip(test_numbers, results["gnfs8_results"])
        ]
        shor4_report = [
            {
                "N": N,
                "factors": r["factors"],
                "valid": r["status"] == "verified",
                "elapsed_seconds": r["elapsed_seconds"],
            }
            for N, r in zip(test_numbers, results["shor4_results"])
        ]

        results["gnfs8_metrics"] = self.benchmark.compute_metrics(gnfs8_report)
        results["shor4_metrics"] = self.benchmark.compute_metrics(
            shor4_report, baseline_report=gnfs8_report)

        results["HONEST_SCOPE"] = {
            "gnfs8": (
                "GNFS-8 is a proposed QICC architecture. "
                "It is NOT equivalent to Shor's algorithm. "
                "Performance on small N does not establish competitiveness "
                "for cryptographically relevant integers."
            ),
            "shor4": (
                "Shor Algorithm 4 is a SOFTWARE-EMULATED architecture. "
                "It is NOT equivalent to Shor's quantum algorithm. "
                "This implementation uses classical trial division, "
                "NOT quantum period-finding."
            ),
            "equivalence": (
                "The expression 'Shor Algorithm 4 ≡ Shor's Algorithm' "
                "must remain a research target rather than a proven theorem (§13.2). "
                "Equivalence requires: correctness proof, complexity bound, "
                "and comparison against established algorithms."
            ),
        }

        return results


# =============================================================================
# SELF-TEST SUITE
# =============================================================================

def smoke_test():
    print("=" * 78)
    print("  GNFS-8 & SHOR ALGORITHM 4 — Smoke Test")
    print("=" * 78)

    # ── T1: Factorization Validator ──────────────────────────────────────
    print("\n [T1] Factorization Validator")

    # Valid factorization
    assert validate_factorization(15, [3, 5]) == True
    assert validate_factorization(21, [3, 7]) == True
    assert validate_factorization(77, [7, 11]) == True
    assert validate_factorization(1001, [7, 11, 13]) == True
    print("   Valid factorizations: PASS")

    # Invalid factorization
    assert validate_factorization(15, [3, 4]) == False  # 3*4 != 15
    assert validate_factorization(15, [1, 15]) == False  # trivial
    assert validate_factorization(15, [3, 5, 7]) == False  # wrong product
    print("   Invalid factorizations: PASS")

    # Prime check
    assert validate_factorization(15, [3, 5]) == True  # both prime
    assert validate_factorization(12, [3, 4]) == False  # 4 is not prime
    print("   Primality check: PASS")

    print("[PASS] T1 Factorization Validator works")

    # ── T2: GNFS-8 Factorizer ────────────────────────────────────────────
    print("\n [T2] GNFS-8 Factorizer")

    gnfs8 = GNFS8Factorizer()

    # Test with small composites
    test_cases = [
        (15, [3, 5]),
        (21, [3, 7]),
        (77, [7, 11]),
        (143, [11, 13]),
        (1001, [7, 11, 13]),
    ]

    for N, expected in test_cases:
        result = gnfs8.factor(N)
        print(f"   N={N}: status={result['status']}, factors={result['factors']}")
        assert result["status"] == "verified", f"Failed for N={N}"
        assert result["factors"] is not None, f"No factors for N={N}"
        assert validate_factorization(N, result["factors"]), f"Invalid factorization for N={N}"

    print("[PASS] T2 GNFS-8 Factorizer works")

    # ── T3: Shor Algorithm 4 ─────────────────────────────────────────────
    print("\n [T3] Shor Algorithm 4")

    shor4 = ShorAlgorithm4()

    for N, expected in test_cases[:3]:  # Test first 3
        result = shor4.factor(N)
        print(f"   N={N}: status={result['status']}, factors={result['factors']}")
        assert result["status"] == "verified", f"Failed for N={N}"
        assert result["factors"] is not None, f"No factors for N={N}"
        assert validate_factorization(N, result["factors"]), f"Invalid factorization for N={N}"

    print("[PASS] T3 Shor Algorithm 4 works")

    # ── T4: Performance Benchmark ────────────────────────────────────────
    print("\n [T4] Performance Benchmark")

    benchmark = PerformanceBenchmark()

    # Run standard tests
    test_numbers = [15, 21, 77, 143, 1001]
    report = benchmark.run_factorization_benchmark(gnfs8, test_numbers)

    print(f"   Tested {len(report)} numbers")
    for r in report:
        print(f"     N={r['N']}: valid={r['valid']}, time={r['elapsed_seconds']:.4f}s")

    # Compute metrics
    metrics = benchmark.compute_metrics(report)
    print(f"   Accuracy: {metrics['accuracy']:.2%}")
    print(f"   Total time: {metrics['total_time_seconds']:.4f}s")

    assert metrics["accuracy"] > 0.5, "Accuracy should be > 50%"
    print("[PASS] T4 Performance Benchmark works")

    # ── T5: Integrated System ────────────────────────────────────────────
    print("\n [T5] Integrated System")

    system = IntegratedGNFS8Shor4System()
    comparison = system.run_comparison(test_numbers)

    print(f"   GNFS-8 accuracy: {comparison['gnfs8_metrics']['accuracy']:.2%}")
    print(f"   Shor4 accuracy: {comparison['shor4_metrics']['accuracy']:.2%}")

    assert comparison["gnfs8_metrics"]["accuracy"] > 0.5
    assert comparison["shor4_metrics"]["accuracy"] > 0.5

    print("[PASS] T5 Integrated System works")

    # ── T6: OPMC Constants ───────────────────────────────────────────────
    print("\n [T6] OPMC Constants")
    k_star = opmc_k_star(1e-8)
    print(f"   k* (Theorem 11.1): {k_star}")
    assert k_star == 23
    print(f"   d(4,4) (Lemma 2.4): {OPMC_D_MN(4, 4)}")
    assert OPMC_D_MN(4, 4) == 320
    print("[PASS] T6 OPMC constants match paper")

    print("\n" + "=" * 78)
    print("  ALL TESTS PASSED.")
    print("")
    print("  CRITICAL HONEST REMINDERS (from the paper):")
    print("  • GNFS-8 is NOT equivalent to Shor's algorithm (§13.3)")
    print("  • Shor Algorithm 4 is SOFTWARE-EMULATED, not quantum (§12.2)")
    print("  • 'Shor Algorithm 4 ≡ Shor's Algorithm' is a RESEARCH TARGET,")
    print("    not a proven theorem (§13.2)")
    print("  • Performance on small N does NOT establish competitiveness")
    print("    for cryptographically relevant integers (§4.7)")
    print("  • This does NOT prove P = NP (§14.4)")
    print("  • Software gates are NOT physical quantum gates (§12.2)")
    print("=" * 78)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
