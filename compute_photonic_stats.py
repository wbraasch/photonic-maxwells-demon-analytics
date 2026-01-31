#!/usr/bin/env python3
"""
Haar-averaged photon-number statistics for:
- Fig. 2: P(n) in a selected output mode (M=3 and M=4, N=3)
- Fig. 3: P(Delta n) with passive/active demon (M=4, N=3)

This scrip  implements the Haar-averaged probabilities described in the supplement:

- Indistinguishable photons: uniform over occupation patterns (Supplement Eq. 27)
- Distinguishable photons: Weingarten-calculus expression (Supplement Eq. 28)

Conventions
-----------
- Modes are 0-indexed in code.
- For distinguishable photons we represent an "output list"
  O = (o_0, o_1, ..., o_{N-1}) where o_k is the output mode of photon k.
  We only enumerate one representative per occupation pattern (sorted tuple),
  and account for multiplicities via mu(s) = prod_m (s_m)! exactly as in Eq. (28).

Dependencies
------------
Required:
- Python >= 3.9
- numpy
- sympy

Optional (recommended):
- haarpy  (for weingarten_unitary)
If haarpy is not available, we fall back to a closed-form Weingarten function
for N=3 only (which is enough for the current paper figures).

Example runs (paper defaults)
-----------------------------
Compute both Fig. 2 and Fig. 3 data (default behavior):
  python compute_photonic_stats.py

Compute only Fig. 2 panels:
  python compute_photonic_stats.py --fig2

Compute only Fig. 3:
  python compute_photonic_stats.py --fig3

Write to a specific directory and filename prefix:
  python compute_photonic_stats.py --outdir out --tag paper_run

Outputs
-------
Writes a JSON file with the following structure:

  {
    "meta": {
      "N": int,
      "selected_mode_fig2": int,
      "used_haarpy": bool,
      "python_version": str,
      "platform": str,
      "numpy_version": str,
      "sympy_version": str,
      "matplotlib_version": str,
      "haarpy_version": str,
      "argv": list[str]            # optional but recommended
    },
    "pattern_probs": {
      "<label>": { "(0, 0, 1)": float, ... },
      ...
    },
    "fig2": {
      "a": {
        "M": int,
        "Pn_distinguishable": [float, ...],   # length N+1
        "Pn_indistinguishable": [float, ...]  # length N+1
      },
      "b": { ... same keys as "a" ... }
    },
    "fig3": {
      "M": int,
      "passive": {
        "Delta_n_distinguishable": { "-N": float, ..., "N": float },
        "Delta_n_indistinguishable": { "-N": float, ..., "N": float }
      },
      "active": {
        "Delta_n_distinguishable": { "-N": float, ..., "N": float },
        "Delta_n_indistinguishable": { "-N": float, ..., "N": float }
      }
    }
  }

Notes:
- JSON forces dict keys to be strings. The plotting script converts Δn keys back to integers.
- The plotting script plot_photonic_stats.py reads only the "fig2" and "fig3" sections above.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import platform
import importlib.metadata as md
import itertools


import numpy as np
from sympy.combinatorics import Permutation
import sympy as sp

# Try to use haarpy if present.
try:
    import haarpy  # type: ignore
    _HAARPY_AVAILABLE = True
except Exception:
    haarpy = None
    _HAARPY_AVAILABLE = False

_USED_HAARPY = _HAARPY_AVAILABLE and (haarpy is not None) and hasattr(haarpy, "weingarten_unitary")

# --------------------------
# Metadata helpers
# --------------------------

def _pkg_version(name: str) -> str:
    """Return installed package version string, or 'not-installed' if missing."""
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "not-installed"


# ----------------------------
# Small data container
# ----------------------------
@dataclass(frozen=True)
class OutcomePattern:
    """
    Container for a single occupation pattern.

    Attributes
    ----------
    rep_tuple:
        Sorted representative output list O = (o_0, o_1, ..., o_{N-1}).
        This list encodes the occupation pattern but keeps an ordering
        (sorted) to give a unique representative. It is a sorted multiset
        of output modes (photons are not individually labeled here).
    occupancy:
        Occupation vector s = (s_0, ..., s_{M-1}) where sum(s_m) = N.
    mu:
        Multiplicity mu(s) = prod_m (s_m)! used in Eq. (28).
    """
    rep_tuple: Tuple[int, ...]
    occupancy: Tuple[int, ...]
    mu: int


def _all_perms(N: int) -> List[Tuple[int, ...]]:
    """Return all permutations of range(N) as tuples in one-line notation."""
    return list(itertools.permutations(range(N)))


def _perm_inverse(p: Tuple[int, ...]) -> Tuple[int, ...]:
    """Return the inverse of a permutation p given in one-line notation."""
    inv = [0] * len(p)
    for k, v in enumerate(p):
        inv[v] = k
    return tuple(inv)


def _occupancy_from_rep(rep_tuple: Tuple[int, ...], M: int) -> Tuple[int, ...]:
    """Convert a representative output tuple into an M-length occupancy vector."""
    counts = [0] * M
    for o in rep_tuple:
        counts[o] += 1
    return tuple(counts)


def _mu_from_occupancy(occ: Tuple[int, ...]) -> int:
    """Return mu(s) = prod_m (s_m)! for an occupancy vector s."""
    mu = 1
    for x in occ:
        mu *= math.factorial(x)
    return mu


def generate_patterns(M: int, N: int) -> List[OutcomePattern]:
    """
    Generate all occupation patterns for N photons in M modes, represented
    by sorted representative output tuples.

    Parameters
    ----------
    M:
        Number of modes.
    N:
        Number of photons.

    Returns
    -------
    patterns:
        List of OutcomePattern objects (one per occupation pattern).
    """
    reps = list(itertools.combinations_with_replacement(range(M), N))
    patterns: List[OutcomePattern] = []
    for rep in reps:
        occ = _occupancy_from_rep(rep, M)
        mu = _mu_from_occupancy(occ)
        patterns.append(OutcomePattern(rep_tuple=tuple(rep), occupancy=occ, mu=mu))
    return patterns


# ----------------------------
# Weingarten values
# ----------------------------
def weingarten_value(M: int, N: int, sigma: Tuple[int, ...]) -> float:
    """
    Compute Wg_{M,N}(sigma) for a permutation sigma in S_N.

    Uses haarpy.weingarten_unitary if available. Otherwise uses a closed-form
    fallback for N=3 only.

    Parameters
    ----------
    M:
        Number of modes (the "R" parameter in the supplement).
    N:
        Number of photons (the "d" parameter in the supplement).
    sigma:
        Permutation in one-line notation, where sigma[k] = sigma(k).
        Example: the transposition (0 1) in S_3 is (1, 0, 2).

    Returns
    -------
    wg:
        Weingarten value as a float.
    """
    if _USED_HAARPY:
        perm_obj = Permutation(list(sigma))
        # Pass a SymPy Integer for the unitary dimension.
        return float(haarpy.weingarten_unitary(perm_obj, sp.Integer(M)))


    # Fallback: closed-form unitary Weingarten for N=3 (S_3 has 3 conjugacy classes).
    if N != 3:
        raise RuntimeError(
            "haarpy is not available, and fallback is implemented only for N=3."
        )

    # Determine cycle type of sigma.
    seen = [False] * N
    cycle_lengths: List[int] = []
    for i in range(N):
        if not seen[i]:
            cur = i
            length = 0
            while not seen[cur]:
                seen[cur] = True
                cur = sigma[cur]
                length += 1
            cycle_lengths.append(length)
    cycle_lengths.sort(reverse=True)

    denom = (M**4 - 5 * M**2 + 4)
    if denom == 0:
        raise ValueError(f"Weingarten denominator is zero for M={M}, N={N}. Require M>=3 for N=3.")


    if cycle_lengths == [1, 1, 1]:
        # identity
        return float((M**2 - 2) / (M * denom))
    if cycle_lengths == [2, 1]:
        # transposition
        return float(-1 / denom)
    if cycle_lengths == [3]:
        # 3-cycle
        return float(2 / (M * denom))

    raise ValueError(f"Unexpected cycle type for sigma={sigma}.")


# ----------------------------
# Eq. (28) implementation
# ----------------------------
def _delta_product(rep_tuple: Tuple[int, ...], pi: Tuple[int, ...], sigma: Tuple[int, ...]) -> bool:
    """
    Return True iff prod_{k=0..N-1} delta_{o_{pi(k)}, o_{pi(sigma(k))}} = 1.

    Parameters
    ----------
    rep_tuple:
        Sorted representative output list O in mode labels.
    pi, sigma:
        Permutations in one-line notation on {0,...,N-1}.
    """
    N = len(rep_tuple)
    # O_prime[k] = o_{pi(k)}
    O_prime = [rep_tuple[pi[k]] for k in range(N)]
    for k in range(N):
        if O_prime[k] != O_prime[sigma[k]]:
            return False
    return True


def expected_prob_distinguishable_eq28(
    pattern: OutcomePattern,
    M: int,
    N: int,
) -> float:
    """
    Haar-averaged probability for a single occupation pattern assuming
    distinguishable photons and a collision-free input (distinct input modes).

    This implements supplement Eq. (28):
        E[p(dist)_{Psi->O}] = (1/mu(s)) * sum_{pi in S_N} sum_{sigma in S_N}
                              ( prod_k delta_{o_{pi(k)}, o_{pi(sigma(k))}} ) * Wg_{M,N}(sigma^{-1})

    Notes
    -----
    - This function does not take an explicit input-mode list because under Haar
      averaging, any collision-free choice of N distinct input modes is equivalent
      up to relabeling. The derivation of Eq. (28) also uses that the input modes
      are distinct (see the supplement text around the tau=identity simplification).
    - If you need repeated input modes (collisions), you need the more general
      formula with the additional permutation sum (not implemented here).

    Parameters
    ----------
    pattern:
        OutcomePattern describing the occupation pattern.
    M:
        Number of modes.
    N:
        Number of photons.

    Returns
    -------
    p:
        Haar-averaged probability for this occupation pattern.
    """
    perms = _all_perms(N)
    accum = 0.0

    for pi in perms:
        for sigma in perms:
            if _delta_product(pattern.rep_tuple, pi, sigma):
                sigma_inv = _perm_inverse(sigma)
                accum += weingarten_value(M, N, sigma_inv)

    return accum / float(pattern.mu)


def expected_prob_indistinguishable_eq27(M: int, N: int) -> float:
    """
    Haar-averaged probability for each occupation pattern assuming indistinguishable photons.

    For indistinguishable photons the Haar-averaged probability is uniform over occupation
    patterns (supplement Eq. (27)):
        1 / binom(M+N-1, N)

    Parameters
    ----------
    M:
        Number of modes.
    N:
        Number of photons.

    Returns
    -------
    p:
        Uniform probability per occupation pattern.
    """
    return 1.0 / math.comb(M + N - 1, N)


# ----------------------------
# Fig. 2 marginals and Fig. 3 Delta n
# ----------------------------
def fig2_marginal_Pn(
    pattern_probs: Dict[Tuple[int, ...], float],
    M: int,
    N: int,
    selected_mode: int = 0,
) -> List[float]:
    """
    Compute P(n) for n=0..N for one selected output mode by marginalizing over patterns.

    Parameters
    ----------
    pattern_probs:
        Map rep_tuple -> probability for that occupation pattern.
    M:
        Number of modes.
    N:
        Number of photons.
    selected_mode:
        Which output mode to measure (0-indexed).

    Returns
    -------
    Pn:
        List of length N+1 with Pn[n] = P(n photons in selected mode).
    """
    Pn = [0.0] * (N + 1)
    for rep, p in pattern_probs.items():
        occ = _occupancy_from_rep(rep, M)
        n = occ[selected_mode]
        Pn[n] += float(p)
    return Pn


def fig3_delta_n_distribution(
    pattern_probs: Dict[Tuple[int, ...], float],
    M: int,
    N: int,
    mode_A_top: int = 0,
    mode_B_top: int = 3,
    subset_A: Tuple[int, int] = (0, 1),
    subset_B: Tuple[int, int] = (2, 3),
    active_demon: bool = False,
) -> Dict[int, float]:
    """
    Compute P(Delta n) for Delta n := n_A - n_B (Fig. 3 style).

    Passive demon:
        Delta n = s[mode_A_top] - s[mode_B_top]

    Active demon:
        If total photons in subset A is less than total photons in subset B,
        swap subsets (implemented as the mode swaps described in the supplement)
        before computing Delta n.

    Parameters
    ----------
    pattern_probs:
        Map rep_tuple -> probability for that occupation pattern.
    M:
        Number of modes (must be 4 for the default Fig. 3 wiring).
    N:
        Number of photons.
    mode_A_top:
        Mode used for n_A (top mode of subset A).
    mode_B_top:
        Mode used for n_B (top mode of subset B).
    subset_A:
        Two modes comprising subset A.
    subset_B:
        Two modes comprising subset B.
    active_demon:
        Whether to apply the conditional swapping rule.

    Returns
    -------
    dist:
        Dict mapping Delta n (integer) -> probability.
    """
    dist: Dict[int, float] = {dn: 0.0 for dn in range(-N, N + 1)}

    for rep, p in pattern_probs.items():
        occ = list(_occupancy_from_rep(rep, M))

        if active_demon:
            NA = occ[subset_A[0]] + occ[subset_A[1]]
            NB = occ[subset_B[0]] + occ[subset_B[1]]
            if NA < NB:
                # Paper wiring: swap subsets via mode swaps (0 <-> 3) and (1 <-> 2) for subset_A=(0,1), subset_B=(2,3).
                occ[subset_A[0]], occ[subset_B[1]] = occ[subset_B[1]], occ[subset_A[0]]
                occ[subset_A[1]], occ[subset_B[0]] = occ[subset_B[0]], occ[subset_A[1]]

        dn = occ[mode_A_top] - occ[mode_B_top]
        dist[dn] += float(p)

    return dist


# ----------------------------
# CLI / main
# ----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute Haar-averaged pattern probabilities and derived marginals for Fig. 2 and Fig. 3."
    )

    p.add_argument("--N", type=int, default=3, help="Photon number N (default: 3).")
    p.add_argument("--selected-mode", type=int, default=0, help="Selected output mode for Fig. 2 P(n) (0-indexed).")

    p.add_argument("--fig2", action="store_true", help="Compute Fig. 2 marginals (panels a and b).")
    p.add_argument("--fig3", action="store_true", help="Compute Fig. 3 Delta n distributions (passive and active).")

    p.add_argument("--M-fig2-a", type=int, default=3, help="Mode number M for Fig. 2(a) (default: 3).")
    p.add_argument("--M-fig2-b", type=int, default=4, help="Mode number M for Fig. 2(b) (default: 4).")
    p.add_argument("--M-fig3", type=int, default=4, help="Mode number M for Fig. 3 (default: 4).")

    p.add_argument("--outdir", type=str, default="results", help="Output directory (default: results).")
    p.add_argument("--tag", type=str, default="haar_stats", help="Filename tag/prefix (default: haar_stats).")

    return p


def compute_pattern_probabilities(M: int, N: int) -> Tuple[Dict[Tuple[int, ...], float], Dict[Tuple[int, ...], float]]:
    """
    Compute full pattern probability tables for both indist and dist.

    Returns
    -------
    dist_probs, indist_probs:
        dict(rep_tuple -> probability) for distinguishable and indistinguishable photons.
    """
    patterns = generate_patterns(M, N)

    # Indist: uniform over patterns
    p_uniform = expected_prob_indistinguishable_eq27(M, N)
    indist_probs = {pat.rep_tuple: p_uniform for pat in patterns}

    # Dist: Eq. (28)
    dist_probs: Dict[Tuple[int, ...], float] = {}
    for pat in patterns:
        dist_probs[pat.rep_tuple] = expected_prob_distinguishable_eq28(pat, M, N)

    # Sanity check sums
    s_dist = sum(dist_probs.values())
    s_ind = sum(indist_probs.values())
    if abs(s_dist - 1.0) > 1e-9:
        raise RuntimeError(f"Distinguishable probabilities do not sum to 1 (sum={s_dist}).")
    if abs(s_ind - 1.0) > 1e-9:
        raise RuntimeError(f"Indistinguishable probabilities do not sum to 1 (sum={s_ind}).")

    return dist_probs, indist_probs


def main() -> None:
    args = build_argparser().parse_args()

    # Default behavior: compute both if neither flag is given.
    do_fig2 = args.fig2 or (not args.fig2 and not args.fig3)
    do_fig3 = args.fig3 or (not args.fig2 and not args.fig3)

    N = int(args.N)
    os.makedirs(args.outdir, exist_ok=True)

    results = {
        "meta": {
            "N": N,
            "selected_mode_fig2": int(args.selected_mode),
            "used_haarpy": bool(_USED_HAARPY),
        },
        "pattern_probs": {},
        "fig2": {},
        "fig3": {},
    }

    results["meta"].update({
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": _pkg_version("numpy"),
        "sympy_version": _pkg_version("sympy"),
        "matplotlib_version": _pkg_version("matplotlib"),
        "haarpy_version": _pkg_version("haarpy"),
    })

    # -------------------
    # Fig. 2
    # -------------------
    if do_fig2:
        for panel, M in [("a", int(args.M_fig2_a)), ("b", int(args.M_fig2_b))]:
            dist_probs, indist_probs = compute_pattern_probabilities(M, N)

            Pn_dist = fig2_marginal_Pn(dist_probs, M, N, selected_mode=int(args.selected_mode))
            Pn_ind = fig2_marginal_Pn(indist_probs, M, N, selected_mode=int(args.selected_mode))

            results["pattern_probs"][f"fig2_{panel}_M{M}_dist"] = {str(k): v for k, v in dist_probs.items()}
            results["pattern_probs"][f"fig2_{panel}_M{M}_indist"] = {str(k): v for k, v in indist_probs.items()}

            results["fig2"][panel] = {
                "M": M,
                "Pn_distinguishable": Pn_dist,
                "Pn_indistinguishable": Pn_ind,
            }

    # -------------------
    # Fig. 3
    # -------------------
    if do_fig3:
        M = int(args.M_fig3)
        dist_probs, indist_probs = compute_pattern_probabilities(M, N)

        # Default wiring matches the paper: subsets A=(0,1), B=(2,3),
        # Delta n = n_A_top - n_B_top with top modes 0 and 3.
        dn_dist_passive = fig3_delta_n_distribution(dist_probs, M, N, active_demon=False)
        def _str_keyed(d: Dict[int, float]) -> Dict[str, float]:
            return {str(k): float(v) for k, v in d.items()}
        dn_ind_passive = fig3_delta_n_distribution(indist_probs, M, N, active_demon=False)
        dn_dist_active = fig3_delta_n_distribution(dist_probs, M, N, active_demon=True)
        dn_ind_active = fig3_delta_n_distribution(indist_probs, M, N, active_demon=True)

        results["pattern_probs"][f"fig3_M{M}_dist"] = {str(k): v for k, v in dist_probs.items()}
        results["pattern_probs"][f"fig3_M{M}_indist"] = {str(k): v for k, v in indist_probs.items()}

        results["fig3"] = {
            "M": M,
            "passive": {
                "Delta_n_distinguishable": _str_keyed(dn_dist_passive),
                "Delta_n_indistinguishable": _str_keyed(dn_ind_passive),
            },
            "active": {
                "Delta_n_distinguishable": _str_keyed(dn_dist_active),
                "Delta_n_indistinguishable": _str_keyed(dn_ind_active),
            },
        }


    outpath = os.path.join(args.outdir, f"{args.tag}_N{N}.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"Wrote results to: {outpath}")
    if not _HAARPY_AVAILABLE:
        print("Note: haarpy not installed; using N=3 closed-form Weingarten fallback.")
    elif not hasattr(haarpy, "weingarten_unitary"):
        print("Note: haarpy installed but no weingarten_unitary found; using N=3 closed-form Weingarten fallback.")



if __name__ == "__main__":
    main()
