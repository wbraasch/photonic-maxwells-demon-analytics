#!/usr/bin/env python3
"""
Plot Fig. 2 and Fig. 3 style charts from the JSON produced by compute_photonic_stats.py.

This script is intended to reproduce the paper's *analytical predictions* from a saved
JSON file.

Expected input JSON schema (high level)
---------------------------------------
The JSON file must contain top-level keys:

  - "fig2": dict with panels "a" and "b"
  - "fig3": dict with sections "passive" and "active"

Each section contains probability data under fixed key names; see plot_fig2/plot_fig3
docstrings for the exact required fields.

Example
-------
  python plot_photonic_stats.py --in results/haar_stats_N3.json --outdir figures --prefix paper

This generates:
  figures/paper_fig2.png
  figures/paper_fig3.png

You can also save PDF:
  python plot_photonic_stats.py --in results/paper_run_N3.json --outdir figures --prefix paper --fmt pdf
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the plotting script."""
    p = argparse.ArgumentParser(description="Plot Fig. 2/3 from photonic stats JSON.")
    p.add_argument("--in", dest="in_path", required=True, help="Input JSON path from compute script.")
    p.add_argument("--outdir", type=str, default="figures", help="Output directory (default: figures).")
    p.add_argument("--prefix", type=str, default="plot", help="Output filename prefix (default: plot).")
    p.add_argument("--fmt", type=str, default="png", choices=["png", "pdf"], help="Output format (png/pdf).")
    p.add_argument("--no-fig2", action="store_true", help="Skip Fig. 2 plot.")
    p.add_argument("--no-fig3", action="store_true", help="Skip Fig. 3 plot.")
    p.add_argument("--show", action="store_true", help="Show plots interactively.")
    return p.parse_args()


def set_plot_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,  # grid behind bars
        "grid.alpha": 0.25,

        # LaTeX-ish math without requiring a TeX install
        "mathtext.fontset": "stix",

        # Embed fonts cleanly in PDF
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# Bar styling (paper-ish colors)
DIST_FACE = "#9ecae1"  # light blue
DIST_EDGE = "#3182bd"  # darker blue outline

IND_FACE = "#fcbba1"   # light red
IND_EDGE = "#de2d26"   # darker red outline

BAR_ALPHA = 0.80
EDGE_LW = 1.2


def load_json(path: str) -> dict:
    """Load and return a JSON file as a Python dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: str) -> None:
    """Create output directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def _as_int_keyed(d: Dict) -> Dict[int, float]:
    """
    Convert a JSON object with string keys representing integers into an int-keyed dict.

    Example:
      {"-1": 0.2, "0": 0.3} -> {-1: 0.2, 0: 0.3}
    """
    out: Dict[int, float] = {}
    for k, v in d.items():
        out[int(k)] = float(v)
    return out


def plot_fig2(data: dict, outpath: str) -> None:
    """
    Plot Fig. 2 (analytical prediction): two panels (a) and (b) showing P(n) for n = 0..N.

    Required JSON fields
    --------------------
    data["fig2"] must be a dict with keys "a" and "b". For each panel key in {"a","b"}:

      - data["fig2"][panel]["Pn_distinguishable"] : list[float] of length N+1
      - data["fig2"][panel]["Pn_indistinguishable"] : list[float] of length N+1
      - data["fig2"][panel]["M"] : int (optional; used only for annotation)

    Interpretation / invariants
    ---------------------------
    - The x-axis uses photon number n = 0,1,...,N inferred from the list length (N = len(Pn)-1).
    - The two probability lists for a given panel must have the same length.
    - Values are treated as probabilities (typically sum to 1 within numerical tolerance).
    """
    fig2 = data.get("fig2", {})
    if not fig2 or "a" not in fig2 or "b" not in fig2:
        raise KeyError("Input JSON does not contain fig2 panels 'a' and 'b' in the expected format.")

    panels = [("a", fig2["a"]), ("b", fig2["b"])]

    # Determine N from length of Pn arrays.
    Pn0 = panels[0][1]["Pn_distinguishable"]
    N = len(Pn0) - 1
    xs = list(range(N + 1))

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    fig.subplots_adjust(top=0.80)
    fig.suptitle("Fig. 2 (analytical prediction)", y=0.98, fontweight="bold")

    for ax, (label, panel) in zip(axes, panels):
        P_dist = panel["Pn_distinguishable"]
        P_ind = panel["Pn_indistinguishable"]
        M = panel.get("M", None)

        width = 0.38
        ax.bar(
            [x - width / 2 for x in xs],
            P_dist,
            width=width,
            label="Distinguishable",
            facecolor=DIST_FACE,
            edgecolor=DIST_EDGE,
            alpha=BAR_ALPHA,
            linewidth=EDGE_LW,
        )
        ax.bar(
            [x + width / 2 for x in xs],
            P_ind,
            width=width,
            label="Indistinguishable",
            facecolor=IND_FACE,
            edgecolor=IND_EDGE,
            alpha=BAR_ALPHA,
            linewidth=EDGE_LW,
        )

        ax.set_xlabel(r"Photon number $n$")
        panel_text = fr"$\mathbf{{({label})}}$"
        if M is not None:
            panel_text += fr"  $M={M},\, N={N}$"

        ax.text(
            0.02, 0.98,
            panel_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5)
        )
        ax.set_xticks(xs)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel(r"Probability $P(n)$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.91),
    )
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig3(data: dict, outpath: str) -> None:
    """
    Plot Fig. 3 (analytical prediction): two panels showing P(Δn) for photon-number difference Δn.

    Required JSON fields
    --------------------
    data["fig3"] must be a dict with keys "passive" and "active". For each section key in
    {"passive","active"}:

      - data["fig3"][section]["Delta_n_distinguishable"] : dict[str, float]
      - data["fig3"][section]["Delta_n_indistinguishable"] : dict[str, float]

    The Δn-probability maps are stored with string keys in JSON (e.g., {"-1": 0.2, "0": 0.3}).
    This function converts keys to integers and plots missing Δn values as probability 0.

    Interpretation / invariants
    ---------------------------
    - The plotted support is inferred from the union of keys present in the distinguishable
      and indistinguishable dictionaries for the passive case.
    - Values are treated as probabilities (typically sum to 1 within numerical tolerance).
    """
    fig3 = data.get("fig3", {})
    if not fig3 or "passive" not in fig3 or "active" not in fig3:
        raise KeyError("Input JSON does not contain fig3 passive/active sections in the expected format.")

    passive = fig3["passive"]
    active = fig3["active"]

    dn_dist_pass = _as_int_keyed(passive["Delta_n_distinguishable"])
    dn_ind_pass = _as_int_keyed(passive["Delta_n_indistinguishable"])
    dn_dist_act = _as_int_keyed(active["Delta_n_distinguishable"])
    dn_ind_act = _as_int_keyed(active["Delta_n_indistinguishable"])

    # Infer support from keys
    dns = sorted(set(dn_dist_pass.keys()) | set(dn_ind_pass.keys()))
    if not dns:
        raise ValueError("No Delta n keys found for fig3 plotting.")
    min_dn, max_dn = min(dns), max(dns)
    xs = list(range(min_dn, max_dn + 1))

    def to_list(d: Dict[int, float]) -> List[float]:
        """Convert an int-keyed probability dict into a list aligned with xs (missing keys -> 0)."""
        return [float(d.get(x, 0.0)) for x in xs]

    P_dist_pass = to_list(dn_dist_pass)
    P_ind_pass = to_list(dn_ind_pass)
    P_dist_act = to_list(dn_dist_act)
    P_ind_act = to_list(dn_ind_act)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    fig.subplots_adjust(top=0.80)
    fig.suptitle("Fig. 3 (analytical prediction)", y=0.98, fontweight="bold")

    for ax, panel_letter, panel_name, P_dist, P_ind in [
        (axes[0], "a", "Passive demon", P_dist_pass, P_ind_pass),
        (axes[1], "b", "Active demon", P_dist_act, P_ind_act),
    ]:

        width = 0.38
        ax.bar(
            [x - width / 2 for x in xs],
            P_dist,
            width=width,
            label="Distinguishable",
            facecolor=DIST_FACE,
            edgecolor=DIST_EDGE,
            alpha=BAR_ALPHA,
            linewidth=EDGE_LW,
        )
        ax.bar(
            [x + width / 2 for x in xs],
            P_ind,
            width=width,
            label="Indistinguishable",
            facecolor=IND_FACE,
            edgecolor=IND_EDGE,
            alpha=BAR_ALPHA,
            linewidth=EDGE_LW,
        )

        ax.set_xlabel(r"Photon-number difference $\Delta n$")
        panel_text = fr"$\mathbf{{({panel_letter})}}$ {panel_name}"
        ax.text(
            0.02, 0.97,
            panel_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5)
        )
        ax.set_xticks(xs)
        ax.grid(True, axis="y")

    axes[0].set_ylabel(r"Probability $P(\Delta n)$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.91),
    )
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Entry point: load JSON, generate requested plots, and write them to disk."""
    args = parse_args()
    set_plot_style()
    data = load_json(args.in_path)

    _ensure_dir(args.outdir)

    if not args.no_fig2:
        outpath = os.path.join(args.outdir, f"{args.prefix}_fig2.{args.fmt}")
        plot_fig2(data, outpath)
        print(f"Wrote: {outpath}")

    if not args.no_fig3:
        outpath = os.path.join(args.outdir, f"{args.prefix}_fig3.{args.fmt}")
        plot_fig3(data, outpath)
        print(f"Wrote: {outpath}")

    if args.show:
        # If you want interactive display, re-run the plot functions without closing,
        # or just open the saved files. Keeping this simple:
        print("Open the saved figure files to view them (or rerun with a custom interactive workflow).")


if __name__ == "__main__":
    main()
