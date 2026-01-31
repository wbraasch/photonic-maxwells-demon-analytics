# Photonic Maxwell's Demon: analytical predictions (Haar-averaged)

Minimal reproducibility code for the **analytical predictions** (Haar-averaged photon-number statistics)
shown in Figs. 2–3 of the paper:

> "Bosonic statistics enhance Maxwell's demon in photonic experiment"

This is **not** the full experiment/simulation pipeline.

## Contents

- `compute_photonic_stats.py`: computes Haar-averaged pattern probabilities and derived marginals; writes JSON
- `plot_photonic_stats.py`: reads the JSON and generates Fig. 2/3-style plots for the analytical prediction
- `examples/haar_stats_both_N3.json`: example output JSON (so plotting can be tested immediately)
- `docs/`: preview images used in this README

## Quick start

### 1) Install dependencies

Python 3.9+ recommended.

```bash
pip install numpy sympy matplotlib
```

(Optional) If available, `haarpy` is used for the unitary Weingarten function; otherwise this code falls back
to a closed-form expression valid for `N=3`.

```bash
pip install haarpy
```

### 2) Recompute the JSON (paper defaults)

```bash
python compute_photonic_stats.py --outdir results --tag haar_stats
```

This writes:
- `results/haar_stats_N3.json`

### 3) Generate the plots

```bash
python plot_photonic_stats.py --in results/haar_stats_N3.json --outdir figures --prefix analytic --fmt png
```

This writes:
- `figures/analytic_fig2.png`
- `figures/analytic_fig3.png`

### 4) Test plotting without recomputing

```bash
python plot_photonic_stats.py --in examples/haar_stats_both_N3.json --outdir figures --prefix analytic --fmt png
```

## Notes

- The JSON output includes basic environment metadata.

## License

See `LICENSE`.
