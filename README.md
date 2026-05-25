# Computable Fairness: Boltzmann–Softmax Control for AI Resource Allocation

Reproduction code for the paper:

> **Computable Fairness: Boltzmann–Softmax Control for AI Resource Allocation**
> Ji-Won Park and Chae Un Kim. Submitted to arXiv, 12 April 2026.
> arXiv: [2605.22827](https://arxiv.org/abs/2605.22827) &nbsp;|&nbsp; DOI: [10.48550/arXiv.2605.22827](https://doi.org/10.48550/arXiv.2605.22827)

This repository contains a single, self-contained script that reproduces all
figures and the main results table from the paper.

---

## What this code produces

Running the script generates every figure and table below into an `output/`
directory (as `.pdf`, `.png`, and `.svg`), plus the Table 5 numbers printed to
the console.

| Output | Paper item | Description |
|--------|-----------|-------------|
| `Fig02_boltzmann_allocation` | Figure 2 | Boltzmann–Softmax allocation probabilities vs. inverse temperature β |
| `Fig03_optimization_landscape` | Figure 3 | Static loss components and λ-weighted total loss vs. β |
| `Fig04_heatmap_stability_corridor` | Figure 4 | Total loss heatmap and Stability Corridor in the (β, λ) plane |
| `Fig05_pareto_frontier` | Figure 5 | Pareto frontier, near-optimal cloud, post-shock trajectories |
| `Fig06_efficiency_entropy_frontier` | Figure 6 | Efficiency–fairness frontier |
| `Fig07_dominance_4policy` | Figure 7 | Top-1 dominance under time-varying events (4 policies) |
| `Fig08_dominance_2policy` | Figure 8 | FixedBetaSoftmax vs. AHC++ dominance comparison |
| `Fig09_beta_controller` | Figure 9 | β(t) trajectory of AHC++ under time-varying λ(t) |
| `Fig10_backlog_dynamics` | Figure 10 | Backlog dynamics across policies |
| `Fig11_sensitivity_burst` | Figure 11 | Cumulative dominance exceedance (AUC) vs. burst factor |
| `Fig12_computational_scalability` | Figure 12 | Single allocation-step scalability benchmark |
| console table | Table 5 | Performance and stability metrics by policy (mean ± 95% CI) |

---

## Requirements

- Python 3.9+
- numpy, matplotlib, scipy, scikit-learn

Install with:

```bash
pip install -r requirements.txt
```

For an exact, pinned environment matching the versions used to verify this code,
use `requirements-lock.txt` instead:

```bash
pip install -r requirements-lock.txt
```

## Running

```bash
python cfd_code.py
```

This runs everything (static figures, the dynamic simulation, the sensitivity
sweep, and the scalability benchmark) and reproduces the paper's figures and
Table 5. Figures are written to `./output/` by default. To choose a different
location:

```bash
CFD_OUT_DIR=/path/to/figures python cfd_code.py
```

### Run options

The dynamic part (Figures 7–11 and Table 5) is the most expensive: it runs the
queueing simulation over 10 seeds plus a sensitivity sweep, with N = 1000 agents
and T = 4200 time steps. On a fast machine the full run is on the order of a
minute or two; on slower hardware or single-threaded NumPy builds it can take
much longer. The following flags let you run only what you need:

| Flag | What it does |
|------|--------------|
| (none) | Full reproduction: all figures + Table 5 (default). |
| `--static-only` | Static figures (2–6) and the scalability benchmark (12) only. Fast. |
| `--dynamic-only` | Dynamic figures (7–11) and Table 5 only. |
| `--quick-test` | Smoke test with reduced N and seed counts. **Does not reproduce the paper numbers** — it only checks that the code runs end to end. The console prints a clear warning in this mode. |
| `--show` | Also call `plt.show()` for interactive display. Off by default; figures are always saved to disk regardless. |

Examples:

```bash
python cfd_code.py --static-only     # quick look at the static-model figures
python cfd_code.py --quick-test      # verify the code runs (NOT a reproduction)
```

The simulation uses fixed seeds, so a full run reproduces the figures and Table 5
across machines (up to floating-point rounding).

---

## Notes on reproducibility

- **Static model and Table 5 are deterministic.** All randomized components use
  fixed seeds (`numpy.random.default_rng` with explicit seeds), so Figures 2–11
  and the Table 5 metrics reproduce to the printed precision.
- **Figure 12 is a wall-clock micro-benchmark.** The qualitative conclusion —
  near-linear scaling, far below the O(N²) pairwise reference — is robust. The
  *exact* runtime ratio, however, depends on your hardware, OS, and NumPy/BLAS
  build. The paper reports roughly a single-digit multiplier when the number of
  agents grows 100×; the precise value you observe may differ from the figure
  caption. This is expected for a timing benchmark and is not a change to the
  method.
- **Fonts.** The script requests Arial. On systems without Arial, matplotlib
  falls back to a default sans-serif font and prints a harmless warning; the
  results are unaffected.
- **β grids differ between figures, by design.** Different figures scan β over
  different grids: the loss-landscape figures use a linear grid on [0.05, 10],
  the Pareto frontier (Fig. 5) uses a denser logarithmic grid over the same
  range for a smoother curve, and the efficiency–fairness frontier (Fig. 6) uses
  a wider range [0.01, 100] to show both extremes (near-uniform and
  near-monopoly). These choices affect resolution and display range, not the
  result: the optimal control value β*(λ) is essentially unchanged across grids
  (it varies only at the level of grid spacing). The figures are therefore
  mutually consistent despite the different grids.

---

## Method summary

The allocation rule is the Boltzmann–Softmax distribution

```
P_i(β) = exp(β z_i) / Σ_j exp(β z_j)
```

where `z_i` is an agent's contribution score and `β` is the inverse temperature
(the control variable). A unified loss balances efficiency and fairness,

```
L_tot(β, λ) = λ · L_eff(β) + (1 − λ) · L_ineq(β)
```

with `L_eff` based on normalized contribution-weighted allocation and `L_ineq`
based on the entropy of the allocation distribution. In the dynamic setting, the
Adaptive Hard-Cap Controller (AHC++) updates `β` online from the error between
observed top-1 dominance and the effective target dominance
`ρ_eff(t) = max(ρ_target(t), 1/K(t))`, where `K(t)` is the number of active agents.

See the paper for full definitions and discussion.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@article{park2026computable,
  title  = {Computable Fairness: Boltzmann--Softmax Control for AI Resource Allocation},
  author = {Park, Ji-Won and Kim, Chae Un},
  year   = {2026},
  eprint = {2605.22827},
  archivePrefix = {arXiv},
  primaryClass  = {physics.app-ph},
  doi    = {10.48550/arXiv.2605.22827}
}
```

---

## License

No license has been assigned yet. The authors retain all rights for now;
a license (expected to be a permissive one such as MIT) will be added after
review. Until then, the code is provided for reading and for reproducing the
results of the paper. If you would like to use it for any other purpose,
please contact the authors.
