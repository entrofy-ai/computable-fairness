"""
=======================================================================
Computable Fairness:
Boltzmann-Softmax Control for AI Resource Allocation
=======================================================================
Ji-Won Park and Chae Un Kim. arXiv:2605.22827 (2026).
https://arxiv.org/abs/2605.22827

Code repository: https://github.com/entrofy-ai/computable-fairness

Full reproduction code for the paper.
  - Static model  : Figures 2–6  (Boltzmann allocation, loss landscape,
                     stability corridor, Pareto frontier, efficiency–
                     fairness frontier)
  - Dynamic model : Figures 7–11 (dominance control, β(t) trajectory,
                     backlog dynamics, sensitivity analysis)
  - Scalability   : Figure 12   (single allocation-step benchmark)
  - Table 5       : Performance and stability metrics by policy

All figure/axis labels are matched to the paper terminology.
  - "Fairness loss" (not "Inequality loss" or "Inequity loss")
  - "Fairness" axis  (not "Equality" or "Entropy")

Usage:
    pip install -r requirements.txt   # or: pip install numpy matplotlib scipy scikit-learn
    python cfd_code.py
=======================================================================
"""

import os
import math
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from sklearn.isotonic import IsotonicRegression

# =============================================================
#  COMMON SETUP
# =============================================================
# Output directory. Defaults to ./output next to this script.
# Override with the environment variable CFD_OUT_DIR if you want a custom path,
# e.g.  CFD_OUT_DIR=/path/to/figs  python cfd_code.py
OUT_DIR = Path(os.environ.get("CFD_OUT_DIR", "./output"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Saving figures to: {OUT_DIR.resolve()}")

FONT = "Arial"
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT, "DejaVu Sans", "Liberation Sans"],
    "font.weight": "bold",
    "font.size": 13,
    "axes.labelsize": 15, "axes.labelweight": "bold",
    "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.linewidth": 1.2,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 12, "lines.linewidth": 2.0,
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "text.usetex": False,
})

C = {
    "blue": np.array([0,114,178])/255,
    "orange": np.array([230,159,0])/255,
    "green": np.array([0,158,115])/255,
    "purple": np.array([204,121,167])/255,
    "red": np.array([213,94,0])/255,
    "gray": np.array([90,90,90])/255,
    "black": np.array([0,0,0]),
    "white": np.array([1,1,1]),
}

def save_fig(fig, name, title="", dpi_png=600):
    fig.tight_layout()
    for fmt in ["pdf","png","svg"]:
        fig.savefig(OUT_DIR/f"{name}.{fmt}", format=fmt,
                    dpi=dpi_png if fmt=="png" else None,
                    bbox_inches="tight", pad_inches=0.1, transparent=True)
    if title:
        print(f"\n{'='*70}")
        print(f"  SAVED: {title}")
        print(f"  File : {name}")
        print(f"{'='*70}")
    else:
        print(f"\n  SAVED: {name}")

def softmax_prob(beta, logits):
    z = beta * logits; z = z - np.max(z)
    w = np.exp(z); s = np.sum(w)
    if s <= 0: return np.ones_like(logits)/len(logits)
    return w / s

def cap_redistribute(prob, rho_eff, max_iter=30):
    p = prob.astype(float, copy=True); n = p.size
    cap = max(0.0, min(1.0, float(rho_eff)))
    if n * cap < 1.0: return np.full_like(p, 1.0/n)
    capped = np.zeros_like(p, dtype=bool)
    for _ in range(max_iter):
        newly = (~capped) & (p > cap)
        if newly.any(): p[newly] = cap; capped[newly] = True
        total = p.sum()
        if abs(total-1.0)<1e-12 and (p<=cap+1e-12).all(): return p
        if total > 1.0: p /= total; return p
        deficit = 1.0 - total; uncapped = ~capped
        if not uncapped.any(): return np.full_like(p, 1.0/n)
        weights = p * uncapped; wsum = weights.sum()
        if wsum <= 0: p[uncapped] += deficit / uncapped.sum()
        else: p[uncapped] += deficit * (weights[uncapped] / wsum)
    p /= p.sum(); return p

def entropy_val(p):
    p = np.clip(p, 1e-18, 1.0); return float(-(p*np.log(p)).sum())

def smooth_vec(x, w):
    x = np.asarray(x, dtype=float)
    if w <= 1: return x
    return np.convolve(x, np.ones(w)/w, mode="same")

def norm01(x):
    return (x - x.min())/(x.max() - x.min() + 1e-12)

N_POP = 1000
RANK = np.arange(1, N_POP+1, dtype=float)
LOGITS = -np.log(RANK); LOGITS -= LOGITS.max()
NORM_LOGITS = norm01(LOGITS)
RNG = np.random.default_rng(7)


# =============================================================
#  FIGURE 2: Boltzmann–Softmax Allocation Probabilities
#  (Paper: "Variation of Boltzmann–Softmax allocation
#   probabilities with inverse temperature β")
# =============================================================
def make_fig02():
    print("\n=== Figure 2 ===")
    betas = [0.01, 2.0, 10.0]
    ls = ["--", "-", "-"]; col = [C["gray"], C["blue"], C["orange"]]
    Pmat = np.vstack([softmax_prob(b, LOGITS) for b in betas]).T
    zipfRef = 1.0/RANK; zipfRef /= zipfRef.sum()

    fig, ax = plt.subplots(figsize=(9.8, 6.8), dpi=150)
    ax.grid(True, which="both", alpha=0.25)
    for k, b in enumerate(betas):
        ax.plot(RANK, Pmat[:,k], linestyle=ls[k], color=col[k], linewidth=3)
    ax.plot(RANK, zipfRef, "k--", linewidth=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1, N_POP); ax.set_ylim(1e-12, 1)
    ax.set_xlabel("Agent rank (log scale)")
    ax.set_ylabel(r"$P_i$ (log scale)")
    ax.legend([r"$\beta=0.01$: Random-like (high entropy)",
               r"$\beta=2.00$: Boltzmann (balanced)",
               r"$\beta=10.0$: Winner-takes-all",
               "Reference: Zipf slope"], loc="upper right", frameon=True)
    save_fig(fig, "Fig02_boltzmann_allocation",
             title="Figure 2. Variation of Boltzmann–Softmax Allocation Probabilities with β")
    plt.show(); plt.close(fig)

# =============================================================
#  FIGURE 3: Static Loss Components and λ-Weighted Total Loss
#  (Paper: "Static loss components and λ-weighted total loss
#   vs. β. The minimum-loss β shifts with policy weight λ.")
# =============================================================
def make_fig03(beta_grid, L_eff_n, L_ineq_n):
    print("\n=== Figure 3 ===")
    lam_family = [0.30, 0.50, 0.70, 0.90]
    fig, ax = plt.subplots(figsize=(9.8, 6.8), dpi=150)
    ax.grid(True, alpha=0.25)
    ax.plot(beta_grid, L_eff_n, "--", color=C["blue"], linewidth=3,
            label="Efficiency loss (norm)")
    ax.plot(beta_grid, L_ineq_n, "--", color=C["orange"], linewidth=3,
            label="Fairness loss (norm)")
    for lam in lam_family:
        Ltot = lam*L_eff_n + (1-lam)*L_ineq_n
        ax.plot(beta_grid, Ltot, "-", color=C["gray"], linewidth=2.5,
                label=fr"Total ($\lambda={lam:.2f}$)")
    ax.set_xlabel(r"Inverse temperature $\beta$"); ax.set_ylabel("Normalized loss")
    ax.set_xlim(0, 10); ax.set_ylim(0, 1)
    handles, labels = ax.get_legend_handles_labels()
    seen = set(); h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen: seen.add(l); h2.append(h); l2.append(l)
    ax.legend(h2, l2, loc="upper right", frameon=True)
    save_fig(fig, "Fig03_optimization_landscape",
             title="Figure 3. Static loss components and λ-weighted total loss vs. β")
    plt.show(); plt.close(fig)

# =============================================================
#  FIGURE 4: Total Loss Heatmap and Stability Corridor
#  (Paper: "Total loss heatmap and Stability Corridor in the
#   (β, λ) plane.")
# =============================================================
def make_fig04():
    print("\n=== Figure 4 ===")
    beta_hm = np.linspace(0.05, 10.0, 300)
    lam_hm = np.linspace(0.0, 1.0, 280)
    L_eff_v = np.zeros(len(beta_hm)); L_ineq_v = np.zeros(len(beta_hm))
    for i, b in enumerate(beta_hm):
        P = softmax_prob(b, LOGITS)
        L_eff_v[i] = 1 - np.dot(P, NORM_LOGITS)
        H = -np.sum(P*np.log(P+1e-300)); L_ineq_v[i] = 1 - H/np.log(N_POP)
    Le = norm01(L_eff_v); Li = norm01(L_ineq_v)
    LAM = lam_hm[:,np.newaxis]
    Loss = LAM*Le[np.newaxis,:] + (1-LAM)*Li[np.newaxis,:]
    LossN = norm01(Loss)
    flat = np.sort(LossN.ravel())
    lo = flat[int(0.02*len(flat))]; hi = flat[int(0.98*len(flat))]
    LossV = np.clip((LossN-lo)/(hi-lo+1e-12), 0, 1)**0.85

    beta_star_raw = np.array([beta_hm[np.argmin(LossN[j])] for j in range(len(lam_hm))])
    ir = IsotonicRegression(increasing=True)
    beta_star_mono = ir.fit_transform(lam_hm, beta_star_raw)
    beta_star_sm = gaussian_filter1d(beta_star_mono, sigma=8)

    tolAbsN=0.12; sigma_sm=10
    beta_lo_raw = np.full(len(lam_hm), np.nan)
    beta_hi_raw = np.full(len(lam_hm), np.nan)
    for j in range(len(lam_hm)):
        row = LossN[j]; idx_star = np.argmin(row)
        ok = row <= (row[idx_star]+tolAbsN)
        labeled, cur = np.zeros(len(ok), dtype=int), 0
        for k in range(len(ok)):
            if ok[k]:
                if k==0 or not ok[k-1]: cur += 1
                labeled[k] = cur
        seg_id = labeled[idx_star]
        if seg_id == 0: continue
        idxs = np.where(labeled==seg_id)[0]
        beta_lo_raw[j] = beta_hm[idxs[0]]; beta_hi_raw[j] = beta_hm[idxs[-1]]
    valid = ~np.isnan(beta_lo_raw) & ~np.isnan(beta_hi_raw)
    beta_lo_sm = gaussian_filter1d(np.where(valid, beta_lo_raw, np.nan), sigma=sigma_sm)
    beta_hi_sm = gaussian_filter1d(np.where(valid, beta_hi_raw, np.nan), sigma=sigma_sm)
    beta_star_clipped = np.clip(beta_star_sm,
        np.where(valid, beta_lo_sm, beta_star_sm),
        np.where(valid, beta_hi_sm, beta_star_sm))

    fig, ax = plt.subplots(figsize=(11, 7.5))
    try: cmap = plt.cm.turbo
    except: cmap = plt.cm.jet
    im = ax.imshow(LossV, extent=[beta_hm[0],beta_hm[-1],lam_hm[0],lam_hm[-1]],
        origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='bilinear')
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('Normalized total loss (contrast-boosted)', fontsize=13, fontweight='bold')
    ax.contour(beta_hm, lam_hm, LossV, levels=np.linspace(0.10,0.95,9),
               colors='white', linewidths=0.8, alpha=0.50)
    ax.fill_betweenx(lam_hm[valid], beta_lo_sm[valid], beta_hi_sm[valid],
        color='#00CFFF', alpha=0.30, linewidth=0)
    for bnd in [beta_lo_sm, beta_hi_sm]:
        ax.plot(bnd[valid], lam_hm[valid], '-', color='black', linewidth=6.0, zorder=4)
        ax.plot(bnd[valid], lam_hm[valid], '-', color='#FFD700', linewidth=3.5, zorder=5)
    ax.plot(beta_star_clipped, lam_hm, 'w-', linewidth=6.0, zorder=6)
    ax.plot(beta_star_clipped, lam_hm, 'k--', linewidth=2.5, zorder=7)

    lam_tip=0.50; j_tip=np.argmin(np.abs(lam_hm-lam_tip))
    beta_mid = beta_star_clipped[j_tip]
    ax.annotate('Stability Corridor', xy=(beta_mid,lam_tip),
        xytext=(beta_mid+3.0, lam_tip+0.18), fontsize=14, color='#FFD700', fontweight='bold',
        arrowprops=dict(arrowstyle='-|>', color='#FFD700', lw=2.5, mutation_scale=18),
        bbox=dict(boxstyle='round,pad=0.4', facecolor=(0.05,0.05,0.05,0.65), edgecolor='none'),
        ha='left', va='center', zorder=10)
    ax.text(0.28, 0.06, 'Random zone\n(low \u03b2)', color='white', fontsize=11, ha='center',
            fontweight='bold', bbox=dict(facecolor=(0,0,0,0.45), edgecolor='none', pad=3))
    ax.text(7.8, 0.93, 'Winner-takes-all\n(high \u03b2)', color='white', fontsize=11, ha='center',
            fontweight='bold', bbox=dict(facecolor=(0,0,0,0.45), edgecolor='none', pad=3))
    corr_patch = mpatches.Patch(facecolor='#00CFFF', alpha=0.30, edgecolor='#FFD700',
        linewidth=1.5, label='Stability Corridor (near-optimal band)')
    bnd_line, = ax.plot([], [], '-', color='#FFD700', linewidth=3.5, label='Corridor boundary')
    opt_line, = ax.plot([], [], 'k--', linewidth=2.5, label='Optimal path \u03b2*(\u03bb)')
    ax.legend(handles=[corr_patch, bnd_line, opt_line], loc='lower right', framealpha=0.85,
        facecolor='#1a1a1a', labelcolor='white', edgecolor='none')
    ax.set_xlim(beta_hm[0], beta_hm[-1]); ax.set_ylim(0, 1)
    ax.set_xlabel('Inverse temperature \u03b2'); ax.set_ylabel('Efficiency weight \u03bb')
    save_fig(fig, "Fig04_heatmap_stability_corridor",
             title="Figure 4. Total Loss Heatmap and Stability Corridor in (β, λ) Plane")
    plt.show(); plt.close(fig)

# =============================================================
#  FIGURE 5: Pareto Frontier and Post-Shock Trajectories
#  (Paper: "Pareto frontier, near-optimal cloud, and post-shock
#   trajectories. (a) ★ = β*(λ=0.60). (b) AHC++ recovers to
#   stability zone; fixed policy collapses (×).")
# =============================================================
def make_fig05():
    print("\n=== Figure 5 ===")
    rng = np.random.default_rng(7)
    beta_scan = np.logspace(np.log10(0.05), np.log10(10.0), 2500)
    Eff_s = np.zeros_like(beta_scan); Eq_s = np.zeros_like(beta_scan)
    for i, b in enumerate(beta_scan):
        P = softmax_prob(b, LOGITS)
        Eff_s[i] = np.sum(P*NORM_LOGITS)
        H = -np.sum(P*np.log(P+1e-12)); Eq_s[i] = H/np.log(N_POP)
    L_eff_s = 1-Eff_s; L_ineq_s = 1-Eq_s
    L_eff_s_n = norm01(L_eff_s); L_ineq_s_n = norm01(L_ineq_s)

    lambda_sweep = np.linspace(0.30, 0.90, 110)
    cloudTol=0.05; samplesPerLam=18; jitter=0.010
    Cloud_X, Cloud_Y = [], []
    for lam in lambda_sweep:
        Loss_s = lam*L_eff_s_n + (1-lam)*L_ineq_s_n
        mL = np.min(Loss_s); ok = np.where(Loss_s <= (1+cloudTol)*mL)[0]
        if ok.size == 0: continue
        ns = min(samplesPerLam, ok.size); pick = rng.choice(ok, size=ns, replace=True)
        Cloud_X.append(np.clip(Eff_s[pick]+(rng.random(ns)-0.5)*jitter, 0, 1))
        Cloud_Y.append(np.clip(Eq_s[pick]+(rng.random(ns)-0.5)*jitter, 0, 1))
    Cloud_X = np.concatenate(Cloud_X) if Cloud_X else np.array([])
    Cloud_Y = np.concatenate(Cloud_Y) if Cloud_Y else np.array([])

    lam0=0.60; Loss0 = lam0*L_eff_s_n + (1-lam0)*L_ineq_s_n
    idx0 = int(np.argmin(Loss0)); beta0 = beta_scan[idx0]
    sx, sy = Eff_s[idx0], Eq_s[idx0]
    beta_path_res = np.linspace(beta0, min(10, beta0*1.6), 25)
    beta_path_col = np.linspace(beta0, 10.0, 25)
    trajR_x = np.interp(beta_path_res, beta_scan, Eff_s)
    trajR_y = np.interp(beta_path_res, beta_scan, Eq_s)
    trajC_x = np.interp(beta_path_col, beta_scan, Eff_s)
    trajC_y = np.interp(beta_path_col, beta_scan, Eq_s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.6), dpi=300)
    ax1.grid(True, alpha=0.25); ax1.set_axisbelow(True)
    if Cloud_X.size:
        ax1.scatter(Cloud_X, Cloud_Y, s=18, c=[C["green"]], alpha=0.20,
                    edgecolors="none", label="Near-optimal cloud")
    ax1.plot(Eff_s, Eq_s, "k--", linewidth=2.2, label="Nominal frontier")
    ax1.plot(sx, sy, marker="p", markersize=14, markerfacecolor=C["purple"],
             markeredgecolor=C["white"], markeredgewidth=1.2, linestyle="None",
             label=r"Optimal Operating Point $\beta^*(\lambda)$")
    ax1.text(sx+0.018, sy+0.035,
             "Optimal Operating Point\n"+rf"$\beta^*(\lambda={lam0:.2f})$",
             color=C["purple"], fontsize=12, fontweight="bold")
    ax1.set_title("(a) Robust Pareto frontier (near-opt cloud)")
    ax1.set_xlabel("Efficiency (throughput proxy)"); ax1.set_ylabel("Fairness (normalized entropy)")
    ax1.set_xlim(0.4, 1.02); ax1.set_ylim(0.0, 1.02); ax1.legend(loc="lower left", frameon=True)

    ax2.grid(True, alpha=0.25); ax2.set_axisbelow(True)
    if Cloud_X.size:
        ax2.scatter(Cloud_X, Cloud_Y, s=14, c=[[0.8,0.8,0.8]], alpha=0.12,
                    edgecolors="none", label="Stability zone (cloud)")
    ax2.plot(Eff_s, Eq_s, "k--", linewidth=1.8, label="Nominal frontier")
    ax2.plot(sx, sy, marker="p", markersize=14, markerfacecolor=C["purple"],
             markeredgecolor=C["white"], markeredgewidth=1.2, linestyle="None",
             label=r"Optimal Operating Point $\beta^*(\lambda)$")
    ax2.plot(trajR_x, trajR_y, "-", color=C["purple"], linewidth=3, label="Recovery (AHC++ control)")
    ax2.plot(trajR_x[-1], trajR_y[-1], "o", color=C["purple"], markersize=6)
    ax2.plot(trajC_x, trajC_y, "-.", color=C["black"], linewidth=2.5, label="Collapse (fixed policy)")
    ax2.plot(trajC_x[-1], trajC_y[-1], "x", color=C["black"], markersize=9, markeredgewidth=2)
    ax2.text(trajR_x[3], trajR_y[3]+0.04, r"Shock $\rightarrow$ Recovery",
             color=C["purple"], fontsize=12, fontweight="bold")
    ax2.text(trajC_x[-1]-0.12, trajC_y[-1]+0.06, "Collapse",
             color=C["black"], fontsize=12, fontweight="bold")
    ax2.set_title("(b) Post-shock trajectories (schematic on frontier)")
    ax2.set_xlabel("Efficiency"); ax2.set_ylabel("Fairness")
    ax2.set_xlim(0.4, 1.02); ax2.set_ylim(0.0, 1.02); ax2.legend(loc="lower left", frameon=True)
    save_fig(fig, "Fig05_pareto_frontier",
             title="Figure 5. Pareto frontier, near-optimal cloud, and post-shock trajectories")
    plt.show(); plt.close(fig)

# =============================================================
#  FIGURE 6: Efficiency–Fairness Frontier
#  (Paper: "Efficiency–fairness frontier generated by the
#   Boltzmann–Softmax rule.")
# =============================================================
def make_fig06():
    print("\n=== Figure 6 ===")
    N = 1000; logits_6 = -np.log(np.arange(1, N+1, dtype=float))
    betas = np.logspace(-2, 2, 220); z01 = norm01(logits_6)
    eff, eq, top1 = [], [], []
    for b in betas:
        p = softmax_prob(float(b), logits_6)
        eff.append(float((p*z01).sum()))
        eq.append(entropy_val(p)/np.log(N)); top1.append(float(p.max()))
    eff = np.array(eff); eq = np.array(eq); top1 = np.array(top1)

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    ax.grid(True, alpha=0.2)
    ax.plot(eq, eff, linewidth=2.5, label="Boltzmann\u2013Softmax frontier")
    for bm in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        idx = int(np.argmin(np.abs(betas-bm)))
        ax.plot(eq[idx], eff[idx], marker="o")
        ax.annotate(f"\u03b2={bm:g}", (eq[idx],eff[idx]),
                    textcoords="offset points", xytext=(6,4), fontsize=8)
    ax.set_xlabel("Fairness (Normalized Entropy, H(P) / log N)")
    ax.set_ylabel(r"Efficiency (sum of $P_i \cdot$ normalized score)")
    ax2 = ax.twinx()
    ax2.plot(eq, top1, linestyle="--", linewidth=1.6, label="Top-1 dominance")
    ax2.set_ylabel(r"Top-1 dominance (max $P_i$)")
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax2.legend(l1+l2, la1+la2, loc="upper right", frameon=True)
    save_fig(fig, "Fig06_efficiency_entropy_frontier",
             title="Figure 6. Efficiency–Fairness Frontier Generated by Boltzmann–Softmax Rule")
    plt.show(); plt.close(fig)


# =============================================================
#  DYNAMIC MODEL: Figures 7–11 + Table 5
#  Queueing simulation with 4 policies under time-varying
#  disturbances (burst, policy change, abuse, capacity reduction)
#  Paper Section 3.2.2: "Dynamic Model Results"
# =============================================================
def run_dynamic_and_plot(N=1000, T=4200, nSeeds=10, sens_seeds=8):
    print("\n=== Dynamic Model Simulation (Figures 7\u201311, Table 5) ===")
    W=400; t_stable=1600
    t_burst_start=1000; t_burst_end=1500; t_policy_change=1400
    t_abuse_start=2200; t_abuse_end=2600; t_cap_start=2800; t_cap_end=3200

    zipf_a=1.10; rank=np.arange(1,N+1)
    pop=rank**(-zipf_a); pop/=pop.sum(); cumPop=np.cumsum(pop)
    score0=-np.log(rank.astype(float)); score0-=score0.max()
    svcMean=2+10*((rank-1)/(N-1))**0.8; svcJitter=0.35
    shockTenant0=0; scoreBoost_val=6.0; forceProb_val=0.55
    lambda_pre=0.50; lambda_post=0.75; rho_low=0.12; rho_high=0.40
    base_arrival=3.0; burst_factor=1.80; headroom=1.10; cap_drop=0.75
    # AHC++ controller parameters (cf. Eq. 16: beta(t+1) = Clip(beta(t) - eta * [y(t) - rho_eff(t)])).
    #   beta_gain  -> eta, the control gain in Eq. 16 (feedback intensity on the dominance error)
    #   beta_min, beta_max -> the Clip(.) range [beta_min, beta_max]
    #   beta_track, beta_smooth -> the reference-tracking and smoothing stabilizers
    #                              mentioned after Eq. 16 as practical add-ons.
    beta_min=0.1; beta_max=8.0; beta_gain=6.0; beta_smooth=0.85; beta_track=0.15

    POL=["RoundRobin","Greedy(z)","FixedBetaSoftmax","AdaptiveHardCap++"]
    nPol=len(POL); pAHC=POL.index("AdaptiveHardCap++")

    lambda_series=np.full(T, lambda_pre); lambda_series[t_policy_change-1:]=lambda_post
    lambda_jobs=np.full(T, base_arrival)
    lambda_jobs[t_burst_start-1:t_burst_end]=base_arrival*burst_factor
    forceProb=np.zeros(T); forceProb[t_abuse_start-1:t_abuse_end]=forceProb_val
    scoreBoost=np.zeros(T); scoreBoost[t_abuse_start-1:t_abuse_end]=scoreBoost_val
    rho_target=rho_low+(rho_high-rho_low)*lambda_series
    E_svc=np.sum(pop*svcMean)*np.exp(0.5*(svcJitter**2))
    cap_base=max(1, int(round(headroom*base_arrival*E_svc)))
    capWork=np.full(T, cap_base, dtype=int)
    capWork[t_cap_start-1:t_cap_end]=max(1, int(round(cap_base*cap_drop)))

    # beta*(lambda) precompute
    logits_d=score0.copy(); norm_logits_d=norm01(logits_d)
    beta_grid=np.linspace(0.1,10.0,320); lam_grid=np.linspace(0.0,1.0,161)
    L_eff_d=np.zeros_like(beta_grid); L_fair_d=np.zeros_like(beta_grid)
    for i, b in enumerate(beta_grid):
        P=softmax_prob(b, logits_d)
        L_eff_d[i]=-np.sum(P*norm_logits_d)
        ent=-np.sum(P*np.log(P+1e-12)); L_fair_d[i]=1-ent/np.log(N)
    L_eff_d=norm01(L_eff_d); L_fair_d=norm01(L_fair_d)
    beta_star=np.zeros_like(lam_grid)
    for j, lam in enumerate(lam_grid):
        L_tot=lam*L_eff_d+(1-lam)*L_fair_d; beta_star[j]=beta_grid[int(np.argmin(L_tot))]
    beta_fixed=np.interp(lambda_pre, lam_grid, beta_star)

    # Storage
    top1_all=np.zeros((nPol,nSeeds,T)); backlog_all=np.zeros((nPol,nSeeds,T))
    beta_all=np.zeros((nPol,nSeeds,T)); K_all=np.zeros((nPol,nSeeds,T))
    rho_eff_all=np.full((nPol,nSeeds,T), np.nan)
    thrpt=np.zeros((nPol,nSeeds)); meanLat=np.full((nPol,nSeeds),np.nan)
    p95Lat=np.full((nPol,nSeeds),np.nan); maxTop1Post=np.zeros((nPol,nSeeds))
    fracGTPost=np.zeros((nPol,nSeeds)); aucTgtPost=np.zeros((nPol,nSeeds))
    aucEffPost=np.zeros((nPol,nSeeds)); backEnd=np.zeros((nPol,nSeeds))

    print(f"  N={N}, T={T}, W={W}, seeds={nSeeds}")
    for s in range(nSeeds):
        rng=np.random.default_rng(1000+(s+1))
        arrTen=[None]*T; arrWork=[None]*T
        for t in range(T):
            kArr=rng.poisson(lambda_jobs[t])
            if kArr<=0: arrTen[t]=np.empty(0,dtype=int); arrWork[t]=np.empty(0,dtype=int); continue
            ten=np.empty(kArr,dtype=int); wk=np.empty(kArr,dtype=int)
            for kk in range(kArr):
                if forceProb[t]>0 and rng.random()<forceProb[t]: j=shockTenant0
                else:
                    u=rng.random(); j=int(np.searchsorted(cumPop,u,side="left"))
                    if j>=N: j=N-1
                m=svcMean[j]; st=max(1,int(round(m*np.exp(svcJitter*rng.standard_normal()))))
                ten[kk]=j; wk[kk]=st
            arrTen[t]=ten; arrWork[t]=wk

        for p_idx, pol in enumerate(POL):
            qArrT=[[] for _ in range(N)]; qRem=[[] for _ in range(N)]
            head=np.zeros(N,dtype=int); qLen=np.zeros(N,dtype=int)
            ring=-np.ones(W,dtype=int); ring_idx=0; ring_fill=0
            win_counts=np.zeros(N,dtype=int); win_total=0
            rr_ptr=0; serverTenant=-1; remWork=0; curArrT=-1
            beta=float(beta_fixed); lat_list=[]; completed=0

            for t in range(T):
                lam_t=lambda_series[t]; rho_tgt=rho_target[t]
                beta_ref=np.interp(lam_t, lam_grid, beta_star)
                ten=arrTen[t]; wk=arrWork[t]
                if ten.size>0:
                    for j, st in zip(ten,wk): qLen[j]+=1; qArrT[j].append(t); qRem[j].append(int(st))
                cstep=int(capWork[t])
                for uu in range(cstep):
                    if remWork<=0:
                        active=np.nonzero(qLen>0)[0]
                        if active.size==0: serverTenant=-1; remWork=0; break
                        if pol=="RoundRobin":
                            cand=active[active>=rr_ptr]
                            if cand.size==0: cand=active
                            chosen=int(cand[0]); rr_ptr=chosen+1
                            if rr_ptr>=N: rr_ptr=0
                        elif pol=="Greedy(z)":
                            svec=score0.copy()
                            if scoreBoost[t]!=0: svec[shockTenant0]+=scoreBoost[t]
                            chosen=int(active[int(np.argmax(svec[active]))])
                        else:
                            svec=score0.copy()
                            if scoreBoost[t]!=0: svec[shockTenant0]+=scoreBoost[t]
                            sa=svec[active]; b=beta_fixed if pol=="FixedBetaSoftmax" else beta
                            prob=softmax_prob(b, sa)
                            if pol=="AdaptiveHardCap++":
                                K_now=int(active.size)
                                rho_eff=max(rho_tgt, 1.0/max(1,K_now))
                                prob=cap_redistribute(prob, rho_eff)
                            r=rng.random(); cdf=np.cumsum(prob)
                            ii=int(np.searchsorted(cdf,r,side="left"))
                            if ii>=active.size: ii=0
                            chosen=int(active[ii])
                        curArrT=qArrT[chosen][head[chosen]]
                        remWork=qRem[chosen][head[chosen]]
                        head[chosen]+=1; qLen[chosen]-=1; serverTenant=chosen
                        if head[chosen]>50 and head[chosen]>len(qArrT[chosen])//2:
                            qArrT[chosen]=qArrT[chosen][head[chosen]:]
                            qRem[chosen]=qRem[chosen][head[chosen]:]; head[chosen]=0
                        if qLen[chosen]==0 and head[chosen]>=len(qArrT[chosen]):
                            qArrT[chosen]=[]; qRem[chosen]=[]; head[chosen]=0
                    if remWork>0:
                        remWork-=1
                        if ring_fill<W: ring_fill+=1
                        else:
                            old=ring[ring_idx]
                            if old!=-1: win_counts[old]-=1; win_total-=1
                        ring[ring_idx]=serverTenant; win_counts[serverTenant]+=1; win_total+=1
                        ring_idx+=1
                        if ring_idx>=W: ring_idx=0
                        if remWork==0: completed+=1; lat_list.append(t-curArrT+1)
                top1=(win_counts.max()/win_total) if win_total>0 else 0.0
                Kt=int(np.count_nonzero(qLen>0)+(1 if remWork>0 else 0))
                if pol=="AdaptiveHardCap++" and win_total>0:
                    rho_eff_now=max(rho_tgt,1.0/max(1,Kt))
                    err=top1-rho_eff_now; beta_prop=beta-beta_gain*err   # Eq. 16: beta - eta*(y - rho_eff)
                    beta_prop+=beta_track*(beta_ref-beta_prop)
                    beta_prop=float(np.clip(beta_prop,beta_min,beta_max))
                    beta=beta_smooth*beta+(1-beta_smooth)*beta_prop
                    rho_eff_all[p_idx,s,t]=rho_eff_now
                else: rho_eff_all[p_idx,s,t]=rho_tgt
                top1_all[p_idx,s,t]=top1
                backlog_all[p_idx,s,t]=float(qLen.sum()+(1 if remWork>0 else 0))
                beta_all[p_idx,s,t]=beta; K_all[p_idx,s,t]=float(Kt)
            thrpt[p_idx,s]=completed/T
            if len(lat_list)>0:
                lat=np.array(lat_list); meanLat[p_idx,s]=lat.mean(); p95Lat[p_idx,s]=np.percentile(lat,95)
            post=slice(t_stable-1,T); top_post=top1_all[p_idx,s,post]; rho_post=rho_target[post]
            rho_eff_post=rho_eff_all[p_idx,s,post]
            maxTop1Post[p_idx,s]=float(np.max(top_post))
            fracGTPost[p_idx,s]=float(np.mean(top_post>rho_post))
            aucTgtPost[p_idx,s]=float(np.mean(np.maximum(0,top_post-rho_post)))
            aucEffPost[p_idx,s]=float(np.mean(np.maximum(0,top_post-rho_eff_post)))
            backEnd[p_idx,s]=backlog_all[p_idx,s,-1]

    # --- Table 5: Performance and stability metrics by policy ---
    print("\n" + "=" * 90)
    print("  Table 5. Performance and stability metrics by policy (mean \u00b1 95% CI)")
    print("=" * 90)

    def mu_ci(x):
        x = np.asarray(x, dtype=float); mu = np.nanmean(x)
        sd = np.nanstd(x, ddof=1)
        ci = 1.96 * sd / math.sqrt(np.count_nonzero(~np.isnan(x)))
        return mu, ci

    metrics = ["thrpt", "meanLat", "p95Lat", "maxTop1", "frac>tgt",
               "AUC_tgt", "AUC_eff", "backEnd"]
    data_arrays = [thrpt, meanLat, p95Lat, maxTop1Post, fracGTPost,
                   aucTgtPost, aucEffPost, backEnd]

    # Header: Metric | RoundRobin | Greedy(z) | FixedBetaSoftmax | AHC++
    header = f"{'Metric':>12s}"
    for name in POL:
        header += f"  {name:>22s}"
    print(header)
    print("-" * 90)

    for m_idx, m_name in enumerate(metrics):
        row = f"{m_name:>12s}"
        for p_idx in range(nPol):
            mu, ci = mu_ci(data_arrays[m_idx][p_idx])
            if m_name in ["thrpt"]:
                row += f"  {mu:>8.4f} \u00b1 {ci:.4f}"
            elif m_name in ["meanLat", "p95Lat"]:
                row += f"  {mu:>8.1f} \u00b1 {ci:5.1f}"
            elif m_name in ["maxTop1", "frac>tgt"]:
                row += f"  {mu:>8.3f} \u00b1 {ci:.3f}"
            elif m_name in ["AUC_tgt", "AUC_eff"]:
                row += f"  {mu:>8.5f} \u00b1 {ci:.5f}"
            elif m_name == "backEnd":
                row += f"  {mu:>8.0f} \u00b1 {ci:5.0f}"
        print(row)
    print("=" * 90)

    # --- FIGURE 7: 4-policy dominance ---
    print("\n=== Figure 7 ===")
    t_arr=np.arange(1,T+1); smWin=35
    top1_mean=top1_all.mean(axis=1); K_mean=K_all.mean(axis=1)
    rho_eff_mean=rho_eff_all[pAHC].mean(axis=0); beta_mean=beta_all.mean(axis=1)
    top1_plot=np.vstack([smooth_vec(top1_mean[p],smWin) for p in range(nPol)])
    K_plot_ahc=smooth_vec(K_mean[pAHC],smWin); rho_eff_plot=smooth_vec(rho_eff_mean,smWin)
    evX=[t_burst_start, t_policy_change, t_abuse_start, t_cap_start]
    evLbl=["burst","policy","abuse","cap"]

    fig, axL=plt.subplots(figsize=(12.8,5.4), dpi=150)
    axL.grid(True, alpha=0.25)
    axL.plot(t_arr, rho_target, "--", color=C["black"], linewidth=2.0, label="\u03c1_target(\u03bb(t))")
    axL.plot(t_arr, rho_eff_plot, ":", color=C["black"], linewidth=2.0, label="\u03c1_eff (AHC++)")
    axL.plot(t_arr, top1_plot[0], "-", color=C["green"], label="RoundRobin")
    axL.plot(t_arr, top1_plot[1], "--", color=C["orange"], label="Greedy(z)")
    axL.plot(t_arr, top1_plot[2], "-.", color=C["blue"], label="FixedBetaSoftmax")
    axL.plot(t_arr, top1_plot[3], "-", color=C["purple"], linewidth=2.6, label="AdaptiveHardCap++")
    axL.set_xlabel("time step"); axL.set_ylabel("top-1 service-share (smoothed)")
    axR=axL.twinx(); axR.plot(t_arr, K_plot_ahc, "--", color=C["red"], linewidth=2.0, label="K(t)")
    axR.set_ylabel("K(t) active tenants (smoothed)")
    yl=axL.get_ylim()
    for x, lab in zip(evX, evLbl):
        axL.vlines(x, yl[0], yl[1], linestyles="--", colors=C["gray"], linewidth=1.0)
        axL.text(x+12, yl[0]+0.92*(yl[1]-yl[0]), lab, rotation=90, color=C["gray"])
    h1,l1=axL.get_legend_handles_labels(); h2,l2=axR.get_legend_handles_labels()
    axL.legend(h1+h2, l1+l2, loc="upper center", bbox_to_anchor=(0.5,1.18), ncol=4, frameon=True)
    save_fig(fig, "Fig07_dominance_4policy",
             title="Figure 7. Smoothed Top-1 Dominance Under Time-Varying Events")
    plt.show(); plt.close(fig)

    # --- FIGURE 8: 2-policy dominance ---
    print("\n=== Figure 8 ===")
    fig, axL=plt.subplots(figsize=(12.8,5.4), dpi=150)
    axL.grid(True, alpha=0.25)
    axL.plot(t_arr, rho_target, "--", color=C["black"], linewidth=2.0, label="\u03c1_target(\u03bb(t))")
    axL.plot(t_arr, rho_eff_plot, ":", color=C["black"], linewidth=2.0, label="\u03c1_eff (AHC++)")
    axL.plot(t_arr, top1_plot[2], "-.", color=C["blue"], linewidth=2.2, label="FixedBetaSoftmax")
    axL.plot(t_arr, top1_plot[3], "-", color=C["purple"], linewidth=2.6, label="AdaptiveHardCap++")
    axL.set_xlabel("time step"); axL.set_ylabel("top-1 service-share (smoothed)")
    axR=axL.twinx(); axR.plot(t_arr, K_plot_ahc, "--", color=C["red"], linewidth=2.0, label="K(t)")
    axR.set_ylabel("K(t) active tenants (smoothed)")
    yl=axL.get_ylim()
    for x, lab in zip(evX, evLbl):
        axL.vlines(x, yl[0], yl[1], linestyles="--", colors=C["gray"], linewidth=1.0)
        axL.text(x+12, yl[0]+0.92*(yl[1]-yl[0]), lab, rotation=90, color=C["gray"])
    h1,l1=axL.get_legend_handles_labels(); h2,l2=axR.get_legend_handles_labels()
    axL.legend(h1+h2, l1+l2, loc="upper center", bbox_to_anchor=(0.5,1.18), ncol=3, frameon=True)
    save_fig(fig, "Fig08_dominance_2policy",
             title="Figure 8. FixedBetaSoftmax vs. AHC++ Dominance Comparison Over Time")
    plt.show(); plt.close(fig)

    # --- FIGURE 9: beta(t) with lambda(t) ---
    print("\n=== Figure 9 ===")
    bts=smooth_vec(beta_mean[pAHC], smWin)
    fig, axL=plt.subplots(figsize=(12.8,5.0), dpi=150)
    axL.grid(True, alpha=0.25)
    axL.plot(t_arr, bts, "-", color=C["purple"], linewidth=2.6, label="\u03b2(t) (AHC++)")
    axL.set_xlabel("time step"); axL.set_ylabel("\u03b2(t)")
    axR=axL.twinx(); axR.plot(t_arr, lambda_series, ":", color=C["gray"], linewidth=2.2, label="\u03bb(t)")
    axR.set_ylabel("\u03bb(t)")
    yl=axL.get_ylim()
    for x in evX: axL.vlines(x, yl[0], yl[1], linestyles="--", colors=C["gray"], linewidth=1.0)
    h1,l1=axL.get_legend_handles_labels(); h2,l2=axR.get_legend_handles_labels()
    axL.legend(h1+h2, l1+l2, loc="upper left", frameon=True)
    save_fig(fig, "Fig09_beta_controller",
             title="Figure 9. β(t) Trajectory of AHC++ Under Time-Varying λ(t)")
    plt.show(); plt.close(fig)

    # --- FIGURE 10: Backlog ---
    print("\n=== Figure 10 ===")
    back_mean=backlog_all.mean(axis=1)
    fig, ax=plt.subplots(figsize=(12.8,5.4), dpi=150)
    ax.grid(True, alpha=0.25)
    ax.plot(t_arr, smooth_vec(back_mean[0],smWin), "-", color=C["green"], label="RoundRobin")
    ax.plot(t_arr, smooth_vec(back_mean[1],smWin), "--", color=C["orange"], label="Greedy(z)")
    ax.plot(t_arr, smooth_vec(back_mean[2],smWin), "-.", color=C["blue"], label="FixedBetaSoftmax")
    ax.plot(t_arr, smooth_vec(back_mean[3],smWin), "-", color=C["purple"], linewidth=2.6, label="AdaptiveHardCap++")
    ax.set_xlabel("time step"); ax.set_ylabel("backlog (jobs)")
    yl=ax.get_ylim()
    for x, lab in zip(evX, evLbl):
        ax.vlines(x, yl[0], yl[1], linestyles="--", colors=C["gray"], linewidth=1.0)
        ax.text(x+12, yl[0]+0.92*(yl[1]-yl[0]), lab, rotation=90, color=C["gray"])
    ax.legend(loc="upper left", ncol=2, frameon=True)
    save_fig(fig, "Fig10_backlog_dynamics",
             title="Figure 10. Backlog Dynamics Across Policies Under Time-Varying Events")
    plt.show(); plt.close(fig)

    # --- FIGURE 11: Sensitivity (FULL simulation loop, no placeholders) ---
    print("\n=== Figure 11: Sensitivity analysis (running additional simulations) ===")
    burst_grid=np.array([1.4, 1.8, 2.2], dtype=float)
    nSeedsSens=sens_seeds
    auc_fix=np.zeros((len(burst_grid), nSeedsSens))
    auc_ahc_tgt=np.zeros((len(burst_grid), nSeedsSens))
    auc_ahc_eff=np.zeros((len(burst_grid), nSeedsSens))

    for k, bf in enumerate(burst_grid):
        lambda_jobs_k=np.full(T, base_arrival, dtype=float)
        lambda_jobs_k[t_burst_start-1:t_burst_end]=base_arrival*bf
        for s in range(nSeedsSens):
            rng_s=np.random.default_rng(5000+31*(s+1)+(k+1))
            arrTen_s=[None]*T; arrWork_s=[None]*T
            for tt in range(T):
                kArr=rng_s.poisson(lambda_jobs_k[tt])
                if kArr<=0: arrTen_s[tt]=np.empty(0,dtype=int); arrWork_s[tt]=np.empty(0,dtype=int); continue
                ten_s=np.empty(kArr,dtype=int); wk_s=np.empty(kArr,dtype=int)
                for kk in range(kArr):
                    if forceProb[tt]>0 and rng_s.random()<forceProb[tt]: j=shockTenant0
                    else:
                        u=rng_s.random(); j=int(np.searchsorted(cumPop,u,side="left"))
                        if j>=N: j=N-1
                    m=svcMean[j]; st=max(1,int(round(m*np.exp(svcJitter*rng_s.standard_normal()))))
                    ten_s[kk]=j; wk_s[kk]=st
                arrTen_s[tt]=ten_s; arrWork_s[tt]=wk_s

            for whichPol in [0, 1]:
                polName="FixedBetaSoftmax" if whichPol==0 else "AdaptiveHardCap++"
                qArrT_s=[[] for _ in range(N)]; qRem_s=[[] for _ in range(N)]
                head_s=np.zeros(N,dtype=int); qLen_s=np.zeros(N,dtype=int)
                ring_s=-np.ones(W,dtype=int); ring_idx_s=0; ring_fill_s=0
                win_counts_s=np.zeros(N,dtype=int); win_total_s=0
                serverTenant_s=-1; remWork_s=0; beta_s=float(beta_fixed)
                top1_ts=np.zeros(T); rhoeff_ts=np.zeros(T)

                for tt in range(T):
                    lam_t=lambda_series[tt]; rho_tgt=rho_target[tt]
                    beta_ref=np.interp(lam_t, lam_grid, beta_star)
                    ten_s2=arrTen_s[tt]; wk_s2=arrWork_s[tt]
                    if ten_s2.size>0:
                        for j, st in zip(ten_s2, wk_s2):
                            qLen_s[j]+=1; qArrT_s[j].append(tt); qRem_s[j].append(int(st))
                    cstep=int(capWork[tt])
                    for uu in range(cstep):
                        if remWork_s<=0:
                            active=np.nonzero(qLen_s>0)[0]
                            if active.size==0: serverTenant_s=-1; remWork_s=0; break
                            svec=score0.copy()
                            if scoreBoost[tt]!=0: svec[shockTenant0]+=scoreBoost[tt]
                            sa=svec[active]
                            b=beta_fixed if polName=="FixedBetaSoftmax" else beta_s
                            prob=softmax_prob(b, sa)
                            if polName=="AdaptiveHardCap++":
                                K_now=int(active.size)
                                rho_eff=max(rho_tgt, 1.0/max(1,K_now))
                                prob=cap_redistribute(prob, rho_eff)
                            r=rng_s.random(); cdf=np.cumsum(prob)
                            ii=int(np.searchsorted(cdf,r,side="left"))
                            if ii>=active.size: ii=0
                            chosen=int(active[ii])
                            curArrT_s=qArrT_s[chosen][head_s[chosen]]
                            remWork_s=qRem_s[chosen][head_s[chosen]]
                            head_s[chosen]+=1; qLen_s[chosen]-=1; serverTenant_s=chosen
                            if head_s[chosen]>50 and head_s[chosen]>len(qArrT_s[chosen])//2:
                                qArrT_s[chosen]=qArrT_s[chosen][head_s[chosen]:]
                                qRem_s[chosen]=qRem_s[chosen][head_s[chosen]:]; head_s[chosen]=0
                            if qLen_s[chosen]==0 and head_s[chosen]>=len(qArrT_s[chosen]):
                                qArrT_s[chosen]=[]; qRem_s[chosen]=[]; head_s[chosen]=0
                        if remWork_s>0:
                            remWork_s-=1
                            if ring_fill_s<W: ring_fill_s+=1
                            else:
                                old=ring_s[ring_idx_s]
                                if old!=-1: win_counts_s[old]-=1; win_total_s-=1
                            ring_s[ring_idx_s]=serverTenant_s
                            win_counts_s[serverTenant_s]+=1; win_total_s+=1
                            ring_idx_s+=1
                            if ring_idx_s>=W: ring_idx_s=0
                    top1_v=(win_counts_s.max()/win_total_s) if win_total_s>0 else 0.0
                    top1_ts[tt]=top1_v
                    Kt=int(np.count_nonzero(qLen_s>0)+(1 if remWork_s>0 else 0))
                    rho_eff_now=max(rho_tgt,1.0/max(1,Kt))
                    rhoeff_ts[tt]=rho_eff_now if polName=="AdaptiveHardCap++" else rho_tgt
                    if polName=="AdaptiveHardCap++" and win_total_s>0:
                        err=top1_v-rho_eff_now; beta_prop=beta_s-beta_gain*err   # Eq. 16: beta - eta*(y - rho_eff)
                        beta_prop+=beta_track*(beta_ref-beta_prop)
                        beta_prop=float(np.clip(beta_prop,beta_min,beta_max))
                        beta_s=beta_smooth*beta_s+(1-beta_smooth)*beta_prop
                post=slice(t_stable-1,T)
                top_post=top1_ts[post]; rho_post=rho_target[post]; rhoeff_post=rhoeff_ts[post]
                aucT=float(np.mean(np.maximum(0, top_post-rho_post)))
                aucE=float(np.mean(np.maximum(0, top_post-rhoeff_post)))
                if whichPol==0: auc_fix[k,s]=aucT
                else: auc_ahc_tgt[k,s]=aucT; auc_ahc_eff[k,s]=aucE
        print(f"  burst_factor={bf:.1f} done")

    def mean_ci_rows(M):
        mu=M.mean(axis=1); sd=M.std(axis=1,ddof=1); ci=1.96*sd/np.sqrt(M.shape[1])
        return mu, ci
    mu_fix, ci_fix = mean_ci_rows(auc_fix)
    mu_ahcT, ci_ahcT = mean_ci_rows(auc_ahc_tgt)
    mu_ahcE, ci_ahcE = mean_ci_rows(auc_ahc_eff)

    fig, ax=plt.subplots(figsize=(9.6, 5.4), dpi=150)
    ax.grid(True, alpha=0.25)
    ax.errorbar(burst_grid, mu_fix, yerr=ci_fix, fmt="o-", color=C["blue"],
                mfc="white", ms=7, lw=2.2, label="FixedBetaSoftmax: AUC_target (mean\u00b195%CI)")
    ax.errorbar(burst_grid, mu_ahcT, yerr=ci_ahcT, fmt="s--", color=C["purple"],
                mfc="white", ms=7, lw=2.2, label="AHC++: AUC_target (mean\u00b195%CI)")
    ax.errorbar(burst_grid, mu_ahcE, yerr=ci_ahcE, fmt="^:", color=C["black"],
                mfc="white", ms=7, lw=2.2, label="AHC++: AUC_eff (mean\u00b195%CI)")
    ax.set_xlabel("burst factor (arrival spike multiplier)")
    ax.set_ylabel("AUC excess (stable post window)")
    ax.legend(loc="upper left", frameon=True)
    save_fig(fig, "Fig11_sensitivity_burst",
             title="Figure 11. Cumulative Dominance Constraint Exceedance (AUC) vs. Burst Factor")
    plt.show(); plt.close(fig)


# =============================================================
#  FIGURE 12: Computational Scalability
#  (Paper Section 3.2.3: "Computational scalability of a single
#   allocation step (N = 10² to 10⁴).")
# =============================================================
def make_fig12():
    # NOTE: This is a wall-clock micro-benchmark. The qualitative result
    # (near-linear scaling, far below the O(N^2) pairwise reference) is robust,
    # but the exact runtime ratios depend on hardware, OS, and BLAS/NumPy build.
    # The reported "~5.5x for 100x agents" in the paper is environment-dependent;
    # expect a single-digit multiplier rather than that precise number on your machine.
    print("\n=== Figure 12: Computational Scalability ===")
    N_bench = [100, 300, 500, 1000, 2500, 5000, 7500, 10000]
    rho_cap = 0.22; beta0 = 2.0; repeats = 50; warmups = 5
    times_med = []; times_p25 = []; times_p75 = []
    for n in N_bench:
        logits_b = -np.log(np.arange(1, n+1, dtype=float))
        for _ in range(warmups):
            _ = cap_redistribute(softmax_prob(beta0, logits_b), rho_cap)
        runs = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = cap_redistribute(softmax_prob(beta0, logits_b), rho_cap)
            runs.append(time.perf_counter() - t0)
        runs = np.array(runs)
        times_med.append(np.median(runs))
        times_p25.append(np.percentile(runs, 25))
        times_p75.append(np.percentile(runs, 75))

    N_pairwise = [100, 300, 500, 1000, 1500]
    pair_med = []
    for n in N_pairwise:
        x = np.random.rand(n).astype(np.float64)
        runs = []
        for _ in range(5):
            t0 = time.perf_counter()
            acc = 0.0; block = 300
            for i in range(0, n, block):
                xi = x[i:i+block]
                acc += np.abs(xi[:, None] - x[None, :]).sum()
            runs.append(time.perf_counter() - t0)
        pair_med.append(float(np.median(runs)))

    anchor_n = N_pairwise[0]; anchor_t = pair_med[0]
    quad_ref = [(n**2) * (anchor_t / (anchor_n**2)) for n in N_bench]

    xlog = np.log(np.array(N_bench, dtype=float))
    ylog = np.log(np.array(times_med, dtype=float))
    slope = np.polyfit(xlog, ylog, 1)[0]
    print(f"  Empirical scaling slope (log-log): {slope:.3f}")
    print(f"  Median time at N={N_bench[-1]}: {times_med[-1]:.6e} seconds")

    N_arr = np.array(N_bench, dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.grid(True, which="both", alpha=0.2)
    ax.plot(N_arr, np.array(times_med), marker="o", linewidth=2.5,
            label="Proposed (Boltzmann-Softmax + Hard-Cap)")
    ax.fill_between(N_arr, np.array(times_p25), np.array(times_p75),
                    alpha=0.18, label="IQR (25-75%)")
    ax.plot(N_arr, np.array(quad_ref), linestyle="--", linewidth=2.0,
            label="O(N^2) reference slope")
    ax.plot(np.array(N_pairwise, dtype=float), np.array(pair_med),
            marker="s", linewidth=2.0, label="Pairwise dummy")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of Agents (N)")
    ax.set_ylabel("Execution Time (seconds)")
    ax.legend(loc="upper left", frameon=True)
    save_fig(fig, "Fig12_computational_scalability",
             title="Figure 12. Computational Scalability of a Single Allocation Step")
    plt.show(); plt.close(fig)


# =============================================================
#  MAIN: Execute all figures and tables in paper order
#  Static  → Figures 2–6
#  Dynamic → Figures 7–11 + Table 5
#  Scalability → Figure 12
# =============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Reproduce figures and Table 5 for Computable Fairness (Park & Kim, 2026).")
    parser.add_argument("--static-only", action="store_true",
                        help="Run only the static-model figures (2-6) and scalability (12).")
    parser.add_argument("--dynamic-only", action="store_true",
                        help="Run only the dynamic-model part (Figures 7-11 and Table 5).")
    parser.add_argument("--quick-test", action="store_true",
                        help="Fast smoke test with reduced N and seed counts. "
                             "NOT a reproduction of the paper numbers; for checking the code runs.")
    parser.add_argument("--show", action="store_true",
                        help="Call plt.show() to display figures interactively "
                             "(off by default; figures are always saved to disk).")
    args = parser.parse_args()

    # By default plots are saved, not shown. Suppress plt.show() unless --show is given,
    # which avoids warnings/blocking in headless environments.
    if not args.show:
        plt.show = lambda *a, **k: None

    run_static = not args.dynamic_only
    run_dynamic = not args.static_only

    # Quick-test parameters: shrink the problem so the code path is exercised quickly.
    # These do NOT reproduce the paper; they only verify the code runs end to end.
    if args.quick_test:
        dyn_N, dyn_T, dyn_seeds, dyn_sens = 200, 4200, 2, 2
        print("!" * 60)
        print("  QUICK-TEST MODE: reduced N/seeds. Results are NOT the paper's")
        print("  reported values; use a full run (no flags) for reproduction.")
        print("!" * 60)
    else:
        dyn_N, dyn_T, dyn_seeds, dyn_sens = 1000, 4200, 10, 8

    print("=" * 60)
    print("  Computable Fairness:")
    print("  Boltzmann–Softmax Control for AI Resource Allocation")
    print("  Paper: Park & Kim, arXiv:2605.22827 (2026)")
    print("  Figures 2–12, Table 5")
    print("=" * 60)

    if run_static:
        # ------ Static model: shared loss landscape data ------
        beta_grid = np.linspace(0.05, 10.0, 400)
        Eff = np.zeros_like(beta_grid); Eq = np.zeros_like(beta_grid)
        for i, b in enumerate(beta_grid):
            P = softmax_prob(b, LOGITS)
            Eff[i] = np.sum(P * NORM_LOGITS)
            H = -np.sum(P * np.log(P + 1e-12)); Eq[i] = H / np.log(N_POP)
        L_eff = 1 - Eff; L_ineq = 1 - Eq
        L_eff_n = norm01(L_eff); L_ineq_n = norm01(L_ineq)

        # ------ Static Model Figures ------
        make_fig02()                                # Figure 2: Boltzmann allocation
        make_fig03(beta_grid, L_eff_n, L_ineq_n)    # Figure 3: Loss landscape
        make_fig04()                                # Figure 4: Heatmap + corridor
        make_fig05()                                # Figure 5: Pareto + trajectories
        make_fig06()                                # Figure 6: Efficiency–fairness frontier

    if run_dynamic:
        # ------ Dynamic Model Figures + Table 5 ------
        run_dynamic_and_plot(N=dyn_N, T=dyn_T, nSeeds=dyn_seeds, sens_seeds=dyn_sens)

    if run_static:
        # ------ Scalability ------
        make_fig12()                                # Figure 12: Scalability

    print("\n" + "=" * 60)
    print(f"  All done. Output: {OUT_DIR.resolve()}")
    print("=" * 60)
