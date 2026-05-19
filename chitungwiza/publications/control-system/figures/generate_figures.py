"""
Figure panels for:
  "Phase-Synchronous Distributed Regulation: A Unified Theory of
   Oscillator-Referenced Cell-Partition Control"

6 panels × 4 charts.  Each panel: white background, minimal text,
at least one 3D subplot per panel.  All charts are data-driven.
"""

import math
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings("ignore")

rng = np.random.default_rng(20260520)

# ── colour palette ─────────────────────────────────────────────────────────────
BLUE   = "#2166AC"
RED    = "#D6604D"
GREEN  = "#4DAC26"
AMBER  = "#E08A1E"
PURPLE = "#762A83"
TEAL   = "#1B9E77"
PAL    = [BLUE, RED, GREEN, AMBER, PURPLE, TEAL]

FIGSIZE = (22, 5.0)
DPI = 150

def make_panel(charts_fn, path):
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    gs = gridspec.GridSpec(1, 4, figure=fig, left=0.06, right=0.97,
                           top=0.88, bottom=0.18, wspace=0.35)
    charts_fn(fig, gs)
    fig.savefig(path, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def style_ax(ax, xlabel="", ylabel="", title="", letter=""):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#AAAAAA")
    ax.tick_params(colors="#555555", labelsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color="#333333")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color="#333333")
    if title:
        ax.set_title(title, fontsize=8.5, color="#222222", pad=3)
    if letter:
        ax.text(-0.12, 1.06, f"({letter})", transform=ax.transAxes,
                fontsize=10, fontweight="bold", color="#111111", va="top")


def style_ax3d(ax, xlabel="", ylabel="", zlabel="", title="", letter=""):
    ax.set_facecolor("white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#DDDDDD")
    ax.yaxis.pane.set_edgecolor("#DDDDDD")
    ax.zaxis.pane.set_edgecolor("#DDDDDD")
    ax.tick_params(labelsize=7, colors="#555555")
    if xlabel: ax.set_xlabel(xlabel, fontsize=7.5, color="#333333", labelpad=2)
    if ylabel: ax.set_ylabel(ylabel, fontsize=7.5, color="#333333", labelpad=2)
    if zlabel: ax.set_zlabel(zlabel, fontsize=7.5, color="#333333", labelpad=2)
    if title:  ax.set_title(title, fontsize=8.5, color="#222222", pad=3)
    if letter:
        ax.text2D(-0.12, 1.06, f"({letter})", transform=ax.transAxes,
                  fontsize=10, fontweight="bold", color="#111111", va="top")


# ── helper functions ───────────────────────────────────────────────────────────

def cauchy_omega(n, gamma):
    raw = rng.standard_cauchy(n) * gamma
    return np.clip(raw, -30 * gamma, 30 * gamma)


def kuramoto_run(n, K, gamma, T=250, dt=0.05):
    omega = cauchy_omega(n, gamma)
    phi = rng.uniform(0, 2 * math.pi, n)
    R_trace = []
    steps = int(T / dt)
    for k in range(steps):
        diff = phi[None, :] - phi[:, None]
        dphi = omega + (K / n) * np.sin(diff).sum(axis=1)
        phi = (phi + dt * dphi) % (2 * math.pi)
        if k % 20 == 0:
            R_trace.append(abs(np.mean(np.exp(1j * phi))))
    return np.array(R_trace), abs(np.mean(np.exp(1j * phi)))


def cell_index(S, n_cells=5, cell_width=20.0):
    return min(int(S / cell_width), n_cells - 1)


def simulate_cell_controller(S0, S_star=50.0, n_cells=5, K_ctrl=0.5,
                              alpha=0.3, h_step=0.1, n_steps=150):
    cell_width = 100.0 / n_cells
    centroids = [(i + 0.5) * cell_width for i in range(n_cells)]
    u_vals = [K_ctrl * (S_star - c) for c in centroids]
    S = S0
    traj = [S]
    V_traj = [(S - S_star) ** 2]
    for _ in range(n_steps):
        i = cell_index(np.clip(S, 0, 99.999), n_cells, cell_width)
        u = u_vals[i]
        dS = -alpha * (S - S_star) + u
        S = np.clip(S + h_step * dS, 0.0, 100.0)
        traj.append(S)
        V_traj.append((S - S_star) ** 2)
    return np.array(traj), np.array(V_traj)


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 1: Phase-Domain Transfer Functions
# ═══════════════════════════════════════════════════════════════════════════════
def panel_01(fig, gs):
    omega_osc = 10.0
    tau_list = [0.5, 1.0, 2.0, 5.0]
    K_p = 1.0

    # ── A: Bode magnitude |G_p(jν)| vs ν for multiple τ ──
    ax = fig.add_subplot(gs[0, 0])
    nu_vals = np.logspace(-3, 1, 400)
    for i, tau in enumerate(tau_list):
        G_mag = K_p / np.sqrt((tau * omega_osc * nu_vals) ** 2 + 1)
        ax.loglog(nu_vals, G_mag, color=PAL[i], lw=1.8,
                  label=f"τ = {tau}")
    ax.legend(fontsize=7, frameon=False)
    style_ax(ax, xlabel="ν  (cycles/rad)", ylabel="|G_p(jν)|",
             title="Phase-Domain Bode Magnitude", letter="A")

    # ── B: Pole migration: s* vs ν* = s*/ω_osc ──
    ax2 = fig.add_subplot(gs[0, 1])
    tau_scatter = np.linspace(0.2, 6.0, 25)
    s_star = -1.0 / tau_scatter
    nu_star = s_star / omega_osc
    sc = ax2.scatter(s_star, nu_star, c=tau_scatter, cmap="viridis",
                     s=40, zorder=3)
    x_line = np.linspace(-5.2, 0, 100)
    ax2.plot(x_line, x_line / omega_osc, color="#AAAAAA", lw=1, ls="--")
    plt.colorbar(sc, ax=ax2, label="τ", pad=0.02).ax.tick_params(labelsize=7)
    style_ax(ax2, xlabel="s*  (rad/s)", ylabel="ν*  (cycles/rad)",
             title="Pole: s* → ν* = s*/ω_osc", letter="B")

    # ── C (3D): |G_p(jν)| surface over (ν, τ·ω_osc) ──
    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    nu_g = np.linspace(0.01, 3.0, 50)
    tau_norm_g = np.linspace(0.5, 10.0, 50)
    NU, TN = np.meshgrid(nu_g, tau_norm_g)
    G_surf = K_p / np.sqrt((TN * NU) ** 2 + 1)
    ax3.plot_surface(NU, TN, G_surf, cmap="Blues", alpha=0.85,
                     linewidth=0, antialiased=True)
    style_ax3d(ax3, xlabel="ν", ylabel="τω_osc", zlabel="|G|",
               title="Bode Surface", letter="C")
    ax3.view_init(elev=28, azim=-50)

    # ── D: Step response: physical vs S-entropy ──
    ax4 = fig.add_subplot(gs[0, 3])
    tau_d = 1.0
    t_vals = np.linspace(0, 6, 300)
    for i, scale in enumerate([60.0, 80.0, 40.0]):
        # S-entropy step response: plant output normalised to [0,100]
        y_phys = K_p * (1 - np.exp(-t_vals / tau_d)) * scale
        ax4.plot(t_vals, y_phys, color=PAL[i], lw=1.6,
                 label=f"range {scale:.0f}")
    ax4.axhline(60.0, color=PAL[0], ls=":", lw=0.8)
    ax4.axhline(80.0, color=PAL[1], ls=":", lw=0.8)
    ax4.axhline(40.0, color=PAL[2], ls=":", lw=0.8)
    ax4.set_ylim(0, 110)
    ax4.legend(fontsize=7, frameon=False)
    style_ax(ax4, xlabel="t  (s)", ylabel="S  ∈ [0,100]",
             title="S-Entropy Normalised Step Response", letter="D")


make_panel(panel_01, "panel_01.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 2: Cell Partition and Describing Function
# ═══════════════════════════════════════════════════════════════════════════════
def panel_02(fig, gs):
    S_star = np.array([50.0, 50.0])
    n_cells = 4
    cell_w = 100.0 / n_cells

    # ── A: 2D cell partition of [0,100]² with action arrows ──
    ax = fig.add_subplot(gs[0, 0])
    for i in range(n_cells):
        for j in range(n_cells):
            cx = (i + 0.5) * cell_w
            cy = (j + 0.5) * cell_w
            rect = plt.Rectangle((i * cell_w, j * cell_w), cell_w, cell_w,
                                  linewidth=0.8, edgecolor="#666666",
                                  facecolor=("#EAF4FF" if (i + j) % 2 == 0
                                             else "#F9F3FF"))
            ax.add_patch(rect)
            u = 0.3 * (S_star[0] - cx)
            v = 0.3 * (S_star[1] - cy)
            ax.annotate("", xy=(cx + u, cy + v), xytext=(cx, cy),
                        arrowprops=dict(arrowstyle="->", color=BLUE,
                                        lw=1.0, mutation_scale=8))
    ax.plot(*S_star, "*", ms=12, color=RED, zorder=5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    style_ax(ax, xlabel="S₁", ylabel="S₂",
             title="Cell Partition: action arrows → S*", letter="A")

    # ── B: Describing function N(A) and -1/N(A) locus ──
    ax2 = fig.add_subplot(gs[0, 1])
    A_vals = np.linspace(0.2, 5.0, 200)
    u_max = 1.0
    N_A = 4 * u_max / (math.pi * A_vals)
    neg_inv_N = -1.0 / N_A
    ax2.plot(A_vals, N_A, color=BLUE, lw=2.0, label="N(A)")
    ax2.plot(A_vals, neg_inv_N, color=RED, lw=2.0, ls="--", label="−1/N(A)")
    ax2.axhline(0, color="#AAAAAA", lw=0.8)
    ax2.legend(fontsize=8, frameon=False)
    style_ax(ax2, xlabel="Input amplitude A", ylabel="Gain",
             title="Describing Function N(A) = 4u_max/(πA)", letter="B")

    # ── C (3D): Nyquist magnitude surface |G(jν)| ──
    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    nu_re = np.linspace(-2.0, 2.0, 60)
    nu_im = np.linspace(0.01, 4.0, 60)
    RE, IM = np.meshgrid(nu_re, nu_im)
    tau_n = 1.0
    omega_n = 10.0
    G_complex = 1.0 / (tau_n * omega_n * (RE + 1j * IM) + 1)
    G_mag3d = np.abs(G_complex)
    ax3.plot_surface(RE, IM, G_mag3d, cmap="Purples", alpha=0.85,
                     linewidth=0, antialiased=True)
    style_ax3d(ax3, xlabel="Re(ν)", ylabel="Im(ν)", zlabel="|G|",
               title="Nyquist Magnitude Surface", letter="C")
    ax3.view_init(elev=30, azim=-40)

    # ── D: Cell centroid proportional law ──
    ax4 = fig.add_subplot(gs[0, 3])
    n_test = 5
    cell_w4 = 100.0 / n_test
    centroids = np.array([(i + 0.5) * cell_w4 for i in range(n_test)])
    errors = 50.0 - centroids
    u_assigned = 0.5 * errors
    ax4.bar(centroids, u_assigned, width=cell_w4 * 0.7,
            color=PAL[:n_test], alpha=0.8, zorder=3)
    ax4.axhline(0, color="#AAAAAA", lw=0.8)
    ax4.set_xticks(centroids)
    ax4.set_xticklabels([f"{c:.0f}" for c in centroids], fontsize=7)
    style_ax(ax4, xlabel="Cell centroid S̄_i", ylabel="u_i = K(S*−S̄_i)",
             title="Nearest-Centroid Control Law", letter="D")


make_panel(panel_02, "panel_02.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 3: Piecewise Lyapunov Stability
# ═══════════════════════════════════════════════════════════════════════════════
def panel_03(fig, gs):
    S_inits = [10.0, 25.0, 75.0, 90.0]
    trajs, V_trajs = [], []
    for S0 in S_inits:
        t, v = simulate_cell_controller(S0)
        trajs.append(t)
        V_trajs.append(v)

    k_arr = np.arange(151)

    # ── A: V(S(k)) vs k ──
    ax = fig.add_subplot(gs[0, 0])
    for i, (v, S0) in enumerate(zip(V_trajs, S_inits)):
        ax.semilogy(k_arr, v + 1e-8, color=PAL[i], lw=1.8,
                    label=f"S₀={S0:.0f}")
    ax.legend(fontsize=7, frameon=False)
    style_ax(ax, xlabel="Iteration k", ylabel="V(S(k)) = ‖S−S*‖²  (log)",
             title="Lyapunov Function Descent", letter="A")

    # ── B: Phase portrait in [0,100] ──
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (traj, S0) in enumerate(zip(trajs, S_inits)):
        ax2.plot(k_arr, traj, color=PAL[i], lw=1.6, label=f"S₀={S0:.0f}")
    ax2.axhline(50.0, color=RED, ls="--", lw=1.2, label="S*")
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=7, frameon=False)
    style_ax(ax2, xlabel="Iteration k", ylabel="S  ∈ [0,100]",
             title="State Convergence to Setpoint", letter="B")

    # ── C (3D): Lyapunov surface V(S₁, S₂) ──
    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    s1 = np.linspace(0, 100, 60)
    s2 = np.linspace(0, 100, 60)
    S1, S2 = np.meshgrid(s1, s2)
    V_surf = (S1 - 50) ** 2 + (S2 - 50) ** 2
    ax3.plot_surface(S1, S2, V_surf, cmap="YlOrRd", alpha=0.88,
                     linewidth=0, antialiased=True)
    ax3.scatter([50], [50], [0], color=RED, s=50, zorder=5)
    style_ax3d(ax3, xlabel="S₁", ylabel="S₂", zlabel="V",
               title="Quadratic Lyapunov Surface", letter="C")
    ax3.view_init(elev=35, azim=-50)

    # ── D: Descent rate dV/dθ per cell ──
    ax4 = fig.add_subplot(gs[0, 3])
    n_cells_d = 5
    cell_w_d = 100.0 / n_cells_d
    S_star_d = 50.0
    alpha_d, K_d = 0.3, 0.5
    descent_rates = []
    for i in range(n_cells_d):
        S_mid = (i + 0.5) * cell_w_d
        u_i = K_d * (S_star_d - (i + 0.5) * cell_w_d)
        dS = -alpha_d * (S_mid - S_star_d) + u_i
        dV = 2 * (S_mid - S_star_d) * dS
        descent_rates.append(dV)
    cell_centers = [(i + 0.5) * cell_w_d for i in range(n_cells_d)]
    colors = [RED if d >= 0 else GREEN for d in descent_rates]
    ax4.bar(cell_centers, descent_rates, width=cell_w_d * 0.7,
            color=colors, alpha=0.85, zorder=3)
    ax4.axhline(0, color="#AAAAAA", lw=0.8)
    ax4.set_xticks(cell_centers)
    ax4.set_xticklabels([f"{c:.0f}" for c in cell_centers], fontsize=7)
    style_ax(ax4, xlabel="Cell centroid S̄_i", ylabel="dV/dθ|_{cell i}",
             title="Descent Rate per Cell", letter="D")


make_panel(panel_03, "panel_03.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 4: CUSUM–ΔP Duality
# ═══════════════════════════════════════════════════════════════════════════════
def panel_04(fig, gs):
    sigma_dp = 0.05
    mu_shift = 0.04
    n_samples = 500
    k_s = sigma_dp
    h_lim = 0.3

    deltaP = np.concatenate([
        rng.normal(0, sigma_dp, 300),
        rng.normal(mu_shift, sigma_dp, 200),
    ])
    k_idx = np.arange(n_samples)

    C_plus = np.zeros(n_samples)
    C_minus = np.zeros(n_samples)
    for k in range(1, n_samples):
        C_plus[k] = max(0.0, C_plus[k - 1] + deltaP[k] - k_s)
        C_minus[k] = min(0.0, C_minus[k - 1] + deltaP[k] + k_s)
    cumsum = np.cumsum(deltaP)

    # ── A: ΔP(k) time series ──
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(k_idx, deltaP, color=BLUE, lw=0.8, alpha=0.8)
    ax.axvline(300, color=RED, ls="--", lw=1.2)
    ax.axhline(0, color="#AAAAAA", lw=0.6)
    style_ax(ax, xlabel="Tick k", ylabel="ΔP(k)",
             title="Timing Deviation Sequence", letter="A")

    # ── B: CUSUM with limits ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(k_idx, C_plus, color=BLUE, lw=1.4, label="C⁺")
    ax2.plot(k_idx, C_minus, color=RED, lw=1.4, label="C⁻")
    ax2.axhline(h_lim, color=GREEN, ls="--", lw=1.2, label=f"h={h_lim}")
    ax2.axhline(-h_lim, color=GREEN, ls="--", lw=1.2)
    ax2.axvline(300, color="#AAAAAA", ls=":", lw=1.0)
    ax2.legend(fontsize=7, frameon=False)
    style_ax(ax2, xlabel="Tick k", ylabel="CUSUM statistic",
             title="CUSUM with Cell Limit ±h", letter="B")

    # ── C (3D): ARL₀ surface over (h, δ) ──
    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    delta_g = np.linspace(0.2, 3.0, 40)
    h_g = np.linspace(0.1, 2.0, 40)
    DG, HG = np.meshgrid(delta_g, h_g)
    ARL0_surf = np.exp(2 * DG * HG) / (DG ** 2)
    ARL0_surf = np.clip(ARL0_surf, 0, 5000)
    ax3.plot_surface(DG, HG, ARL0_surf, cmap="Blues", alpha=0.85,
                     linewidth=0, antialiased=True)
    style_ax3d(ax3, xlabel="δ", ylabel="h", zlabel="ARL₀",
               title="ARL₀ = exp(2δh)/δ²", letter="C")
    ax3.view_init(elev=30, azim=-55)

    # ── D: ARL₁ vs shift magnitude for different h ──
    ax4 = fig.add_subplot(gs[0, 3])
    shift_vals = np.linspace(0.01, 0.5, 200)
    h_list = [0.2, 0.5, 1.0]
    for i, h_val in enumerate(h_list):
        ARL1 = (sigma_dp / (h_val * shift_vals)) ** 2
        ax4.semilogy(shift_vals, ARL1, color=PAL[i], lw=1.8,
                     label=f"h={h_val}")
    ax4.legend(fontsize=7, frameon=False)
    style_ax(ax4, xlabel="|μ₁−μ₀|", ylabel="ARL₁  (log)",
             title="Detection Speed vs Shift Size", letter="D")


make_panel(panel_04, "panel_04.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 5: Banach Contraction and Multi-Scale Hierarchy
# ═══════════════════════════════════════════════════════════════════════════════
def panel_05(fig, gs):
    alpha_b = 0.3
    L_F = 0.5
    h_b = 0.3
    rho_theory = math.sqrt(1 - 2 * h_b * alpha_b + h_b ** 2 * L_F ** 2)

    # ── A: ||S(k) - S*|| vs k showing geometric decay ──
    ax = fig.add_subplot(gs[0, 0])
    n_steps = 80
    S_star_b = 50.0
    init_errors = [40.0, 30.0, 20.0, 10.0]
    k_arr = np.arange(n_steps + 1)
    for i, err0 in enumerate(init_errors):
        S0 = S_star_b + err0
        traj, _ = simulate_cell_controller(S0, n_steps=n_steps)
        errors = np.abs(traj - S_star_b)
        ax.semilogy(k_arr, errors + 1e-8, color=PAL[i], lw=1.8,
                    label=f"|S₀−S*|={err0}")
        # Theoretical bound
        bound = err0 * rho_theory ** k_arr
        ax.semilogy(k_arr, bound + 1e-8, color=PAL[i], lw=0.9, ls="--",
                    alpha=0.6)
    ax.legend(fontsize=7, frameon=False)
    style_ax(ax, xlabel="Iteration k", ylabel="‖S(k)−S*‖  (log)",
             title="Geometric Convergence: ρᵏ bound (dashed)", letter="A")

    # ── B: Measured ρ vs theoretical ρ(h) for different h ──
    ax2 = fig.add_subplot(gs[0, 1])
    h_range = np.linspace(0.02, 1.4, 50)
    rho_theory_arr = np.sqrt(np.clip(
        1 - 2 * h_range * alpha_b + h_range ** 2 * L_F ** 2, 0, 1))
    rho_empirical_arr = []
    for h_val in h_range:
        S1, S2 = 40.0, 42.0
        ratios_local = []
        for _ in range(40):
            d0 = abs(S1 - S2)
            i1 = cell_index(np.clip(S1, 0, 99.999))
            u1 = 0.5 * (50.0 - (i1 + 0.5) * 20.0)
            i2 = cell_index(np.clip(S2, 0, 99.999))
            u2 = 0.5 * (50.0 - (i2 + 0.5) * 20.0)
            S1 = np.clip(S1 + h_val * (-alpha_b * (S1 - 50) + u1), 0, 100)
            S2 = np.clip(S2 + h_val * (-alpha_b * (S2 - 50) + u2), 0, 100)
            d1 = abs(S1 - S2)
            if d0 > 1e-10:
                ratios_local.append(d1 / d0)
        rho_empirical_arr.append(float(np.median(ratios_local)) if ratios_local else 1.0)
    ax2.plot(h_range, rho_theory_arr, color=BLUE, lw=2.0, label="ρ(h) theory")
    ax2.plot(h_range, rho_empirical_arr, color=RED, lw=1.5, ls="--",
             label="ρ empirical")
    ax2.axhline(1.0, color="#AAAAAA", lw=0.8, ls=":")
    ax2.set_ylim(0, 1.2)
    ax2.legend(fontsize=7, frameon=False)
    style_ax(ax2, xlabel="Step size h",
             ylabel="Contraction factor ρ",
             title="Theory vs Empirical ρ(h)", letter="B")

    # ── C (3D): ρ² surface over (h, α) ──
    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    h_g = np.linspace(0.01, 1.5, 50)
    alpha_g = np.linspace(0.05, 1.0, 50)
    HG, AG = np.meshgrid(h_g, alpha_g)
    rho2_surf = np.clip(1 - 2 * HG * AG + HG ** 2 * L_F ** 2, 0, 1.2)
    ax3.plot_surface(HG, AG, rho2_surf, cmap="RdYlGn_r", alpha=0.85,
                     linewidth=0, antialiased=True)
    stable_h = 2 * alpha_g / L_F ** 2
    ax3.plot(stable_h, alpha_g, np.ones_like(alpha_g),
             color=RED, lw=1.5, zorder=5)
    style_ax3d(ax3, xlabel="h", ylabel="α", zlabel="ρ²",
               title="ρ² = 1−2hα+h²L_F²  (red = stability boundary)", letter="C")
    ax3.view_init(elev=30, azim=-40)

    # ── D: Multi-scale frequency hierarchy ──
    ax4 = fig.add_subplot(gs[0, 3])
    freq_levels = [100.0, 10.0, 1.0]
    bandwidths  = [50.0,   5.0,  0.5]
    level_labels = ["L1 (fast)", "L2 (medium)", "L3 (slow)"]
    y_pos = [3, 2, 1]
    for i, (f, bw, lbl, yp) in enumerate(
            zip(freq_levels, bandwidths, level_labels, y_pos)):
        ax4.barh(yp, f, height=0.4, color=PAL[i], alpha=0.85, label=lbl)
        ax4.barh(yp - 0.25, bw, height=0.2, color=PAL[i], alpha=0.45)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(level_labels, fontsize=8)
    ax4.set_xscale("log")
    ax4.legend(fontsize=7, frameon=False)
    style_ax(ax4, xlabel="Frequency / bandwidth  (Hz, log)",
             ylabel="",
             title="Multi-Scale Hierarchy: f₁ >> f₂ >> f₃", letter="D")


make_panel(panel_05, "panel_05.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 6: Information-Optimal Sensing and Security
# ═══════════════════════════════════════════════════════════════════════════════
def panel_06(fig, gs):
    n_sensors = 20
    Sigma = 10.0
    beta = rng.uniform(0.5, 8.5, n_sensors)
    c_cost = rng.uniform(0.5, 3.5, n_sensors)
    C_budget = 12.0
    v = np.log(Sigma / (Sigma - beta))

    # Greedy selection
    order = np.argsort(-v / c_cost)
    remaining = C_budget
    selected = np.zeros(n_sensors, dtype=bool)
    for idx in order:
        if c_cost[idx] <= remaining:
            selected[idx] = True
            remaining -= c_cost[idx]

    # ── A: Sensor knapsack scatter ──
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(c_cost[selected], v[selected], color=BLUE, s=60, zorder=4,
               label="Admitted")
    ax.scatter(c_cost[~selected], v[~selected], color=RED, s=40, marker="x",
               lw=1.5, zorder=4, label="Rejected")
    ax.legend(fontsize=7, frameon=False)
    style_ax(ax, xlabel="Cost c_i", ylabel="Value v_i = log(Σ/(Σ−β_i))",
             title="Knapsack Sensor Selection", letter="A")

    # ── B: Catalytic composition κ(i∘j) ──
    ax2 = fig.add_subplot(gs[0, 1])
    kappa_single = beta / Sigma
    kappa_pairs = []
    sum_pairs = []
    for i in range(n_sensors):
        for j in range(i + 1, n_sensors):
            ki, kj = kappa_single[i], kappa_single[j]
            kij = 1 - (1 - ki) * (1 - kj)
            kappa_pairs.append(kij)
            sum_pairs.append(ki + kj)
    kappa_pairs = np.array(kappa_pairs)
    sum_pairs = np.array(sum_pairs)
    ax2.scatter(sum_pairs, kappa_pairs, color=TEAL, s=12, alpha=0.5, zorder=3)
    diag = np.linspace(0, 2.0, 100)
    ax2.plot(diag, diag, color="#AAAAAA", lw=1.0, ls="--", label="κ_i+κ_j")
    ax2.set_xlim(0, 2.0)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=7, frameon=False)
    style_ax(ax2, xlabel="κ_i + κ_j", ylabel="κ(i∘j)",
             title="Catalytic Composition: submodularity κ(i∘j)≤κ_i+κ_j", letter="B")

    # ── C (3D): Total information gain over (n_selected, budget) ──
    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    budget_g = np.linspace(1.0, 20.0, 30)
    n_sel_g = np.arange(1, 15)
    BG, NG = np.meshgrid(budget_g, n_sel_g.astype(float))
    # Approximate: gain ~ n_sel * log(budget/n_sel + 1)
    gain_surf = NG * np.log(BG / NG + 1)
    gain_surf = np.clip(gain_surf, 0, None)
    ax3.plot_surface(BG, NG, gain_surf, cmap="Greens", alpha=0.85,
                     linewidth=0, antialiased=True)
    style_ax3d(ax3, xlabel="Budget C", ylabel="n selected",
               zlabel="Info gain",
               title="Information Gain vs Budget & Selection", letter="C")
    ax3.view_init(elev=30, azim=-50)

    # ── D: Delay attack detection at cell boundary ──
    ax4 = fig.add_subplot(gs[0, 3])
    k_vals = np.arange(200)
    sigma_noise = 0.02
    h_sec = 0.25  # cell half-width / detection threshold
    in_ctrl = rng.normal(0, sigma_noise, 120)
    delay_ramp = np.linspace(0, 0.5, 80)
    deltaP_attack = np.concatenate([in_ctrl, delay_ramp])
    cumsum_attack = np.cumsum(deltaP_attack)
    ax4.plot(k_vals, deltaP_attack, color=BLUE, lw=0.9, alpha=0.7,
             label="ΔP(k)")
    ax4.plot(k_vals, cumsum_attack, color=PURPLE, lw=1.6, label="Σ ΔP")
    ax4.axhline(h_sec, color=RED, ls="--", lw=1.4, label=f"±h={h_sec}")
    ax4.axhline(-h_sec, color=RED, ls="--", lw=1.4)
    detect_k = np.where(np.abs(cumsum_attack) > h_sec)[0]
    if len(detect_k) > 0:
        ax4.axvline(detect_k[0], color=GREEN, lw=1.5, ls=":")
        ax4.scatter([detect_k[0]], [cumsum_attack[detect_k[0]]],
                    color=GREEN, s=50, zorder=5)
    ax4.legend(fontsize=7, frameon=False)
    style_ax(ax4, xlabel="Tick k", ylabel="ΔP / Σ ΔP",
             title="Delay Attack Detection at Cell Boundary", letter="D")


make_panel(panel_06, "panel_06.png")

print("\nAll 6 panels generated.")
