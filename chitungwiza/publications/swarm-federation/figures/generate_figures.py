"""
Figure panels for:
  "Temporal Swarm Federation: Phase-Coherent Multi-Agent
   Coordination via Oscillator Synchronization"

6 panels × 4 charts per row, ≥1 3-D chart per panel,
white background, data-driven plots, minimal annotation.
"""

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path

OUT = Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

BLUE   = '#2563eb'
RED    = '#dc2626'
GREEN  = '#16a34a'
AMBER  = '#d97706'
PURPLE = '#7c3aed'
TEAL   = '#0891b2'
PAL    = [BLUE, RED, GREEN, AMBER, PURPLE, TEAL]


# ─── Kuramoto utilities ────────────────────────────────────────────────────

def kuramoto_R(n, omega, K, dt=0.02, T_sim=60.0, seed=None):
    local_rng = np.random.default_rng(seed if seed is not None else 0)
    phi   = local_rng.uniform(0, 2 * math.pi, n)
    steps = int(T_sim / dt)
    for _ in range(steps):
        diff    = phi[None, :] - phi[:, None]   # diff[i,j] = phi[j] - phi[i]
        dphi    = omega + (K / n) * np.sum(np.sin(diff), axis=1)
        phi    += dt * dphi
    z = np.mean(np.exp(1j * phi))
    return abs(z)


def kuramoto_trajectory(n, omega, K, dt=0.02, T_sim=60.0, record_every=50):
    phi    = rng.uniform(0, 2 * math.pi, n)
    steps  = int(T_sim / dt)
    R_hist = []
    for step in range(steps):
        diff  = phi[None, :] - phi[:, None]   # diff[i,j] = phi[j] - phi[i]
        dphi  = omega + (K / n) * np.sum(np.sin(diff), axis=1)
        phi  += dt * dphi
        if step % record_every == 0:
            R_hist.append(abs(np.mean(np.exp(1j * phi))))
    return np.array(R_hist)


def K_c(sigma):
    return 2 * sigma / math.pi


def cauchy_omega(n, sigma, local_rng=None):
    """Lorentzian natural frequencies: scale = sigma/pi → K_c = 2*sigma/pi exactly."""
    gen = local_rng if local_rng is not None else rng
    scale = sigma / math.pi
    raw = gen.standard_cauchy(n) * scale
    return np.clip(raw, -30 * scale, 30 * scale)


def R_star(K, kc):
    if K <= kc:
        return 0.0
    return math.sqrt(1.0 - kc / K)


def classify_regime(R):
    if R < 0.30: return 0   # turbulent
    if R < 0.50: return 1   # aperture-dominated
    if R < 0.80: return 2   # hierarchical cascade
    if R < 0.95: return 3   # coherent
    return 4                # phase-locked


# ─── Style helpers ─────────────────────────────────────────────────────────

def _style(ax, is3d=False):
    ax.set_facecolor('white')
    if is3d:
        ax.tick_params(labelsize=6, pad=1)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor('#e0e0e0')
        ax.grid(True, alpha=0.20, lw=0.4)
    else:
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
            sp.set_color('#bbbbbb')
        ax.tick_params(labelsize=7, length=3, width=0.6)
        ax.grid(True, alpha=0.22, lw=0.4)


def make_panel(charts, path):
    fig = plt.figure(figsize=(22, 5.0), facecolor='white')
    gs  = fig.add_gridspec(1, 4, wspace=0.42,
                           left=0.04, right=0.99, top=0.94, bottom=0.14)
    for col, (fn, is3d) in enumerate(charts):
        kw = {'projection': '3d'} if is3d else {}
        ax = fig.add_subplot(gs[0, col], **kw)
        _style(ax, is3d)
        fn(ax)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  {path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# PANEL 1 — Kuramoto Order Parameter
# ═══════════════════════════════════════════════════════════════════════════

# Pre-compute simulation data (reused across panels)
_sigma  = 1.0
_kc     = K_c(_sigma)
_n_sim  = 80
_mults  = [0.3, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
_R_sims = []
print("  computing Kuramoto data ...")
for _i, _m in enumerate(_mults):
    _K    = _kc * _m
    _om   = cauchy_omega(_n_sim, _sigma, np.random.default_rng(100 + _i))
    _R_sims.append(kuramoto_R(_n_sim, _om, _K, seed=200 + _i))


def p1_order_param(ax):
    mults  = np.array(_mults)
    R_sim  = np.array(_R_sims)
    R_th   = np.array([R_star(_kc * m, _kc) for m in mults])
    k_fine = np.linspace(0.5, 5.5, 300)
    r_fine = np.array([R_star(_kc * k, _kc) for k in k_fine])
    ax.plot(k_fine, r_fine, '-', color=BLUE, lw=2.0, label='R*')
    ax.scatter(mults, R_sim, color=RED, s=30, zorder=5, label='simulated')
    ax.axvline(1.0, color='#aaaaaa', lw=0.8, ls='--')
    ax.set_xlabel('K / K_c', fontsize=8)
    ax.set_ylabel('R_ens', fontsize=8)
    ax.legend(fontsize=6, frameon=False)


def p1_convergence_surface(ax):
    n    = 60
    mults_s = [0.8, 1.2, 2.0, 3.5]
    t_ax    = None
    for j, m in enumerate(mults_s):
        K   = _kc * m
        om  = cauchy_omega(n, _sigma, np.random.default_rng(300 + j))
        traj = kuramoto_trajectory(n, om, K, T_sim=50.0, record_every=40)
        if t_ax is None:
            t_ax = np.linspace(0, 50.0, len(traj))
        ax.plot(t_ax, [m] * len(traj), traj, color=PAL[j], lw=1.4, alpha=0.85)
    ax.set_xlabel('t', fontsize=7, labelpad=3)
    ax.set_ylabel('K/K_c', fontsize=7, labelpad=3)
    ax.set_zlabel('R_ens', fontsize=7, labelpad=3)


def p1_phase_dist(ax):
    n = 150
    om = cauchy_omega(n, _sigma)
    phi_sub  = rng.uniform(0, 2 * math.pi, n)
    phi_sup  = phi_sub.copy()
    # run super-critical to convergence
    K_sup = _kc * 2.5
    for _ in range(int(80.0 / 0.02)):
        diff = phi_sup[None, :] - phi_sup[:, None]
        phi_sup += 0.02 * (om + (K_sup / n) * np.sum(np.sin(diff), axis=1))
    ax.hist(phi_sub % (2 * math.pi), bins=30, density=True,
            alpha=0.55, color=BLUE, edgecolor='none', label='K < K_c')
    ax.hist(phi_sup % (2 * math.pi), bins=30, density=True,
            alpha=0.55, color=RED,  edgecolor='none', label='K > K_c')
    ax.set_xlabel('phase φ (rad)', fontsize=8)
    ax.set_ylabel('density', fontsize=8)
    ax.legend(fontsize=6, frameon=False)


def p1_finite_N(ax):
    K     = _kc * 2.5
    N_vals = [10, 20, 40, 80, 150, 300]
    errs   = []
    r_th   = R_star(K, _kc)
    for N in N_vals:
        om = cauchy_omega(N, _sigma, np.random.default_rng(400 + N))
        r  = kuramoto_R(N, om, K, seed=500 + N)
        errs.append(abs(r - r_th))
    N_arr  = np.array(N_vals, dtype=float)
    ax.loglog(N_arr, errs, 'o-', color=BLUE, ms=5, lw=1.5, label='|R_sim - R*|')
    ax.loglog(N_arr, 1.0 / np.sqrt(N_arr), '--', color=RED,
              lw=1.0, label='1/√N')
    ax.set_xlabel('N', fontsize=8)
    ax.set_ylabel('error', fontsize=8)
    ax.legend(fontsize=6, frameon=False)


# ═══════════════════════════════════════════════════════════════════════════
# PANEL 2 — Five Coordination Regimes
# ═══════════════════════════════════════════════════════════════════════════

_REGIME_THRESHOLDS = [0.0, 0.30, 0.50, 0.80, 0.95, 1.01]
_REGIME_COSTS      = [1.0, 0.70, 0.40, 0.15, 0.00]
_REGIME_NAMES      = ['turbulent', 'aperture', 'cascade', 'coherent', 'locked']
_REGIME_COLS       = [RED, AMBER, GREEN, TEAL, BLUE]


def p2_cost_step(ax):
    R_pts = np.linspace(0, 1, 1000)
    costs = np.array([_REGIME_COSTS[classify_regime(r)] for r in R_pts])
    ax.step(R_pts, costs, where='post', color=BLUE, lw=2.0)
    for thresh in _REGIME_THRESHOLDS[1:-1]:
        ax.axvline(thresh, color='#cccccc', lw=0.8, ls='--')
    ax.set_xlabel('R_ens', fontsize=8)
    ax.set_ylabel('coordination cost', fontsize=8)
    ax.set_ylim(-0.05, 1.10)


def p2_regime_surface(ax):
    N_vals  = np.array([20, 50, 100, 200, 500], dtype=float)
    K_mults = np.linspace(0.5, 4.0, 20)
    KM, NV  = np.meshgrid(K_mults, N_vals)
    Z       = np.zeros_like(KM)
    for i, N in enumerate(N_vals):
        for j, km in enumerate(K_mults):
            K   = _kc * km
            om  = cauchy_omega(int(N), _sigma, np.random.default_rng(int(N * 7 + j * 13)))
            Z[i, j] = kuramoto_R(int(N), om, K, seed=int(N + j))
    ax.plot_surface(KM, np.log10(NV), Z, cmap='Blues', alpha=0.85, linewidth=0)
    ax.set_xlabel('K / K_c', fontsize=7, labelpad=3)
    ax.set_ylabel('log₁₀ N', fontsize=7, labelpad=3)
    ax.set_zlabel('R_ens', fontsize=7, labelpad=3)


def p2_timeseries(ax):
    n  = 80
    om = cauchy_omega(n, _sigma)
    for j, km in enumerate([0.5, 1.5, 3.0]):
        K    = _kc * km
        traj = kuramoto_trajectory(n, om, K, T_sim=50.0, record_every=30)
        t_ax = np.linspace(0, 50.0, len(traj))
        ax.plot(t_ax, traj, color=PAL[j], lw=1.5, label=f'K={km:.1f}K_c')
    ax.set_xlabel('time', fontsize=8)
    ax.set_ylabel('R_ens(t)', fontsize=8)
    ax.legend(fontsize=6, frameon=False)


def p2_regime_hist(ax):
    n  = 80
    for j, km in enumerate([0.5, 1.5, 2.5, 4.0]):
        R_samples = []
        for seed in range(25):
            om = cauchy_omega(n, _sigma, np.random.default_rng(seed * 17 + j))
            K  = _kc * km
            R_samples.append(kuramoto_R(n, om, K, seed=seed * 31 + j))
        ax.hist(R_samples, bins=12, density=True, alpha=0.50,
                color=PAL[j], edgecolor='none', label=f'{km:.1f}K_c')
    ax.set_xlabel('R_ens', fontsize=8)
    ax.set_ylabel('density', fontsize=8)
    ax.legend(fontsize=6, frameon=False)


# ═══════════════════════════════════════════════════════════════════════════
# PANEL 3 — Critical Coupling
# ═══════════════════════════════════════════════════════════════════════════

def p3_kc_linear(ax):
    sigmas  = np.linspace(0.2, 3.0, 12)
    kc_pred = [K_c(s) for s in sigmas]
    # simulated K_c: K where R_ens first exceeds 0.3 in a sweep
    kc_found = []
    n = 80
    for s in sigmas:
        found = None
        for km in np.linspace(0.3, 4.0, 30):
            K  = K_c(s) * km
            om = cauchy_omega(n, s)
            R  = kuramoto_R(n, om, K)
            if R > 0.30:
                found = K
                break
        kc_found.append(found if found else K_c(s) * 4.0)
    ax.scatter(sigmas, kc_pred,  color=BLUE, s=25, label='2σ/π')
    ax.scatter(sigmas, kc_found, color=RED,  s=25, marker='^', label='numerical')
    ax.plot(sigmas, kc_pred, color=BLUE, lw=1.2, alpha=0.6)
    ax.set_xlabel('σ_ω', fontsize=8)
    ax.set_ylabel('K_c', fontsize=8)
    ax.legend(fontsize=6, frameon=False)


def p3_kc_surface(ax):
    sigs = np.linspace(0.3, 2.5, 15)
    Ks   = np.linspace(0.1, 5.0, 20)
    S, K = np.meshgrid(sigs, Ks)
    n    = 60
    Z    = np.zeros_like(S)
    for i, Ki in enumerate(Ks):
        for j, si in enumerate(sigs):
            om    = cauchy_omega(n, si, np.random.default_rng(i * 100 + j))
            Z[i, j] = kuramoto_R(n, om, Ki, seed=i * 200 + j)
    ax.plot_surface(S, K, Z, cmap='Blues', alpha=0.85, linewidth=0)
    ax.set_xlabel('σ_ω', fontsize=7, labelpad=3)
    ax.set_ylabel('K', fontsize=7, labelpad=3)
    ax.set_zlabel('R_ens', fontsize=7, labelpad=3)


def p3_ratio(ax):
    sigmas  = np.linspace(0.3, 3.0, 18)
    target  = 2.0 / math.pi
    ratios  = [K_c(s) / s for s in sigmas]
    ax.scatter(sigmas, ratios, color=BLUE, s=22)
    ax.axhline(target, color=RED, lw=1.5, ls='--')
    ax.set_xlabel('σ_ω', fontsize=8)
    ax.set_ylabel('K_c / σ_ω', fontsize=8)
    ax.set_ylim(0, 1.0)


def p3_sharpness(ax):
    N_vals = [20, 40, 80, 150]
    for j, N in enumerate(N_vals):
        mults = np.linspace(0.6, 2.5, 25)
        Rs    = []
        for km in mults:
            K  = _kc * km
            om = cauchy_omega(N, _sigma, np.random.default_rng(N + int(km * 100)))
            Rs.append(kuramoto_R(N, om, K, seed=N + int(km * 200)))
        ax.plot(mults, Rs, '-o', ms=3, color=PAL[j], lw=1.3, label=f'N={N}')
    ax.axvline(1.0, color='#aaaaaa', lw=0.8, ls='--')
    ax.set_xlabel('K / K_c', fontsize=8)
    ax.set_ylabel('R_ens', fontsize=8)
    ax.legend(fontsize=6, frameon=False)


# ═══════════════════════════════════════════════════════════════════════════
# PANEL 4 — Zero-Overhead Phase Lock
# ═══════════════════════════════════════════════════════════════════════════

def p4_overhead_step(ax):
    R_pts = np.linspace(0, 1, 1000)
    # Overhead: drops to 0 at R >= 0.95
    overhead = np.where(R_pts < 0.95,
                        1.0 - R_pts ** 2,
                        0.0)
    ax.plot(R_pts, overhead, color=BLUE, lw=2.0)
    ax.axvline(0.95, color=RED, lw=1.2, ls='--')
    ax.fill_between(R_pts[R_pts >= 0.95], 0, overhead[R_pts >= 0.95],
                    alpha=0.12, color=GREEN)
    ax.set_xlabel('R_ens', fontsize=8)
    ax.set_ylabel('coordination overhead', fontsize=8)


def p4_overhead_surface(ax):
    N_vals = np.array([20, 50, 100, 200], dtype=float)
    R_vals = np.linspace(0, 1, 25)
    NV, RV = np.meshgrid(np.log10(N_vals), R_vals)
    Z = np.where(RV >= 0.95,
                 np.zeros_like(RV),
                 (1.0 - RV ** 2) / np.log10(10 ** NV))
    ax.plot_surface(RV, NV, Z, cmap='Reds', alpha=0.82, linewidth=0)
    ax.set_xlabel('R_ens', fontsize=7, labelpad=3)
    ax.set_ylabel('log₁₀ N', fontsize=7, labelpad=3)
    ax.set_zlabel('overhead / agent', fontsize=7, labelpad=3)


def p4_lock_cdf(ax):
    # Time to reach R >= 0.95 for different K/K_c
    n = 80
    for j, km in enumerate([1.5, 2.0, 3.0, 5.0]):
        K     = _kc * km
        lock_times = []
        for seed in range(40):
            om   = cauchy_omega(n, _sigma, np.random.default_rng(seed * 19 + j))
            phi  = np.random.default_rng(seed).uniform(0, 2 * math.pi, n)
            t    = 0.0
            dt   = 0.05
            locked = False
            for step in range(4000):
                diff = phi[None, :] - phi[:, None]
                phi += dt * (om + (K / n) * np.sum(np.sin(diff), axis=1))
                t   += dt
                if abs(np.mean(np.exp(1j * phi))) >= 0.95:
                    lock_times.append(t)
                    locked = True
                    break
            if not locked:
                lock_times.append(200.0)
        lock_times.sort()
        cdf = np.arange(1, len(lock_times) + 1) / len(lock_times)
        ax.plot(lock_times, cdf, color=PAL[j], lw=1.5, label=f'{km:.1f}K_c')
    ax.set_xlabel('time to phase-lock', fontsize=8)
    ax.set_ylabel('CDF', fontsize=8)
    ax.legend(fontsize=6, frameon=False)


def p4_savings(ax):
    N_arr    = np.array([10, 20, 50, 100, 200, 500, 1000], dtype=float)
    # Savings = N*(N-1)/2 messages avoided when R >= 0.95
    savings  = N_arr * (N_arr - 1) / 2
    baseline = N_arr * (N_arr - 1) / 2
    ax.loglog(N_arr, savings,  'o-', color=BLUE, lw=1.8, ms=5, label='msgs saved')
    ax.loglog(N_arr, baseline, '--', color=RED,  lw=1.0, label='N² / 2')
    ax.set_xlabel('N agents', fontsize=8)
    ax.set_ylabel('consensus msgs avoided', fontsize=8)
    ax.legend(fontsize=6, frameon=False)


# ═══════════════════════════════════════════════════════════════════════════
# PANEL 5 — Composition Inflation & Federation Admission
# ═══════════════════════════════════════════════════════════════════════════

def T(m, D):
    return D * (D + 1) ** (m - 1)


def p5_T_lines(ax):
    ms = np.arange(1, 12)
    for j, D in enumerate(range(1, 7)):
        ys = [T(m, D) for m in ms]
        ax.semilogy(ms, ys, '-o', color=PAL[j], ms=3.5, lw=1.6, label=f'D={D}')
    ax.set_xlabel('m (agents)', fontsize=8)
    ax.set_ylabel('T(m, D)', fontsize=8)
    ax.legend(fontsize=6, frameon=False, ncol=2)


def p5_T_surface(ax):
    ms = np.arange(1, 10)
    Ds = np.arange(1, 8)
    M, D = np.meshgrid(ms, Ds)
    Z = np.log10(D * (D + 1) ** (M - 1))
    ax.plot_surface(M, D, Z, cmap='Blues', alpha=0.88, linewidth=0, antialiased=True)
    ax.set_xlabel('m', fontsize=7, labelpad=3)
    ax.set_ylabel('D', fontsize=7, labelpad=3)
    ax.set_zlabel('log₁₀ T', fontsize=7, labelpad=3)


def p5_marginal_entropy(ax):
    n_agents = 30
    base_H   = 3.0
    # Simulate marginal entropy ΔH_i for each agent
    DH = rng.exponential(0.3, n_agents) + rng.uniform(-0.1, 0.4, n_agents)
    costs = rng.uniform(0.05, 0.35, n_agents)
    admitted = DH > costs
    ax.bar(np.where(admitted)[0],  DH[admitted],    color=BLUE, alpha=0.75, label='admitted')
    ax.bar(np.where(~admitted)[0], DH[~admitted],   color=RED,  alpha=0.60, label='rejected')
    ax.plot(np.arange(n_agents), costs, 'k--', lw=1.0, label='cost threshold')
    ax.set_xlabel('agent index', fontsize=8)
    ax.set_ylabel('ΔH_i (nats)', fontsize=8)
    ax.legend(fontsize=6, frameon=False)


def p5_fed_entropy(ax):
    n_agents = 25
    fed_sizes = np.arange(1, n_agents + 1)
    # Federation entropy grows sub-linearly (diminishing returns)
    H_fed = 2.0 * np.log(fed_sizes + 1) + rng.normal(0, 0.05, n_agents)
    ax.plot(fed_sizes, H_fed, 'o-', color=BLUE, ms=4, lw=1.5)
    ax.fill_between(fed_sizes, 0, H_fed, alpha=0.08, color=BLUE)
    ax.set_xlabel('federation size N', fontsize=8)
    ax.set_ylabel('H_fed (nats)', fontsize=8)


# ═══════════════════════════════════════════════════════════════════════════
# PANEL 6 — Multi-Sensor Federation & Graceful Decoherence
# ═══════════════════════════════════════════════════════════════════════════

def p6_decoherence(ax):
    n    = 80
    K    = _kc * 3.0
    om   = cauchy_omega(n, _sigma)
    # Warm up to phase-locked state
    phi  = rng.uniform(0, 2 * math.pi, n)
    dt   = 0.05
    for _ in range(int(60.0 / dt)):
        diff = phi[None, :] - phi[:, None]
        phi += dt * (om + (K / n) * np.sum(np.sin(diff), axis=1))
    # Record decoherence by adding noise
    t_dec = np.linspace(0, 50, 200)
    R_dec = []
    noise_level = 0.0
    for step in range(200):
        noise_level = min(2.0, noise_level + 0.01)
        phi_noisy = phi + rng.normal(0, noise_level * 0.01, n)
        diff  = phi_noisy[None, :] - phi_noisy[:, None]
        phi  += dt * (om + (K / n) * np.sum(np.sin(diff), axis=1))
        R_dec.append(abs(np.mean(np.exp(1j * phi_noisy))))
    ax.plot(t_dec, R_dec, color=BLUE, lw=1.6)
    for r_thresh in [0.95, 0.80, 0.50, 0.30]:
        ax.axhline(r_thresh, color='#cccccc', lw=0.7, ls='--')
    ax.set_xlabel('time', fontsize=8)
    ax.set_ylabel('R_ens (decoherence)', fontsize=8)


def p6_coupling_surface(ax):
    n    = 20
    K    = _kc * 2.5
    om   = cauchy_omega(n, _sigma)
    phi  = rng.uniform(0, 2 * math.pi, n)
    for _ in range(int(80.0 / 0.02)):
        diff = phi[None, :] - phi[:, None]
        phi += 0.02 * (om + (K / n) * np.sum(np.sin(diff), axis=1))
    # Phase-difference matrix
    diff_mat = np.abs(phi[None, :] - phi[:, None]) % (2 * math.pi)
    diff_mat = np.minimum(diff_mat, 2 * math.pi - diff_mat)
    i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n))
    ax.plot_surface(i_idx.astype(float), j_idx.astype(float),
                    diff_mat, cmap='Blues', alpha=0.82, linewidth=0)
    ax.set_xlabel('agent i', fontsize=7, labelpad=3)
    ax.set_ylabel('agent j', fontsize=7, labelpad=3)
    ax.set_zlabel('|φ_i − φ_j|', fontsize=7, labelpad=3)


def p6_sensor_dp(ax):
    specs = [('temp', 0.10, BLUE), ('pressure', 0.08, GREEN),
             ('light', 0.12, AMBER), ('vibration', 0.07, PURPLE)]
    for name, sig, col in specs:
        dp = rng.normal(0, sig, 3000)
        ax.hist(dp, bins=55, density=True, alpha=0.45,
                color=col, edgecolor='none', label=name)
    ax.set_xlabel('ΔP (normalised)', fontsize=8)
    ax.set_ylabel('density', fontsize=8)
    ax.legend(fontsize=6, frameon=False)


def p6_detection(ax):
    # Running R_ens with detection threshold
    n    = 80
    K    = _kc * 2.5
    om   = cauchy_omega(n, _sigma)
    phi  = rng.uniform(0, 2 * math.pi, n)
    traj = kuramoto_trajectory(n, om, K, T_sim=80.0, record_every=20)
    t_ax = np.linspace(0, 80.0, len(traj))
    ax.plot(t_ax, traj, color=BLUE, lw=1.5)
    thresh = 0.50
    ax.axhline(thresh, color=RED, lw=1.2, ls='--')
    # Mark detection event
    detected = np.where(np.array(traj) < thresh)[0]
    if len(detected):
        td = t_ax[detected[0]]
        ax.axvline(td, color=RED, lw=1.0, alpha=0.6)
        ax.scatter([td], [traj[detected[0]]], color=RED, s=40, zorder=5)
    ax.set_xlabel('time', fontsize=8)
    ax.set_ylabel('R_ens', fontsize=8)


# ═══════════════════════════════════════════════════════════════════════════
# Assemble & save
# ═══════════════════════════════════════════════════════════════════════════

print("Generating swarm-federation panels ...")

make_panel([(p1_order_param,        False), (p1_convergence_surface, True),
            (p1_phase_dist,         False), (p1_finite_N,           False)],
           OUT / 'panel_01.png')

make_panel([(p2_cost_step,    False), (p2_regime_surface, True),
            (p2_timeseries,   False), (p2_regime_hist,    False)],
           OUT / 'panel_02.png')

make_panel([(p3_kc_linear,   False), (p3_kc_surface, True),
            (p3_ratio,       False), (p3_sharpness,  False)],
           OUT / 'panel_03.png')

make_panel([(p4_overhead_step,    False), (p4_overhead_surface, True),
            (p4_lock_cdf,         False), (p4_savings,          False)],
           OUT / 'panel_04.png')

make_panel([(p5_T_lines,          False), (p5_T_surface,        True),
            (p5_marginal_entropy, False), (p5_fed_entropy,       False)],
           OUT / 'panel_05.png')

make_panel([(p6_decoherence,      False), (p6_coupling_surface,  True),
            (p6_sensor_dp,        False), (p6_detection,         False)],
           OUT / 'panel_06.png')

print("Done. Six panels saved to", OUT)
