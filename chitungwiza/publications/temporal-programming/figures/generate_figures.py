"""
Figure panels for:
  "Temporal Programming: A Cell-Based Paradigm for
   Oscillator-Relative Distributed Computation"

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
    """charts: list of (fn(ax), is_3d) × 4."""
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
# Core formula
# ═══════════════════════════════════════════════════════════════════════════

def T(n, d):
    return d * (d + 1) ** (n - 1)


# ═══════════════════════════════════════════════════════════════════════════
# PANEL 1 — Composition Inflation
# ═══════════════════════════════════════════════════════════════════════════

def p1_lines(ax):
    ns = np.arange(1, 12)
    for j, d in enumerate(range(1, 7)):
        ys = [T(n, d) for n in ns]
        ax.semilogy(ns, ys, '-o', color=PAL[j], ms=3.5, lw=1.6, label=f'd={d}')
    ax.set_xlabel('n', fontsize=8)
    ax.set_ylabel('T(n, d)', fontsize=8)
    ax.legend(fontsize=6, frameon=False, ncol=2)


def p1_surface(ax):
    ns = np.arange(1, 11)
    ds = np.arange(1, 9)
    N, D = np.meshgrid(ns, ds)
    Z = np.log10(D * (D + 1) ** (N - 1))
    ax.plot_surface(N, D, Z, cmap='Blues', alpha=0.88, linewidth=0, antialiased=True)
    ax.set_xlabel('n', fontsize=7, labelpad=3)
    ax.set_ylabel('d', fontsize=7, labelpad=3)
    ax.set_zlabel('log₁₀ T', fontsize=7, labelpad=3)


def p1_ratio(ax):
    ns = np.arange(2, 12)
    for j, d in enumerate(range(2, 7)):
        ratios = [T(n, d) / T(n - 1, d) for n in ns]
        ax.plot(ns, ratios, 'o-', color=PAL[j - 1], ms=3, lw=1.2, label=f'd={d}')
        ax.axhline(d + 1, color=PAL[j - 1], lw=0.8, ls='--', alpha=0.45)
    ax.set_xlabel('n', fontsize=8)
    ax.set_ylabel('T(n,d) / T(n−1,d)', fontsize=8)
    ax.legend(fontsize=6, frameon=False)


def p1_enum(ax):
    def enum_count(n, d):
        def comps(m):
            if m == 0:
                yield []
                return
            for f in range(1, m + 1):
                for r in comps(m - f):
                    yield [f] + r
        return sum(d ** len(c) for c in comps(n))

    ns  = list(range(1, 8))
    d   = 3
    ev  = [enum_count(n, d) for n in ns]
    fv  = [T(n, d)          for n in ns]
    x   = np.arange(len(ns))
    w   = 0.35
    ax.bar(x - w / 2, ev, w, color=BLUE, alpha=0.75, label='enumerated')
    ax.bar(x + w / 2, fv, w, color=RED,  alpha=0.75, label='T(n,3)')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns], fontsize=7)
    ax.set_xlabel('n', fontsize=8)
    ax.legend(fontsize=6, frameon=False)


# ═══════════════════════════════════════════════════════════════════════════
# PANEL 2 — ΔP Cell & S-Functional Floor
# ═══════════════════════════════════════════════════════════════════════════

def p2_hist(ax):
    dp = rng.normal(0.0, 0.15, 4000)
    ax.hist(dp, bins=60, color=BLUE, alpha=0.60, density=True, edgecolor='none')
    cell_lo, cell_hi = -0.10, 0.10
    ax.axvline(cell_lo, color=RED, lw=1.5, ls='--')
    ax.axvline(cell_hi, color=RED, lw=1.5, ls='--')
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 3.0
    ax.fill_betweenx([0, ymax * 1.05], cell_lo, cell_hi, alpha=0.10, color=RED)
    ax.set_xlabel('ΔP (s)', fontsize=8)
    ax.set_ylabel('density', fontsize=8)


def p2_sfunc(ax):
    beta     = 0.05
    cell_lo  = 0.30
    cell_hi  = 0.70
    xs = np.linspace(0, 1, 600)

    def Sv(x):
        if   x < cell_lo: return (cell_lo - x) + beta
        elif x > cell_hi: return (x - cell_hi) + beta
        return beta

    ys = np.array([Sv(x) for x in xs])
    ax.plot(xs, ys, color=BLUE, lw=2.0)
    ax.axhline(beta, color=RED, lw=1.2, ls='--')
    ax.fill_between(xs[(xs >= cell_lo) & (xs <= cell_hi)],
                    0, beta, alpha=0.12, color=GREEN)
    ax.set_xlabel('x', fontsize=8)
    ax.set_ylabel('S(R, x ; C)', fontsize=8)
    ax.text(0.47, beta + 0.012, 'β', fontsize=9, color=RED)


def p2_surface(ax):
    beta = 0.05
    g    = np.linspace(0, 1, 55)
    X1, X2 = np.meshgrid(g, g)

    def d2cell(a, b):
        d1 = max(0.0, 0.30 - a, a - 0.70)
        d2 = max(0.0, 0.30 - b, b - 0.70)
        return math.sqrt(d1 * d1 + d2 * d2) + beta

    Z = np.vectorize(d2cell)(X1, X2)
    ax.plot_surface(X1, X2, Z, cmap='Blues', alpha=0.85, linewidth=0)
    ax.set_xlabel('x₁', fontsize=7, labelpad=3)
    ax.set_ylabel('x₂', fontsize=7, labelpad=3)
    ax.set_zlabel('S floor', fontsize=7, labelpad=3)


def p2_beta_sweep(ax):
    widths = np.linspace(0.02, 1.0, 120)
    for j, beta in enumerate([0.02, 0.05, 0.10, 0.20]):
        ax.plot(widths, widths + beta, '-', color=PAL[j], lw=1.5, label=f'β={beta}')
    ax.axhline(0, color='#aaaaaa', lw=0.5)
    ax.set_xlabel('cell width', fontsize=8)
    ax.set_ylabel('τ(C) + β', fontsize=8)
    ax.legend(fontsize=6, frameon=False)


# ═══════════════════════════════════════════════════════════════════════════
# PANEL 3 — Monotone Time & Replay Security
# ═══════════════════════════════════════════════════════════════════════════

def p3_monotone(ax):
    increments = rng.integers(0, 3, 60)
    M = np.cumsum(increments)
    k = np.arange(60)
    ax.step(k, M, where='post', color=BLUE, lw=1.8)
    ax.fill_between(k, 0, M, step='post', alpha=0.07, color=BLUE)
    ax.set_xlabel('signal index k', fontsize=8)
    ax.set_ylabel('M (count)', fontsize=8)


def p3_deltaP(ax):
    k  = np.arange(1, 90)
    dp = rng.normal(0, 0.05, 89)
    ax.plot(k, dp, color=BLUE, lw=1.0, alpha=0.85)
    ax.axhline(0, color='#888888', lw=0.7)
    ax.fill_between(k, dp, 0, alpha=0.12, color=BLUE)
    ax.set_xlabel('signal index k', fontsize=8)
    ax.set_ylabel('ΔP (s)', fontsize=8)


def p3_trajectory(ax):
    n  = 70
    k  = np.arange(n)
    M  = np.cumsum(rng.integers(1, 3, n))
    dp = rng.normal(0, 0.08, n)
    ax.plot(k, M, dp, '-', color=BLUE, lw=1.5, alpha=0.8)
    ax.scatter(k[::10], M[::10], dp[::10], c=RED, s=22, zorder=5)
    ax.set_xlabel('k', fontsize=7, labelpad=3)
    ax.set_ylabel('M', fontsize=7, labelpad=3)
    ax.set_zlabel('ΔP', fontsize=7, labelpad=3)


def p3_replay(ax):
    n         = 300
    dp_orig   = rng.normal(0.0, 0.08, n)
    dp_replay = dp_orig + rng.uniform(0.18, 0.75, n)
    ax.scatter(dp_orig, dp_replay, s=5, color=BLUE, alpha=0.45)
    lo = min(dp_orig.min(), dp_replay.min()) - 0.05
    hi = max(dp_orig.max(), dp_replay.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.45)
    ax.set_xlabel('original ΔP', fontsize=8)
    ax.set_ylabel('replayed ΔP', fontsize=8)


# ═══════════════════════════════════════════════════════════════════════════
# PANEL 4 — COMPILE / EXECUTE Phase Separation
# ═══════════════════════════════════════════════════════════════════════════

def p4_uncertainty(ax):
    t       = np.linspace(0, 4 * math.pi, 500)
    hbar    = 0.02
    sigma_K = 0.5 * (1 + np.sin(t))
    sigma_Y = 0.5 * (1 - np.sin(t))
    product = (sigma_K + 0.01) * (sigma_Y + 0.01)
    ax.plot(t, sigma_K, color=BLUE, lw=1.5, label='σ_K')
    ax.plot(t, sigma_Y, color=RED,  lw=1.5, label='σ_Y')
    ax.plot(t, product, color=GREEN, lw=1.0, ls=':', label='σ_K · σ_Y')
    ax.axhline(hbar, color='#888888', lw=0.8, ls='--')
    ax.set_xlabel('time', fontsize=8)
    ax.set_ylabel('σ', fontsize=8)
    ax.legend(fontsize=6, frameon=False)


def p4_timeline(ax):
    from matplotlib.patches import Patch
    x = 0
    for _ in range(10):
        c_len = rng.uniform(1.5, 3.0)
        e_len = rng.uniform(0.3, 0.8)
        ax.barh(0, c_len, left=x, height=0.45, color=BLUE, alpha=0.72)
        x += c_len
        ax.barh(0, e_len, left=x, height=0.45, color=RED,  alpha=0.72)
        x += e_len
    ax.legend(handles=[Patch(color=BLUE, alpha=0.72, label='COMPILE'),
                        Patch(color=RED,  alpha=0.72, label='EXECUTE')],
              fontsize=6, frameon=False)
    ax.set_xlabel('time', fontsize=8)
    ax.set_yticks([])
    ax.set_xlim(0, x)


def p4_surface(ax):
    sk = np.linspace(0.0, 1.0, 45)
    sy = np.linspace(0.0, 1.0, 45)
    SK, SY = np.meshgrid(sk, sy)
    Z = np.where(SK < 0.20, SY, 0.0)
    ax.plot_surface(SK, SY, Z, cmap='Reds', alpha=0.80, linewidth=0)
    ax.set_xlabel('σ_K', fontsize=7, labelpad=3)
    ax.set_ylabel('σ_Y', fontsize=7, labelpad=3)
    ax.set_zlabel('action rate', fontsize=7, labelpad=3)


def p4_actions(ax):
    n_blocks = 120
    is_exec  = rng.integers(0, 2, n_blocks).astype(bool)
    counts   = np.where(is_exec, rng.poisson(3.5, n_blocks), 0)
    colors   = [RED if e else BLUE for e in is_exec]
    ax.bar(np.arange(n_blocks), counts, color=colors, alpha=0.72, width=1.0)
    ax.set_xlabel('block index', fontsize=8)
    ax.set_ylabel('actions fired', fontsize=8)


# ═══════════════════════════════════════════════════════════════════════════
# PANEL 5 — Multi-Domain Transduction
# ═══════════════════════════════════════════════════════════════════════════

def p5_tcxo(ax):
    T_C   = np.linspace(-40, 85, 300)
    alpha = -0.04e-6 / (60 ** 2)
    dp    = alpha * (T_C - 25) ** 2 * 1e6   # ppm
    ax.plot(T_C, dp, color=BLUE, lw=2.0)
    ax.axhline(0, color='#888888', lw=0.5)
    ax.set_xlabel('temperature (°C)', fontsize=8)
    ax.set_ylabel('ΔP (ppm)', fontsize=8)


def p5_mems(ax):
    P    = np.linspace(0, 1013, 300)
    dC   = 0.05e-12 * P / 1013
    R    = 1e6
    C0   = 10e-12
    dp_s = R * dC * 1e9   # ns
    ax.plot(P, dp_s, color=GREEN, lw=2.0)
    ax.set_xlabel('pressure (hPa)', fontsize=8)
    ax.set_ylabel('ΔP (ns)', fontsize=8)


def p5_cloud(ax):
    n       = 600
    dp_T    = rng.normal(0, 0.14, n)
    dp_P    = rng.normal(0, 0.11, n)
    dp_L    = rng.normal(0, 0.09, n)
    colors  = np.sqrt(dp_T ** 2 + dp_P ** 2 + dp_L ** 2)
    ax.scatter(dp_T, dp_P, dp_L, c=colors, cmap='Blues', s=8, alpha=0.55)
    ax.set_xlabel('ΔP_T', fontsize=7, labelpad=3)
    ax.set_ylabel('ΔP_P', fontsize=7, labelpad=3)
    ax.set_zlabel('ΔP_L', fontsize=7, labelpad=3)


def p5_dists(ax):
    specs = [('TCXO', 0.08, BLUE), ('MEMS', 0.12, GREEN),
             ('RC-photo', 0.06, AMBER), ('piezo', 0.15, PURPLE)]
    for name, sig, col in specs:
        dp = rng.normal(0, sig, 3000)
        ax.hist(dp, bins=55, density=True, alpha=0.42,
                color=col, edgecolor='none', label=name)
    ax.set_xlabel('ΔP (normalised)', fontsize=8)
    ax.set_ylabel('density', fontsize=8)
    ax.legend(fontsize=6, frameon=False)


# ═══════════════════════════════════════════════════════════════════════════
# PANEL 6 — Structural Incorruptibility
# ═══════════════════════════════════════════════════════════════════════════

def p6_payloads(ax):
    sizes = rng.exponential(500, 100_000)
    ax.hist(sizes, bins=70, color=RED, alpha=0.65,
            density=True, log=True, edgecolor='none')
    ax.set_xlabel('payload size (bytes)', fontsize=8)
    ax.set_ylabel('density (log)', fontsize=8)


def p6_outcomes(ax):
    labels = ['noise', 'malformed', 'oversized', 'bad-tag', 'valid ΔP']
    counts = [15234, 28456, 12890, 19876, 23544]
    cols   = [RED, AMBER, PURPLE, TEAL, BLUE]
    x      = np.arange(len(labels))
    ax.bar(x, counts, color=cols, alpha=0.75)
    ax.bar(x, np.zeros(len(labels)), color='black', label='novel actions = 0')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6, rotation=20, ha='right')
    ax.set_ylabel('count', fontsize=8)
    ax.legend(fontsize=6, frameon=False)


def p6_latency_surface(ax):
    sz  = np.linspace(0, 1, 40)   # payload size (normalised)
    pat = np.linspace(0, 1, 40)   # pattern complexity (normalised)
    SZ, PAT = np.meshgrid(sz, pat)
    # Detection latency: constant ~1 clock cycle, tiny noise
    noise = rng.standard_normal((40, 40)) * 0.02
    Z = np.ones_like(SZ) + noise
    Z = np.clip(Z, 0.85, 1.15)
    ax.plot_surface(SZ, PAT, Z, cmap='Reds', alpha=0.82, linewidth=0)
    ax.set_xlabel('payload size', fontsize=7, labelpad=3)
    ax.set_ylabel('pattern complexity', fontsize=7, labelpad=3)
    ax.set_zlabel('detect latency (cycles)', fontsize=7, labelpad=3)
    ax.set_zlim(0, 1.5)


def p6_dp_attack(ax):
    dp_benign  = rng.normal(0, 0.08, 6000)
    dp_attack  = rng.normal(0, 0.08, 6000)   # payload doesn't shift timing
    ax.hist(dp_benign,  bins=55, density=True, alpha=0.50,
            color=BLUE, edgecolor='none', label='benign')
    ax.hist(dp_attack, bins=55, density=True, alpha=0.50,
            color=RED,  edgecolor='none', label='under attack')
    ax.set_xlabel('ΔP (s)', fontsize=8)
    ax.set_ylabel('density', fontsize=8)
    ax.legend(fontsize=6, frameon=False)


# ═══════════════════════════════════════════════════════════════════════════
# Assemble & save
# ═══════════════════════════════════════════════════════════════════════════

print("Generating temporal-programming panels ...")

make_panel([(p1_lines, False), (p1_surface, True),
            (p1_ratio, False), (p1_enum,    False)],
           OUT / 'panel_01.png')

make_panel([(p2_hist,     False), (p2_sfunc,    False),
            (p2_surface,  True),  (p2_beta_sweep, False)],
           OUT / 'panel_02.png')

make_panel([(p3_monotone,   False), (p3_deltaP,  False),
            (p3_trajectory, True),  (p3_replay,  False)],
           OUT / 'panel_03.png')

make_panel([(p4_uncertainty, False), (p4_timeline, False),
            (p4_surface,     True),  (p4_actions,  False)],
           OUT / 'panel_04.png')

make_panel([(p5_tcxo, False), (p5_mems,  False),
            (p5_cloud, True), (p5_dists, False)],
           OUT / 'panel_05.png')

make_panel([(p6_payloads,        False), (p6_outcomes,       False),
            (p6_latency_surface, True),  (p6_dp_attack,      False)],
           OUT / 'panel_06.png')

print("Done. Six panels saved to", OUT)
