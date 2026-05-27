import sys
import os
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize

OUT = os.path.join(os.path.dirname(__file__), 'panels')
os.makedirs(OUT, exist_ok=True)

# ── Core bijection Phi: Z+ -> (n, l, m, s) ───────────────────────────────────
def phi(M):
    count = 0
    n = 1
    while True:
        cap = 2 * n * n
        if count + cap >= M:
            rem = M - count - 1
            idx = 0
            for l in range(n):
                for m in range(-l, l + 1):
                    for s in [0.5, -0.5]:
                        if idx == rem:
                            return (n, l, m, s)
                        idx += 1
        count += cap
        n += 1

def C1(n): return 2
def C2(n): return 2 * (2 * n + 1)
def C3(n): return 2 * n * n
def N_state(n): return n * (n + 1) * (2 * n + 1) // 3

# ── Precompute datasets ────────────────────────────────────────────────────────
ns = np.arange(1, 21)
c1v = np.array([C1(n) for n in ns])
c2v = np.array([C2(n) for n in ns])
c3v = np.array([C3(n) for n in ns])
Nv  = np.array([N_state(n) for n in ns])

M_max = 200
states_all = [phi(M) for M in range(1, M_max + 1)]
n_arr = np.array([s[0] for s in states_all])
l_arr = np.array([s[1] for s in states_all])
m_arr = np.array([s[2] for s in states_all])
s_arr = np.array([s[3] for s in states_all])
M_arr = np.arange(1, M_max + 1)

# Full partition space for n=1..7
full_states = []
for n in range(1, 8):
    for l in range(n):
        for m in range(-l, l + 1):
            for s in [0.5, -0.5]:
                full_states.append((n, l, m, s))
fs_n = np.array([s[0] for s in full_states])
fs_l = np.array([s[1] for s in full_states])
fs_m = np.array([s[2] for s in full_states])
fs_s = np.array([s[3] for s in full_states])

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         9,
    'axes.linewidth':    0.8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'xtick.major.size':  3,
    'ytick.major.size':  3,
})

C = ['#1a1a2e', '#16213e', '#0f3460', '#533483', '#e94560']
plasma = cm.plasma
viridis = cm.viridis

def save_panel(fig, stem):
    for ext in ('png', 'pdf'):
        path = os.path.join(OUT, f'{stem}.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  saved {stem}')


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — Dimensional Capacity Hierarchy
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 5), facecolor='white')
fig.subplots_adjust(wspace=0.38)

# 1a — capacity curves C_d(n)
ax = fig.add_subplot(141)
ax.plot(ns, c1v, 'o-', color=C[0], lw=1.8, ms=4)
ax.plot(ns, c2v, 's-', color=C[2], lw=1.8, ms=4)
ax.plot(ns, c3v, '^-', color=C[4], lw=1.8, ms=4)
ax.set_xlabel('n', labelpad=4)
ax.set_ylabel('$C_d(n)$', labelpad=4)
ax.set_title('$C_1$, $C_2$, $C_3$ per Level', pad=8)

# 1b — log-log growth (d=2 and d=3)
ax = fig.add_subplot(142)
ax.loglog(ns[1:], c2v[1:], 's-', color=C[2], lw=1.8, ms=4)
ax.loglog(ns[1:], c3v[1:], '^-', color=C[4], lw=1.8, ms=4)
ref_n = ns[1:].astype(float)
ax.loglog(ref_n, ref_n,       '--', color='gray',  lw=0.8, alpha=0.5)
ax.loglog(ref_n, ref_n ** 2,  '--', color='black', lw=0.8, alpha=0.5)
ax.set_xlabel('$n$', labelpad=4)
ax.set_ylabel('$\\log C_d$', labelpad=4)
ax.set_title('Growth Scaling (log–log)', pad=8)

# 1c — 3D: sublevel degeneracy (n, l) -> 2(2l+1)
ax = fig.add_subplot(143, projection='3d')
sub_n, sub_l, sub_deg = [], [], []
for nn in range(1, 12):
    for ll in range(nn):
        sub_n.append(nn);  sub_l.append(ll);  sub_deg.append(2 * (2 * ll + 1))
sc = ax.scatter(sub_n, sub_l, sub_deg, c=sub_deg, cmap='plasma', s=22, alpha=0.85)
ax.set_xlabel('$n$', fontsize=7, labelpad=2)
ax.set_ylabel('$\\ell$', fontsize=7, labelpad=2)
ax.set_zlabel('$2(2\\ell+1)$', fontsize=7, labelpad=2)
ax.set_title('Sublevel Degeneracy', pad=8)
ax.tick_params(labelsize=6)

# 1d — cumulative count N_state(n)
ax = fig.add_subplot(144)
ax.fill_between(ns, Nv, alpha=0.15, color=C[3])
ax.plot(ns, Nv, 'o-', color=C[3], lw=1.8, ms=4)
ax.set_xlabel('$n$', labelpad=4)
ax.set_ylabel('$N_{state}(n)$', labelpad=4)
ax.set_title('Cumulative State Count', pad=8)

save_panel(fig, 'panel_01_capacity_hierarchy')


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2 — Bijection Structure
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 5), facecolor='white')
fig.subplots_adjust(wspace=0.38)

# 2a — M -> level n
ax = fig.add_subplot(141)
ax.scatter(M_arr, n_arr, c=n_arr, cmap='plasma', s=7, alpha=0.75, linewidths=0)
ax.set_xlabel('$M$', labelpad=4)
ax.set_ylabel('$n(M)$', labelpad=4)
ax.set_title('Level Assignment $\\Phi(M)$', pad=8)

# 2b — (n, l) projection coloured by M
ax = fig.add_subplot(142)
sc = ax.scatter(n_arr, l_arr, c=M_arr, cmap='viridis', s=10, alpha=0.75, linewidths=0)
plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
ax.set_xlabel('$n$', labelpad=4)
ax.set_ylabel('$\\ell$', labelpad=4)
ax.set_title('$(n, \\ell)$ Projection', pad=8)

# 2c — 3D scatter (n, l, m) coloured by spin s
ax = fig.add_subplot(143, projection='3d')
ax.scatter(n_arr, l_arr, m_arr, c=s_arr, cmap='RdBu', s=10, alpha=0.75, linewidths=0)
ax.set_xlabel('$n$', fontsize=7, labelpad=2)
ax.set_ylabel('$\\ell$', fontsize=7, labelpad=2)
ax.set_zlabel('$m$', fontsize=7, labelpad=2)
ax.set_title('$(n,\\ell,m)$ State Space', pad=8)
ax.tick_params(labelsize=6)

# 2d — level boundary bar chart
ax = fig.add_subplot(144)
bounds = [0] + [N_state(n) for n in range(1, 16)]
for i in range(len(bounds) - 1):
    ax.barh(i + 1, bounds[i + 1] - bounds[i], left=bounds[i], height=0.75,
            color=plasma(i / 14), alpha=0.85)
ax.set_xlabel('$M$', labelpad=4)
ax.set_ylabel('$n$', labelpad=4)
ax.set_title('Level Boundaries', pad=8)

save_panel(fig, 'panel_02_bijection_structure')


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 3 — Within-Level Partition Geometry
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 5), facecolor='white')
fig.subplots_adjust(wspace=0.38)

# 3a — d=1: only spin, no angular variation
ax = fig.add_subplot(141)
for n in range(1, 11):
    ax.scatter([n, n], [0.5, -0.5], c=[C[0]], s=28, zorder=4)
ax.set_xlabel('$n$', labelpad=4)
ax.set_ylabel('$s$', labelpad=4)
ax.set_ylim(-1, 1)
ax.set_title('$d{=}1$: Spin Only', pad=8)

# 3b — d=2: (n, m) scatter
ax = fig.add_subplot(142)
for n in range(1, 9):
    for l in range(n):
        for m in range(-l, l + 1):
            ax.scatter(n, m, c=[C[2]], s=12, alpha=0.7, linewidths=0)
ax.set_xlabel('$n$', labelpad=4)
ax.set_ylabel('$m$', labelpad=4)
ax.set_title('$d{=}2$: $(n,m)$ Structure', pad=8)

# 3c — 3D: (n, l, m) for d=3, n=1..6
ax = fig.add_subplot(143, projection='3d')
ns3d, ls3d, ms3d = [], [], []
for n in range(1, 7):
    for l in range(n):
        for m in range(-l, l + 1):
            ns3d.append(n);  ls3d.append(l);  ms3d.append(m)
ax.scatter(ns3d, ls3d, ms3d, c=ns3d, cmap='plasma', s=14, alpha=0.75, linewidths=0)
ax.set_xlabel('$n$', fontsize=7, labelpad=2)
ax.set_ylabel('$\\ell$', fontsize=7, labelpad=2)
ax.set_zlabel('$m$', fontsize=7, labelpad=2)
ax.set_title('$d{=}3$: $(n,\\ell,m)$ Space', pad=8)
ax.tick_params(labelsize=6)

# 3d — angular dimension comparison
ax = fig.add_subplot(144)
d_vals    = [1, 2, 3]
ang_dims  = [0, 1, 2]
cap_n5    = [C1(5), C2(5), C3(5)]
ax2b = ax.twinx()
ax.bar(d_vals, ang_dims, alpha=0.35, color=C[3], width=0.4)
ax2b.plot(d_vals, cap_n5, 'o-', color=C[4], lw=2, ms=7)
ax.set_xticks([1, 2, 3])
ax.set_xlabel('$d$', labelpad=4)
ax.set_ylabel('angular dim', color=C[3], labelpad=4)
ax2b.set_ylabel('$C_d(5)$', color=C[4], labelpad=4)
ax.set_title('Angular Dimension vs $d$', pad=8)

save_panel(fig, 'panel_03_partition_geometry')


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 4 — SO(3) Rotational Structure and Axis Exchange
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 5), facecolor='white')
fig.subplots_adjust(wspace=0.38)

theta_c = np.linspace(0, 2 * np.pi, 300)

# 4a — SO(2) orbit in xy-plane
ax = fig.add_subplot(141)
ax.plot(np.cos(theta_c), np.sin(theta_c), '-', color=C[2], lw=2)
ax.scatter([1, 0, -1, 0], [0, 1, 0, -1], c=[C[4]]*4, s=50, zorder=5)
ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
ax.axvline(0, color='gray', lw=0.5, alpha=0.5)
ax.set_aspect('equal')
ax.set_xlabel('$x$', labelpad=4)
ax.set_ylabel('$y$', labelpad=4)
ax.set_title('$SO(2)$ Orbit', pad=8)

# 4b — SO(3) sphere with equator + meridians
ax = fig.add_subplot(142, projection='3d')
phi_g = np.linspace(0, np.pi, 40)
th_g  = np.linspace(0, 2 * np.pi, 40)
Ph, Th = np.meshgrid(phi_g, th_g)
Xs = np.sin(Ph) * np.cos(Th)
Ys = np.sin(Ph) * np.sin(Th)
Zs = np.cos(Ph)
ax.plot_surface(Xs, Ys, Zs, alpha=0.06, color=C[2], linewidth=0)
ax.plot(np.cos(theta_c), np.sin(theta_c), np.zeros_like(theta_c), color=C[4], lw=1.8)
ax.plot(np.cos(theta_c), np.zeros_like(theta_c), np.sin(theta_c), color=C[0], lw=1.2, alpha=0.7)
ax.plot(np.zeros_like(theta_c), np.cos(theta_c), np.sin(theta_c), color=C[2], lw=1.0, alpha=0.5)
ax.set_xlabel('$x$', fontsize=7, labelpad=1)
ax.set_ylabel('$y$', fontsize=7, labelpad=1)
ax.set_zlabel('$z$', fontsize=7, labelpad=1)
ax.set_title('$SO(3)$ Orbit Space', pad=8)
ax.tick_params(labelsize=6)

# 4c — 3D axis exchange (x,y,z) -> (y,x,-z)  [det = +1, proper rotation]
ax = fig.add_subplot(143, projection='3d')
def quiv3(ax, dx, dy, dz, col, a=1.0):
    ax.quiver(0, 0, 0, dx, dy, dz, color=col,
              arrow_length_ratio=0.18, lw=2, alpha=a)
# Original frame
quiv3(ax, 1, 0,  0, C[4])
quiv3(ax, 0, 1,  0, C[0])
quiv3(ax, 0, 0,  1, C[2])
# Exchanged frame (dashed-style: lower alpha)
quiv3(ax, 0, 1,  0, C[4], 0.38)
quiv3(ax, 1, 0,  0, C[0], 0.38)
quiv3(ax, 0, 0, -1, C[2], 0.38)
ax.set_xlim(-1.3, 1.3);  ax.set_ylim(-1.3, 1.3);  ax.set_zlim(-1.3, 1.3)
ax.set_xlabel('$x$', fontsize=7, labelpad=1)
ax.set_ylabel('$y$', fontsize=7, labelpad=1)
ax.set_zlabel('$z$', fontsize=7, labelpad=1)
ax.set_title('Axis Exchange $\\in SO(3)$', pad=8)
ax.tick_params(labelsize=6)

# 4d — m-label under SO(2) vs axis-exchange SO(3) relabeling
ax = fig.add_subplot(144)
m_in = np.arange(-4, 5)
ax.plot(m_in, m_in,   'o-', color=C[0], lw=2, ms=5)   # SO(2): m invariant
ax.plot(m_in, -m_in,  's--', color=C[4], lw=2, ms=5)  # SO(3) exchange: m -> -m
ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
ax.axvline(0, color='gray', lw=0.5, alpha=0.5)
ax.set_xlabel('$m$ (original)', labelpad=4)
ax.set_ylabel('$m$ (relabeled)', labelpad=4)
ax.set_title('$m$-Label Transformation', pad=8)

save_panel(fig, 'panel_04_SO3_symmetry')


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 5 — Cyclic Categorical Closure
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 5), facecolor='white')
fig.subplots_adjust(wspace=0.38)

Mmax = 50
states50 = [phi(M) for M in range(1, Mmax + 1)]
n50 = np.array([s[0] for s in states50])
l50 = np.array([s[1] for s in states50])
m50 = np.array([s[2] for s in states50])

# 5a — cyclic successor on circle (M=1..Mmax)
ax = fig.add_subplot(141)
ang50 = np.linspace(0, 2 * np.pi, Mmax, endpoint=False)
cx50 = np.cos(ang50);  cy50 = np.sin(ang50)
for i in range(Mmax):
    j = (i + 1) % Mmax
    ax.annotate('', xy=(cx50[j], cy50[j]), xytext=(cx50[i], cy50[i]),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.35))
ax.scatter(cx50, cy50, c=np.arange(Mmax), cmap='plasma', s=28, zorder=5, linewidths=0)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Cyclic Successor $\\sigma$', pad=8)

# 5b — level traversal n(M)
ax = fig.add_subplot(142)
ax.fill_between(range(1, Mmax + 1), n50, alpha=0.18, color=C[2])
ax.step(range(1, Mmax + 1), n50, '-', color=C[2], lw=1.6, where='post')
ax.set_xlabel('$M$', labelpad=4)
ax.set_ylabel('$n(M)$', labelpad=4)
ax.set_title('Level Traversal', pad=8)

# 5c — 3D trajectory in (n, l, m) state space
ax = fig.add_subplot(143, projection='3d')
ax.plot(n50, l50, m50, '-', color=C[3], lw=0.9, alpha=0.55)
ax.scatter(n50, l50, m50, c=np.arange(Mmax), cmap='plasma', s=14, alpha=0.85, linewidths=0)
ax.set_xlabel('$n$', fontsize=7, labelpad=2)
ax.set_ylabel('$\\ell$', fontsize=7, labelpad=2)
ax.set_zlabel('$m$', fontsize=7, labelpad=2)
ax.set_title('Trajectory in $(n,\\ell,m)$', pad=8)
ax.tick_params(labelsize=6)

# 5d — level sizes C_3(n) = 2n^2
ax = fig.add_subplot(144)
ns_bar = np.arange(1, 16)
lvl_sizes = np.array([C3(n) for n in ns_bar])
ax.bar(ns_bar, lvl_sizes, color=[plasma(i / 14) for i in range(15)], alpha=0.85)
ax.set_xlabel('$n$', labelpad=4)
ax.set_ylabel('$C_3(n) = 2n^2$', labelpad=4)
ax.set_title('Level Sizes $2n^2$', pad=8)

save_panel(fig, 'panel_05_cyclic_closure')


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 6 — Negation–Identity Duality
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 5), facecolor='white')
fig.subplots_adjust(wspace=0.38)

total = len(full_states)

# 6a — |N(Phi(M))| for each M (constant = total-1)
ax = fig.add_subplot(141)
m_plot = np.arange(1, M_max + 1)
neg_sz = np.full(M_max, total - 1)
ax.fill_between(m_plot, neg_sz, alpha=0.15, color=C[3])
ax.plot(m_plot, neg_sz, '-', color=C[3], lw=1.8)
ax.scatter([1], [total - 1], color=C[4], s=60, zorder=6)
ax.set_xlabel('$M$', labelpad=4)
ax.set_ylabel('$|\\mathcal{N}(\\Phi(M))|$', labelpad=4)
ax.set_title('Negation Sequence Size', pad=8)

# 6b — partition space: P = Phi(1) highlighted, rest = N(P)
ax = fig.add_subplot(142)
is_ground = (fs_n == 1) & (fs_l == 0) & (fs_m == 0) & (fs_s == 0.5)
ax.scatter(fs_n[~is_ground], fs_l[~is_ground], c=[C[2]], s=10, alpha=0.45, linewidths=0)
ax.scatter(fs_n[is_ground],  fs_l[is_ground],  c=[C[4]], s=70, zorder=6, linewidths=0)
ax.set_xlabel('$n$', labelpad=4)
ax.set_ylabel('$\\ell$', labelpad=4)
ax.set_title('$\\Phi(1)$ vs $\\mathcal{N}(\\Phi(1))$', pad=8)

# 6c — 3D: full partition space, ground state marked
ax = fig.add_subplot(143, projection='3d')
ax.scatter(fs_n[~is_ground], fs_l[~is_ground], fs_m[~is_ground],
           c=[C[2]], s=9, alpha=0.45, linewidths=0)
ax.scatter(fs_n[is_ground], fs_l[is_ground], fs_m[is_ground],
           c=[C[4]], s=60, zorder=6, linewidths=0)
ax.set_xlabel('$n$', fontsize=7, labelpad=2)
ax.set_ylabel('$\\ell$', fontsize=7, labelpad=2)
ax.set_zlabel('$m$', fontsize=7, labelpad=2)
ax.set_title('Negation Volume', pad=8)
ax.tick_params(labelsize=6)

# 6d — negation density |N| / |Parts_n| converges to 1 as n_max grows
ax = fig.add_subplot(144)
n_maxv = np.arange(1, 16)
cum = np.array([N_state(n) for n in n_maxv])
density = (cum - 1) / cum
ax.plot(n_maxv, density, 'o-', color=C[4], lw=2, ms=5)
ax.axhline(1.0, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.set_xlabel('$n_{\\max}$', labelpad=4)
ax.set_ylabel('$|\\mathcal{N}| / |\\mathcal{P}|$', labelpad=4)
ax.set_ylim(0, 1.08)
ax.set_title('Negation Density', pad=8)

save_panel(fig, 'panel_06_negation_identity')

print('\nAll 6 panels complete. Output:', OUT)
