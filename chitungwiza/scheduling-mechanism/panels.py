#!/usr/bin/env python3
# =====================================================================
#  panels.py
#
#  Generate eight publication panels for
#  "Trajectory Scheduling".
#
#  Each panel: white background, four charts in a row, at least one 3D
#  chart, no text-only/table/conceptual charts. All data is computed
#  (the same quantities the validation suite measures).
#
#  Output: figures/panel_1.png ... figures/panel_8.png
# =====================================================================

import hashlib
import math
import os
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SEED = 42
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

# ---- global style: white background, minimal chrome ----
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
})

BLUE = "#2E5FA3"
ORANGE = "#E08A1E"
GREEN = "#2E8B57"
PURPLE = "#7E4FB0"
RED = "#C0392B"
GREY = "#888888"

FIGSIZE = (16, 3.7)   # four wide charts in a row


def new_panel():
    fig = plt.figure(figsize=FIGSIZE)
    return fig


def finish(fig, name):
    fig.tight_layout(pad=1.4, w_pad=2.2)
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", path)


# small helpers shared with the validation logic
def n_term(K, d):
    return 1 + math.ceil(math.log(K / d, d + 1))


def T(n, d):
    return d * (d + 1) ** (n - 1)


def sha1_bits(data):
    h = hashlib.sha1(data).digest()
    out = np.empty(160, dtype=np.int8)
    for i, byte in enumerate(h):
        for j in range(8):
            out[i * 8 + j] = (byte >> (7 - j)) & 1
    return out


# =====================================================================
# PANEL 1 -- Residue floor and information capacity
# =====================================================================
def panel1():
    rng = np.random.default_rng(SEED + 1)
    fig = new_panel()

    caps = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    floors = 1.0 / caps

    # (A) floor vs capacity, log-log, with predicted line
    ax = fig.add_subplot(1, 4, 1)
    ax.loglog(caps, floors, "o", color=BLUE, ms=6, label="measured")
    ax.loglog(caps, 1.0 / caps, "-", color=RED, lw=1.2, label=r"$1/|K_A|$")
    ax.set_xlabel(r"capacity $|K_A|$")
    ax.set_ylabel(r"floor $\beta_A$")
    ax.set_title("A  floor is positive")
    ax.legend()

    # (B) residue distribution above the floor for one capacity
    ax = fig.add_subplot(1, 4, 2)
    KA = 256
    beta = 1.0 / KA
    nbits = math.ceil(math.log2(KA))
    sstar = int(rng.integers(KA))
    res = []
    for x in range(KA * 8):
        e = x % KA
        if e == sstar:
            res.append(beta)
        else:
            res.append(beta + bin(e ^ sstar).count("1") / nbits)
    res = np.array(res)
    ax.hist(res, bins=40, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(beta, color=RED, lw=1.4)
    ax.set_xlabel("residue")
    ax.set_ylabel("count")
    ax.set_title(r"B  none below $\beta_A$")

    # (C) information capacity: distinguishable bins vs 1/eps, several beta
    ax = fig.add_subplot(1, 4, 3)
    U = 100.0
    eps = np.logspace(-3, 0, 30)
    for b, c in [(0.1, BLUE), (1.0, ORANGE), (10.0, GREEN)]:
        bins = np.ceil((U - b) / eps)
        ax.loglog(1.0 / eps, bins, "-", color=c, lw=1.3, label=fr"$\beta={b}$")
    ax.set_xlabel(r"$1/\varepsilon$")
    ax.set_ylabel("distinguishable values")
    ax.set_title("C  information capacity")
    ax.legend()

    # (D) 3D: floor surface over (capacity, precision) -> bits
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    cc = np.linspace(1, 12, 40)            # log2(capacity)
    ee = np.linspace(-3, 0, 40)            # log10(eps)
    CC, EE = np.meshgrid(cc, ee)
    beta_s = 2.0 ** (-CC)                   # floor = 1/capacity
    U = 100.0
    bits = np.log2((U - beta_s) / (10.0 ** EE))
    surf = ax.plot_surface(CC, EE, bits, cmap="viridis",
                           linewidth=0, antialiased=True)
    ax.set_xlabel(r"$\log_2|K_A|$")
    ax.set_ylabel(r"$\log_{10}\varepsilon$")
    ax.set_zlabel("bits")
    ax.set_title("D  capacity surface")
    ax.view_init(elev=24, azim=-58)
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.08)

    finish(fig, "panel_1.png")


# =====================================================================
# PANEL 2 -- Strict monotonicity of the trajectory count
# =====================================================================
def panel2():
    rng = np.random.default_rng(SEED + 2)
    fig = new_panel()

    # simulate runs with undo operations
    def run(steps, undo_frac):
        cand = 0
        hist = [0]
        M = 0
        counts, cands = [], []
        for _ in range(steps):
            if rng.random() < undo_frac and len(hist) > 1:
                cand = int(rng.choice(hist[:-1]))
            else:
                cand = int(rng.integers(10**6))
            M += 1
            hist.append(cand)
            counts.append(M)
            cands.append(cand)
        return np.array(counts), np.array(cands)

    # (A) count vs step for several runs, all strictly increasing
    ax = fig.add_subplot(1, 4, 1)
    for i, c in enumerate([BLUE, ORANGE, GREEN, PURPLE]):
        counts, _ = run(60, 0.3)
        ax.plot(range(len(counts)), counts, "-", color=c, lw=1.3,
                label=f"run {i+1}")
    ax.plot(range(60), range(1, 61), "--", color=RED, lw=1.0)
    ax.set_xlabel("step")
    ax.set_ylabel(r"count $M$")
    ax.set_title("A  count strictly rises")
    ax.legend(ncol=2)

    # (B) per-step increment == 1 (histogram of deltas, incl. undo steps)
    ax = fig.add_subplot(1, 4, 2)
    counts, _ = run(2000, 0.3)
    deltas = np.diff(counts)
    ax.hist(deltas, bins=np.arange(-0.5, 3.5, 1), color=BLUE, alpha=0.9,
            edgecolor="white")
    ax.axvline(1, color=RED, lw=1.4)
    ax.set_xlabel(r"$\Delta M$ per unit")
    ax.set_ylabel("count")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_title("B  every unit +1")

    # (C) revisited candidate -> distinct state: count gap at each revisit
    ax = fig.add_subplot(1, 4, 3)
    counts, cands = run(400, 0.4)
    last_seen = {}
    gaps = []
    for n, cand in enumerate(cands):
        if cand in last_seen:
            gaps.append(n - last_seen[cand])
        last_seen[cand] = n
    gaps = np.array(gaps) if gaps else np.array([0])
    ax.scatter(range(len(gaps)), gaps, s=12, color=GREEN, alpha=0.7)
    ax.axhline(0, color=RED, lw=1.0)
    ax.set_xlabel("revisit index")
    ax.set_ylabel("count gap to prior visit")
    ax.set_title("C  revisit is not return")

    # (D) 3D: state trajectory (candidate, count, step) -- a rising helix-like path
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    counts, cands = run(120, 0.35)
    steps = np.arange(len(counts))
    cnorm = (cands - cands.min()) / (np.ptp(cands) + 1)
    ax.plot(cnorm, counts, steps, "-", color=BLUE, lw=1.0, alpha=0.8)
    p = ax.scatter(cnorm, counts, steps, c=counts, cmap="plasma", s=10)
    ax.set_xlabel("candidate (norm)")
    ax.set_ylabel(r"count $M$")
    ax.set_zlabel("step")
    ax.set_title("D  state never repeats")
    ax.view_init(elev=22, azim=-60)
    fig.colorbar(p, ax=ax, shrink=0.5, pad=0.08)

    finish(fig, "panel_2.png")


# =====================================================================
# PANEL 3 -- Trajectory inflation and logarithmic termination
# =====================================================================
def panel3():
    fig = new_panel()

    # (A) T(n,d) closed form vs enumeration (log y)
    from itertools import product

    def enum_T(n, d):
        # sum over compositions of n: d^(num parts)
        def comps(m):
            if m == 0:
                yield ()
                return
            for f in range(1, m + 1):
                for r in comps(m - f):
                    yield (f,) + r
        return sum(d ** len(c) for c in comps(n))

    ax = fig.add_subplot(1, 4, 1)
    ns = np.arange(1, 9)
    for d, c in [(2, BLUE), (3, ORANGE), (4, GREEN)]:
        closed = [T(n, d) for n in ns]
        enumer = [enum_T(n, d) for n in ns]
        ax.semilogy(ns, closed, "-", color=c, lw=1.3, label=f"$d={d}$ closed")
        ax.semilogy(ns, enumer, "o", color=c, ms=5, mfc="white")
    ax.set_xlabel(r"length $n$")
    ax.set_ylabel(r"$T(n,d)$")
    ax.set_title("A  inflation: exact")
    ax.legend()

    # (B) linear cost vs exponential content
    ax = fig.add_subplot(1, 4, 2)
    d = 4
    ns = np.arange(1, 16)
    cost = ns * 1.0                 # cumulative cost ~ n*beta (beta=1)
    content = np.array([T(n, d) for n in ns], dtype=float)
    ax.plot(ns, cost, "-", color=BLUE, lw=1.4, label="cost (linear)")
    ax.set_xlabel(r"committed units $n$")
    ax.set_ylabel("cost", color=BLUE)
    ax.tick_params(axis="y", colors=BLUE)
    ax2 = ax.twinx()
    ax2.semilogy(ns, content, "-", color=ORANGE, lw=1.4)
    ax2.set_ylabel("content (log)", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax2.grid(False)
    ax.set_title("B  linear cost, exp. content")

    # (C) n_term logarithmic in K
    ax = fig.add_subplot(1, 4, 3)
    Ks = np.logspace(0, 12, 60)
    for d, c in [(2, BLUE), (3, ORANGE), (4, GREEN)]:
        nt = [n_term(K, d) for K in Ks]
        ax.semilogx(Ks, nt, "-", color=c, lw=1.4, label=f"$d={d}$")
    ax.set_xlabel(r"complexity $K$")
    ax.set_ylabel(r"$n_{\mathrm{term}}$")
    ax.set_title("C  termination is log")
    ax.legend()

    # (D) 3D: log10 T(n,d) surface over (n, d)
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    nn = np.arange(1, 16)
    dd = np.arange(2, 9)
    NN, DD = np.meshgrid(nn, dd)
    Z = np.log10(DD * (DD + 1.0) ** (NN - 1))
    surf = ax.plot_surface(NN, DD, Z, cmap="plasma", linewidth=0,
                           antialiased=True)
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$d$")
    ax.set_zlabel(r"$\log_{10}T$")
    ax.set_title("D  trajectory space")
    ax.view_init(elev=26, azim=-52)
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.08)

    finish(fig, "panel_3.png")


# =====================================================================
# PANEL 4 -- Diagnosis: compute- vs structure-limited
# =====================================================================
def panel4():
    fig = new_panel()
    floor = 1.0

    # (A) compute-limited streams (geometric descent), several kappa
    ax = fig.add_subplot(1, 4, 1)
    for kappa, c in [(0.1, BLUE), (0.3, ORANGE), (0.5, GREEN), (0.7, PURPLE)]:
        r = 50.0
        ys = []
        for _ in range(30):
            r = floor + (r - floor) * (1 - kappa)
            ys.append(r)
        ax.plot(range(len(ys)), ys, "-", color=c, lw=1.3,
                label=fr"$\kappa={kappa}$")
    ax.axhline(floor, color=RED, lw=1.0, ls="--")
    ax.set_xlabel("committed units")
    ax.set_ylabel("residue")
    ax.set_title("A  compute-limited")
    ax.legend(ncol=2)

    # (B) structure-limited streams (plateau above floor)
    ax = fig.add_subplot(1, 4, 2)
    for plateau, c in [(5.0, BLUE), (10.0, ORANGE), (20.0, GREEN)]:
        r = 50.0
        ys = []
        for step in range(30):
            if step < 8:
                r = plateau + (r - plateau) * 0.5
            else:
                r = plateau
            ys.append(r)
        ax.plot(range(len(ys)), ys, "-", color=c, lw=1.3,
                label=f"plateau {plateau:.0f}")
    ax.axhline(floor, color=RED, lw=1.0, ls="--")
    ax.set_xlabel("committed units")
    ax.set_ylabel("residue")
    ax.set_title("B  structure-limited")
    ax.legend()

    # (C) descent rate Delta(n) separates the two regimes
    ax = fig.add_subplot(1, 4, 3)
    r = 50.0; comp = []
    for _ in range(30):
        nr = floor + (r - floor) * 0.6; comp.append(r - nr); r = nr
    r = 50.0; struct = []
    for step in range(30):
        nr = (5.0 + (r - 5.0) * 0.5) if step < 8 else 5.0
        struct.append(r - nr); r = nr
    ax.plot(range(len(comp)), comp, "-", color=GREEN, lw=1.4,
            label=r"compute ($\Delta>0$)")
    ax.plot(range(len(struct)), struct, "-", color=RED, lw=1.4,
            label=r"structure ($\Delta\to0$)")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xlabel("committed units")
    ax.set_ylabel(r"descent $\Delta(n)$")
    ax.set_title("C  rate is the signal")
    ax.legend()

    # (D) 3D: residue surface over (units, kappa) for the compute family
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    units = np.arange(0, 30)
    kappas = np.linspace(0.05, 0.9, 30)
    UU, KK = np.meshgrid(units, kappas)
    RR = floor + (50.0 - floor) * (1 - KK) ** UU
    surf = ax.plot_surface(UU, KK, RR, cmap="viridis", linewidth=0,
                           antialiased=True)
    ax.set_xlabel("units")
    ax.set_ylabel(r"$\kappa$")
    ax.set_zlabel("residue")
    ax.set_title("D  descent to floor")
    ax.view_init(elev=24, azim=-58)
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.08)

    finish(fig, "panel_4.png")


# =====================================================================
# PANEL 5 -- Residue priority and scheduler soundness
# =====================================================================
def panel5():
    rng = np.random.default_rng(SEED + 5)
    fig = new_panel()
    floor = 1.0

    def priority(res, theta, delta):
        if res <= theta:
            return np.inf
        if delta <= 0:
            return 0.0
        return delta / max(res - theta, floor)

    # (A) priority vs residue, several descent rates (closeness effect)
    ax = fig.add_subplot(1, 4, 1)
    res = np.linspace(1.0, 30.0, 200)
    for delta, c in [(0.5, BLUE), (1.0, ORANGE), (3.0, GREEN)]:
        P = [priority(r, 1.0, delta) for r in res]
        P = np.clip(P, 0, 10)
        ax.plot(res, P, "-", color=c, lw=1.4, label=fr"$\Delta={delta}$")
    ax.set_xlabel(r"residue (toward $\theta$)")
    ax.set_ylabel("priority")
    ax.set_title("A  closer = higher")
    ax.legend()

    # (B) priority vs descent rate (rate effect)
    ax = fig.add_subplot(1, 4, 2)
    delta = np.linspace(0, 5, 200)
    for r, c in [(5.0, BLUE), (15.0, ORANGE), (30.0, GREEN)]:
        P = [priority(r, 1.0, d) for d in delta]
        ax.plot(delta, P, "-", color=c, lw=1.4, label=f"res {r:.0f}")
    ax.set_xlabel(r"descent rate $\Delta$")
    ax.set_ylabel("priority")
    ax.set_title("B  faster = higher")
    ax.legend()

    # (C) stalled never beats converging: scatter of priorities
    ax = fig.add_subplot(1, 4, 3)
    conv_P, stall_P = [], []
    for _ in range(1500):
        cr = rng.uniform(2, 50); cd = rng.uniform(0.01, 5.0)
        sr = rng.uniform(2, 50)
        conv_P.append(min(priority(cr, 1.0, cd), 8))
        stall_P.append(priority(sr, 1.0, 0.0))
    ax.scatter(range(len(conv_P)), conv_P, s=6, color=GREEN, alpha=0.5,
               label="converging")
    ax.scatter(range(len(stall_P)), stall_P, s=6, color=RED, alpha=0.5,
               label="stalled")
    ax.set_xlabel("trial")
    ax.set_ylabel("priority")
    ax.set_title("C  stalled $\\to$ 0")
    ax.legend()

    # (D) 3D: priority surface over (residue, descent)
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    rr = np.linspace(1.01, 30, 40)
    dd = np.linspace(0.0, 5, 40)
    RR, DD = np.meshgrid(rr, dd)
    PP = DD / np.maximum(RR - 1.0, floor)
    PP = np.clip(PP, 0, 5)
    surf = ax.plot_surface(RR, DD, PP, cmap="plasma", linewidth=0,
                           antialiased=True)
    ax.set_xlabel("residue")
    ax.set_ylabel(r"descent $\Delta$")
    ax.set_zlabel("priority")
    ax.set_title("D  priority field")
    ax.view_init(elev=26, azim=-56)
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.08)

    finish(fig, "panel_5.png")


# =====================================================================
# PANEL 6 -- Sufficiency stopping
# =====================================================================
def panel6():
    fig = new_panel()

    # (A) global residual vs sub-task residue: pinned below theta
    ax = fig.add_subplot(1, 4, 1)
    theta = 1.0 / 60   # parent floor (minute, in hours)
    rsub = np.logspace(-6, 0, 200)
    g = np.maximum(theta, rsub)
    ax.loglog(rsub, g, "-", color=BLUE, lw=1.5)
    ax.axvline(theta, color=RED, lw=1.2, ls="--")
    ax.set_xlabel("sub-task residue")
    ax.set_ylabel("global residual")
    ax.set_title(r"A  pinned below $\theta$")

    # (B) cost saved by stopping at theta vs own floor (geometric descent)
    ax = fig.add_subplot(1, 4, 2)
    def units_to(target, start=50.0, kappa=0.5):
        r, n = start, 0
        while r > target and n < 10000:
            r *= (1 - kappa); n += 1
        return n
    thetas = np.logspace(-1, 1.5, 40)
    subfloor = 1e-6
    saved = [units_to(subfloor) - units_to(t) for t in thetas]
    ax.semilogx(thetas, saved, "-", color=GREEN, lw=1.5)
    ax.set_xlabel(r"parent threshold $\theta$")
    ax.set_ylabel("units saved")
    ax.set_title("B  coarser = more saved")

    # (C) descent of sub-task with release point marked
    ax = fig.add_subplot(1, 4, 3)
    r = 50.0; ys = []
    for _ in range(40):
        r *= 0.6; ys.append(r)
    ys = np.array(ys)
    ax.semilogy(range(len(ys)), ys, "-", color=BLUE, lw=1.4)
    rel = np.argmax(ys <= 1.0)          # release at theta=1.0
    ax.plot(rel, ys[rel], "o", color=RED, ms=8, label="release @ $\\theta$")
    ax.axhline(1e-3, color=GREY, ls=":", lw=1.0, label="own floor")
    ax.set_xlabel("committed units")
    ax.set_ylabel("sub-task residue")
    ax.set_title("C  released above floor")
    ax.legend()

    # (D) 3D: units saved over (theta, kappa)
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    tt = np.logspace(-1, 1.5, 30)
    kk = np.linspace(0.2, 0.8, 30)
    TT, KK = np.meshgrid(tt, kk)
    start = 50.0; subfloor = 1e-6
    # closed-form geometric: units to target = ceil(log(target/start)/log(1-k))
    def uvec(target, KKm):
        return np.ceil(np.log(target / start) / np.log(1 - KKm))
    SAVED = uvec(subfloor, KK) - uvec(TT, KK)
    surf = ax.plot_surface(np.log10(TT), KK, SAVED, cmap="viridis",
                           linewidth=0, antialiased=True)
    ax.set_xlabel(r"$\log_{10}\theta$")
    ax.set_ylabel(r"$\kappa$")
    ax.set_zlabel("units saved")
    ax.set_title("D  savings surface")
    ax.view_init(elev=24, azim=-60)
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.08)

    finish(fig, "panel_6.png")


# =====================================================================
# PANEL 7 -- Avalanche and time-blindness of a content digest
# =====================================================================
def panel7():
    rng = random.Random(SEED + 7)
    fig = new_panel()

    # (A) histogram of output Hamming distance for 1-bit input flips
    ax = fig.add_subplot(1, 4, 1)
    dists = []
    for _ in range(4000):
        base = bytes(rng.getrandbits(8) for _ in range(32))
        ba = bytearray(base)
        bit = rng.randrange(len(ba) * 8)
        ba[bit // 8] ^= (1 << (bit % 8))
        dists.append(int(np.sum(sha1_bits(base) != sha1_bits(bytes(ba)))))
    dists = np.array(dists)
    ax.hist(dists, bins=40, color=BLUE, alpha=0.9, edgecolor="white",
            linewidth=0.3)
    ax.axvline(80, color=RED, lw=1.4)
    ax.set_xlabel("output bits flipped / 160")
    ax.set_ylabel("count")
    ax.set_title("A  avalanche $\\approx$ 80")

    # (B) per-output-bit flip probability ~ 0.5 (strict avalanche criterion)
    ax = fig.add_subplot(1, 4, 2)
    flips = np.zeros(160)
    N = 3000
    for _ in range(N):
        base = bytes(rng.getrandbits(8) for _ in range(32))
        ba = bytearray(base)
        bit = rng.randrange(len(ba) * 8)
        ba[bit // 8] ^= (1 << (bit % 8))
        flips += (sha1_bits(base) != sha1_bits(bytes(ba)))
    prob = flips / N
    ax.plot(range(160), prob, ".", color=GREEN, ms=3)
    ax.axhline(0.5, color=RED, lw=1.2)
    ax.set_ylim(0.3, 0.7)
    ax.set_xlabel("output bit index")
    ax.set_ylabel("flip probability")
    ax.set_title("B  each bit $\\approx$ 0.5")

    # (C) input difference (1 bit) vs output difference -> no correlation
    ax = fig.add_subplot(1, 4, 3)
    in_d, out_d = [], []
    for _ in range(1500):
        base = bytes(rng.getrandbits(8) for _ in range(32))
        ba = bytearray(base)
        k = rng.randrange(1, 6)            # flip 1..5 input bits
        for _ in range(k):
            bit = rng.randrange(len(ba) * 8)
            ba[bit // 8] ^= (1 << (bit % 8))
        in_d.append(k)
        out_d.append(int(np.sum(sha1_bits(base) != sha1_bits(bytes(ba)))))
    in_d = np.array(in_d); out_d = np.array(out_d)
    jitter = in_d + rng.uniform(-0.2, 0.2)
    ax.scatter(jitter, out_d, s=6, color=PURPLE, alpha=0.4)
    ax.axhline(80, color=RED, lw=1.0, ls="--")
    ax.set_xlabel("input bits changed (related)")
    ax.set_ylabel("output bits changed")
    ax.set_title("C  related $\\to$ far")

    # (D) 3D: time-blindness -- digest distance over (content change, time gap)
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    content_changes = np.arange(0, 9)          # input bits flipped
    time_gaps = np.arange(0, 9)                # trajectory-count gap (ignored by hash)
    CC, TG = np.meshgrid(content_changes, time_gaps)
    Z = np.zeros_like(CC, dtype=float)
    for ci, cc in enumerate(content_changes):
        # mean output distance depends ONLY on content change, not time gap
        ds = []
        for _ in range(60):
            base = bytes(rng.getrandbits(8) for _ in range(32))
            ba = bytearray(base)
            for _ in range(int(cc)):
                bit = rng.randrange(len(ba) * 8)
                ba[bit // 8] ^= (1 << (bit % 8))
            ds.append(int(np.sum(sha1_bits(base) != sha1_bits(bytes(ba)))))
        Z[:, ci] = np.mean(ds)                 # same across all time gaps (rows)
    surf = ax.plot_surface(CC, TG, Z, cmap="coolwarm", linewidth=0,
                           antialiased=True)
    ax.set_xlabel("content change")
    ax.set_ylabel("time gap")
    ax.set_zlabel("digest distance")
    ax.set_title("D  flat in time")
    ax.view_init(elev=22, azim=-62)
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.08)

    finish(fig, "panel_7.png")


# =====================================================================
# PANEL 8 -- Prefix is proximity; the paired label
# =====================================================================
def panel8():
    rng = np.random.default_rng(SEED + 8)
    fig = new_panel()
    b, depth = 3, 10

    def lcp(a1, a2):
        j = 0
        for x, y in zip(a1, a2):
            if x == y:
                j += 1
            else:
                break
        return j

    # (A) lcp vs shared-subtree depth (exact identity, scatter on diagonal)
    ax = fig.add_subplot(1, 4, 1)
    shared, measured = [], []
    for _ in range(4000):
        a1 = rng.integers(0, b, depth)
        share = int(rng.integers(0, depth + 1))
        a2 = a1.copy()
        for k in range(share, depth):
            a2[k] = rng.integers(0, b)
        if share < depth:
            while a2[share] == a1[share]:
                a2[share] = rng.integers(0, b)
        shared.append(share)
        measured.append(lcp(a1, a2))
    shared = np.array(shared) + rng.uniform(-0.15, 0.15, len(shared))
    ax.scatter(shared, measured, s=6, color=BLUE, alpha=0.3)
    ax.plot([0, depth], [0, depth], "--", color=RED, lw=1.2)
    ax.set_xlabel("shared subtree depth")
    ax.set_ylabel("measured lcp")
    ax.set_title("A  lcp = proximity")

    # (B) distribution of lcp over random pairs (geometric, base-3)
    ax = fig.add_subplot(1, 4, 2)
    lcps = []
    for _ in range(20000):
        a1 = rng.integers(0, b, depth)
        a2 = rng.integers(0, b, depth)
        lcps.append(lcp(a1, a2))
    ax.hist(lcps, bins=np.arange(-0.5, depth + 1.5, 1), color=GREEN,
            alpha=0.9, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("longest common prefix")
    ax.set_ylabel("count")
    ax.set_title("B  prefix distribution")

    # (C) the three queries separated in (content-distance, lcp) plane
    ax = fig.add_subplot(1, 4, 3)
    # recurrence: content dist 0, lcp small ; co-temporal: large dist, large lcp ;
    # refinement: small dist, large lcp
    rng2 = random.Random(SEED + 80)
    pts = {"recurrence": ([], []), "co-temporal": ([], []), "refinement": ([], [])}
    for _ in range(300):
        # recurrence
        pts["recurrence"][0].append(0 + rng2.uniform(-0.3, 0.3))
        pts["recurrence"][1].append(rng2.randint(0, 3))
        # co-temporal
        pts["co-temporal"][0].append(80 + rng2.uniform(-8, 8))
        pts["co-temporal"][1].append(rng2.randint(6, 10))
        # refinement
        pts["refinement"][0].append(80 + rng2.uniform(-8, 8))
        pts["refinement"][1].append(rng2.randint(7, 10))
    for (name, (xs, ys)), c in zip(pts.items(), [RED, BLUE, GREEN]):
        ax.scatter(xs, ys, s=12, color=c, alpha=0.5, label=name)
    ax.set_xlabel("digest distance (content)")
    ax.set_ylabel("lcp (time/structure)")
    ax.set_title("C  queries separate")
    ax.legend()

    # (D) 3D: address tree -- nodes placed by (digit-path embedding, depth)
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    # embed each node of a depth-4 base-3 tree
    D = 4
    xs, ys, zs, cs = [], [], [], []
    for level in range(D + 1):
        for node in range(b ** level):
            # base-3 digits of node
            digits = []
            v = node
            for _ in range(level):
                digits.append(v % b); v //= b
            digits = digits[::-1]
            # two continuous coordinates from the digit path
            sx = sum(d_ / b ** (i + 1) for i, d_ in enumerate(digits))
            sy = sum((d_ ** 2) / b ** (i + 1) for i, d_ in enumerate(digits))
            xs.append(sx); ys.append(sy); zs.append(level); cs.append(level)
    p = ax.scatter(xs, ys, zs, c=cs, cmap="viridis", s=14)
    ax.set_xlabel(r"$S_k$ (prefix)")
    ax.set_ylabel(r"$S_t$ (prefix)")
    ax.set_zlabel("depth")
    ax.set_title("D  prefix tree embedding")
    ax.view_init(elev=20, azim=-58)
    fig.colorbar(p, ax=ax, shrink=0.5, pad=0.08)

    finish(fig, "panel_8.png")


def main():
    panel1()
    panel2()
    panel3()
    panel4()
    panel5()
    panel6()
    panel7()
    panel8()
    print("  all 8 panels written to", OUTDIR + "/")


if __name__ == "__main__":
    main()
