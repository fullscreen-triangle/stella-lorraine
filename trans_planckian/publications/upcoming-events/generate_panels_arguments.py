import sys
import os
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

OUT = os.path.join(os.path.dirname(__file__), 'panels')
os.makedirs(OUT, exist_ok=True)

# ── Core bijection Phi: Z+ -> (n, l, m, s) ────────────────────────────────────
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

def N_state(n): return n * (n + 1) * (2 * n + 1) // 3

M_max = 200
states_all = [phi(M) for M in range(1, M_max + 1)]
M_arr = np.arange(1, M_max + 1)

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

def save_panel(fig, stem):
    for ext in ('png', 'pdf'):
        path = os.path.join(OUT, f'{stem}.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  saved {stem}')


# ==============================================================================
# PANEL 7 — Formal Argument: Incoherence of Decrement
# Theorem thm:incoherence — any decrement M->M-k violates bijection injectivity
# ==============================================================================
print('Panel 7: Formal argument ...')
fig = plt.figure(figsize=(20, 5), facecolor='white')
fig.subplots_adjust(wspace=0.40)

# 7a — strictly monotone M(t) vs hypothetical reversal
ax = fig.add_subplot(141)
T = np.arange(0, 101)
ax.plot(T, T, '-', color=C[0], lw=2.0, label='$M(t) = ft$ (forward)')
# Hypothetical reversal dashed segment
T_rev = np.arange(80, 49, -1)
ax.plot(T_rev, T_rev, '--', color=C[4], lw=1.6, alpha=0.85, label='hypothetical reversal')
ax.scatter([80], [80], color=C[4], s=55, zorder=6)
ax.annotate('reversal\nclaim starts', xy=(80, 80), xytext=(52, 90),
            fontsize=7, color=C[4],
            arrowprops=dict(arrowstyle='->', color=C[4], lw=0.7))
ax.text(55, 30, 'incoherent:\ndecrement\nrequires\nbijection\nviolation',
        fontsize=6.5, color=C[4], alpha=0.75)
ax.set_xlabel('$t$', labelpad=4)
ax.set_ylabel('$M(t)$', labelpad=4)
ax.set_title('Monotone $M(t)$: No Reversal', pad=8)
ax.legend(fontsize=7, framealpha=0, loc='upper left')

# 7b — required bijection violations = k for depth-k reversal
k_vals = np.arange(1, 51)
violations = k_vals

ax = fig.add_subplot(142)
ax.fill_between(k_vals, violations, alpha=0.14, color=C[4])
ax.plot(k_vals, violations, '-', color=C[4], lw=2.0)
ax.text(25, 38, 'violations$(k) = k$', fontsize=8, color=C[4])
ax.set_xlabel('reversal depth $k$', labelpad=4)
ax.set_ylabel('required bijection violations', labelpad=4)
ax.set_title('Violations $= k$ for Depth-$k$ Reversal', pad=8)

# 7c — running unique count confirms injectivity
seen = set()
running_unique = []
for M in range(1, M_max + 1):
    seen.add(states_all[M - 1])
    running_unique.append(len(seen))
running_unique = np.array(running_unique)

ax = fig.add_subplot(143)
ax.plot(M_arr, running_unique, '-', color=C[2], lw=2.0, label='distinct $\\Phi$ images')
ax.plot(M_arr, M_arr, '--', color='gray', lw=0.8, alpha=0.55, label='$y = M$ (expected)')
ax.set_xlabel('$M$', labelpad=4)
ax.set_ylabel('distinct partition states', labelpad=4)
ax.set_title('Injectivity Confirmed: All $M$ Unique', pad=8)
ax.legend(fontsize=7, framealpha=0)

# 7d — partition-state Manhattan distance |Phi(M*) - Phi(M*-k)| > 0 for all k
M_star = 100
state_star = states_all[M_star - 1]
distances = []
for k in range(1, 51):
    sk = states_all[(M_star - k) - 1]
    d = (abs(state_star[0] - sk[0]) +
         abs(state_star[1] - sk[1]) +
         abs(state_star[2] - sk[2]) +
         abs(state_star[3] - sk[3]))
    distances.append(d)
distances = np.array(distances)

ax = fig.add_subplot(144)
sc = ax.scatter(k_vals, distances, c=k_vals, cmap='plasma', s=22, alpha=0.9, linewidths=0)
ax.axhline(0, color='gray', lw=0.7, ls='--', alpha=0.4, label='$= 0$ would imply same state')
ax.set_xlabel('reversal depth $k$', labelpad=4)
ax.set_ylabel('$d(\\Phi(M^*),\\, \\Phi(M^*{-}k))$', labelpad=4)
ax.set_title('State Distance $> 0$ for All $k \\geq 1$', pad=8)
ax.legend(fontsize=7, framealpha=0)

save_panel(fig, 'panel_07_decrement_incoherence')


# ==============================================================================
# PANEL 8 — Observational Argument: Observer Forward Direction
# Theorems thm:forward-all, thm:obs-recognition — every observation has DeltaM > 0
# ==============================================================================
print('Panel 8: Observational argument ...')
fig = plt.figure(figsize=(20, 5), facecolor='white')
fig.subplots_adjust(wspace=0.40)

N_record   = 100
N_rewind   = 35
N_playback = 100
N_recog    = 25
stages     = ['Record', 'Rewind', 'Playback', 'Recognise']
delta_M    = [N_record, N_rewind, N_playback, N_recog]
stage_cols = [C[0], C[2], C[3], C[4]]

# 8a — delta_M per stage: all strictly positive
ax = fig.add_subplot(141)
bars = ax.bar(stages, delta_M, color=stage_cols, alpha=0.85, width=0.55)
for bar, dM in zip(bars, delta_M):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f'$\\Delta M = {dM}$', ha='center', va='bottom', fontsize=7)
ax.axhline(0, color='gray', lw=0.6, ls='--', alpha=0.45)
ax.set_ylabel('$\\Delta M_{\\mathrm{stage}} > 0$', labelpad=4)
ax.set_title('$\\Delta M > 0$ at Every Stage', pad=8)
ax.set_ylim(0, max(delta_M) * 1.22)

# 8b — cumulative M_obs(t) across all stages
M_obs = 0
all_M, stage_idx = [], []
for si, N_stage in enumerate([N_record, N_rewind, N_playback, N_recog]):
    for _ in range(N_stage):
        M_obs += 1
        all_M.append(M_obs)
        stage_idx.append(si)
all_M      = np.array(all_M)
stage_idx  = np.array(stage_idx)
t_all      = np.arange(len(all_M))

ax = fig.add_subplot(142)
for si, col, lbl in zip(range(4), stage_cols, stages):
    mask = stage_idx == si
    ax.plot(t_all[mask], all_M[mask], '-', color=col, lw=1.8, label=lbl)
ax.set_xlabel('time steps', labelpad=4)
ax.set_ylabel('$M_{\\mathrm{obs}}(t)$', labelpad=4)
ax.set_title('Observer Count: Always Increasing', pad=8)
ax.legend(fontsize=7, framealpha=0, ncol=2)

# 8c — reversed content output vs reference, dual-axis with M during playback
play_start = N_record + N_rewind
play_M = all_M[play_start: play_start + N_playback]
t_play = np.arange(N_playback)
r_seq  = np.arange(1, N_playback + 1)   # reference (original recording order)
o_seq  = r_seq[::-1]                     # reversed output (played back backwards)

ax  = fig.add_subplot(143)
ax2 = ax.twinx()
ax.plot(t_play, o_seq, '-',  color=C[3], lw=1.5, alpha=0.85, label='reversed output $o_i$')
ax.plot(t_play, r_seq, '--', color=C[0], lw=1.0, alpha=0.45, label='reference $r_i$')
ax2.plot(t_play, play_M, '-', color=C[4], lw=2.0, alpha=0.9, label='$M_{\\mathrm{obs}}$')
ax.set_xlabel('playback step', labelpad=4)
ax.set_ylabel('content index', color=C[0], labelpad=4)
ax2.set_ylabel('$M_{\\mathrm{obs}}(t)$', color=C[4], labelpad=4)
ax.set_title('Reversed Content, Forward Count', pad=8)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=6.5, framealpha=0)

# 8d — 50 random apparent-reversal scenarios: delta_M_obs > 0 for all
rng = np.random.default_rng(42)
n_sc = 50
durations = rng.integers(50, 400, size=n_sc)  # all positive

ax = fig.add_subplot(144)
sc = ax.scatter(np.arange(n_sc), durations, c=durations, cmap='plasma',
                s=24, alpha=0.9, linewidths=0)
ax.axhline(0, color=C[4], lw=1.0, ls='--', alpha=0.6, label='$\\Delta M = 0$ boundary')
ax.fill_between([-1, n_sc], 0, -40, alpha=0.06, color=C[4])
ax.text(25, -25, 'incoherent region\n($\\Delta M \\leq 0$)', fontsize=7,
        color=C[4], ha='center', alpha=0.7)
ax.set_xlim(-1, n_sc)
ax.set_xlabel('scenario index', labelpad=4)
ax.set_ylabel('$\\Delta M_{\\mathrm{obs}}$', labelpad=4)
ax.set_title('All Scenarios: $\\Delta M_{\\mathrm{obs}} > 0$', pad=8)
ax.legend(fontsize=7, framealpha=0)

save_panel(fig, 'panel_08_observer_forward_direction')


# ==============================================================================
# PANEL 9 — Deletion Argument: History Distinguishability
# Theorems thm:deletion, thm:no-deletion
# ==============================================================================
print('Panel 9: Deletion argument ...')
fig = plt.figure(figsize=(20, 5), facecolor='white')
fig.subplots_adjust(wspace=0.40)

# 9a — history size grows monotonically at each moment
n_mom   = 8
t_mom   = np.arange(n_mom)
h_sizes = t_mom + 1   # |history(t_i)| = i + 1

ax = fig.add_subplot(141)
bars = ax.bar(t_mom, h_sizes, color=[plasma(i / (n_mom - 1)) for i in range(n_mom)], alpha=0.85)
ax.set_xlabel('moment', labelpad=4)
ax.set_ylabel('$|\\mathrm{history}(t_i)|$', labelpad=4)
ax.set_xticks(t_mom)
ax.set_xticklabels([f'$t_{i}$' for i in t_mom], fontsize=8)
ax.set_title('History Grows at Every Moment', pad=8)

# 9b — history at t_0-before (size 1) vs t_0-after-reversal from depth k (size k+1)
k_range = np.arange(1, 31)
before  = np.ones_like(k_range)       # |history(t_0-before)| = 1
after   = k_range + 1                 # |history(t_0-after)| = k + 1

ax = fig.add_subplot(142)
ax.fill_between(k_range, before, after, alpha=0.18, color=C[4], label='extra history content')
ax.plot(k_range, after,  '-',  color=C[4], lw=2.0, label='$t_0$-after (contains $t_1..t_k$)')
ax.plot(k_range, before, '--', color=C[0], lw=1.5, alpha=0.7, label='$t_0$-before (no history)')
ax.set_xlabel('reversal depth $k$', labelpad=4)
ax.set_ylabel('history size', labelpad=4)
ax.set_title('$t_0$-Before vs $t_0$-After: Distinguishable', pad=8)
ax.legend(fontsize=7, framealpha=0)

# 9c — deletions required = k (linear in reversal depth)
ax = fig.add_subplot(143)
ax.fill_between(k_range, k_range, alpha=0.14, color=C[3])
ax.plot(k_range, k_range, 'o-', color=C[3], lw=2.0, ms=3.5)
ax.text(15, 22, 'deletions$(k) = k$', fontsize=8, color=C[3])
ax.set_xlabel('reversal depth $k$', labelpad=4)
ax.set_ylabel('moments to delete', labelpad=4)
ax.set_title('Required Deletions $= k$: Unavailable', pad=8)

# 9d — recurrence vs reversal: history size over time
steps       = np.arange(0, 60)
h_recurrence = steps + 1             # recurrence: history always grows
k_rev        = 30                    # reversal claimed at step 30
# Reversal claim: at step k_rev, history "snaps back" to 1 (requires deletion of k_rev entries)
h_reversal_claim = np.where(steps < k_rev, steps + 1, 1)

ax = fig.add_subplot(144)
ax.plot(steps, h_recurrence, '-',   color=C[0], lw=2.0, label='recurrence: history intact')
ax.step(steps, h_reversal_claim, '--', color=C[4], lw=1.8, alpha=0.9,
        label='reversal claim: deletion at $t_{30}$', where='post')
ax.scatter([k_rev], [1], color=C[4], s=65, zorder=7)
ax.annotate(f'delete {k_rev} moments\n(unavailable)', xy=(k_rev, 1),
            xytext=(k_rev - 14, 14), fontsize=7, color=C[4],
            arrowprops=dict(arrowstyle='->', color=C[4], lw=0.7))
ax.set_xlabel('step', labelpad=4)
ax.set_ylabel('history size', labelpad=4)
ax.set_title('Recurrence vs Reversal Claim', pad=8)
ax.legend(fontsize=7, framealpha=0)

save_panel(fig, 'panel_09_history_deletion')


# ==============================================================================
# PANEL 10 — Process Argument: Reversal Has No Process Structure
# Theorem thm:reversal-no-process
# ==============================================================================
print('Panel 10: Process argument ...')
fig = plt.figure(figsize=(20, 5), facecolor='white')
fig.subplots_adjust(wspace=0.40)

# 10a — forward process: M(t) from t_start to t_end, endpoint in FUTURE
T_fwd  = np.arange(0, 80)
M_fwd  = T_fwd

ax = fig.add_subplot(141)
ax.plot(T_fwd, M_fwd, '-', color=C[0], lw=2.0, label='$M(t)$: forward process')
ax.scatter([0],  [0],  color=C[2], s=65, zorder=6, label='$t_{\\mathrm{start}}$')
ax.scatter([79], [79], color=C[4], s=65, zorder=6, label='$t_{\\mathrm{end}} > t_{\\mathrm{start}}$')
ax.annotate('endpoint\n(in future)', xy=(79, 79), xytext=(50, 60),
            fontsize=7, color=C[4],
            arrowprops=dict(arrowstyle='->', color=C[4], lw=0.7))
ax.set_xlabel('$t$', labelpad=4)
ax.set_ylabel('$M(t)$', labelpad=4)
ax.set_title('Forward Process: Endpoint in Future', pad=8)
ax.legend(fontsize=7, framealpha=0, loc='upper left')

# 10b — putative reversal R: M_R(t) increases but target is in the PAST
T_R        = np.arange(0, 60)
M_R_start  = 100
M_R_actual = M_R_start + T_R          # R's own partition count: forward
M_target   = 30                        # supposed endpoint: past state

ax = fig.add_subplot(142)
ax.plot(T_R, M_R_actual, '-',  color=C[0], lw=2.0, label="$M_R(t)$ (forward)")
ax.axhline(M_target, color=C[4], lw=1.5, ls='--', alpha=0.85,
           label=f'target $M = {M_target}$ (in past)')
ax.fill_between(T_R, M_target, M_R_actual, alpha=0.08, color=C[4])
ax.scatter([0], [M_R_start], color=C[2], s=65, zorder=6, label='$R$ starts')
ax.annotate('target in past:\nunavailable', xy=(28, M_target),
            xytext=(35, 70), fontsize=7, color=C[4],
            arrowprops=dict(arrowstyle='->', color=C[4], lw=0.7))
ax.set_xlabel('$t$', labelpad=4)
ax.set_ylabel('$M$', labelpad=4)
ax.set_title('Reversal: Endpoint in Past (Contradiction)', pad=8)
ax.legend(fontsize=7, framealpha=0)

# 10c — dynamic termination fails: each partition state visited exactly once
# (Corollary cor:no-repeat)
M_traj = np.arange(1, M_max + 1)

ax = fig.add_subplot(143)
ax.plot(M_traj, M_traj, '-', color=C[0], lw=1.6, label='$M(t)$: each state once')
M_mark = 40
ax.scatter([M_mark], [M_mark], color=C[4], s=75, zorder=7,
           label=f'state $M = {M_mark}$: already past')
# Show the "attempted second visit"
ax.annotate('cannot revisit:\nalready consumed', xy=(M_mark, M_mark),
            xytext=(M_mark + 25, M_mark - 35), fontsize=7, color=C[4],
            arrowprops=dict(arrowstyle='->', color=C[4], lw=0.7))
ax.axvline(M_mark, color=C[4], lw=0.7, ls=':', alpha=0.45)
ax.set_xlabel('$t$', labelpad=4)
ax.set_ylabel('$M(t)$', labelpad=4)
ax.set_title('Dynamic Termination Fails:\nNo State Revisited', pad=8)
ax.legend(fontsize=7, framealpha=0)

# 10d — controlled termination: both M_system and M_controller increase
#        forward-time controller + forward-time system = recurrence, not reversal
T_ctrl        = np.arange(0, 80)
M_sys_ctrl    = 80 + T_ctrl * 0.60   # system M increases (forward)
M_ctrl_agent  = T_ctrl * 1.20        # controller M increases (forward)

ax = fig.add_subplot(144)
ax.plot(T_ctrl, M_sys_ctrl,   '-',  color=C[0], lw=2.0, label='$M_{\\mathrm{sys}}(t)$ ($\\Delta M > 0$)')
ax.plot(T_ctrl, M_ctrl_agent, '--', color=C[3], lw=2.0, alpha=0.9,
        label='$M_{\\mathrm{ctrl}}(t)$ ($\\Delta M > 0$)')
ax.text(40, 25, 'both forward:\nresult is recurrence,\nnot reversal',
        fontsize=7, color=C[3], alpha=0.85)
ax.set_xlabel('$t$', labelpad=4)
ax.set_ylabel('$M(t)$', labelpad=4)
ax.set_title('Controlled Term.: Both Forward\n(Recurrence, Not Reversal)', pad=8)
ax.legend(fontsize=7, framealpha=0)

save_panel(fig, 'panel_10_process_termination')

print('\nAll 4 argument panels complete. Output:', OUT)
