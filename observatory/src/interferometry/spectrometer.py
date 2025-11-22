"""
Figure 17: Spectrometer as Categorical Process
Demonstrates that the spectrometer exists only in categorical states during measurement
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, FancyBboxPatch
from matplotlib import patches
import matplotlib.gridspec as gridspec

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {
    'wrong': '#D32F2F',
    'correct': '#388E3C',
    'category': '#F18F01',
    'fft': '#7B1FA2'
}

if __name__ == "__main__":
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3,
                        height_ratios=[1, 1.2, 1])

    # ============================================================================
    # PANEL A: TRADITIONAL VIEW (WRONG)
    # ============================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')

    # Draw physical spectrometer
    spec_box = FancyBboxPatch((0.2, 0.3), 0.6, 0.4,
                            boxstyle="round,pad=0.05",
                            facecolor=colors['wrong'], alpha=0.3,
                            edgecolor=colors['wrong'], linewidth=3)
    ax1.add_patch(spec_box)

    # Add components
    ax1.text(0.5, 0.6, '🔬', ha='center', va='center', fontsize=60)
    ax1.text(0.5, 0.4, 'Physical\nSpectrometer', ha='center', va='center',
            fontsize=12, fontweight='bold')

    # Add properties (WRONG)
    properties = [
        '• Persistent object',
        '• Continuous existence',
        '• Fixed spatial location',
        '• Physical device'
    ]

    for i, prop in enumerate(properties):
        ax1.text(0.1, 0.15 - i*0.05, prop, ha='left', va='top',
                fontsize=10, color=colors['wrong'])

    # Add X mark
    ax1.text(0.05, 0.95, '❌', ha='left', va='top', fontsize=40,
            color=colors['wrong'])

    ax1.text(0.15, 0.95, 'INCORRECT VIEW', ha='left', va='top',
            fontsize=14, fontweight='bold', color=colors['wrong'])

    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_title('A. Traditional View (Wrong)',
                fontsize=14, fontweight='bold', pad=20,
                color=colors['wrong'])

    # ============================================================================
    # PANEL B: CATEGORICAL VIEW (CORRECT)
    # ============================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    # Draw categorical states sequence
    n_states = 5
    for i in range(n_states):
        x = 0.1 + i * 0.18

        # Category circle
        circle = Circle((x, 0.5), 0.06, color=colors['correct'],
                    alpha=0.6, zorder=3)
        ax2.add_patch(circle)
        ax2.text(x, 0.5, f'C_{i+1}', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white', zorder=4)

        # Arrow to next state
        if i < n_states - 1:
            arrow = FancyArrowPatch((x + 0.07, 0.5), (x + 0.11, 0.5),
                                arrowstyle='->', mutation_scale=15,
                                linewidth=2, color=colors['correct'])
            ax2.add_patch(arrow)

    # Add label
    ax2.text(0.5, 0.65, 'Sequence of Categorical States', ha='center',
            fontsize=11, fontweight='bold')

    # Add properties (CORRECT)
    properties = [
        '• Observation process',
        '• Discrete existence',
        '• Categorical space (no location)',
        '• Created by measurement'
    ]

    for i, prop in enumerate(properties):
        ax2.text(0.1, 0.35 - i*0.05, prop, ha='left', va='top',
                fontsize=10, color=colors['correct'])

    # Add checkmark
    ax2.text(0.05, 0.95, '✓', ha='left', va='top', fontsize=40,
            color=colors['correct'])

    ax2.text(0.15, 0.95, 'CORRECT VIEW', ha='left', va='top',
            fontsize=14, fontweight='bold', color=colors['correct'])

    # Add mathematical expression
    ax2.text(0.5, 0.05, r'$S(t) = \sum_i \delta(t - t_i) \times C_i$',
            ha='center', va='bottom', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_title('B. Categorical View (Correct)',
                fontsize=14, fontweight='bold', pad=20,
                color=colors['correct'])

    # ============================================================================
    # PANEL C: SINGLE SPECTROMETER, MULTIPLE LEVELS
    # ============================================================================
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')

    # Timeline
    timeline_y = 0.5
    ax3.plot([0.05, 0.95], [timeline_y, timeline_y], 'k-',
            linewidth=3, alpha=0.5)

    # Measurement events at different categorical states
    events = [
        {'t': 0.1, 'cat': 'C₁', 'level': 'Level 0\n(All molecules)', 'color': '#E53935'},
        {'t': 0.25, 'cat': 'C₂', 'level': 'Level 1\n(Slower subset)', 'color': '#FB8C00'},
        {'t': 0.4, 'cat': 'C₃', 'level': 'Level 2\n(Even slower)', 'color': '#FFB300'},
        {'t': 0.55, 'cat': 'C₄', 'level': 'Level 3\n(Slowest)', 'color': '#7CB342'},
        {'t': 0.7, 'cat': 'C₅', 'level': 'Level 4', 'color': '#039BE5'},
        {'t': 0.85, 'cat': 'C₆', 'level': 'Level 5', 'color': '#5E35B1'},
    ]

    for i, event in enumerate(events):
        t = event['t']
        cat = event['cat']
        level = event['level']
        color = event['color']

        # Event marker
        ax3.scatter([t], [timeline_y], s=300, color=color,
                zorder=5, marker='o', edgecolors='black', linewidths=2)

        # Category label
        ax3.text(t, timeline_y + 0.08, cat, ha='center', va='bottom',
                fontsize=12, fontweight='bold', color=color)

        # Level label
        ax3.text(t, timeline_y - 0.08, level, ha='center', va='top',
                fontsize=9, style='italic')

        # Time marker
        ax3.text(t, timeline_y - 0.22, f't_{i+1}', ha='center', va='top',
                fontsize=8, color='gray')

        # Existence indicator
        if i < len(events) - 1:
            # Exists during measurement
            ax3.plot([t, t], [timeline_y - 0.02, timeline_y + 0.02],
                    color=color, linewidth=4, alpha=0.8)

            # Doesn't exist between measurements
            t_next = events[i+1]['t']
            ax3.plot([t + 0.01, t_next - 0.01], [timeline_y, timeline_y],
                    'k--', linewidth=1, alpha=0.3)

    # Add labels
    ax3.text(0.5, 0.85, 'Spectrometer exists only at discrete measurement moments',
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    ax3.text(0.5, 0.15, 'Each categorical state = One cascade level',
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Add existence equation
    ax3.text(0.5, 0.05,
            r'$S(t) \neq 0 \Leftrightarrow \exists i: t = t_i$ (measurement moment)',
            ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.set_title('C. Single Spectrometer, Multiple Levels (Sequential Categorical States)',
                fontsize=14, fontweight='bold', pad=20)

    # ============================================================================
    # PANEL D: FFT RECONSTRUCTION
    # ============================================================================
    ax4 = fig.add_subplot(gs[2, :])

    # Generate FFT spectrum showing all cascade levels
    frequencies = np.linspace(0, 100, 1000)

    # Multiple peaks corresponding to different cascade levels
    peaks = [
        {'f': 10, 'A': 8000, 'label': 'C₁ (Level 0)', 'color': '#E53935'},
        {'f': 25, 'A': 4000, 'label': 'C₂ (Level 1)', 'color': '#FB8C00'},
        {'f': 40, 'A': 2000, 'label': 'C₃ (Level 2)', 'color': '#FFB300'},
        {'f': 55, 'A': 1000, 'label': 'C₄ (Level 3)', 'color': '#7CB342'},
        {'f': 70, 'A': 500, 'label': 'C₅ (Level 4)', 'color': '#039BE5'},
        {'f': 85, 'A': 250, 'label': 'C₆ (Level 5)', 'color': '#5E35B1'},
    ]

    # Build spectrum
    spectrum = np.zeros_like(frequencies)
    for peak in peaks:
        spectrum += peak['A'] * np.exp(-((frequencies - peak['f'])/2)**2)

    # Plot spectrum
    ax4.plot(frequencies, spectrum, linewidth=2, color=colors['fft'], alpha=0.7)
    ax4.fill_between(frequencies, spectrum, alpha=0.3, color=colors['fft'])

    # Mark peaks
    for peak in peaks:
        ax4.axvline(x=peak['f'], color=peak['color'], linestyle='--',
                alpha=0.6, linewidth=2)
        ax4.scatter([peak['f']], [peak['A']], s=200, color=peak['color'],
                zorder=5, edgecolors='black', linewidths=2)
        ax4.text(peak['f'], peak['A'] + 400, peak['label'],
                ha='center', fontsize=9, rotation=45,
                bbox=dict(boxstyle='round', facecolor=peak['color'],
                        alpha=0.3))

    ax4.set_xlabel('Frequency (THz)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    ax4.set_title('D. FFT Reconstruction (All Levels Simultaneously)',
                fontsize=14, fontweight='bold', pad=20)

    # Add explanation
    ax4.text(0.5, 0.95,
            'FFT spectrum contains all categorical states simultaneously\n'
            'Each peak = One cascade level (measured sequentially but reconstructed together)',
            transform=ax4.transAxes, ha='center', va='top',
            fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 100])
    ax4.set_ylim([0, 9000])

    # ============================================================================
    # OVERALL TITLE AND ANNOTATIONS
    # ============================================================================
    fig.suptitle('Spectrometer as Categorical Process: Existence Only in Measurement States',
                fontsize=16, fontweight='bold', y=0.99)

    # Add key insight box
    fig.text(0.5, 0.01,
            'KEY INSIGHT: The virtual spectrometer does not exist as a persistent physical device.\n'
            'It exists only in categorical states created during measurement. What we call "the spectrometer"\n'
            'is actually the observation process itself—a sequence of categorical completions.',
            ha='center', va='bottom', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5, pad=10))

    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    plt.savefig('figure_17_spectrometer_categorical_process.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 17 saved: figure_17_spectrometer_categorical_process.png")
    plt.close()
