"""
Reflectance Cascade Data Analysis and Visualization
Publication-quality figures for Nature/Science-tier journals

Author: Kundai Sachikonye
Date: 2025-11-21
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

# Nature journal style parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['figure.titlesize'] = 11

# Color scheme (Nature-compatible)
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Purple
    'accent': '#F18F01',       # Orange
    'success': '#06A77D',      # Green
    'planck': '#E63946',       # Red (for Planck limit)
    'grid': '#CCCCCC',         # Gray
}

# Physical constants
PLANCK_TIME = 5.391247e-44  # seconds
SPEED_OF_LIGHT = 299792458  # m/s

# ============================================================================
# DATA EXTRACTION
# ============================================================================

def extract_cascade_data(log_text):
    """
    Extract cascade performance data from console output.

    Parameters
    ----------
    log_text : str
        Raw console output from cascade experiment

    Returns
    -------
    dict
        Structured data containing:
        - system_params: Initial system configuration
        - cascade_results: Performance at each reflection depth
        - network_stats: Graph topology metrics
    """

    data = {
        'system_params': {},
        'cascade_results': [],
        'network_stats': {}
    }

    # Extract system parameters
    base_freq_match = re.search(r'Base frequency: ([\d.e+-]+) Hz', log_text)
    if base_freq_match:
        data['system_params']['base_frequency'] = float(base_freq_match.group(1))

    bmd_depth_match = re.search(r'BMD depth: (\d+) \(creates (\d+) parallel channels\)', log_text)
    if bmd_depth_match:
        data['system_params']['bmd_depth'] = int(bmd_depth_match.group(1))
        data['system_params']['bmd_channels'] = int(bmd_depth_match.group(2))

    nodes_match = re.search(r'Network nodes: (\d+)', log_text)
    if nodes_match:
        data['network_stats']['nodes'] = int(nodes_match.group(1))

    edges_match = re.search(r'Network edges: ([\d,]+)', log_text)
    if edges_match:
        data['network_stats']['edges'] = int(edges_match.group(1).replace(',', ''))

    avg_degree_match = re.search(r'Average degree: ([\d.]+)', log_text)
    if avg_degree_match:
        data['network_stats']['avg_degree'] = float(avg_degree_match.group(1))

    # Extract cascade results for each reflection depth
    # Handle both with and without INFO: prefixes
    cascade_blocks = re.findall(
        r'Starting cascade with (\d+) reflections?.*?'
        r'(?:INFO:[^:]+:)?Precision achieved: ([\d.e+-]+) s.*?'
        r'(?:INFO:[^:]+:)?Orders below Planck: ([-\d.]+).*?'
        r'(?:INFO:[^:]+:)?Total enhancement: ([\d.e+-]+)×',
        log_text,
        re.DOTALL
    )

    for reflections, precision, orders_below, enhancement in cascade_blocks:
        result = {
            'reflections': int(reflections),
            'precision': float(precision),
            'orders_below_planck': float(orders_below),
            'enhancement': float(enhancement)
        }

        # Extract per-reflection details
        reflection_pattern = r'--- Reflection (\d+)/' + reflections + r' ---.*?' \
                           r'(?:INFO:[^:]+:)?Cumulative frequency: ([\d.e+-]+) Hz'

        reflection_data = re.findall(reflection_pattern, log_text, re.DOTALL)
        result['per_reflection_freq'] = [
            (int(r[0]), float(r[1])) for r in reflection_data
        ]

        data['cascade_results'].append(result)

    return data


def parse_log_file(filepath):
    """Load and parse log file."""
    with open(filepath, 'r') as f:
        return extract_cascade_data(f.read())


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def power_law_model(x, a, b):
    """Power law: y = a * x^b"""
    return a * np.power(x, b)


def exponential_model(x, a, b):
    """Exponential: y = a * exp(b*x)"""
    return a * np.exp(b * x)


def analyze_scaling(data):
    """
    Analyze scaling behavior of cascade performance.

    Returns
    -------
    dict
        Fitted parameters and goodness-of-fit metrics
    """

    results = data['cascade_results']

    reflections = np.array([r['reflections'] for r in results])
    precision = np.array([r['precision'] for r in results])
    enhancement = np.array([r['enhancement'] for r in results])

    analysis = {}

    # Fit precision scaling (should be exponential or power law)
    try:
        # Try power law fit
        popt_power, pcov_power = curve_fit(
            power_law_model,
            reflections,
            precision,
            p0=[precision[0], -2],
            maxfev=10000
        )

        # Calculate R² for power law
        residuals_power = precision - power_law_model(reflections, *popt_power)
        ss_res_power = np.sum(residuals_power**2)
        ss_tot = np.sum((precision - np.mean(precision))**2)
        r2_power = 1 - (ss_res_power / ss_tot)

        analysis['precision_power_law'] = {
            'params': popt_power,
            'covariance': pcov_power,
            'r_squared': r2_power,
            'formula': f'τ = {popt_power[0]:.2e} × n^{popt_power[1]:.2f}'
        }
    except:
        analysis['precision_power_law'] = None

    # Try exponential fit
    try:
        popt_exp, pcov_exp = curve_fit(
            exponential_model,
            reflections,
            np.log(precision),  # Log transform for exponential
            p0=[np.log(precision[0]), -1],
            maxfev=10000
        )

        # Convert back from log space
        pred_exp = np.exp(exponential_model(reflections, *popt_exp))
        residuals_exp = precision - pred_exp
        ss_res_exp = np.sum(residuals_exp**2)
        r2_exp = 1 - (ss_res_exp / ss_tot)

        analysis['precision_exponential'] = {
            'params': popt_exp,
            'covariance': pcov_exp,
            'r_squared': r2_exp,
            'formula': f'τ = exp({popt_exp[0]:.2f} + {popt_exp[1]:.2f}×n)'
        }
    except:
        analysis['precision_exponential'] = None

    # Fit enhancement scaling (should be linear or power law)
    try:
        popt_enh, pcov_enh = curve_fit(
            power_law_model,
            reflections,
            enhancement,
            p0=[enhancement[0], 1],
            maxfev=10000
        )

        residuals_enh = enhancement - power_law_model(reflections, *popt_enh)
        ss_res_enh = np.sum(residuals_enh**2)
        ss_tot_enh = np.sum((enhancement - np.mean(enhancement))**2)
        r2_enh = 1 - (ss_res_enh / ss_tot_enh)

        analysis['enhancement_scaling'] = {
            'params': popt_enh,
            'covariance': pcov_enh,
            'r_squared': r2_enh,
            'formula': f'E = {popt_enh[0]:.2e} × n^{popt_enh[1]:.2f}'
        }
    except:
        analysis['enhancement_scaling'] = None

    # Calculate per-reflection improvement factor
    if len(reflections) > 1:
        precision_ratios = precision[:-1] / precision[1:]
        analysis['avg_improvement_per_reflection'] = np.mean(precision_ratios)
        analysis['std_improvement_per_reflection'] = np.std(precision_ratios)

    return analysis


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_figure_1_precision_scaling(data, analysis, save_path='figure1_precision_scaling.pdf'):
    """
    Figure 1: Precision Scaling with Reflection Depth

    Multi-panel figure showing:
    A) Precision vs reflections (log scale)
    B) Enhancement vs reflections
    C) Orders below Planck time
    D) Per-reflection frequency accumulation
    """

    fig = plt.figure(figsize=(7.2, 6))  # 180mm width (Nature full page)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    results = data['cascade_results']
    reflections = np.array([r['reflections'] for r in results])
    precision = np.array([r['precision'] for r in results])
    enhancement = np.array([r['enhancement'] for r in results])
    orders_below = np.array([r['orders_below_planck'] for r in results])

    # ========================================================================
    # Panel A: Precision vs Reflections (Log Scale)
    # ========================================================================
    ax_a = fig.add_subplot(gs[0, 0])

    # Plot data points
    ax_a.scatter(reflections, precision, s=50, color=COLORS['primary'],
                 zorder=3, label='Measured', edgecolors='white', linewidth=1)

    # Plot Planck time limit
    ax_a.axhline(PLANCK_TIME, color=COLORS['planck'], linestyle='--',
                 linewidth=1.5, label='Planck time', zorder=2)

    # Plot fitted curve if available
    if analysis.get('precision_power_law'):
        fit_x = np.linspace(reflections.min(), reflections.max(), 100)
        fit_y = power_law_model(fit_x, *analysis['precision_power_law']['params'])
        ax_a.plot(fit_x, fit_y, color=COLORS['secondary'], linestyle='-',
                  linewidth=1.5, label='Power law fit', zorder=1, alpha=0.7)

        # Add R² annotation
        r2 = analysis['precision_power_law']['r_squared']
        ax_a.text(0.05, 0.95, f'$R^2$ = {r2:.4f}',
                  transform=ax_a.transAxes, fontsize=7,
                  verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax_a.set_yscale('log')
    ax_a.set_xlabel('Number of Reflections', fontweight='bold')
    ax_a.set_ylabel('Time Precision (s)', fontweight='bold')
    ax_a.set_title('A. Precision Scaling', fontweight='bold', loc='left')
    ax_a.grid(True, alpha=0.3, linestyle=':', color=COLORS['grid'])
    ax_a.legend(frameon=True, fancybox=True, shadow=True)

    # Add shaded region below Planck time
    ax_a.fill_between([reflections.min()-0.5, reflections.max()+0.5],
                       [PLANCK_TIME, PLANCK_TIME],
                       [precision.min()/10, precision.min()/10],
                       alpha=0.1, color=COLORS['success'],
                       label='Trans-Planckian regime')

    # ========================================================================
    # Panel B: Enhancement Factor
    # ========================================================================
    ax_b = fig.add_subplot(gs[0, 1])

    # Plot data
    ax_b.scatter(reflections, enhancement, s=50, color=COLORS['accent'],
                 zorder=3, edgecolors='white', linewidth=1)

    # Plot fitted curve
    if analysis.get('enhancement_scaling'):
        fit_x = np.linspace(reflections.min(), reflections.max(), 100)
        fit_y = power_law_model(fit_x, *analysis['enhancement_scaling']['params'])
        ax_b.plot(fit_x, fit_y, color=COLORS['secondary'], linestyle='-',
                  linewidth=1.5, alpha=0.7)

        # Add formula annotation
        formula = analysis['enhancement_scaling']['formula']
        ax_b.text(0.05, 0.95, formula,
                  transform=ax_b.transAxes, fontsize=7,
                  verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax_b.set_yscale('log')
    ax_b.set_xlabel('Number of Reflections', fontweight='bold')
    ax_b.set_ylabel('Enhancement Factor (×)', fontweight='bold')
    ax_b.set_title('B. Enhancement Scaling', fontweight='bold', loc='left')
    ax_b.grid(True, alpha=0.3, linestyle=':', color=COLORS['grid'])

    # ========================================================================
    # Panel C: Orders Below Planck Time
    # ========================================================================
    ax_c = fig.add_subplot(gs[1, 0])

    # Bar plot
    bars = ax_c.bar(reflections, -orders_below, width=0.6,
                    color=COLORS['primary'], edgecolor='white', linewidth=1)

    # Color bars by regime
    for i, bar in enumerate(bars):
        if -orders_below[i] > 0:  # Trans-Planckian
            bar.set_color(COLORS['success'])
        else:
            bar.set_color(COLORS['primary'])

    ax_c.axhline(0, color=COLORS['planck'], linestyle='--', linewidth=1.5)
    ax_c.set_xlabel('Number of Reflections', fontweight='bold')
    ax_c.set_ylabel('Orders of Magnitude Below Planck Time', fontweight='bold')
    ax_c.set_title('C. Trans-Planckian Depth', fontweight='bold', loc='left')
    ax_c.grid(True, alpha=0.3, linestyle=':', color=COLORS['grid'], axis='y')

    # Add annotation for maximum achieved
    max_orders = -orders_below.max()
    ax_c.text(0.95, 0.95, f'Max: {max_orders:.1f} orders\nbelow Planck',
              transform=ax_c.transAxes, fontsize=7,
              verticalalignment='top', horizontalalignment='right',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ========================================================================
    # Panel D: Cumulative Frequency Accumulation
    # ========================================================================
    ax_d = fig.add_subplot(gs[1, 1])

    # Plot frequency accumulation for each cascade
    for i, result in enumerate(results):
        if result['per_reflection_freq']:
            ref_nums, cum_freqs = zip(*result['per_reflection_freq'])
            ax_d.plot(ref_nums, cum_freqs, marker='o', markersize=4,
                     linewidth=1.5, label=f'{result["reflections"]} reflections',
                     alpha=0.8)

    ax_d.set_yscale('log')
    ax_d.set_xlabel('Reflection Step', fontweight='bold')
    ax_d.set_ylabel('Cumulative Frequency (Hz)', fontweight='bold')
    ax_d.set_title('D. Frequency Accumulation', fontweight='bold', loc='left')
    ax_d.grid(True, alpha=0.3, linestyle=':', color=COLORS['grid'])
    ax_d.legend(frameon=True, fancybox=True, shadow=True, fontsize=6)

    # ========================================================================
    # Overall figure title
    # ========================================================================
    fig.suptitle('Trans-Planckian Timekeeping: Reflectance Cascade Performance',
                 fontsize=12, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved: {save_path}")

    return fig


def create_figure_2_network_topology(data, save_path='figure2_network_topology.pdf'):
    """
    Figure 2: Network Topology and BMD Structure

    Shows:
    A) Network statistics
    B) BMD hierarchical decomposition
    C) Graph enhancement mechanism
    """

    fig = plt.figure(figsize=(7.2, 4))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

    network = data['network_stats']
    system = data['system_params']

    # ========================================================================
    # Panel A: Network Statistics
    # ========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.axis('off')

    stats_text = f"""
    Network Topology
    ═══════════════════

    Nodes: {network['nodes']:,}
    Edges: {network['edges']:,}
    Avg Degree: {network['avg_degree']:.2f}

    Density: {network['edges'] / (network['nodes'] * (network['nodes']-1) / 2):.3f}

    BMD Configuration
    ═══════════════════

    Depth: {system['bmd_depth']}
    Channels: {system['bmd_channels']}
    Base Freq: {system['base_frequency']:.2e} Hz
    """

    ax_a.text(0.1, 0.9, stats_text, transform=ax_a.transAxes,
              fontsize=8, verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    ax_a.set_title('A. System Configuration', fontweight='bold', loc='left', pad=20)

    # ========================================================================
    # Panel B: BMD Hierarchical Structure
    # ========================================================================
    ax_b = fig.add_subplot(gs[0, 1])

    # Draw BMD tree structure
    depth = system['bmd_depth']

    def draw_bmd_tree(ax, depth, max_width=3):
        """Draw hierarchical BMD structure"""

        # Calculate positions
        levels = depth + 1
        y_positions = np.linspace(1, 0, levels)

        for level in range(levels):
            n_nodes = 3**level
            if n_nodes > max_width:
                # Show only representative nodes
                x_positions = np.linspace(0, 1, max_width)
                ax.scatter(x_positions, [y_positions[level]]*max_width,
                          s=100, color=COLORS['primary'], zorder=3,
                          edgecolors='white', linewidth=1)

                # Add ellipsis
                ax.text(0.5, y_positions[level], f'... ({n_nodes} nodes)',
                       ha='center', va='bottom', fontsize=6)
            else:
                x_positions = np.linspace(0, 1, n_nodes)
                ax.scatter(x_positions, [y_positions[level]]*n_nodes,
                          s=100, color=COLORS['primary'], zorder=3,
                          edgecolors='white', linewidth=1)

            # Draw connections to next level
            if level < depth:
                n_next = 3**(level+1)
                if n_nodes <= max_width and n_next <= max_width:
                    x_next = np.linspace(0, 1, n_next)
                    for i, x in enumerate(x_positions):
                        # Each node connects to 3 children
                        for j in range(3):
                            child_idx = i*3 + j
                            if child_idx < len(x_next):
                                ax.plot([x, x_next[child_idx]],
                                       [y_positions[level], y_positions[level+1]],
                                       color=COLORS['grid'], linewidth=0.5,
                                       alpha=0.5, zorder=1)

        # Labels
        ax.text(-0.1, 1, 'Root\n(1 demon)', ha='right', va='center', fontsize=7)
        ax.text(-0.1, 0, f'Depth {depth}\n({3**depth} demons)',
               ha='right', va='center', fontsize=7)

        ax.set_xlim(-0.15, 1.05)
        ax.set_ylim(-0.1, 1.1)
        ax.axis('off')

    draw_bmd_tree(ax_b, depth)
    ax_b.set_title('B. BMD Hierarchy', fontweight='bold', loc='left')

    # ========================================================================
    # Panel C: Graph Enhancement Mechanism
    # ========================================================================
    ax_c = fig.add_subplot(gs[0, 2])

    # Show enhancement factors from cascade results
    results = data['cascade_results']

    # Extract graph enhancement (if available in logs)
    # For now, show theoretical enhancement
    reflections = np.arange(1, 6)

    # Graph enhancement = (edges/nodes) per reflection
    if network['edges'] > 0 and network['nodes'] > 0:
        base_enhancement = network['edges'] / network['nodes']
        graph_enhancements = base_enhancement ** reflections

        ax_c.semilogy(reflections, graph_enhancements, marker='o',
                     markersize=6, linewidth=2, color=COLORS['accent'])

        ax_c.set_xlabel('Reflection Depth', fontweight='bold')
        ax_c.set_ylabel('Graph Enhancement Factor', fontweight='bold')
        ax_c.set_title('C. Network Amplification', fontweight='bold', loc='left')
        ax_c.grid(True, alpha=0.3, linestyle=':', color=COLORS['grid'])

        # Add formula
        formula = f'$G_n = ({base_enhancement:.0f})^n$'
        ax_c.text(0.05, 0.95, formula, transform=ax_c.transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ========================================================================
    # Overall title
    # ========================================================================
    fig.suptitle('Network Topology and BMD Structure',
                 fontsize=12, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Saved: {save_path}")

    return fig


def create_summary_table(data, analysis, save_path='table1_cascade_summary.csv'):
    """
    Create summary table of all cascade results.
    """

    results = data['cascade_results']

    table_data = []
    for r in results:
        row = {
            'Reflections': r['reflections'],
            'Precision (s)': f"{r['precision']:.2e}",
            'Orders Below Planck': f"{r['orders_below_planck']:.2f}",
            'Enhancement Factor': f"{r['enhancement']:.2e}",
            'Cumulative Frequency (Hz)': f"{r['per_reflection_freq'][-1][1]:.2e}" if r['per_reflection_freq'] else 'N/A'
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)
    df.to_csv(save_path, index=False)
    print(f"✓ Saved: {save_path}")

    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(log_file='cascade.md'):
    """
    Main execution function.

    Parameters
    ----------
    log_file : str
        Path to cascade log file
    """

    print("="*70)
    print("REFLECTANCE CASCADE DATA ANALYSIS")
    print("="*70)
    print()

    # Load and parse data
    print("Loading data...")
    data = parse_log_file(log_file)
    print(f"✓ Loaded {len(data['cascade_results'])} cascade runs")
    print()

    # System configuration
    print("System Configuration:")
    print(f"  Base frequency: {data['system_params']['base_frequency']:.2e} Hz")
    print(f"  BMD depth: {data['system_params']['bmd_depth']}")
    print(f"  BMD channels: {data['system_params']['bmd_channels']}")
    print(f"  Network nodes: {data['network_stats']['nodes']}")
    print(f"  Network edges: {data['network_stats']['edges']:,}")
    print()

    # Perform analysis
    print("Performing statistical analysis...")
    analysis = analyze_scaling(data)
    print("✓ Analysis complete")
    print()

    # Print key findings
    print("Key Findings:")
    print("-" * 70)

    if analysis.get('precision_power_law'):
        print(f"Precision scaling: {analysis['precision_power_law']['formula']}")
        print(f"  R² = {analysis['precision_power_law']['r_squared']:.4f}")

    if analysis.get('enhancement_scaling'):
        print(f"Enhancement scaling: {analysis['enhancement_scaling']['formula']}")
        print(f"  R² = {analysis['enhancement_scaling']['r_squared']:.4f}")

    if 'avg_improvement_per_reflection' in analysis:
        print(f"Avg improvement per reflection: {analysis['avg_improvement_per_reflection']:.2f}×")
        print(f"  (σ = {analysis['std_improvement_per_reflection']:.2f})")

    print()

    # Generate visualizations
    print("Generating figures...")

    if len(data['cascade_results']) == 0:
        print("⚠ No cascade results found in log file. Cannot generate figures.")
        print("   Check that the log file contains CASCADE COMPLETE blocks.")
        return data, analysis

    create_figure_1_precision_scaling(data, analysis)
    create_figure_2_network_topology(data)
    create_summary_table(data, analysis)
    print()

    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    return data, analysis


if __name__ == "__main__":
    import os
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cascade_file = os.path.join(script_dir, 'cascade.md')
    data, analysis = main(cascade_file)
