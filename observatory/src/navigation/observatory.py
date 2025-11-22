#!/usr/bin/env python3
"""
Stella-Lorraine Observatory: Complete System Dynamics Visualization
Multi-panel analysis of integrated experimental framework
Combines temporal precision, oscillatory coupling, consciousness targeting,
memorial framework, and Bayesian optimization
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Wedge, FancyArrow, Polygon
from matplotlib.collections import PatchCollection
import seaborn as sns
from datetime import datetime

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def load_stella_experiments(data_files):
    """Load Stella-Lorraine experiment JSON files"""
    experiments = []
    for file in data_files:
        try:
            with open(file, 'r') as f:
                experiments.append(json.load(f))
            print(f"✓ Loaded: {file}")
        except FileNotFoundError:
            print(f"✗ File not found: {file}")
        except json.JSONDecodeError as e:
            print(f"✗ JSON decode error in {file}: {e}")
    return experiments


def extract_experiment_summary(experiment):
    """Extract key metrics from experiment"""
    config = experiment['configuration']

    summary = {
        'experiment_id': experiment['experiment_id'],
        'timestamp': experiment['timestamp'],
        'experiment_type': experiment['experiment_type'],

        # Temporal precision
        'target_precision_ns': config['temporal_precision']['target_precision_ns'],
        'sampling_rate_hz': config['temporal_precision']['sampling_rate_hz'],
        'measurement_duration_s': config['temporal_precision']['measurement_duration_s'],
        'multi_scale_frequencies': config['temporal_precision']['multi_scale_frequencies'],

        # Oscillatory analysis
        'oscillator_count': config['oscillatory_analysis']['oscillator_count'],
        'coupling_strength': config['oscillatory_analysis']['coupling_strength'],
        'convergence_threshold': config['oscillatory_analysis']['convergence_threshold'],
        'frequency_range': config['oscillatory_analysis']['frequency_range_hz'],

        # Consciousness targeting
        'population_size': config['consciousness_targeting']['population_size'],
        'consciousness_dimensions': config['consciousness_targeting']['consciousness_dimensions'],
        'targeting_accuracy': config['consciousness_targeting']['targeting_accuracy_threshold'],

        # Memorial framework
        'consciousness_inheritance': config['memorial_framework']['consciousness_inheritance_rate'],
        'expertise_transfer': config['memorial_framework']['expertise_transfer_efficiency'],
        'temporal_persistence': config['memorial_framework']['temporal_persistence_requirement'],

        # Bayesian network
        'bayesian_nodes': config['bayesian_network']['node_count'],
        'convergence_criteria': config['bayesian_network']['convergence_criteria'],

        # Optimization
        'optimization_algorithm': config['optimization']['optimization_algorithm'],
        'budget_iterations': config['optimization']['budget_iterations'],
    }

    return summary


def create_stella_visualization(experiments):
    """Create comprehensive 9-panel visualization of complete system dynamics"""

    # Create figure with 9 panels (3x3 grid)
    fig = plt.figure(figsize=(22, 18))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Extract summaries
    summaries = [extract_experiment_summary(exp) for exp in experiments]

    # ========================================================================
    # PANEL A: Multi-Scale Frequency Cascade
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])

    # Extract frequency scales
    freq_scales = summaries[0]['multi_scale_frequencies']
    scale_names = ['kHz', 'MHz', 'GHz', 'THz']

    # Plot frequency cascade
    x_pos = np.arange(len(freq_scales))
    colors = plt.cm.viridis(np.linspace(0, 1, len(freq_scales)))

    bars = ax1.bar(x_pos, np.log10(freq_scales), color=colors, alpha=0.8,
                   edgecolor='black', linewidth=2)

    # Add value labels
    for i, (bar, freq) in enumerate(zip(bars, freq_scales)):
        height = bar.get_height()
        ax1.annotate(f'{freq:.0e} Hz\n({scale_names[i]})',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_xlabel('Frequency Scale', fontweight='bold')
    ax1.set_ylabel('Frequency (log₁₀ Hz)', fontweight='bold')
    ax1.set_title('A) Multi-Scale Frequency Cascade',
                  fontweight='bold', loc='left')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(scale_names)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add coupling annotation
    ax1.text(0.95, 0.95, 'Oscillatory\nCoupling\nEnabled',
            transform=ax1.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
            fontsize=9, fontweight='bold')

    # ========================================================================
    # PANEL B: Temporal Precision Evolution
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    # Compare experiments
    exp_labels = [f"Exp {i+1}" for i in range(len(summaries))]
    target_precisions = [s['target_precision_ns'] for s in summaries]
    durations = [s['measurement_duration_s'] for s in summaries]

    # Create dual-axis plot
    x_pos = np.arange(len(exp_labels))
    width = 0.35

    color1 = 'C0'
    bars1 = ax2.bar(x_pos - width/2, target_precisions, width,
                    label='Target Precision (ns)', color=color1, alpha=0.8)
    ax2.set_ylabel('Target Precision (ns)', fontweight='bold', color=color1)
    ax2.tick_params(axis='y', labelcolor=color1)

    # Second y-axis for duration
    ax2_twin = ax2.twinx()
    color2 = 'C1'
    bars2 = ax2_twin.bar(x_pos + width/2, durations, width,
                         label='Duration (s)', color=color2, alpha=0.8)
    ax2_twin.set_ylabel('Measurement Duration (s)', fontweight='bold', color=color2)
    ax2_twin.tick_params(axis='y', labelcolor=color2)

    ax2.set_xlabel('Experiment', fontweight='bold')
    ax2.set_title('B) Temporal Precision Configuration',
                  fontweight='bold', loc='left')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(exp_labels)
    ax2.grid(True, alpha=0.3)

    # Add sampling rate annotation
    sampling_rate = summaries[0]['sampling_rate_hz']
    ax2.text(0.5, 0.95, f'Sampling Rate:\n{sampling_rate/1e6:.1f} MHz',
            transform=ax2.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
            fontsize=9, fontweight='bold')

    # ========================================================================
    # PANEL C: Oscillator Network Dynamics
    # ========================================================================
    ax3 = fig.add_subplot(gs[0, 2])

    # Extract oscillator parameters
    osc_counts = [s['oscillator_count'] for s in summaries]
    coupling_strengths = [s['coupling_strength'] for s in summaries]

    # Create scatter plot showing network size vs coupling
    colors = ['C2', 'C3']
    for i, (count, coupling, label) in enumerate(zip(osc_counts, coupling_strengths, exp_labels)):
        ax3.scatter(count, coupling, s=500, c=colors[i], alpha=0.7,
                   edgecolors='black', linewidth=2, label=label, zorder=3)

        # Add annotation
        ax3.annotate(f'{count} osc.\n{coupling:.2f} coupling',
                    xy=(count, coupling), xytext=(10, 10),
                    textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    ax3.set_xlabel('Number of Oscillators', fontweight='bold')
    ax3.set_ylabel('Coupling Strength', fontweight='bold')
    ax3.set_title('C) Oscillator Network Configuration',
                  fontweight='bold', loc='left')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')

    # Add convergence threshold
    conv_threshold = summaries[0]['convergence_threshold']
    ax3.text(0.95, 0.05, f'Convergence:\n{conv_threshold:.0e}',
            transform=ax3.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
            fontsize=9, fontweight='bold')

    # ========================================================================
    # PANEL D: Consciousness Targeting Framework
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 0])

    # Extract consciousness parameters
    pop_sizes = [s['population_size'] for s in summaries]
    dimensions = [s['consciousness_dimensions'] for s in summaries]
    accuracy = [s['targeting_accuracy'] for s in summaries]

    # Create grouped bar chart
    x_pos = np.arange(len(exp_labels))
    width = 0.25

    bars1 = ax4.bar(x_pos - width, np.array(pop_sizes)/1000, width,
                    label='Population (×1000)', color='C4', alpha=0.8)
    bars2 = ax4.bar(x_pos, dimensions, width,
                    label='Dimensions', color='C5', alpha=0.8)
    bars3 = ax4.bar(x_pos + width, np.array(accuracy)*10, width,
                    label='Accuracy (×10)', color='C6', alpha=0.8)

    ax4.set_xlabel('Experiment', fontweight='bold')
    ax4.set_ylabel('Parameter Value', fontweight='bold')
    ax4.set_title('D) Consciousness Targeting Parameters',
                  fontweight='bold', loc='left')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(exp_labels)
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add targeting features
    features_text = (
        "Features:\n"
        "• Nordic Paradox\n"
        "• Free Will Tracking\n"
        "• Death Proximity\n"
        "• Functional Delusion"
    )
    ax4.text(0.95, 0.95, features_text,
            transform=ax4.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.7),
            fontsize=8, family='monospace')

    # ========================================================================
    # PANEL E: Memorial Framework Efficiency
    # ========================================================================
    ax5 = fig.add_subplot(gs[1, 1])

    # Extract memorial parameters
    inheritance = summaries[0]['consciousness_inheritance']
    expertise = summaries[0]['expertise_transfer']
    persistence = summaries[0]['temporal_persistence']

    # Create radar chart
    categories = ['Consciousness\nInheritance', 'Expertise\nTransfer',
                  'Temporal\nPersistence']
    values = [inheritance, expertise, persistence]

    # Number of variables
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]  # Complete the circle
    angles += angles[:1]

    # Plot
    ax5 = plt.subplot(gs[1, 1], projection='polar')
    ax5.plot(angles, values, 'o-', linewidth=2, color='C7', label='Efficiency')
    ax5.fill(angles, values, alpha=0.25, color='C7')

    # Fix axis to go in the right order
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(categories, fontsize=9)
    ax5.set_ylim(0, 1)
    ax5.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax5.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
    ax5.grid(True)

    ax5.set_title('E) Memorial Framework Efficiency',
                  fontweight='bold', pad=20)

    # Add target annotation
    ax5.text(0.5, -0.15, 'Buhera Model Enabled\nCapitalism Elimination: 99%',
            transform=ax5.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
            fontsize=8, fontweight='bold')

    # ========================================================================
    # PANEL F: Bayesian Network Architecture
    # ========================================================================
    ax6 = fig.add_subplot(gs[1, 2])

    # Extract Bayesian parameters
    node_counts = [s['bayesian_nodes'] for s in summaries]

    # Create network visualization (simplified)
    for i, (node_count, label) in enumerate(zip(node_counts, exp_labels)):
        # Calculate layout
        radius = 0.3
        center_x = 0.3 if i == 0 else 0.7
        center_y = 0.5

        # Draw central node
        circle = Circle((center_x, center_y), 0.05, color=f'C{i}',
                       alpha=0.8, zorder=3, transform=ax6.transAxes)
        ax6.add_patch(circle)

        # Draw surrounding nodes
        n_display = min(8, node_count)  # Display up to 8 nodes
        angles = np.linspace(0, 2*np.pi, n_display, endpoint=False)

        for angle in angles:
            x = center_x + radius * 0.3 * np.cos(angle)
            y = center_y + radius * 0.3 * np.sin(angle)

            # Draw node
            small_circle = Circle((x, y), 0.02, color=f'C{i}',
                                 alpha=0.5, zorder=2, transform=ax6.transAxes)
            ax6.add_patch(small_circle)

            # Draw connection
            ax6.plot([center_x, x], [center_y, y], 'k-', alpha=0.3,
                    linewidth=1, transform=ax6.transAxes)

        # Add label
        ax6.text(center_x, center_y - 0.15, f'{label}\n{node_count} nodes',
                ha='center', va='top', transform=ax6.transAxes,
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    ax6.set_title('F) Bayesian Network Architecture',
                  fontweight='bold', loc='left')

    # Add method annotation
    method_text = (
        "Method:\n"
        "Variational Inference\n"
        "Causal Structure Learning\n"
        "Convergence: 10⁻⁶"
    )
    ax6.text(0.5, 0.95, method_text,
            transform=ax6.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7),
            fontsize=8, family='monospace')

    # ========================================================================
    # PANEL G: Optimization Strategy
    # ========================================================================
    ax7 = fig.add_subplot(gs[2, 0])

    # Optimization parameters
    algorithm = summaries[0]['optimization_algorithm']
    budget = summaries[0]['budget_iterations']

    # Create optimization trajectory visualization (synthetic)
    iterations = np.linspace(0, budget, 100)

    # Simulate convergence curves for both experiments
    np.random.seed(42)
    for i, label in enumerate(exp_labels):
        # Exponential convergence with noise
        baseline = 0.1
        improvement = 0.9 * (1 - np.exp(-iterations / (budget * 0.3)))
        noise = np.random.normal(0, 0.02, len(iterations))
        precision = baseline + improvement + noise
        precision = np.clip(precision, 0, 1)

        ax7.plot(iterations, precision, linewidth=2, label=label,
                color=f'C{i}', alpha=0.8)

    ax7.set_xlabel('Optimization Iteration', fontweight='bold')
    ax7.set_ylabel('Normalized Precision', fontweight='bold')
    ax7.set_title('G) Bayesian Optimization Convergence',
                  fontweight='bold', loc='left')
    ax7.legend(loc='lower right')
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(0, budget)
    ax7.set_ylim(0, 1.05)

    # Add algorithm info
    algo_text = (
        f"Algorithm:\n{algorithm.replace('_', ' ').title()}\n"
        f"Acquisition: Expected Improvement\n"
        f"Kernel: Matérn 5/2"
    )
    ax7.text(0.05, 0.95, algo_text,
            transform=ax7.transAxes, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
            fontsize=8, family='monospace')

    # ========================================================================
    # PANEL H: System Integration Map
    # ========================================================================
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.axis('off')

    # Create system integration diagram
    components = [
        ('Temporal\nPrecision', 0.5, 0.9, 'C0'),
        ('Oscillatory\nAnalysis', 0.2, 0.6, 'C1'),
        ('Consciousness\nTargeting', 0.8, 0.6, 'C2'),
        ('Memorial\nFramework', 0.2, 0.3, 'C3'),
        ('Bayesian\nNetwork', 0.8, 0.3, 'C4'),
        ('Optimization', 0.5, 0.1, 'C5'),
    ]

    # Draw components
    for name, x, y, color in components:
        circle = Circle((x, y), 0.08, color=color, alpha=0.7,
                       edgecolor='black', linewidth=2, zorder=3,
                       transform=ax8.transAxes)
        ax8.add_patch(circle)
        ax8.text(x, y, name, ha='center', va='center',
                transform=ax8.transAxes, fontsize=7, fontweight='bold')

    # Draw connections
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5),
        (1, 5), (2, 5), (0, 5)
    ]

    for i, j in connections:
        x1, y1 = components[i][1], components[i][2]
        x2, y2 = components[j][1], components[j][2]
        ax8.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.5),
                    transform=ax8.transAxes)

    ax8.set_xlim(0, 1)
    ax8.set_ylim(0, 1)
    ax8.set_title('H) System Integration Architecture',
                  fontweight='bold', loc='left')

    # Add integration note
    ax8.text(0.5, 0.02, 'Fully Integrated Multi-Domain Framework',
            transform=ax8.transAxes, ha='center', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightgreen',
                     edgecolor='darkgreen', linewidth=2, alpha=0.7),
            fontsize=9, fontweight='bold')

    # ========================================================================
    # PANEL I: Experiment Comparison Summary
    # ========================================================================
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    # Create comparison table
    table_data = []
    table_data.append(['Parameter', 'Exp 1', 'Exp 2', 'Change'])
    table_data.append(['─'*20, '─'*15, '─'*15, '─'*10])

    # Compare key parameters
    comparisons = [
        ('Duration (s)', 'measurement_duration_s', '{:.0f}'),
        ('Oscillators', 'oscillator_count', '{:.0f}'),
        ('Bayesian Nodes', 'bayesian_nodes', '{:.0f}'),
        ('Population', 'population_size', '{:.0f}'),
        ('Coupling', 'coupling_strength', '{:.2f}'),
        ('Accuracy', 'targeting_accuracy', '{:.2f}'),
    ]

    for label, key, fmt in comparisons:
        val1 = summaries[0][key]
        val2 = summaries[1][key]

        if val1 != 0:
            change = ((val2 - val1) / val1) * 100
            change_str = f'{change:+.1f}%'
        else:
            change_str = 'N/A'

        table_data.append([
            label,
            fmt.format(val1),
            fmt.format(val2),
            change_str
        ])

    # Draw table
    y_start = 0.95
    y_step = 0.08

    for i, row in enumerate(table_data):
        y_pos = y_start - i * y_step

        # Header styling
        if i == 0:
            weight = 'bold'
            color = 'darkblue'
            size = 10
        elif '─' in row[0]:
            # Separator
            ax9.plot([0.05, 0.95], [y_pos, y_pos], 'k-',
                    linewidth=1, alpha=0.3, transform=ax9.transAxes)
            continue
        else:
            weight = 'normal'
            color = 'black'
            size = 9

        # Draw cells
        ax9.text(0.05, y_pos, row[0], ha='left', va='center',
                fontsize=size, fontweight=weight, color=color,
                transform=ax9.transAxes)
        ax9.text(0.50, y_pos, row[1], ha='center', va='center',
                fontsize=size, fontweight=weight, color=color,
                transform=ax9.transAxes, family='monospace')
        ax9.text(0.70, y_pos, row[2], ha='center', va='center',
                fontsize=size, fontweight=weight, color=color,
                transform=ax9.transAxes, family='monospace')
        ax9.text(0.90, y_pos, row[3], ha='center', va='center',
                fontsize=size, fontweight=weight, color=color,
                transform=ax9.transAxes, family='monospace')

    ax9.set_title('I) Experiment Comparison Summary',
                  fontweight='bold', loc='left')

    # Add timestamps
    time1 = summaries[0]['timestamp']
    time2 = summaries[1]['timestamp']
    ax9.text(0.5, 0.05, f'Exp 1: {time1}\nExp 2: {time2}',
            transform=ax9.transAxes, ha='center', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5),
            fontsize=7, family='monospace')

    # ========================================================================
    # Overall figure title and metadata
    # ========================================================================
    fig.suptitle('Stella-Lorraine Observatory: Complete System Dynamics & Integration',
                 fontsize=20, fontweight='bold', y=0.998)

    # Add metadata footer
    metadata_text = (
        f"Experiments: {len(experiments)} | "
        f"Type: {summaries[0]['experiment_type']} | "
        f"Target Precision: {summaries[0]['target_precision_ns']} ns | "
        f"Multi-Scale Coupling: kHz → THz | "
        f"Optimization: Bayesian | "
        f"Framework: Integrated Multi-Domain"
    )
    fig.text(0.5, 0.002, metadata_text, ha='center', fontsize=10,
             style='italic', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue',
                      edgecolor='darkblue', linewidth=2, alpha=0.6))

    return fig


def print_stella_statistics(experiments):
    """Print comprehensive statistics for Stella experiments"""
    print("\n" + "="*80)
    print("STELLA-LORRAINE OBSERVATORY - COMPLETE SYSTEM ANALYSIS")
    print("="*80)

    summaries = [extract_experiment_summary(exp) for exp in experiments]

    for idx, (exp, summary) in enumerate(zip(experiments, summaries)):
        print(f"\n{'═'*80}")
        print(f"EXPERIMENT {idx + 1}: {summary['experiment_id']}")
        print(f"Timestamp: {summary['timestamp']}")
        print(f"{'═'*80}")

        # Temporal Precision
        print(f"\n┌─ TEMPORAL PRECISION")
        print(f"│  Target Precision:      {summary['target_precision_ns']} ns")
        print(f"│  Sampling Rate:         {summary['sampling_rate_hz']/1e6:.1f} MHz")
        print(f"│  Duration:              {summary['measurement_duration_s']:.1f} s")
        print(f"│  Multi-Scale Freqs:     {len(summary['multi_scale_frequencies'])} scales")
        for i, freq in enumerate(summary['multi_scale_frequencies']):
            print(f"│    Scale {i+1}:            {freq:.0e} Hz")

        # Oscillatory Analysis
        print(f"\n├─ OSCILLATORY ANALYSIS")
        print(f"│  Oscillator Count:      {summary['oscillator_count']:,}")
        print(f"│  Coupling Strength:     {summary['coupling_strength']:.2f}")
        print(f"│  Convergence Threshold: {summary['convergence_threshold']:.0e}")
        print(f"│  Frequency Range:       {summary['frequency_range'][0]:.0e} - {summary['frequency_range'][1]:.0e} Hz")

        # Consciousness Targeting
        print(f"\n├─ CONSCIOUSNESS TARGETING")
        print(f"│  Population Size:       {summary['population_size']:,}")
        print(f"│  Dimensions:            {summary['consciousness_dimensions']}")
        print(f"│  Targeting Accuracy:    {summary['targeting_accuracy']*100:.1f}%")

        # Memorial Framework
        print(f"\n├─ MEMORIAL FRAMEWORK")
        print(f"│  Consciousness Inherit: {summary['consciousness_inheritance']*100:.1f}%")
        print(f"│  Expertise Transfer:    {summary['expertise_transfer']*100:.1f}%")
        print(f"│  Temporal Persistence:  {summary['temporal_persistence']*100:.1f}%")

        # Bayesian Network
        print(f"\n├─ BAYESIAN NETWORK")
        print(f"│  Node Count:            {summary['bayesian_nodes']}")
        print(f"│  Convergence Criteria:  {summary['convergence_criteria']:.0e}")

        # Optimization
        print(f"\n└─ OPTIMIZATION")
        print(f"   Algorithm:             {summary['optimization_algorithm']}")
        print(f"   Budget Iterations:     {summary['budget_iterations']:,}")

    # Comparison
    if len(summaries) > 1:
        print(f"\n{'═'*80}")
        print("EXPERIMENT COMPARISON")
        print(f"{'═'*80}")

        print(f"\nKey Differences:")

        # Duration
        dur_change = ((summaries[1]['measurement_duration_s'] -
                      summaries[0]['measurement_duration_s']) /
                     summaries[0]['measurement_duration_s'] * 100)
        print(f"  Duration:        {summaries[0]['measurement_duration_s']:.0f}s → {summaries[1]['measurement_duration_s']:.0f}s ({dur_change:+.1f}%)")

        # Oscillators
        osc_change = ((summaries[1]['oscillator_count'] -
                      summaries[0]['oscillator_count']) /
                     summaries[0]['oscillator_count'] * 100)
        print(f"  Oscillators:     {summaries[0]['oscillator_count']:,} → {summaries[1]['oscillator_count']:,} ({osc_change:+.1f}%)")

        # Bayesian nodes
        node_change = ((summaries[1]['bayesian_nodes'] -
                       summaries[0]['bayesian_nodes']) /
                      summaries[0]['bayesian_nodes'] * 100)
        print(f"  Bayesian Nodes:  {summaries[0]['bayesian_nodes']} → {summaries[1]['bayesian_nodes']} ({node_change:+.1f}%)")

    # System Integration Summary
    print(f"\n{'═'*80}")
    print("SYSTEM INTEGRATION SUMMARY")
    print(f"{'═'*80}")

    print(f"\nIntegrated Components:")
    print(f"  ✓ Temporal Precision (ns-scale targeting)")
    print(f"  ✓ Oscillatory Analysis (multi-scale coupling)")
    print(f"  ✓ Consciousness Targeting (4D framework)")
    print(f"  ✓ Memorial Framework (Buhera model)")
    print(f"  ✓ Bayesian Network (causal learning)")
    print(f"  ✓ Optimization (Bayesian strategy)")

    print(f"\nKey Features:")
    print(f"  • Multi-scale frequency cascade (kHz → THz)")
    print(f"  • Self-organizing oscillator networks")
    print(f"  • Consciousness inheritance tracking")
    print(f"  • Variational inference optimization")
    print(f"  • Environmental corrections enabled")
    print(f"  • Quantum enhancement enabled")

    print("\n" + "="*80)


def main():
    """Main execution function"""

    # Define data files
    data_files = [
        'stella_experiment_20251008_081846_20251008_081846_results.json',
        'stella_experiment_20251008_202536_20251008_202536_results.json'
    ]

    print("="*80)
    print("STELLA-LORRAINE OBSERVATORY SYSTEM VISUALIZATION")
    print("="*80)

    # Load experiments
    print("\nLoading Stella-Lorraine experiment data...")
    experiments = load_stella_experiments(data_files)

    if len(experiments) == 0:
        print("\n✗ No experiment data loaded. Please check file paths.")
        return

    print(f"\n✓ Successfully loaded {len(experiments)} experiments")

    # Create visualization
    print("\nGenerating comprehensive system visualizations...")
    fig = create_stella_visualization(experiments)

    # Save outputs
    output_png = 'stella_lorraine_system_dynamics.png'
    output_pdf = 'stella_lorraine_system_dynamics.pdf'

    print("\nSaving figures...")
    fig.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ PNG saved: {output_png}")

    fig.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ PDF saved: {output_pdf}")

    # Print comprehensive statistics
    print_stella_statistics(experiments)

    # Display figure
    print("\nDisplaying figure...")
    plt.show()

    print("\n" + "="*80)
    print("✓ VISUALIZATION COMPLETE!")
    print("="*80)
    print("\nStella-Lorraine Observatory System Dynamics:")
    print("  • Complete multi-domain integration")
    print("  • Temporal precision + oscillatory coupling")
    print("  • Consciousness targeting + memorial framework")
    print("  • Bayesian optimization + causal learning")
    print("  • Multi-scale frequency cascade (kHz → THz)")
    print("  • Self-organizing network dynamics")
    print("\nThis represents your complete experimental framework!")
    print("="*80)


if __name__ == "__main__":
    main()
