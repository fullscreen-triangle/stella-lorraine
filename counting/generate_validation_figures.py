#!/usr/bin/env python3
"""
Generate Panel Figures from Validation Results
==============================================

Creates publication-quality 4-panel charts for each validation section.
Each panel includes at least one 3D visualization.

Usage:
    python generate_validation_figures.py validation_results_YYYYMMDD_HHMMSS.json
    python generate_validation_figures.py --all  # Process all result files
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['figure.titlesize'] = 11


class ValidationFigureGenerator:
    """Generate panel figures from validation results."""

    def __init__(self, output_dir: str = "figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.colormap = 'viridis'

    def generate_all_panels(self, results_file: str):
        """Generate all panel charts from a results file."""
        print(f"\nProcessing: {results_file}")

        with open(results_file, 'r') as f:
            data = json.load(f)

        # Extract base name for output files
        base_name = Path(results_file).stem

        # Generate each panel
        print("  Generating oscillator validation panel...")
        self.generate_oscillator_panel(data['oscillator_validation'], base_name)

        print("  Generating partition coordinates panel...")
        self.generate_partition_panel(data['partition_coordinates'], base_name)

        print("  Generating categorical temperature panel...")
        self.generate_temperature_panel(data['categorical_temperature'], base_name)

        print("  Generating ion trajectory panel...")
        self.generate_trajectory_panel(data['ion_trajectory_validation'], base_name)

        if data.get('pipeline_validation'):
            print("  Generating pipeline validation panel...")
            self.generate_pipeline_panel(data['pipeline_validation'], base_name)

        if data.get('statistical_analysis'):
            print("  Generating statistical analysis panel...")
            self.generate_statistics_panel(data['statistical_analysis'], base_name)

        print(f"  [OK] All panels saved to: {self.output_dir}")

    def generate_oscillator_panel(self, data: Dict[str, Any], base_name: str):
        """Panel 1: Hardware Oscillator Validation (4 charts)."""
        fig = plt.figure(figsize=(12, 3))

        measurements = data['measurements']
        durations = [m['duration_s'] for m in measurements]
        cycles = [m['cycles_counted'] for m in measurements]
        errors = [m['error'] for m in measurements]

        # Chart 1: Cycle Count vs Duration (log-log)
        ax1 = plt.subplot(141)
        ax1.loglog(durations, cycles, 'o-', color='#2E86AB', linewidth=2,
                   markersize=6, markerfacecolor='white', markeredgewidth=2)
        ax1.set_xlabel('Duration (s)')
        ax1.set_ylabel('Cycles Counted')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_title('A', loc='left', fontweight='bold')

        # Chart 2: Reconstruction Error
        ax2 = plt.subplot(142)
        # Add small offset to avoid log(0)
        errors_plot = [max(e, 1e-15) for e in errors]
        ax2.semilogy(range(len(errors_plot)), errors_plot, 's-', color='#A23B72',
                     linewidth=2, markersize=6, markerfacecolor='white',
                     markeredgewidth=2)
        ax2.axhline(y=1e-10, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Measurement Index')
        ax2.set_ylabel('Relative Error')
        ax2.set_ylim([1e-16, 1e-8])
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_title('B', loc='left', fontweight='bold')

        # Chart 3: 3D Surface - Time vs Cycles vs Frequency
        ax3 = plt.subplot(143, projection='3d')
        freq = data['frequency_hz']

        # Create meshgrid
        t_range = np.logspace(-6, -2, 50)
        f_range = np.linspace(freq * 0.5, freq * 1.5, 50)
        T, F = np.meshgrid(t_range, f_range)
        C = T * F  # Cycles = time × frequency

        surf = ax3.plot_surface(np.log10(T), F/1e6, C/1e3, cmap=self.colormap,
                               alpha=0.8, edgecolor='none')
        ax3.scatter(np.log10(durations), [freq/1e6]*len(durations),
                   np.array(cycles)/1e3, c='red', s=50, marker='o',
                   edgecolors='white', linewidths=1.5)
        ax3.set_xlabel('log₁₀(Time) (s)')
        ax3.set_ylabel('Freq (MHz)')
        ax3.set_zlabel('Cycles (×10³)')
        ax3.set_title('C', loc='left', fontweight='bold')
        ax3.view_init(elev=20, azim=45)

        # Chart 4: Fundamental Identity Verification
        ax4 = plt.subplot(144)
        expected = [d * freq for d in durations]
        ax4.plot(expected, cycles, 'o', color='#F18F01', markersize=8,
                markerfacecolor='white', markeredgewidth=2)
        lim_max = max(max(expected), max(cycles))
        ax4.plot([0, lim_max], [0, lim_max], 'k--', linewidth=1, alpha=0.5)
        ax4.set_xlabel('Expected Cycles')
        ax4.set_ylabel('Measured Cycles')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_title('D', loc='left', fontweight='bold')
        ax4.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        output_file = self.output_dir / f"{base_name}_panel_1_oscillator.png"
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def generate_partition_panel(self, data: Dict[str, Any], base_name: str):
        """Panel 2: Partition Coordinates Validation (4 charts)."""
        fig = plt.figure(figsize=(12, 3))

        vals = data['validations']
        M = [v['state_count_M'] for v in vals]
        n = [v['n'] for v in vals]
        capacity = [v['capacity'] for v in vals]
        cumulative = [v['cumulative_capacity'] for v in vals]

        # Chart 1: n vs M (shows square root relationship)
        ax1 = plt.subplot(141)
        ax1.plot(M, n, 'o-', color='#2E86AB', linewidth=2, markersize=6,
                markerfacecolor='white', markeredgewidth=2)
        # Theoretical line
        M_theory = np.linspace(min(M), max(M), 100)
        n_theory = np.sqrt(M_theory / 2) + 1
        ax1.plot(M_theory, n_theory, 'r--', linewidth=1, alpha=0.5)
        ax1.set_xlabel('State Count M')
        ax1.set_ylabel('Principal Number n')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_title('A', loc='left', fontweight='bold')

        # Chart 2: Capacity C(n) = 2n²
        ax2 = plt.subplot(142)
        ax2.plot(n, capacity, 'o', color='#A23B72', markersize=8,
                markerfacecolor='white', markeredgewidth=2, label='Measured')
        n_range = np.linspace(min(n), max(n), 100)
        C_theory = 2 * n_range**2
        ax2.plot(n_range, C_theory, 'r--', linewidth=1, alpha=0.5, label='Theory')
        ax2.set_xlabel('Principal Number n')
        ax2.set_ylabel('Capacity C(n)')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_title('B', loc='left', fontweight='bold')
        ax2.legend(frameon=False)

        # Chart 3: 3D - (n, ℓ, m) space
        ax3 = plt.subplot(143, projection='3d')

        # Extract quantum numbers
        n_vals = np.array([v['n'] for v in vals])
        l_vals = np.array([v['l'] for v in vals])
        m_vals = np.array([v['m'] for v in vals])

        # Color by capacity
        colors = capacity
        norm = Normalize(vmin=min(colors), vmax=max(colors))

        scatter = ax3.scatter(n_vals, l_vals, m_vals, c=colors, cmap=self.colormap,
                             s=100, alpha=0.8, edgecolors='white', linewidths=1)
        ax3.set_xlabel('n')
        ax3.set_ylabel('ℓ')
        ax3.set_zlabel('m')
        ax3.set_title('C', loc='left', fontweight='bold')
        ax3.view_init(elev=20, azim=45)

        # Chart 4: Cumulative Capacity
        ax4 = plt.subplot(144)
        ax4.plot(n, cumulative, 'o-', color='#F18F01', linewidth=2,
                markersize=6, markerfacecolor='white', markeredgewidth=2)
        # Theoretical cumulative
        cumul_theory = [sum(2*i**2 for i in range(1, ni+1)) for ni in n]
        ax4.plot(n, cumul_theory, 'r--', linewidth=1, alpha=0.5)
        ax4.set_xlabel('Max n')
        ax4.set_ylabel('Cumulative States')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_title('D', loc='left', fontweight='bold')

        plt.tight_layout()
        output_file = self.output_dir / f"{base_name}_panel_2_partition.png"
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def generate_temperature_panel(self, data: Dict[str, Any], base_name: str):
        """Panel 3: Categorical Temperature Validation (4 charts)."""
        fig = plt.figure(figsize=(12, 3))

        vals = data['validations']
        M = [v['state_count_M'] for v in vals]
        T_cat = [v['T_categorical_K'] for v in vals]
        T_exp = [v['T_expected_K'] for v in vals]
        suppression = [v['suppression_factor'] for v in vals]
        energy = vals[0]['energy_eV']

        # Chart 1: T_categorical vs M (log-log)
        ax1 = plt.subplot(141)
        ax1.loglog(M, T_cat, 'o-', color='#2E86AB', linewidth=2,
                  markersize=6, markerfacecolor='white', markeredgewidth=2,
                  label='Categorical T')
        ax1.loglog(M, T_exp, 'r--', linewidth=1, alpha=0.5, label='Theory')
        ax1.set_xlabel('State Count M')
        ax1.set_ylabel('Temperature (K)')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_title('A', loc='left', fontweight='bold')
        ax1.legend(frameon=False)

        # Chart 2: Suppression Factor 1/M
        ax2 = plt.subplot(142)
        ax2.loglog(M, suppression, 's-', color='#A23B72', linewidth=2,
                  markersize=6, markerfacecolor='white', markeredgewidth=2,
                  label='Measured')
        M_range = np.logspace(0, 6, 100)
        supp_theory = 1.0 / M_range
        ax2.loglog(M_range, supp_theory, 'r--', linewidth=1, alpha=0.5,
                  label='1/M')
        ax2.set_xlabel('State Count M')
        ax2.set_ylabel('Suppression Factor')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_title('B', loc='left', fontweight='bold')
        ax2.legend(frameon=False)

        # Chart 3: 3D Surface - T(E, M)
        ax3 = plt.subplot(143, projection='3d')

        # Create meshgrid for E and M
        E_range = np.linspace(1, 50, 50)
        M_range = np.logspace(0, 6, 50)
        E_grid, M_grid = np.meshgrid(E_range, M_range)

        # Calculate T = 2E/(3k_B × M)
        k_B = 1.380649e-23
        e_charge = 1.602176634e-19
        T_grid = 2 * E_grid * e_charge / (3 * k_B * M_grid)

        surf = ax3.plot_surface(E_grid, np.log10(M_grid), T_grid,
                               cmap=self.colormap, alpha=0.8, edgecolor='none')

        # Plot measured points
        E_measured = [energy] * len(M)
        ax3.scatter(E_measured, np.log10(M), T_cat, c='red', s=50,
                   marker='o', edgecolors='white', linewidths=1.5)

        ax3.set_xlabel('Energy (eV)')
        ax3.set_ylabel('log₁₀(M)')
        ax3.set_zlabel('T (K)')
        ax3.set_title('C', loc='left', fontweight='bold')
        ax3.view_init(elev=20, azim=45)

        # Chart 4: T_measured vs T_expected
        ax4 = plt.subplot(144)
        ax4.loglog(T_exp, T_cat, 'o', color='#F18F01', markersize=8,
                  markerfacecolor='white', markeredgewidth=2)
        lim = [min(min(T_exp), min(T_cat)), max(max(T_exp), max(T_cat))]
        ax4.loglog(lim, lim, 'k--', linewidth=1, alpha=0.5)
        ax4.set_xlabel('Expected T (K)')
        ax4.set_ylabel('Measured T (K)')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_title('D', loc='left', fontweight='bold')
        ax4.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        output_file = self.output_dir / f"{base_name}_panel_3_temperature.png"
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def generate_trajectory_panel(self, data: Dict[str, Any], base_name: str):
        """Panel 4: Ion Trajectory Validation (4 charts)."""
        fig = plt.figure(figsize=(12, 3))

        vals = data['validations']

        # Extract data
        mz_vals = [v['ion']['mz'] for v in vals]
        charges = [v['ion']['charge'] for v in vals]
        energies = [v['ion']['energy'] for v in vals]
        state_counts = [v['total_state_count'] for v in vals]
        total_times = [v['total_time_s'] * 1e6 for v in vals]  # Convert to μs

        # Chart 1: State Count vs m/z
        ax1 = plt.subplot(141)
        ax1.plot(mz_vals, state_counts, 'o-', color='#2E86AB', linewidth=2,
                markersize=8, markerfacecolor='white', markeredgewidth=2)
        ax1.set_xlabel('m/z')
        ax1.set_ylabel('Total State Count')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_title('A', loc='left', fontweight='bold')

        # Chart 2: Journey Time vs m/z
        ax2 = plt.subplot(142)
        ax2.plot(mz_vals, total_times, 's-', color='#A23B72', linewidth=2,
                markersize=8, markerfacecolor='white', markeredgewidth=2)
        ax2.set_xlabel('m/z')
        ax2.set_ylabel('Journey Time (μs)')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_title('B', loc='left', fontweight='bold')

        # Chart 3: 3D - (m/z, charge, state_count)
        ax3 = plt.subplot(143, projection='3d')

        colors = energies
        norm = Normalize(vmin=min(colors), vmax=max(colors))

        scatter = ax3.scatter(mz_vals, charges, state_counts,
                             c=colors, cmap=self.colormap, s=200, alpha=0.8,
                             edgecolors='white', linewidths=1.5)
        ax3.set_xlabel('m/z')
        ax3.set_ylabel('Charge')
        ax3.set_zlabel('State Count')
        ax3.set_title('C', loc='left', fontweight='bold')
        ax3.view_init(elev=20, azim=45)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3, pad=0.1, shrink=0.8)
        cbar.set_label('Energy (eV)', rotation=270, labelpad=15)

        # Chart 4: Stage Breakdown (stacked for first ion)
        ax4 = plt.subplot(144)

        # Get stage breakdown from first ion
        if vals and 'stage_breakdown' in vals[0]:
            stages = vals[0]['stage_breakdown']
            stage_names = [s['to_stage'].replace('_', ' ').title() for s in stages]
            delta_M = [s['delta_M'] for s in stages]

            # Create stacked bar
            y_pos = np.arange(len(stage_names))
            colors_stages = plt.cm.viridis(np.linspace(0.2, 0.8, len(stage_names)))

            ax4.barh(y_pos, delta_M, color=colors_stages, edgecolor='white',
                    linewidth=1)
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(stage_names, fontsize=7)
            ax4.set_xlabel('State Count ΔM')
            ax4.grid(True, alpha=0.3, linestyle='--', axis='x')
            ax4.set_title('D', loc='left', fontweight='bold')
        else:
            # Fallback: plot state count vs time
            ax4.plot(total_times, state_counts, 'o', color='#F18F01',
                    markersize=8, markerfacecolor='white', markeredgewidth=2)
            ax4.set_xlabel('Time (μs)')
            ax4.set_ylabel('State Count')
            ax4.grid(True, alpha=0.3, linestyle='--')
            ax4.set_title('D', loc='left', fontweight='bold')

        plt.tight_layout()
        output_file = self.output_dir / f"{base_name}_panel_4_trajectory.png"
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def generate_pipeline_panel(self, data: Dict[str, Any], base_name: str):
        """Panel 5: Pipeline Validation (4 charts)."""
        fig = plt.figure(figsize=(12, 3))

        validations = data['validations']

        # Extract trans-planckian data
        tp_vals = validations['trans_planckian']['validations'][:50]  # Limit for visualization
        mz_tp = [v['mz'] for v in tp_vals]
        state_count = [v['state_count_M'] for v in tp_vals]
        capacity = [v['cumulative_capacity'] for v in tp_vals]
        n_vals = [v['n'] for v in tp_vals]

        # Chart 1: State Count Distribution
        ax1 = plt.subplot(141)
        ax1.hist(state_count, bins=20, color='#2E86AB', alpha=0.7,
                edgecolor='white', linewidth=1)
        ax1.axvline(np.mean(state_count), color='red', linestyle='--',
                   linewidth=2, alpha=0.7)
        ax1.set_xlabel('State Count M')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax1.set_title('A', loc='left', fontweight='bold')

        # Chart 2: Capacity Bounds Check
        ax2 = plt.subplot(142)
        fraction_used = np.array(state_count) / np.array(capacity)
        ax2.scatter(mz_tp, fraction_used, c=n_vals, cmap=self.colormap,
                   s=50, alpha=0.7, edgecolors='white', linewidths=0.5)
        ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel('m/z')
        ax2.set_ylabel('M / Capacity')
        ax2.set_ylim([0, 1.1])
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_title('B', loc='left', fontweight='bold')

        # Chart 3: 3D - (m/z, n, state_count)
        ax3 = plt.subplot(143, projection='3d')

        scatter = ax3.scatter(mz_tp, n_vals, state_count,
                             c=capacity, cmap=self.colormap, s=100, alpha=0.7,
                             edgecolors='white', linewidths=0.5)
        ax3.set_xlabel('m/z')
        ax3.set_ylabel('n')
        ax3.set_zlabel('State Count M')
        ax3.set_title('C', loc='left', fontweight='bold')
        ax3.view_init(elev=20, azim=45)

        # Chart 4: CatScript Validation - n distribution and validity
        ax4 = plt.subplot(144)

        cs_vals = validations['catscript']['validations'][:50]
        n_cs = [v['n'] for v in cs_vals]
        n_correct = [v['n_correct'] for v in cs_vals]
        l_valid = [v['l_valid'] for v in cs_vals]
        m_valid = [v['m_valid'] for v in cs_vals]
        s_valid = [v['s_valid'] for v in cs_vals]

        # Count validation results
        all_valid = [nc and lv and mv and sv for nc, lv, mv, sv in
                    zip(n_correct, l_valid, m_valid, s_valid)]

        # Create stacked validation bars
        x_pos = np.arange(min(len(n_cs), 20))  # Limit to 20 for visibility
        colors_valid = ['#2E86AB' if v else '#C73E1D' for v in all_valid[:20]]

        ax4.bar(x_pos, n_cs[:20], color=colors_valid, alpha=0.7,
               edgecolor='white', linewidth=1)
        ax4.set_xlabel('Ion Index')
        ax4.set_ylabel('Principal Number n')
        ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax4.set_title('D', loc='left', fontweight='bold')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#2E86AB', label='Valid'),
                          Patch(facecolor='#C73E1D', label='Invalid')]
        ax4.legend(handles=legend_elements, frameon=False, loc='upper right')

        plt.tight_layout()
        output_file = self.output_dir / f"{base_name}_panel_5_pipeline.png"
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

    def generate_statistics_panel(self, data: Dict[str, Any], base_name: str):
        """Panel 6: Statistical Analysis (4 charts)."""
        fig = plt.figure(figsize=(12, 3))

        tp = data.get('trans_planckian', {})
        cs = data.get('catscript', {})
        cc = data.get('categorical_cryogenics', {})
        regime_dist = data.get('regime_distribution', {})

        # Chart 1: Trans-Planckian Statistics
        ax1 = plt.subplot(141)
        if tp:
            metrics = ['Mean M', 'Std M', 'Min M', 'Max M']
            values = [
                tp.get('mean_state_count', 0),
                tp.get('std_state_count', 0),
                tp.get('min_state_count', 0),
                tp.get('max_state_count', 0)
            ]
            colors_bar = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
            bars = ax1.bar(range(len(metrics)), values, color=colors_bar,
                          alpha=0.7, edgecolor='white', linewidth=1.5)
            ax1.set_xticks(range(len(metrics)))
            ax1.set_xticklabels(metrics, rotation=45, ha='right')
            ax1.set_ylabel('State Count')
            ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax1.set_title('A', loc='left', fontweight='bold')

        # Chart 2: CatScript Statistics
        ax2 = plt.subplot(142)
        if cs:
            metrics = ['Mean n', 'Std n', 'Min n', 'Max n']
            values = [
                cs.get('mean_n', 0),
                cs.get('std_n', 0),
                cs.get('min_n', 0),
                cs.get('max_n', 0)
            ]
            ax2.bar(range(len(metrics)), values, color='#2E86AB',
                   alpha=0.7, edgecolor='white', linewidth=1.5)
            ax2.set_xticks(range(len(metrics)))
            ax2.set_xticklabels(metrics, rotation=45, ha='right')
            ax2.set_ylabel('n value')
            ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax2.set_title('B', loc='left', fontweight='bold')

        # Chart 3: 3D - Temperature Statistics
        ax3 = plt.subplot(143, projection='3d')
        if cc:
            # Create synthetic distribution around mean
            mean_T = cc.get('mean_temperature_K', 0)
            std_T = cc.get('std_temperature_K', 1e-8)

            # Generate points in 3D (T, suppression, frequency)
            n_points = 100
            T_samples = np.random.normal(mean_T, std_T, n_points)
            supp_samples = np.random.normal(cc.get('mean_suppression', 0),
                                           std_T / mean_T if mean_T > 0 else 1e-10,
                                           n_points)
            freq_samples = np.arange(n_points)

            scatter = ax3.scatter(T_samples * 1e3, supp_samples * 1e6, freq_samples,
                                 c=freq_samples, cmap=self.colormap, s=30,
                                 alpha=0.6, edgecolors='white', linewidths=0.3)
            ax3.set_xlabel('T (mK)')
            ax3.set_ylabel('Suppression (×10⁻⁶)')
            ax3.set_zlabel('Sample Index')
            ax3.set_title('C', loc='left', fontweight='bold')
            ax3.view_init(elev=20, azim=45)

        # Chart 4: Regime Distribution
        ax4 = plt.subplot(144)
        if regime_dist:
            regimes = list(regime_dist.keys())
            counts = list(regime_dist.values())
            colors_regime = plt.cm.viridis(np.linspace(0.2, 0.8, len(regimes)))

            ax4.pie(counts, labels=regimes, autopct='%1.1f%%',
                   colors=colors_regime, startangle=90,
                   textprops={'fontsize': 7})
            ax4.set_title('D', loc='left', fontweight='bold')
        else:
            # Show validation fraction
            if tp and cs and cc:
                labels = ['Trans-\nPlanckian', 'CatScript', 'Categorical\nCryogenics']
                fractions = [
                    tp.get('fraction_bounded', 0),
                    cs.get('fraction_valid', 0),
                    cc.get('fraction_matching', 0)
                ]
                colors_frac = ['#2E86AB', '#A23B72', '#F18F01']
                bars = ax4.bar(range(len(labels)), fractions, color=colors_frac,
                              alpha=0.7, edgecolor='white', linewidth=1.5)
                ax4.set_xticks(range(len(labels)))
                ax4.set_xticklabels(labels, fontsize=7)
                ax4.set_ylabel('Fraction Valid')
                ax4.set_ylim([0, 1.1])
                ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
                ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
                ax4.set_title('D', loc='left', fontweight='bold')

        plt.tight_layout()
        output_file = self.output_dir / f"{base_name}_panel_6_statistics.png"
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate panel figures from validation results"
    )
    parser.add_argument(
        'results_file',
        nargs='?',
        help='Path to validation results JSON file'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all validation result files'
    )
    parser.add_argument(
        '--output-dir',
        default='figures',
        help='Output directory for figures (default: figures/)'
    )

    args = parser.parse_args()

    generator = ValidationFigureGenerator(output_dir=args.output_dir)

    if args.all:
        # Find all validation result files
        results_dir = Path('validation_results')
        result_files = list(results_dir.glob('validation_results_*.json'))

        if not result_files:
            print("No validation result files found in validation_results/")
            return

        print(f"Found {len(result_files)} result file(s)")
        for result_file in result_files:
            generator.generate_all_panels(str(result_file))

    elif args.results_file:
        generator.generate_all_panels(args.results_file)

    else:
        print("Please specify a results file or use --all flag")
        print("\nUsage:")
        print("  python generate_validation_figures.py results.json")
        print("  python generate_validation_figures.py --all")
        return

    print("\n" + "=" * 70)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 70)
    print(f"All panels saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
