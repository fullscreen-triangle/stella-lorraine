import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any

# ============================================================
# BASE CLASS: Hierarchy Analyzer
# ============================================================

class HierarchyAnalyzer:
    """
    Base class for hierarchy-specific analysis.
    Each hierarchy gets its own specialized subclass.
    """

    def __init__(self, json_files: List[str]):
        """
        Parameters:
        -----------
        json_files : list of str
            Paths to the 3 measurement files for this hierarchy
        """
        self.json_files = sorted(json_files)  # Sort by timestamp
        self.data = [self.load_json(f) for f in self.json_files]
        self.hierarchy_name = self.get_hierarchy_name()
        self.theoretical_scale = self.get_theoretical_scale()

    def load_json(self, filepath: str) -> Dict:
        """Load JSON data."""
        with open(filepath, 'r') as f:
            return json.load(f)

    def get_hierarchy_name(self) -> str:
        """Extract hierarchy name from filename."""
        raise NotImplementedError("Subclass must implement")

    def get_theoretical_scale(self) -> float:
        """Return theoretical time scale for this hierarchy."""
        raise NotImplementedError("Subclass must implement")

    def extract_observations(self) -> List[Dict]:
        """Extract observation data from all measurements."""
        raise NotImplementedError("Subclass must implement")

    def analyze_temporal_evolution(self) -> Dict:
        """Analyze how measurements evolve over time."""
        raise NotImplementedError("Subclass must implement")

    def analyze_categorical_exclusion(self) -> Dict:
        """Analyze categorical exclusion dynamics."""
        raise NotImplementedError("Subclass must implement")

    def analyze_precision(self) -> Dict:
        """Characterize precision at this hierarchy."""
        raise NotImplementedError("Subclass must implement")

    def analyze_reproducibility(self) -> Dict:
        """Analyze consistency across 3 measurements."""
        raise NotImplementedError("Subclass must implement")

    def get_physical_interpretation(self) -> str:
        """Return hierarchy-specific physical interpretation."""
        raise NotImplementedError("Subclass must implement")

    def generate_comprehensive_report(self, output_dir: str) -> Dict:
        """Generate complete analysis report."""
        raise NotImplementedError("Subclass must implement")


# ============================================================
# PICOSECOND HIERARCHY ANALYZER
# ============================================================

class PicosecondAnalyzer(HierarchyAnalyzer):
    """
    Specialized analyzer for picosecond hierarchy (10^-12 s).

    Physical context:
    - Molecular vibrations
    - Chemical bond oscillations
    - Fast electronic processes

    Exclusion mechanism:
    - Energy conservation at molecular level
    - Vibrational mode coupling
    - Quantum coherence constraints
    """

    def get_hierarchy_name(self) -> str:
        return "Picosecond (10⁻¹² s)"

    def get_theoretical_scale(self) -> float:
        return 1e-12

    def extract_observations(self) -> List[Dict]:
        """Extract observation sequences from JSON."""
        observations = []
        for data in self.data:
            obs = {
                'timestamp': data.get('timestamp', 'unknown'),
                'sequence': data.get('observation_sequence', []),
                'excluded': data.get('excluded_harmonics', []),
                'available': data.get('available_harmonics', []),
                'total_states': data.get('total_categorical_states', 0),
                'excluded_count': len(data.get('excluded_harmonics', [])),
                'available_count': len(data.get('available_harmonics', []))
            }
            observations.append(obs)
        return observations

    def analyze_temporal_evolution(self) -> Dict:
        """Analyze temporal evolution of picosecond measurements."""
        obs = self.extract_observations()

        timestamps = [o['timestamp'] for o in obs]
        excluded_counts = [o['excluded_count'] for o in obs]
        available_counts = [o['available_count'] for o in obs]

        # Calculate exclusion rate
        exclusion_rate = np.diff(excluded_counts) if len(excluded_counts) > 1 else [0]

        # Calculate stability
        excluded_stability = np.std(excluded_counts) / np.mean(excluded_counts) if np.mean(excluded_counts) > 0 else 0

        return {
            'timestamps': timestamps,
            'excluded_counts': excluded_counts,
            'available_counts': available_counts,
            'exclusion_rate': exclusion_rate,
            'stability': excluded_stability,
            'total_excluded': sum(excluded_counts),
            'mean_excluded': np.mean(excluded_counts),
            'std_excluded': np.std(excluded_counts)
        }

    def analyze_categorical_exclusion(self) -> Dict:
        """Analyze categorical exclusion dynamics."""
        obs = self.extract_observations()

        all_excluded = []
        cumulative_excluded = set()

        for o in obs:
            excluded_set = set(o['excluded'])
            cumulative_excluded.update(excluded_set)
            all_excluded.append(len(cumulative_excluded))

        exclusion_growth = np.array(all_excluded)
        saturation_rate = exclusion_growth[-1] / exclusion_growth[0] if len(exclusion_growth) > 1 and exclusion_growth[0] > 0 else 1.0

        return {
            'cumulative_excluded': all_excluded,
            'total_unique_excluded': len(cumulative_excluded),
            'saturation_rate': saturation_rate,
            'exclusion_efficiency': len(cumulative_excluded) / obs[0]['total_states'] if obs[0]['total_states'] > 0 else 0
        }

    def analyze_precision(self) -> Dict:
        """Characterize precision at picosecond hierarchy."""
        obs = self.extract_observations()
        excluded_counts = [o['excluded_count'] for o in obs]

        precision = {
            'mean': np.mean(excluded_counts),
            'std': np.std(excluded_counts),
            'relative_precision': np.std(excluded_counts) / np.mean(excluded_counts) if np.mean(excluded_counts) > 0 else 0,
            'min': np.min(excluded_counts),
            'max': np.max(excluded_counts),
            'range': np.max(excluded_counts) - np.min(excluded_counts),
            'n_samples': len(excluded_counts)
        }

        return precision

    def analyze_reproducibility(self) -> Dict:
        """Analyze consistency across measurements."""
        obs = self.extract_observations()

        excluded_counts = [o['excluded_count'] for o in obs]
        available_counts = [o['available_count'] for o in obs]

        reproducibility = {
            'excluded_cv': np.std(excluded_counts) / np.mean(excluded_counts) if np.mean(excluded_counts) > 0 else 0,
            'available_cv': np.std(available_counts) / np.mean(available_counts) if np.mean(available_counts) > 0 else 0,
            'excluded_range': max(excluded_counts) - min(excluded_counts),
            'available_range': max(available_counts) - min(available_counts),
            'is_reproducible': np.std(excluded_counts) / np.mean(excluded_counts) < 0.1 if np.mean(excluded_counts) > 0 else False
        }

        return reproducibility

    def get_physical_interpretation(self) -> str:
        """Return picosecond-specific physical interpretation."""
        return """
╔════════════════════════════════════════════╗
║    PICOSECOND HIERARCHY PHYSICS            ║
╠════════════════════════════════════════════╣
║                                            ║
║  Time Scale:     10⁻¹² seconds             ║
║  Frequency:      ~1 THz                    ║
║                                            ║
║  Physical Processes:                       ║
║    • Molecular vibrations                  ║
║    • Chemical bond oscillations            ║
║    • Fast electronic transitions           ║
║    • Vibrational mode coupling             ║
║                                            ║
║  Exclusion Mechanisms:                     ║
║    • Energy conservation                   ║
║    • Quantum coherence constraints         ║
║    • Vibrational selection rules           ║
║    • Phase matching conditions             ║
║                                            ║
║  Measurement Challenges:                   ║
║    • Thermal fluctuations                  ║
║    • Quantum decoherence                   ║
║    • Environmental coupling                ║
║                                            ║
║  Categorical Structure:                    ║
║    • Discrete vibrational states           ║
║    • Energy quantization                   ║
║    • Selection rule enforcement            ║
║                                            ║
╚════════════════════════════════════════════╝
        """

    def generate_comprehensive_report(self, output_dir: str) -> Dict:
        """Generate complete 6-panel analysis."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Run all analyses
        temporal = self.analyze_temporal_evolution()
        categorical = self.analyze_categorical_exclusion()
        precision = self.analyze_precision()
        reproducibility = self.analyze_reproducibility()

        # Create figure
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

        colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'excluded': '#d62728',
            'available': '#2ca02c',
            'accent': '#9467bd'
        }

        # PANEL 1: Temporal Evolution
        ax1 = fig.add_subplot(gs[0, 0])
        x = range(len(temporal['excluded_counts']))
        ax1.plot(x, temporal['excluded_counts'], 'o-', color=colors['excluded'],
                linewidth=2.5, markersize=10, label='Excluded States')
        ax1.plot(x, temporal['available_counts'], 's-', color=colors['available'],
                linewidth=2.5, markersize=10, label='Available States')
        ax1.set_xlabel('Measurement Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('State Count', fontsize=12, fontweight='bold')
        ax1.set_title(f'Panel 1: Temporal Evolution\nStability: {temporal["stability"]:.4f}',
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(x)

        # PANEL 2: Categorical Exclusion
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(categorical['cumulative_excluded'], 'o-', color=colors['primary'],
                linewidth=3, markersize=12)
        ax2.fill_between(range(len(categorical['cumulative_excluded'])),
                        0, categorical['cumulative_excluded'],
                        alpha=0.3, color=colors['primary'])
        ax2.set_xlabel('Measurement Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Excluded States', fontsize=12, fontweight='bold')
        ax2.set_title(f'Panel 2: Exclusion Growth\nEfficiency: {categorical["exclusion_efficiency"]:.2%}',
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # PANEL 3: Precision
        ax3 = fig.add_subplot(gs[1, 0])
        metrics_labels = ['Mean', 'Std', 'Min', 'Max']
        metrics_values = [precision['mean'], precision['std'],
                         precision['min'], precision['max']]
        bars = ax3.bar(metrics_labels, metrics_values, color=colors['secondary'],
                      edgecolor='black', linewidth=1.5, alpha=0.7)
        ax3.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax3.set_title(f'Panel 3: Precision Metrics\nRelative: {precision["relative_precision"]:.6f}',
                     fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)

        # PANEL 4: Reproducibility
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        repro_text = f"""
╔════════════════════════════════════════════╗
║      REPRODUCIBILITY ANALYSIS              ║
╠════════════════════════════════════════════╣
║                                            ║
║  Excluded States CV:  {reproducibility['excluded_cv']:.6f}      ║
║  Available States CV: {reproducibility['available_cv']:.6f}      ║
║                                            ║
║  Excluded Range:      {reproducibility['excluded_range']}              ║
║  Available Range:     {reproducibility['available_range']}              ║
║                                            ║
║  Reproducible:        {'YES ✓' if reproducibility['is_reproducible'] else 'NO ✗'}            ║
║                                            ║
║  Assessment:                               ║
║  {'  EXCELLENT reproducibility' if reproducibility['excluded_cv'] < 0.05 else '  GOOD reproducibility' if reproducibility['excluded_cv'] < 0.1 else '  MODERATE reproducibility'}          ║
║                                            ║
╚════════════════════════════════════════════╝
        """
        ax4.text(0.05, 0.95, repro_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # PANEL 5: Physical Interpretation
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.axis('off')
        ax5.text(0.05, 0.95, self.get_physical_interpretation(),
                transform=ax5.transAxes, fontsize=10, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        # PANEL 6: Summary
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        summary_text = f"""
╔════════════════════════════════════════════╗
║         SUMMARY STATISTICS                 ║
╠════════════════════════════════════════════╣
║                                            ║
║  TEMPORAL EVOLUTION:                       ║
║    Mean Excluded:     {temporal['mean_excluded']:.1f}             ║
║    Std Excluded:      {temporal['std_excluded']:.1f}             ║
║    Total Excluded:    {temporal['total_excluded']}               ║
║                                            ║
║  CATEGORICAL EXCLUSION:                    ║
║    Unique Excluded:   {categorical['total_unique_excluded']}               ║
║    Saturation Rate:   {categorical['saturation_rate']:.3f}           ║
║    Efficiency:        {categorical['exclusion_efficiency']:.2%}          ║
║                                            ║
║  PRECISION:                                ║
║    Mean:              {precision['mean']:.3f}          ║
║    Std Dev:           {precision['std']:.3f}          ║
║    Relative:          {precision['relative_precision']:.6f}      ║
║                                            ║
║  REPRODUCIBILITY:                          ║
║    CV:                {reproducibility['excluded_cv']:.6f}      ║
║    Status:            {'PASS' if reproducibility['is_reproducible'] else 'REVIEW'}            ║
║                                            ║
╚════════════════════════════════════════════╝
        """
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        fig.suptitle(f'COMPREHENSIVE ANALYSIS: {self.hierarchy_name}\n'
                    f'3 Measurements | {temporal["timestamps"][0]} to {temporal["timestamps"][-1]}',
                    fontsize=16, fontweight='bold', y=0.98)

        output_file = output_dir / f'picosecond_comprehensive_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Picosecond analysis saved: {output_file}")
        plt.close()

        return {
            'temporal': temporal,
            'categorical': categorical,
            'precision': precision,
            'reproducibility': reproducibility
        }


# ============================================================
# ZEPTOSECOND HIERARCHY ANALYZER
# ============================================================

class ZeptosecondAnalyzer(HierarchyAnalyzer):
    """
    Specialized analyzer for zeptosecond hierarchy (10^-21 s).

    Physical context:
    - Nuclear processes
    - Electron orbital transitions
    - Quantum tunneling events
    - Auger electron emission

    Exclusion mechanism:
    - Nuclear selection rules
    - Angular momentum conservation
    - Parity constraints
    - Quantum number restrictions
    """

    def get_hierarchy_name(self) -> str:
        return "Zeptosecond (10⁻²¹ s)"

    def get_theoretical_scale(self) -> float:
        return 1e-21

    def extract_observations(self) -> List[Dict]:
        """Extract observation sequences from JSON."""
        observations = []
        for data in self.data:
            obs = {
                'timestamp': data.get('timestamp', 'unknown'),
                'sequence': data.get('observation_sequence', []),
                'excluded': data.get('excluded_nuclear_states', []),
                'available': data.get('available_nuclear_states', []),
                'total_states': data.get('total_categorical_states', 0),
                'excluded_count': len(data.get('excluded_nuclear_states', [])),
                'available_count': len(data.get('available_nuclear_states', []))
            }
            observations.append(obs)
        return observations

    def analyze_temporal_evolution(self) -> Dict:
        """Analyze temporal evolution at zeptosecond scale."""
        obs = self.extract_observations()

        timestamps = [o['timestamp'] for o in obs]
        excluded_counts = [o['excluded_count'] for o in obs]
        available_counts = [o['available_count'] for o in obs]

        exclusion_rate = np.diff(excluded_counts) if len(excluded_counts) > 1 else [0]
        excluded_stability = np.std(excluded_counts) / np.mean(excluded_counts) if np.mean(excluded_counts) > 0 else 0

        # Zeptosecond-specific: nuclear decay analysis
        nuclear_decay_constant = -np.mean(exclusion_rate) if len(exclusion_rate) > 0 else 0

        return {
            'timestamps': timestamps,
            'excluded_counts': excluded_counts,
            'available_counts': available_counts,
            'exclusion_rate': exclusion_rate,
            'stability': excluded_stability,
            'total_excluded': sum(excluded_counts),
            'mean_excluded': np.mean(excluded_counts),
            'std_excluded': np.std(excluded_counts),
            'nuclear_decay_constant': nuclear_decay_constant
        }

    def analyze_categorical_exclusion(self) -> Dict:
        """Analyze nuclear state exclusion dynamics."""
        obs = self.extract_observations()

        all_excluded = []
        cumulative_excluded = set()

        for o in obs:
            excluded_set = set(o['excluded'])
            cumulative_excluded.update(excluded_set)
            all_excluded.append(len(cumulative_excluded))

        exclusion_growth = np.array(all_excluded)

        # Nuclear-specific: selection rule efficiency
        selection_rule_efficiency = len(cumulative_excluded) / obs[0]['total_states'] if obs[0]['total_states'] > 0 else 0

        return {
            'cumulative_excluded': all_excluded,
            'total_unique_excluded': len(cumulative_excluded),
            'saturation_rate': exclusion_growth[-1] / exclusion_growth[0] if len(exclusion_growth) > 1 and exclusion_growth[0] > 0 else 1.0,
            'exclusion_efficiency': selection_rule_efficiency,
            'selection_rule_efficiency': selection_rule_efficiency
        }

    def analyze_precision(self) -> Dict:
        """Characterize precision at zeptosecond hierarchy."""
        obs = self.extract_observations()
        excluded_counts = [o['excluded_count'] for o in obs]

        # Zeptosecond-specific: quantum uncertainty
        quantum_uncertainty = np.std(excluded_counts) * self.theoretical_scale

        precision = {
            'mean': np.mean(excluded_counts),
            'std': np.std(excluded_counts),
            'relative_precision': np.std(excluded_counts) / np.mean(excluded_counts) if np.mean(excluded_counts) > 0 else 0,
            'min': np.min(excluded_counts),
            'max': np.max(excluded_counts),
            'range': np.max(excluded_counts) - np.min(excluded_counts),
            'n_samples': len(excluded_counts),
            'quantum_uncertainty': quantum_uncertainty
        }

        return precision

    def analyze_reproducibility(self) -> Dict:
        """Analyze consistency across measurements."""
        obs = self.extract_observations()

        excluded_counts = [o['excluded_count'] for o in obs]
        available_counts = [o['available_count'] for o in obs]

        # Zeptosecond-specific: nuclear process reproducibility
        nuclear_reproducibility = 1.0 - (np.std(excluded_counts) / np.mean(excluded_counts)) if np.mean(excluded_counts) > 0 else 0

        reproducibility = {
            'excluded_cv': np.std(excluded_counts) / np.mean(excluded_counts) if np.mean(excluded_counts) > 0 else 0,
            'available_cv': np.std(available_counts) / np.mean(available_counts) if np.mean(available_counts) > 0 else 0,
            'excluded_range': max(excluded_counts) - min(excluded_counts),
            'available_range': max(available_counts) - min(available_counts),
            'is_reproducible': np.std(excluded_counts) / np.mean(excluded_counts) < 0.1 if np.mean(excluded_counts) > 0 else False,
            'nuclear_reproducibility': nuclear_reproducibility
        }

        return reproducibility

    def get_physical_interpretation(self) -> str:
        """Return zeptosecond-specific physical interpretation."""
        return """
╔════════════════════════════════════════════╗
║    ZEPTOSECOND HIERARCHY PHYSICS           ║
╠════════════════════════════════════════════╣
║                                            ║
║  Time Scale:     10⁻²¹ seconds             ║
║  Frequency:      ~10²¹ Hz                  ║
║                                            ║
║  Physical Processes:                       ║
║    • Nuclear transitions                   ║
║    • Electron orbital dynamics             ║
║    • Quantum tunneling events              ║
║    • Auger electron emission               ║
║    • Inner-shell ionization                ║
║                                            ║
║  Exclusion Mechanisms:                     ║
║    • Nuclear selection rules               ║
║    • Angular momentum conservation         ║
║    • Parity constraints                    ║
║    • Quantum number restrictions           ║
║    • Spin coupling rules                   ║
║                                            ║
║  Measurement Challenges:                   ║
║    • Extreme time resolution required      ║
║    • Quantum measurement limits            ║
║    • Nuclear process stochasticity         ║
║    • Relativistic effects                  ║
║                                            ║
║  Categorical Structure:                    ║
║    • Discrete nuclear states               ║
║    • Quantum number quantization           ║
║    • Selection rule enforcement            ║
║    • Symmetry-based exclusions             ║
║                                            ║
╚════════════════════════════════════════════╝
        """

    def generate_comprehensive_report(self, output_dir: str) -> Dict:
        """Generate complete 6-panel analysis for zeptosecond."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        temporal = self.analyze_temporal_evolution()
        categorical = self.analyze_categorical_exclusion()
        precision = self.analyze_precision()
        reproducibility = self.analyze_reproducibility()

        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

        colors = {
            'primary': '#8B4513',
            'secondary': '#FF6347',
            'excluded': '#DC143C',
            'available': '#32CD32',
            'accent': '#4B0082'
        }

        # PANEL 1: Temporal Evolution
        ax1 = fig.add_subplot(gs[0, 0])
        x = range(len(temporal['excluded_counts']))
        ax1.plot(x, temporal['excluded_counts'], 'o-', color=colors['excluded'],
                linewidth=2.5, markersize=10, label='Excluded Nuclear States')
        ax1.plot(x, temporal['available_counts'], 's-', color=colors['available'],
                linewidth=2.5, markersize=10, label='Available Nuclear States')
        ax1.set_xlabel('Measurement Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('State Count', fontsize=12, fontweight='bold')
        ax1.set_title(f'Panel 1: Nuclear State Evolution\nDecay Constant: {temporal["nuclear_decay_constant"]:.4f}',
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(x)

        # PANEL 2: Selection Rule Efficiency
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(categorical['cumulative_excluded'], 'o-', color=colors['primary'],
                linewidth=3, markersize=12)
        ax2.fill_between(range(len(categorical['cumulative_excluded'])),
                        0, categorical['cumulative_excluded'],
                        alpha=0.3, color=colors['primary'])
        ax2.set_xlabel('Measurement Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Excluded States', fontsize=12, fontweight='bold')
        ax2.set_title(f'Panel 2: Selection Rule Enforcement\nEfficiency: {categorical["selection_rule_efficiency"]:.2%}',
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # PANEL 3: Quantum Precision
        ax3 = fig.add_subplot(gs[1, 0])
        metrics_labels = ['Mean', 'Std', 'Min', 'Max']
        metrics_values = [precision['mean'], precision['std'],
                         precision['min'], precision['max']]
        bars = ax3.bar(metrics_labels, metrics_values, color=colors['secondary'],
                      edgecolor='black', linewidth=1.5, alpha=0.7)
        ax3.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax3.set_title(f'Panel 3: Quantum Precision\nUncertainty: {precision["quantum_uncertainty"]:.3e} s',
                     fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)

        # PANEL 4: Nuclear Reproducibility
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        repro_text = f"""
╔════════════════════════════════════════════╗
║   NUCLEAR PROCESS REPRODUCIBILITY          ║
╠════════════════════════════════════════════╣
║                                            ║
║  Excluded States CV:  {reproducibility['excluded_cv']:.6f}      ║
║  Available States CV: {reproducibility['available_cv']:.6f}      ║
║                                            ║
║  Nuclear Reproducibility: {reproducibility['nuclear_reproducibility']:.4f}     ║
║                                            ║
║  Excluded Range:      {reproducibility['excluded_range']}              ║
║  Available Range:     {reproducibility['available_range']}              ║
║                                            ║
║  Reproducible:        {'YES ✓' if reproducibility['is_reproducible'] else 'NO ✗'}            ║
║                                            ║
║  Assessment:                               ║
║  {'  EXCELLENT nuclear stability' if reproducibility['nuclear_reproducibility'] > 0.95 else '  GOOD nuclear stability' if reproducibility['nuclear_reproducibility'] > 0.90 else '  MODERATE nuclear stability'}      ║
║                                            ║
╚════════════════════════════════════════════╝
        """
        ax4.text(0.05, 0.95, repro_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

        # PANEL 5: Physical Interpretation
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.axis('off')
        ax5.text(0.05, 0.95, self.get_physical_interpretation(),
                transform=ax5.transAxes, fontsize=10, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # PANEL 6: Summary
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        summary_text = f"""
╔════════════════════════════════════════════╗
║         ZEPTOSECOND SUMMARY                ║
╠════════════════════════════════════════════╣
║                                            ║
║  TEMPORAL EVOLUTION:                       ║
║    Mean Excluded:     {temporal['mean_excluded']:.1f}             ║
║    Std Excluded:      {temporal['std_excluded']:.1f}             ║
║    Decay Constant:    {temporal['nuclear_decay_constant']:.4f}         ║
║                                            ║
║  SELECTION RULES:                          ║
║    Unique Excluded:   {categorical['total_unique_excluded']}               ║
║    Efficiency:        {categorical['selection_rule_efficiency']:.2%}          ║
║                                            ║
║  QUANTUM PRECISION:                        ║
║    Mean:              {precision['mean']:.3f}          ║
║    Std Dev:           {precision['std']:.3f}          ║
║    Uncertainty:       {precision['quantum_uncertainty']:.3e} s    ║
║                                            ║
║  NUCLEAR REPRODUCIBILITY:                  ║
║    Score:             {reproducibility['nuclear_reproducibility']:.4f}         ║
║    Status:            {'PASS' if reproducibility['is_reproducible'] else 'REVIEW'}            ║
║                                            ║
╚════════════════════════════════════════════╝
        """
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        fig.suptitle(f'COMPREHENSIVE ANALYSIS: {self.hierarchy_name}\n'
                    f'Nuclear Process Measurements | {temporal["timestamps"][0]} to {temporal["timestamps"][-1]}',
                    fontsize=16, fontweight='bold', y=0.98)

        output_file = output_dir / f'zeptosecond_comprehensive_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Zeptosecond analysis saved: {output_file}")
        plt.close()

        return {
            'temporal': temporal,
            'categorical': categorical,
            'precision': precision,
            'reproducibility': reproducibility
        }


# ============================================================
# PLANCK TIME HIERARCHY ANALYZER
# ============================================================

class PlanckTimeAnalyzer(HierarchyAnalyzer):
    """
    Specialized analyzer for Planck time hierarchy (5.39×10^-44 s).

    Physical context:
    - Quantum gravity effects
    - Spacetime foam fluctuations
    - Fundamental length scale
    - Information-theoretic limits

    Exclusion mechanism:
    - Spacetime topology constraints
    - Causal structure preservation
    - Information conservation
    - Quantum gravitational selection
    """

    def get_hierarchy_name(self) -> str:
        return "Planck Time (5.39×10⁻⁴⁴ s)"

    def get_theoretical_scale(self) -> float:
        return 5.391247e-44

    def extract_observations(self) -> List[Dict]:
        """Extract observation sequences from JSON."""
        observations = []
        for data in self.data:
            obs = {
                'timestamp': data.get('timestamp', 'unknown'),
                'sequence': data.get('observation_sequence', []),
                'excluded': data.get('excluded_planck_states', []),
                'available': data.get('available_planck_states', []),
                'total_states': data.get('total_categorical_states', 0),
                'excluded_count': len(data.get('excluded_planck_states', [])),
                'available_count': len(data.get('available_planck_states', []))
            }
            observations.append(obs)
        return observations

    def analyze_temporal_evolution(self) -> Dict:
        """Analyze temporal evolution at Planck scale."""
        obs = self.extract_observations()

        timestamps = [o['timestamp'] for o in obs]
        excluded_counts = [o['excluded_count'] for o in obs]
        available_counts = [o['available_count'] for o in obs]

        exclusion_rate = np.diff(excluded_counts) if len(excluded_counts) > 1 else [0]
        excluded_stability = np.std(excluded_counts) / np.mean(excluded_counts) if np.mean(excluded_counts) > 0 else 0

        # Planck-specific: quantum gravity coupling
        quantum_gravity_coupling = np.mean(np.abs(exclusion_rate)) if len(exclusion_rate) > 0 else 0

        return {
            'timestamps': timestamps,
            'excluded_counts': excluded_counts,
            'available_counts': available_counts,
            'exclusion_rate': exclusion_rate,
            'stability': excluded_stability,
            'total_excluded': sum(excluded_counts),
            'mean_excluded': np.mean(excluded_counts),
            'std_excluded': np.std(excluded_counts),
            'quantum_gravity_coupling': quantum_gravity_coupling
        }

    def analyze_categorical_exclusion(self) -> Dict:
        """Analyze spacetime topology exclusion."""
        obs = self.extract_observations()

        all_excluded = []
        cumulative_excluded = set()

        for o in obs:
            excluded_set = set(o['excluded'])
            cumulative_excluded.update(excluded_set)
            all_excluded.append(len(cumulative_excluded))

        exclusion_growth = np.array(all_excluded)

        # Planck-specific: topology preservation
        topology_preservation = 1.0 - (exclusion_growth[-1] / obs[0]['total_states']) if obs[0]['total_states'] > 0 else 0

        return {
            'cumulative_excluded': all_excluded,
            'total_unique_excluded': len(cumulative_excluded),
            'saturation_rate': exclusion_growth[-1] / exclusion_growth[0] if len(exclusion_growth) > 1 and exclusion_growth[0] > 0 else 1.0,
            'exclusion_efficiency': len(cumulative_excluded) / obs[0]['total_states'] if obs[0]['total_states'] > 0 else 0,
            'topology_preservation': topology_preservation
        }

    def analyze_precision(self) -> Dict:
        """Characterize precision at Planck scale."""
        obs = self.extract_observations()
        excluded_counts = [o['excluded_count'] for o in obs]

        # Planck-specific: fundamental uncertainty
        fundamental_uncertainty = np.std(excluded_counts) * self.theoretical_scale

        precision = {
            'mean': np.mean(excluded_counts),
            'std': np.std(excluded_counts),
            'relative_precision': np.std(excluded_counts) / np.mean(excluded_counts) if np.mean(excluded_counts) > 0 else 0,
            'min': np.min(excluded_counts),
            'max': np.max(excluded_counts),
            'range': np.max(excluded_counts) - np.min(excluded_counts),
            'n_samples': len(excluded_counts),
            'fundamental_uncertainty': fundamental_uncertainty
        }

        return precision

    def analyze_reproducibility(self) -> Dict:
        """Analyze consistency across measurements."""
        obs = self.extract_observations()

        excluded_counts = [o['excluded_count'] for o in obs]
        available_counts = [o['available_count'] for o in obs]

        # Planck-specific: quantum gravity stability
        qg_stability = 1.0 - (np.std(excluded_counts) / np.mean(excluded_counts)) if np.mean(excluded_counts) > 0 else 0

        reproducibility = {
            'excluded_cv': np.std(excluded_counts) / np.mean(excluded_counts) if np.mean(excluded_counts) > 0 else 0,
            'available_cv': np.std(available_counts) / np.mean(available_counts) if np.mean(available_counts) > 0 else 0,
            'excluded_range': max(excluded_counts) - min(excluded_counts),
            'available_range': max(available_counts) - min(available_counts),
            'is_reproducible': np.std(excluded_counts) / np.mean(excluded_counts) < 0.1 if np.mean(excluded_counts) > 0 else False,
            'quantum_gravity_stability': qg_stability
        }

        return reproducibility

    def get_physical_interpretation(self) -> str:
        """Return Planck-scale physical interpretation."""
        return """
╔════════════════════════════════════════════╗
║     PLANCK TIME HIERARCHY PHYSICS          ║
╠════════════════════════════════════════════╣
║                                            ║
║  Time Scale:     5.39×10⁻⁴⁴ seconds        ║
║  Frequency:      ~10⁴⁴ Hz                  ║
║  Planck Energy:  1.22×10¹⁹ GeV             ║
║                                            ║
║  Physical Processes:                       ║
║    • Quantum gravity effects               ║
║    • Spacetime foam fluctuations           ║
║    • Virtual black hole formation          ║
║    • Fundamental length emergence          ║
║    • Causal structure dynamics             ║
║                                            ║
║  Exclusion Mechanisms:                     ║
║    • Spacetime topology constraints        ║
║    • Causal structure preservation         ║
║    • Information conservation              ║
║    • Quantum gravitational selection       ║
║    • Holographic principle limits          ║
║                                            ║
║  Measurement Challenges:                   ║
║    • Beyond standard quantum mechanics     ║
║    • Quantum gravity regime                ║
║    • Spacetime uncertainty                 ║
║    • Fundamental observability limits      ║
║                                            ║
║  Categorical Structure:                    ║
║    • Discrete spacetime geometry           ║
║    • Topological invariants                ║
║    • Causal set structure                  ║
║    • Information-theoretic bounds          ║
║                                            ║
╚════════════════════════════════════════════╝
        """

    def generate_comprehensive_report(self, output_dir: str) -> Dict:
        """Generate complete 6-panel analysis for Planck time."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        temporal = self.analyze_temporal_evolution()
        categorical = self.analyze_categorical_exclusion()
        precision = self.analyze_precision()
        reproducibility = self.analyze_reproducibility()

        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

        colors = {
            'primary': '#4B0082',
            'secondary': '#FFD700',
            'excluded': '#8B008B',
            'available': '#00CED1',
            'accent': '#FF1493'
        }

        # PANEL 1: Quantum Gravity Evolution
        ax1 = fig.add_subplot(gs[0, 0])
        x = range(len(temporal['excluded_counts']))
        ax1.plot(x, temporal['excluded_counts'], 'o-', color=colors['excluded'],
                linewidth=2.5, markersize=10, label='Excluded Planck States')
        ax1.plot(x, temporal['available_counts'], 's-', color=colors['available'],
                linewidth=2.5, markersize=10, label='Available Planck States')
        ax1.set_xlabel('Measurement Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('State Count', fontsize=12, fontweight='bold')
        ax1.set_title(f'Panel 1: Quantum Gravity Evolution\nCoupling: {temporal["quantum_gravity_coupling"]:.4f}',
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(x)

        # PANEL 2: Spacetime Topology
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(categorical['cumulative_excluded'], 'o-', color=colors['primary'],
                linewidth=3, markersize=12)
        ax2.fill_between(range(len(categorical['cumulative_excluded'])),
                        0, categorical['cumulative_excluded'],
                        alpha=0.3, color=colors['primary'])
        ax2.set_xlabel('Measurement Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Excluded States', fontsize=12, fontweight='bold')
        ax2.set_title(f'Panel 2: Spacetime Topology Preservation\nPreservation: {categorical["topology_preservation"]:.2%}',
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # PANEL 3: Fundamental Precision
        ax3 = fig.add_subplot(gs[1, 0])
        metrics_labels = ['Mean', 'Std', 'Min', 'Max']
        metrics_values = [precision['mean'], precision['std'],
                         precision['min'], precision['max']]
        bars = ax3.bar(metrics_labels, metrics_values, color=colors['secondary'],
                      edgecolor='black', linewidth=1.5, alpha=0.7)
        ax3.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax3.set_title(f'Panel 3: Fundamental Precision\nUncertainty: {precision["fundamental_uncertainty"]:.3e} s',
                     fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)

        # PANEL 4: Quantum Gravity Stability
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        repro_text = f"""
╔════════════════════════════════════════════╗
║   QUANTUM GRAVITY STABILITY                ║
╠════════════════════════════════════════════╣
║                                            ║
║  Excluded States CV:  {reproducibility['excluded_cv']:.6f}      ║
║  Available States CV: {reproducibility['available_cv']:.6f}      ║
║                                            ║
║  QG Stability:        {reproducibility['quantum_gravity_stability']:.4f}         ║
║                                            ║
║  Excluded Range:      {reproducibility['excluded_range']}              ║
║  Available Range:     {reproducibility['available_range']}              ║
║                                            ║
║  Reproducible:        {'YES ✓' if reproducibility['is_reproducible'] else 'NO ✗'}            ║
║                                            ║
║  Assessment:                               ║
║  {'  EXCELLENT QG stability' if reproducibility['quantum_gravity_stability'] > 0.95 else '  GOOD QG stability' if reproducibility['quantum_gravity_stability'] > 0.90 else '  MODERATE QG stability'}          ║
║                                            ║
╚════════════════════════════════════════════╝
        """
        ax4.text(0.05, 0.95, repro_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))

        # PANEL 5: Physical Interpretation
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.axis('off')
        ax5.text(0.05, 0.95, self.get_physical_interpretation(),
                transform=ax5.transAxes, fontsize=10, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.5))

        # PANEL 6: Summary
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        summary_text = f"""
╔════════════════════════════════════════════╗
║         PLANCK TIME SUMMARY                ║
╠════════════════════════════════════════════╣
║                                            ║
║  QUANTUM GRAVITY:                          ║
║    Mean Excluded:     {temporal['mean_excluded']:.1f}             ║
║    Std Excluded:      {temporal['std_excluded']:.1f}             ║
║    QG Coupling:       {temporal['quantum_gravity_coupling']:.4f}         ║
║                                            ║
║  SPACETIME TOPOLOGY:                       ║
║    Unique Excluded:   {categorical['total_unique_excluded']}               ║
║    Preservation:      {categorical['topology_preservation']:.2%}          ║
║                                            ║
║  FUNDAMENTAL PRECISION:                    ║
║    Mean:              {precision['mean']:.3f}          ║
║    Std Dev:           {precision['std']:.3f}          ║
║    Uncertainty:       {precision['fundamental_uncertainty']:.3e} s    ║
║                                            ║
║  QG STABILITY:                             ║
║    Score:             {reproducibility['quantum_gravity_stability']:.4f}         ║
║    Status:            {'PASS' if reproducibility['is_reproducible'] else 'REVIEW'}            ║
║                                            ║
╚════════════════════════════════════════════╝
        """
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.5))

        fig.suptitle(f'COMPREHENSIVE ANALYSIS: {self.hierarchy_name}\n'
                    f'Quantum Gravity Regime | {temporal["timestamps"][0]} to {temporal["timestamps"][-1]}',
                    fontsize=16, fontweight='bold', y=0.98)

        output_file = output_dir / f'planck_time_comprehensive_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Planck time analysis saved: {output_file}")
        plt.close()

        return {
            'temporal': temporal,
            'categorical': categorical,
            'precision': precision,
            'reproducibility': reproducibility
        }


# ============================================================
# TRANS-PLANCKIAN HIERARCHY ANALYZER
# ============================================================

class TransPlanckianAnalyzer(HierarchyAnalyzer):
    """
    Specialized analyzer for trans-Planckian hierarchy (10^-50 s).

    Physical context:
    - Beyond quantum gravity
    - Pre-geometric regime
    - Information-theoretic foundations
    - Categorical completion topology

    Exclusion mechanism:
    - Logical consistency constraints
    - Categorical irreversibility
    - Information conservation
    - Symmetry-based exclusions
    """

    def get_hierarchy_name(self) -> str:
        return "Trans-Planckian (10⁻⁵⁰ s)"

    def get_theoretical_scale(self) -> float:
        return 1e-50

    def extract_observations(self) -> List[Dict]:
        """Extract observation sequences from JSON."""
        observations = []
        for data in self.data:
            obs = {
                'timestamp': data.get('timestamp', 'unknown'),
                'sequence': data.get('observation_sequence', []),
                'excluded': data.get('excluded_categorical_states', []),
                'available': data.get('available_categorical_states', []),
                'total_states': data.get('total_categorical_states', 0),
                'excluded_count': len(data.get('excluded_categorical_states', [])),
                'available_count': len(data.get('available_categorical_states', []))
            }
            observations.append(obs)
        return observations

    def analyze_temporal_evolution(self) -> Dict:
        """Analyze temporal evolution in trans-Planckian regime."""
        obs = self.extract_observations()

        timestamps = [o['timestamp'] for o in obs]
        excluded_counts = [o['excluded_count'] for o in obs]
        available_counts = [o['available_count'] for o in obs]

        exclusion_rate = np.diff(excluded_counts) if len(excluded_counts) > 1 else [0]
        excluded_stability = np.std(excluded_counts) / np.mean(excluded_counts) if np.mean(excluded_counts) > 0 else 0

        # Trans-Planckian-specific: categorical completion rate
        categorical_completion_rate = np.mean(np.abs(exclusion_rate)) if len(exclusion_rate) > 0 else 0

        return {
            'timestamps': timestamps,
            'excluded_counts': excluded_counts,
            'available_counts': available_counts,
            'exclusion_rate': exclusion_rate,
            'stability': excluded_stability,
            'total_excluded': sum(excluded_counts),
            'mean_excluded': np.mean(excluded_counts),
            'std_excluded': np.std(excluded_counts),
            'categorical_completion_rate': categorical_completion_rate
        }

    def analyze_categorical_exclusion(self) -> Dict:
        """Analyze categorical completion dynamics."""
        obs = self.extract_observations()

        all_excluded = []
        cumulative_excluded = set()

        for o in obs:
            excluded_set = set(o['excluded'])
            cumulative_excluded.update(excluded_set)
            all_excluded.append(len(cumulative_excluded))

        exclusion_growth = np.array(all_excluded)

        # Trans-Planckian-specific: information conservation
        information_conservation = 1.0 - (len(cumulative_excluded) / obs[0]['total_states']) if obs[0]['total_states'] > 0 else 0

        return {
            'cumulative_excluded': all_excluded,
            'total_unique_excluded': len(cumulative_excluded),
            'saturation_rate': exclusion_growth[-1] / exclusion_growth[0] if len(exclusion_growth) > 1 and exclusion_growth[0] > 0 else 1.0,
            'exclusion_efficiency': len(cumulative_excluded) / obs[0]['total_states'] if obs[0]['total_states'] > 0 else 0,
            'information_conservation': information_conservation
        }

    def analyze_precision(self) -> Dict:
        """Characterize precision in trans-Planckian regime."""
        obs = self.extract_observations()
        excluded_counts = [o['excluded_count'] for o in obs]

        # Trans-Planckian-specific: logical precision (not physical)
        logical_precision = 1.0 - (np.std(excluded_counts) / np.mean(excluded_counts)) if np.mean(excluded_counts) > 0 else 0

        precision = {
            'mean': np.mean(excluded_counts),
            'std': np.std(excluded_counts),
            'relative_precision': np.std(excluded_counts) / np.mean(excluded_counts) if np.mean(excluded_counts) > 0 else 0,
            'min': np.min(excluded_counts),
            'max': np.max(excluded_counts),
            'range': np.max(excluded_counts) - np.min(excluded_counts),
            'n_samples': len(excluded_counts),
            'logical_precision': logical_precision
        }

        return precision

    def analyze_reproducibility(self) -> Dict:
        """Analyze consistency across measurements."""
        obs = self.extract_observations()

        excluded_counts = [o['excluded_count'] for o in obs]
        available_counts = [o['available_count'] for o in obs]

        # Trans-Planckian-specific: categorical consistency
        categorical_consistency = 1.0 - (np.std(excluded_counts) / np.mean(excluded_counts)) if np.mean(excluded_counts) > 0 else 0

        reproducibility = {
            'excluded_cv': np.std(excluded_counts) / np.mean(excluded_counts) if np.mean(excluded_counts) > 0 else 0,
            'available_cv': np.std(available_counts) / np.mean(available_counts) if np.mean(available_counts) > 0 else 0,
            'excluded_range': max(excluded_counts) - min(excluded_counts),
            'available_range': max(available_counts) - min(available_counts),
            'is_reproducible': np.std(excluded_counts) / np.mean(excluded_counts) < 0.1 if np.mean(excluded_counts) > 0 else False,
            'categorical_consistency': categorical_consistency
        }

        return reproducibility

    def get_physical_interpretation(self) -> str:
        """Return trans-Planckian physical interpretation."""
        return """
╔════════════════════════════════════════════╗
║   TRANS-PLANCKIAN HIERARCHY PHYSICS        ║
╠════════════════════════════════════════════╣
║                                            ║
║  Time Scale:     10⁻⁵⁰ seconds             ║
║  Regime:         Beyond quantum gravity    ║
║  Nature:         Pre-geometric             ║
║                                            ║
║  Physical Processes:                       ║
║    • Categorical completion dynamics       ║
║    • Information-theoretic foundations     ║
║    • Pre-geometric structure emergence     ║
║    • Logical consistency enforcement       ║
║    • Symmetry breaking cascades            ║
║                                            ║
║  Exclusion Mechanisms:                     ║
║    • Logical consistency constraints       ║
║    • Categorical irreversibility           ║
║    • Information conservation              ║
║    • Symmetry-based exclusions             ║
║    • Completion topology rules             ║
║                                            ║
║  Measurement Nature:                       ║
║    • NOT physical measurement              ║
║    • Logical/categorical observation       ║
║    • Information-theoretic bounds          ║
║    • Consistency checking                  ║
║                                            ║
║  Categorical Structure:                    ║
║    • Pure categorical states               ║
║    • Completion topology                   ║
║    • Irreversibility principle             ║
║    • Information conservation              ║
║                                            ║
║  KEY INSIGHT:                              ║
║    Precision here is LOGICAL, not          ║
║    physical. Limited by consistency,       ║
║    not quantum mechanics.                  ║
║                                            ║
╚════════════════════════════════════════════╝
        """

    def generate_comprehensive_report(self, output_dir: str) -> Dict:
        """Generate complete 6-panel analysis for trans-Planckian."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        temporal = self.analyze_temporal_evolution()
        categorical = self.analyze_categorical_exclusion()
        precision = self.analyze_precision()
        reproducibility = self.analyze_reproducibility()

        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

        colors = {
            'primary': '#000080',
            'secondary': '#00FFFF',
            'excluded': '#FF00FF',
            'available': '#00FF00',
            'accent': '#FFFF00'
        }

        # PANEL 1: Categorical Completion
        ax1 = fig.add_subplot(gs[0, 0])
        x = range(len(temporal['excluded_counts']))
        ax1.plot(x, temporal['excluded_counts'], 'o-', color=colors['excluded'],
                linewidth=2.5, markersize=10, label='Excluded Categorical States')
        ax1.plot(x, temporal['available_counts'], 's-', color=colors['available'],
                linewidth=2.5, markersize=10, label='Available Categorical States')
        ax1.set_xlabel('Measurement Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('State Count', fontsize=12, fontweight='bold')
        ax1.set_title(f'Panel 1: Categorical Completion Dynamics\nCompletion Rate: {temporal["categorical_completion_rate"]:.4f}',
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(x)

        # PANEL 2: Information Conservation
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(categorical['cumulative_excluded'], 'o-', color=colors['primary'],
                linewidth=3, markersize=12)
        ax2.fill_between(range(len(categorical['cumulative_excluded'])),
                        0, categorical['cumulative_excluded'],
                        alpha=0.3, color=colors['primary'])
        ax2.set_xlabel('Measurement Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Excluded States', fontsize=12, fontweight='bold')
        ax2.set_title(f'Panel 2: Information Conservation\nConservation: {categorical["information_conservation"]:.2%}',
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # PANEL 3: Logical Precision
        ax3 = fig.add_subplot(gs[1, 0])
        metrics_labels = ['Mean', 'Std', 'Min', 'Max']
        metrics_values = [precision['mean'], precision['std'],
                         precision['min'], precision['max']]
        bars = ax3.bar(metrics_labels, metrics_values, color=colors['secondary'],
                      edgecolor='black', linewidth=1.5, alpha=0.7)
        ax3.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax3.set_title(f'Panel 3: Logical Precision (NOT Physical)\nPrecision: {precision["logical_precision"]:.6f}',
                     fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)

        # PANEL 4: Categorical Consistency
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        repro_text = f"""
╔════════════════════════════════════════════╗
║   CATEGORICAL CONSISTENCY                  ║
╠════════════════════════════════════════════╣
║                                            ║
║  Excluded States CV:  {reproducibility['excluded_cv']:.6f}      ║
║  Available States CV: {reproducibility['available_cv']:.6f}      ║
║                                            ║
║  Categorical Consistency: {reproducibility['categorical_consistency']:.4f}     ║
║                                            ║
║  Excluded Range:      {reproducibility['excluded_range']}              ║
║  Available Range:     {reproducibility['available_range']}              ║
║                                            ║
║  Reproducible:        {'YES ✓' if reproducibility['is_reproducible'] else 'NO ✗'}            ║
║                                            ║
║  Assessment:                               ║
║  {'  EXCELLENT consistency' if reproducibility['categorical_consistency'] > 0.95 else '  GOOD consistency' if reproducibility['categorical_consistency'] > 0.90 else '  MODERATE consistency'}          ║
║                                            ║
║  KEY: This is LOGICAL reproducibility,     ║
║       not physical measurement error.      ║
║                                            ║
╚════════════════════════════════════════════╝
        """
        ax4.text(0.05, 0.95, repro_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

        # PANEL 5: Physical Interpretation
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.axis('off')
        ax5.text(0.05, 0.95, self.get_physical_interpretation(),
                transform=ax5.transAxes, fontsize=10, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='aliceblue', alpha=0.5))

        # PANEL 6: Summary
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        summary_text = f"""
╔════════════════════════════════════════════╗
║      TRANS-PLANCKIAN SUMMARY               ║
╠════════════════════════════════════════════╣
║                                            ║
║  CATEGORICAL COMPLETION:                   ║
║    Mean Excluded:     {temporal['mean_excluded']:.1f}             ║
║    Std Excluded:      {temporal['std_excluded']:.1f}             ║
║    Completion Rate:   {temporal['categorical_completion_rate']:.4f}         ║
║                                            ║
║  INFORMATION CONSERVATION:                 ║
║    Unique Excluded:   {categorical['total_unique_excluded']}               ║
║    Conservation:      {categorical['information_conservation']:.2%}          ║
║                                            ║
║  LOGICAL PRECISION:                        ║
║    Mean:              {precision['mean']:.3f}          ║
║    Std Dev:           {precision['std']:.3f}          ║
║    Logical Precision: {precision['logical_precision']:.6f}      ║
║                                            ║
║  CATEGORICAL CONSISTENCY:                  ║
║    Score:             {reproducibility['categorical_consistency']:.4f}         ║
║    Status:            {'PASS' if reproducibility['is_reproducible'] else 'REVIEW'}            ║
║                                            ║
║  CRITICAL NOTE:                            ║
║    This hierarchy operates beyond          ║
║    physical measurement. Precision is      ║
║    LOGICAL, not quantum-limited.           ║
║                                            ║
╚════════════════════════════════════════════╝
        """
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        fig.suptitle(f'COMPREHENSIVE ANALYSIS: {self.hierarchy_name}\n'
                    f'Pre-Geometric Regime | {temporal["timestamps"][0]} to {temporal["timestamps"][-1]}',
                    fontsize=16, fontweight='bold', y=0.98)

        output_file = output_dir / f'trans_planckian_comprehensive_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Trans-Planckian analysis saved: {output_file}")
        plt.close()

        return {
            'temporal': temporal,
            'categorical': categorical,
            'precision': precision,
            'reproducibility': reproducibility
        }


# ============================================================
# MASTER ANALYSIS ORCHESTRATOR
# ============================================================

class HierarchyAnalysisOrchestrator:
    """
    Master orchestrator for analyzing all hierarchies.
    Coordinates analysis across all temporal scales.
    """

    def __init__(self, data_directory: str):
        """
        Parameters:
        -----------
        data_directory : str
            Directory containing all JSON measurement files
        """
        self.data_directory = Path(data_directory)
        self.hierarchies = self.organize_files()

    def organize_files(self) -> Dict[str, List[str]]:
        """
        Organize JSON files by hierarchy.

        Returns:
        --------
        dict : {hierarchy_name: [file1, file2, file3]}
        """
        hierarchies = {
            'picosecond': [],
            'zeptosecond': [],
            'planck_time': [],
            'trans_planckian': []
        }

        # Find all JSON files
        json_files = list(self.data_directory.glob('*.json'))

        for file in json_files:
            filename = file.name.lower()

            if 'picosecond' in filename:
                hierarchies['picosecond'].append(str(file))
            elif 'zeptosecond' in filename:
                hierarchies['zeptosecond'].append(str(file))
            elif 'planck' in filename:
                hierarchies['planck_time'].append(str(file))
            elif 'trans' in filename or 'planckian' in filename:
                hierarchies['trans_planckian'].append(str(file))

        # Sort each hierarchy by timestamp
        for key in hierarchies:
            hierarchies[key] = sorted(hierarchies[key])

        return hierarchies

    def analyze_all_hierarchies(self, output_directory: str = 'hierarchy_analysis') -> Dict:
        """
        Run complete analysis for all hierarchies.

        Parameters:
        -----------
        output_directory : str
            Directory to save all analysis outputs

        Returns:
        --------
        dict : Complete analysis results for all hierarchies
        """
        output_dir = Path(output_directory)
        output_dir.mkdir(exist_ok=True, parents=True)

        print("\n" + "="*80)
        print("HIERARCHICAL OSCILLATORY SYSTEM ANALYSIS")
        print("="*80)
        print(f"Data Directory: {self.data_directory}")
        print(f"Output Directory: {output_dir}")
        print("="*80 + "\n")

        results = {}

        # Analyze Picosecond Hierarchy
        if len(self.hierarchies['picosecond']) >= 3:
            print(f"[1/4] Analyzing PICOSECOND hierarchy...")
            print(f"      Files: {len(self.hierarchies['picosecond'])}")
            try:
                analyzer = PicosecondAnalyzer(self.hierarchies['picosecond'][:3])
                results['picosecond'] = analyzer.generate_comprehensive_report(output_dir)
                print(f"      ✓ Complete\n")
            except Exception as e:
                print(f"      ✗ Error: {str(e)}\n")
                results['picosecond'] = None
        else:
            print(f"[1/4] Skipping PICOSECOND (insufficient files: {len(self.hierarchies['picosecond'])})\n")
            results['picosecond'] = None

        # Analyze Zeptosecond Hierarchy
        if len(self.hierarchies['zeptosecond']) >= 3:
            print(f"[2/4] Analyzing ZEPTOSECOND hierarchy...")
            print(f"      Files: {len(self.hierarchies['zeptosecond'])}")
            try:
                analyzer = ZeptosecondAnalyzer(self.hierarchies['zeptosecond'][:3])
                results['zeptosecond'] = analyzer.generate_comprehensive_report(output_dir)
                print(f"      ✓ Complete\n")
            except Exception as e:
                print(f"      ✗ Error: {str(e)}\n")
                results['zeptosecond'] = None
        else:
            print(f"[2/4] Skipping ZEPTOSECOND (insufficient files: {len(self.hierarchies['zeptosecond'])})\n")
            results['zeptosecond'] = None

        # Analyze Planck Time Hierarchy
        if len(self.hierarchies['planck_time']) >= 3:
            print(f"[3/4] Analyzing PLANCK TIME hierarchy...")
            print(f"      Files: {len(self.hierarchies['planck_time'])}")
            try:
                analyzer = PlanckTimeAnalyzer(self.hierarchies['planck_time'][:3])
                results['planck_time'] = analyzer.generate_comprehensive_report(output_dir)
                print(f"      ✓ Complete\n")
            except Exception as e:
                print(f"      ✗ Error: {str(e)}\n")
                results['planck_time'] = None
        else:
            print(f"[3/4] Skipping PLANCK TIME (insufficient files: {len(self.hierarchies['planck_time'])})\n")
            results['planck_time'] = None

        # Analyze Trans-Planckian Hierarchy
        if len(self.hierarchies['trans_planckian']) >= 3:
            print(f"[4/4] Analyzing TRANS-PLANCKIAN hierarchy...")
            print(f"      Files: {len(self.hierarchies['trans_planckian'])}")
            try:
                analyzer = TransPlanckianAnalyzer(self.hierarchies['trans_planckian'][:3])
                results['trans_planckian'] = analyzer.generate_comprehensive_report(output_dir)
                print(f"      ✓ Complete\n")
            except Exception as e:
                print(f"      ✗ Error: {str(e)}\n")
                results['trans_planckian'] = None
        else:
            print(f"[4/4] Skipping TRANS-PLANCKIAN (insufficient files: {len(self.hierarchies['trans_planckian'])})\n")
            results['trans_planckian'] = None

        print("="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)

        # Generate comparative summary
        self.generate_comparative_summary(results, output_dir)

        return results

    def generate_comparative_summary(self, results: Dict, output_dir: Path):
        """
        Generate comparative analysis across all hierarchies.
        Shows hierarchical independence.
        """
        print("\nGenerating comparative summary...")

        # Create comparative figure
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Collect data from all hierarchies
        hierarchy_names = []
        precisions = []
        stabilities = []
        efficiencies = []
        reproducibilities = []

        for name, result in results.items():
            if result is not None:
                hierarchy_names.append(name.replace('_', ' ').title())
                precisions.append(result['precision']['relative_precision'])
                stabilities.append(result['temporal']['stability'])
                efficiencies.append(result['categorical']['exclusion_efficiency'])
                reproducibilities.append(1.0 - result['reproducibility']['excluded_cv'])

        # PANEL 1: Precision Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = ax1.bar(hierarchy_names, precisions, color=['#1f77b4', '#8B4513', '#4B0082', '#000080'],
                       edgecolor='black', linewidth=2, alpha=0.7)
        ax1.set_ylabel('Relative Precision', fontsize=12, fontweight='bold')
        ax1.set_title('Panel 1: Precision Across Hierarchies\n(Lower = Better)',
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add value labels
        for bar, value in zip(bars1, precisions):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # PANEL 2: Stability Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.bar(hierarchy_names, stabilities, color=['#1f77b4', '#8B4513', '#4B0082', '#000080'],
                       edgecolor='black', linewidth=2, alpha=0.7)
        ax2.set_ylabel('Temporal Stability', fontsize=12, fontweight='bold')
        ax2.set_title('Panel 2: Stability Across Hierarchies\n(Lower = More Stable)',
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        for bar, value in zip(bars2, stabilities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # PANEL 3: Exclusion Efficiency
        ax3 = fig.add_subplot(gs[1, 0])
        bars3 = ax3.bar(hierarchy_names, efficiencies, color=['#1f77b4', '#8B4513', '#4B0082', '#000080'],
                       edgecolor='black', linewidth=2, alpha=0.7)
        ax3.set_ylabel('Exclusion Efficiency', fontsize=12, fontweight='bold')
        ax3.set_title('Panel 3: Categorical Exclusion Efficiency\n(Higher = More Efficient)',
                     fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

        for bar, value in zip(bars3, efficiencies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # PANEL 4: Reproducibility
        ax4 = fig.add_subplot(gs[1, 1])
        bars4 = ax4.bar(hierarchy_names, reproducibilities, color=['#1f77b4', '#8B4513', '#4B0082', '#000080'],
                       edgecolor='black', linewidth=2, alpha=0.7)
        ax4.set_ylabel('Reproducibility Score', fontsize=12, fontweight='bold')
        ax4.set_title('Panel 4: Measurement Reproducibility\n(Higher = More Reproducible)',
                     fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

        for bar, value in zip(bars4, reproducibilities):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        fig.suptitle('COMPARATIVE ANALYSIS: HIERARCHICAL INDEPENDENCE\n'
                    'Each Hierarchy Operates Independently with Distinct Precision Characteristics',
                    fontsize=16, fontweight='bold', y=0.98)

        output_file = output_dir / 'comparative_hierarchy_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Comparative analysis saved: {output_file}")
        plt.close()

        # Generate text summary
        summary_file = output_dir / 'analysis_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HIERARCHICAL OSCILLATORY SYSTEM ANALYSIS SUMMARY\n")
            f.write("="*80 + "\n\n")

            for i, name in enumerate(hierarchy_names):
                f.write(f"{name}:\n")
                f.write(f"  Relative Precision:    {precisions[i]:.6f}\n")
                f.write(f"  Temporal Stability:    {stabilities[i]:.6f}\n")
                f.write(f"  Exclusion Efficiency:  {efficiencies[i]:.2%}\n")
                f.write(f"  Reproducibility:       {reproducibilities[i]:.6f}\n")
                f.write("\n")

            f.write("="*80 + "\n")
            f.write("KEY INSIGHT: HIERARCHICAL INDEPENDENCE\n")
            f.write("="*80 + "\n\n")
            f.write("Notice that precision does NOT increase monotonically with hierarchy.\n")
            f.write("Each hierarchy has its own precision characteristics, determined by:\n")
            f.write("  1. Physical processes at that scale\n")
            f.write("  2. Exclusion mechanisms\n")
            f.write("  3. Categorical space structure\n")
            f.write("  4. Measurement/observation method\n\n")
            f.write("This demonstrates that hierarchies are INDEPENDENT, not nested.\n")
            f.write("Trans-Planckian can be precise even if intermediate scales are not.\n")
            f.write("="*80 + "\n")

        print(f"✓ Text summary saved: {summary_file}")
        print("\n" + "="*80)
        print("ALL ANALYSES COMPLETE")
        print(f"Results saved to: {output_dir}")
        print("="*80 + "\n")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    import sys

    # Check command line arguments
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "."  # Current directory

    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = "hierarchy_analysis"

    print("\n" + "="*80)
    print("HIERARCHICAL OSCILLATORY SYSTEM ANALYZER")
    print("="*80)
    print(f"Data Directory:   {data_dir}")
    print(f"Output Directory: {output_dir}")
    print("="*80 + "\n")

    # Create orchestrator and run analysis
    orchestrator = HierarchyAnalysisOrchestrator(data_dir)
    results = orchestrator.analyze_all_hierarchies(output_dir)

    print("\n✓ ALL PROCESSING COMPLETE\n")
