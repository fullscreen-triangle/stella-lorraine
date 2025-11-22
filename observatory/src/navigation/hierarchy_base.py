import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from pathlib import Path
from datetime import datetime
import pandas as pd

# ============================================================
# BASE CLASS: Hierarchy Analyzer
# ============================================================

class HierarchyAnalyzer:
    """
    Base class for hierarchy-specific analysis.
    Each hierarchy gets its own specialized subclass.
    """

    def __init__(self, json_files):
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

    def load_json(self, filepath):
        """Load JSON data."""
        with open(filepath, 'r') as f:
            return json.load(f)

    def get_hierarchy_name(self):
        """Extract hierarchy name from filename."""
        raise NotImplementedError("Subclass must implement")

    def get_theoretical_scale(self):
        """Return theoretical time scale for this hierarchy."""
        raise NotImplementedError("Subclass must implement")

    def extract_observations(self):
        """Extract observation data from all measurements."""
        raise NotImplementedError("Subclass must implement")

    def analyze_temporal_evolution(self):
        """Analyze how measurements evolve over time."""
        raise NotImplementedError("Subclass must implement")

    def analyze_categorical_exclusion(self):
        """Analyze categorical exclusion dynamics."""
        raise NotImplementedError("Subclass must implement")

    def analyze_precision(self):
        """Characterize precision at this hierarchy."""
        raise NotImplementedError("Subclass must implement")

    def analyze_reproducibility(self):
        """Analyze consistency across 3 measurements."""
        raise NotImplementedError("Subclass must implement")

    def generate_comprehensive_report(self, output_dir):
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

    def get_hierarchy_name(self):
        return "Picosecond (10⁻¹² s)"

    def get_theoretical_scale(self):
        return 1e-12

    def extract_observations(self):
        """
        Extract observation sequences from JSON.
        Picosecond data structure:
        {
          "timestamp": "...",
          "observation_sequence": [...],
          "excluded_harmonics": [...],
          "available_harmonics": [...]
        }
        """
        observations = []
        for data in self.data:
            obs = {
                'timestamp': data.get('timestamp'),
                'sequence': data.get('observation_sequence', []),
                'excluded': data.get('excluded_harmonics', []),
                'available': data.get('available_harmonics', []),
                'total_states': data.get('total_categorical_states', 0),
                'excluded_count': len(data.get('excluded_harmonics', [])),
                'available_count': len(data.get('available_harmonics', []))
            }
            observations.append(obs)
        return observations

    def analyze_temporal_evolution(self):
        """
        Analyze how picosecond measurements evolve.

        Key questions:
        - Do excluded states accumulate?
        - Is there temporal drift?
        - What's the exclusion rate?
        """
        obs = self.extract_observations()

        timestamps = [o['timestamp'] for o in obs]
        excluded_counts = [o['excluded_count'] for o in obs]
        available_counts = [o['available_count'] for o in obs]

        # Calculate exclusion rate
        if len(excluded_counts) > 1:
            exclusion_rate = np.diff(excluded_counts)
        else:
            exclusion_rate = [0]

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

    def analyze_categorical_exclusion(self):
        """
        Analyze categorical exclusion dynamics at picosecond scale.

        Picosecond-specific:
        - Molecular vibrational modes
        - Energy quantization
        - Phase coherence
        """
        obs = self.extract_observations()

        # Build exclusion history
        all_excluded = []
        cumulative_excluded = set()

        for o in obs:
            excluded_set = set(o['excluded'])
            cumulative_excluded.update(excluded_set)
            all_excluded.append(len(cumulative_excluded))

        # Analyze exclusion patterns
        exclusion_growth = np.array(all_excluded)

        # Check for saturation
        if len(exclusion_growth) > 1:
            saturation_rate = exclusion_growth[-1] / exclusion_growth[0] if exclusion_growth[0] > 0 else 0
        else:
            saturation_rate = 1.0

        return {
            'cumulative_excluded': all_excluded,
            'total_unique_excluded': len(cumulative_excluded),
            'saturation_rate': saturation_rate,
            'exclusion_efficiency': len(cumulative_excluded) / obs[0]['total_states'] if obs[0]['total_states'] > 0 else 0
        }

    def analyze_precision(self):
        """
        Characterize precision at picosecond hierarchy.

        Precision limited by:
        - Thermal fluctuations
        - Quantum decoherence
        - Measurement apparatus
        """
        obs = self.extract_observations()

        # Extract observation sequences
        sequences = [o['sequence'] for o in obs]

        # Calculate precision metrics
        if all(len(s) > 0 for s in sequences):
            # Convert sequences to numeric if possible
            numeric_sequences = []
            for seq in sequences:
                try:
                    numeric_sequences.append([float(x) for x in seq])
                except:
                    # If not numeric, use length as proxy
                    numeric_sequences.append([len(seq)])

            # Calculate statistics
            all_values = [v for seq in numeric_sequences for v in seq]

            precision = {
                'mean': np.mean(all_values),
                'std': np.std(all_values),
                'relative_precision': np.std(all_values) / np.abs(np.mean(all_values)) if np.mean(all_values) != 0 else 0,
                'min': np.min(all_values),
                'max': np.max(all_values),
                'range': np.max(all_values) - np.min(all_values),
                'n_samples': len(all_values)
            }
        else:
            # Use excluded counts as proxy
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

    def analyze_reproducibility(self):
        """
        Analyze consistency across 3 picosecond measurements.

        Key metrics:
        - Inter-measurement variance
        - Correlation between measurements
        - Systematic drift
        """
        obs = self.extract_observations()

        excluded_counts = [o['excluded_count'] for o in obs]
        available_counts = [o['available_count'] for o in obs]

        # Calculate reproducibility metrics
        reproducibility = {
            'excluded_cv': np.std(excluded_counts) / np.mean(excluded_counts) if np.mean(excluded_counts) > 0 else 0,
            'available_cv': np.std(available_counts) / np.mean(available_counts) if np.mean(available_counts) > 0 else 0,
            'excluded_range': max(excluded_counts) - min(excluded_counts),
            'available_range': max(available_counts) - min(available_counts),
            'is_reproducible': np.std(excluded_counts) / np.mean(excluded_counts) < 0.1 if np.mean(excluded_counts) > 0 else False
        }

        return reproducibility

    def generate_comprehensive_report(self, output_dir):
        """
        Generate complete 6-panel analysis for picosecond hierarchy.

        Panels:
        1. Temporal Evolution of Exclusions
        2. Categorical Exclusion Dynamics
        3. Precision Characterization
        4. Reproducibility Analysis
        5. Physical Interpretation
        6. Summary Statistics
        """
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

        # Color scheme
        colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'excluded': '#d62728',
            'available': '#2ca02c',
            'accent': '#9467bd'
        }

        # ========== PANEL 1: Temporal Evolution ==========
        ax1 = fig.add_subplot(gs[0, 0])

        x = range(len(temporal['excluded_counts']))
        ax1.plot(x, temporal['excluded_counts'], 'o-', color=colors['excluded'],
                linewidth=2.5, markersize=10, label='Excluded States')
        ax1.plot(x, temporal['available_counts'], 's-', color=colors['available'],
                linewidth=2.5, markersize=10, label='Available States')

        ax1.set_xlabel('Measurement Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('State Count', fontsize=12, fontweight='bold')
        ax1.set_title('Panel 1: Temporal Evolution of Categorical States\n'
                     f'Stability: {temporal["stability"]:.4f}',
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(x)
        ax1.set_xticklabels([t.split('_')[1] for t in temporal['timestamps']], rotation=45)

        # ========== PANEL 2: Categorical Exclusion Dynamics ==========
        ax2 = fig.add_subplot(gs[0, 1])

        ax2.plot(categorical['cumulative_excluded'], 'o-', color=colors['primary'],
                linewidth=3, markersize=12)
        ax2.fill_between(range(len(categorical['cumulative_excluded'])),
                        0, categorical['cumulative_excluded'],
                        alpha=0.3, color=colors['primary'])

        ax2.set_xlabel('Measurement Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Excluded States', fontsize=12, fontweight='bold')
        ax2.set_title('Panel 2: Categorical Exclusion Growth\n'
                     f'Efficiency: {categorical["exclusion_efficiency"]:.2%}',
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # ========== PANEL 3: Precision Characterization ==========
        ax3 = fig.add_subplot(gs[1, 0])

        metrics_labels = ['Mean', 'Std', 'Min', 'Max']
        metrics_values = [precision['mean'], precision['std'],
                         precision['min'], precision['max']]

        bars = ax3.bar(metrics_labels, metrics_values, color=colors['secondary'],
                      edgecolor='black', linewidth=1.5, alpha=0.7)

        ax3.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax3.set_title('Panel 3: Precision Metrics\n'
                     f'Relative Precision: {precision["relative_precision"]:.6f}',
                     fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3e}', ha='center', va='bottom', fontsize=9)

        # ========== PANEL 4: Reproducibility Analysis ==========
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
║  {'  Measurements are consistent' if reproducibility['is_reproducible'] else '  Measurements show variation'}        ║
║                                            ║
╚════════════════════════════════════════════╝
        """

        ax4.text(0.05, 0.95, repro_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # ========== PANEL 5: Physical Interpretation ==========
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.axis('off')

        physics_text = f"""
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

        ax5.text(0.05, 0.95, physics_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        # ========== PANEL 6: Summary Statistics ==========
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
║    Mean:              {precision['mean']:.3e}      ║
║    Std Dev:           {precision['std']:.3e}      ║
║    Relative:          {precision['relative_precision']:.6f}      ║
║    Samples:           {precision['n_samples']}               ║
║                                            ║
║  REPRODUCIBILITY:                          ║
║    CV (Excluded):     {reproducibility['excluded_cv']:.6f}      ║
║    Status:            {'PASS' if reproducibility['is_reproducible'] else 'REVIEW'}            ║
║                                            ║
╚════════════════════════════════════════════╝
        """

        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        # Overall title
        fig.suptitle(f'COMPREHENSIVE ANALYSIS: {self.hierarchy_name}\n'
                    f'3 Measurements | {temporal["timestamps"][0]} to {temporal["timestamps"][-1]}',
                    fontsize=16, fontweight='bold', y=0.98)

        # Save
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
# I'll create similar specialized analyzers for:
# - ZeptosecondAnalyzer
# - PlanckTimeAnalyzer
# - TransPlanckianAnalyzer
# ============================================================
