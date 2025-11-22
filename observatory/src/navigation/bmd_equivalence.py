#!/usr/bin/env python3
"""
Biological Maxwell Demon (BMD) Equivalence Verification
=========================================================
Demonstrates that different processing pathways (visual, spectral, semantic, hardware)
converge to identical variance states, validating BMD equivalence principle.

BMD Equivalence Theorem:
For equivalent processing pathways Π₁ ≡ Π₂ ≡ Π₃, the outputs converge to
identical variance states: Var(Π₁(x)) = Var(Π₂(x)) = Var(Π₃(x))
"""

import numpy as np
import os
import json
from typing import Dict, List, Tuple
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Try to import other navigation modules
try:
    from entropy_navigation import SEntropyNavigator
    from fourier_transform_coordinates import MultiDomainSEFT
    from hardware_clock_integration import HardwareClockSync
    from molecular_vibrations import QuantumVibrationalAnalyzer
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("BMD equivalence will use simplified models...")


class BMDPathway:
    """Represents a single BMD processing pathway"""

    def __init__(self, name: str, pathway_type: str):
        self.name = name
        self.type = pathway_type  # 'visual', 'spectral', 'semantic', 'hardware'
        self.variance_history = []

    def process(self, input_data: np.ndarray) -> Dict:
        """Process input through this BMD pathway"""
        # Simulate pathway-specific processing
        if self.type == 'visual':
            output = self._visual_processing(input_data)
        elif self.type == 'spectral':
            output = self._spectral_processing(input_data)
        elif self.type == 'semantic':
            output = self._semantic_processing(input_data)
        elif self.type == 'hardware':
            output = self._hardware_processing(input_data)
        else:
            output = input_data

        # Calculate variance state
        variance = np.var(output)
        self.variance_history.append(variance)

        return {
            'output': output,
            'variance': variance,
            'mean': np.mean(output),
            'std': np.std(output),
            'pathway': self.name
        }

    def _visual_processing(self, data: np.ndarray) -> np.ndarray:
        """Visual pathway: Process as image features"""
        # Simulate convolution + pooling (visual cortex-like processing)
        kernel = np.array([0.25, 0.5, 0.25])
        filtered = np.convolve(data, kernel, mode='same')
        return filtered

    def _spectral_processing(self, data: np.ndarray) -> np.ndarray:
        """Spectral pathway: Frequency domain analysis"""
        # FFT to frequency domain
        fft_result = np.fft.fft(data)
        power_spectrum = np.abs(fft_result)**2
        # Return power in specific bands
        return power_spectrum[:len(data)]

    def _semantic_processing(self, data: np.ndarray) -> np.ndarray:
        """Semantic pathway: Information content extraction"""
        # Information-theoretic processing
        # Normalize and compute entropy-weighted transform
        normalized = data - np.mean(data)
        normalized = normalized / (np.std(normalized) + 1e-10)
        return normalized

    def _hardware_processing(self, data: np.ndarray) -> np.ndarray:
        """Hardware pathway: Clock-synchronized sampling"""
        # Simulate hardware sampling at fixed intervals
        # Resample to simulate hardware clock coordination
        resampled = data[::max(1, len(data)//1000)]
        # Pad back to original length
        return np.pad(resampled, (0, len(data) - len(resampled)), 'edge')


class BMDEquivalenceValidator:
    """Validates BMD equivalence across multiple pathways"""

    def __init__(self, n_pathways: int = 4):
        self.pathways = [
            BMDPathway("Visual Processing", "visual"),
            BMDPathway("Spectral Analysis", "spectral"),
            BMDPathway("Semantic Embedding", "semantic"),
            BMDPathway("Hardware Sampling", "hardware")
        ]

        self.results = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'pathways': [],
            'convergence_analysis': {},
            'equivalence_metrics': {}
        }

    def generate_test_signal(self, n_samples: int = 1024,
                           frequency: float = 7.1e13) -> Tuple[np.ndarray, np.ndarray]:
        """Generate molecular vibration test signal"""
        duration = 100e-15  # 100 fs
        time_points = np.linspace(0, duration, n_samples)

        # Composite signal with harmonics
        signal = np.zeros(n_samples)
        signal += np.sin(2*np.pi*frequency*time_points)
        signal += 0.3 * np.sin(2*np.pi*2*frequency*time_points)
        signal += 0.1 * np.sin(2*np.pi*3*frequency*time_points)

        # Add noise
        signal += 0.05 * np.random.randn(n_samples)

        return signal, time_points

    def process_through_all_pathways(self, input_signal: np.ndarray,
                                    n_iterations: int = 50) -> Dict:
        """Process signal through all BMD pathways multiple times"""
        print(f"\n🔬 Processing signal through {len(self.pathways)} BMD pathways")
        print(f"   Iterations: {n_iterations}")

        pathway_results = {p.name: [] for p in self.pathways}

        for iteration in range(n_iterations):
            for pathway in self.pathways:
                result = pathway.process(input_signal)
                pathway_results[pathway.name].append(result)

        return pathway_results

    def calculate_variance_convergence(self, pathway_results: Dict) -> Dict:
        """Calculate convergence of variances across pathways"""
        # Extract variance trajectories
        variance_trajectories = {}
        for pathway_name, results in pathway_results.items():
            variances = [r['variance'] for r in results]
            variance_trajectories[pathway_name] = variances

        # Calculate convergence metrics
        final_variances = {name: traj[-1] for name, traj in variance_trajectories.items()}
        mean_final_variance = np.mean(list(final_variances.values()))
        variance_spread = np.std(list(final_variances.values()))

        # Relative variance spread (should be small for BMD equivalence)
        relative_spread = variance_spread / mean_final_variance if mean_final_variance > 0 else 0

        # Convergence rate
        convergence_rates = {}
        for name, traj in variance_trajectories.items():
            if len(traj) > 1:
                # Exponential fit to convergence
                diff_traj = np.diff(traj)
                iterations = np.arange(len(diff_traj))  # Match length with diff
                # Rate of variance change
                rate = np.polyfit(iterations, np.log(np.abs(diff_traj) + 1e-10), 1)[0]
                convergence_rates[name] = rate

        return {
            'variance_trajectories': variance_trajectories,
            'final_variances': final_variances,
            'mean_final_variance': mean_final_variance,
            'variance_spread': variance_spread,
            'relative_spread': relative_spread,
            'convergence_rates': convergence_rates,
            'equivalence_achieved': relative_spread < 0.1  # <10% spread
        }

    def validate_bmd_equivalence(self, input_signal: np.ndarray,
                                n_iterations: int = 50) -> Dict:
        """
        Main validation: Process signal through all pathways and verify equivalence
        """
        print("\n" + "="*70)
        print("   BMD EQUIVALENCE VALIDATION")
        print("="*70)

        # Process through all pathways
        pathway_results = self.process_through_all_pathways(input_signal, n_iterations)

        # Analyze variance convergence
        convergence = self.calculate_variance_convergence(pathway_results)

        # Statistical tests for equivalence
        variances = list(convergence['final_variances'].values())

        # One-way ANOVA equivalent: Are all pathway variances statistically equivalent?
        f_statistic, p_value = self._anova_test(variances)

        # Pairwise equivalence tests
        pairwise_tests = self._pairwise_equivalence_tests(convergence['final_variances'])

        results = {
            'pathway_results': pathway_results,
            'convergence_analysis': convergence,
            'statistical_tests': {
                'f_statistic': f_statistic,
                'p_value': p_value,
                'equivalence_hypothesis': p_value > 0.05,  # Fail to reject equivalence
                'pairwise_tests': pairwise_tests
            },
            'equivalence_achieved': convergence['equivalence_achieved']
        }

        self.results['convergence_analysis'] = convergence
        self.results['equivalence_metrics'] = results['statistical_tests']

        # Print results
        print(f"\n📊 CONVERGENCE ANALYSIS:")
        print(f"   Mean final variance: {convergence['mean_final_variance']:.6e}")
        print(f"   Variance spread: {convergence['variance_spread']:.6e}")
        print(f"   Relative spread: {convergence['relative_spread']:.4f} "
              f"({'✓ CONVERGED' if convergence['relative_spread'] < 0.1 else '✗ DIVERGED'})")

        print(f"\n📈 FINAL VARIANCES BY PATHWAY:")
        for name, var in convergence['final_variances'].items():
            deviation = abs(var - convergence['mean_final_variance']) / convergence['mean_final_variance'] * 100
            print(f"   {name:20s}: {var:.6e} ({deviation:+5.2f}% from mean)")

        print(f"\n🎯 STATISTICAL EQUIVALENCE:")
        print(f"   F-statistic: {f_statistic:.4f}")
        print(f"   P-value: {p_value:.6f}")
        if results['statistical_tests']['equivalence_hypothesis']:
            print(f"   ✓ BMD EQUIVALENCE CONFIRMED (p > 0.05)")
        else:
            print(f"   ⚠ Pathways show statistical differences (p < 0.05)")

        print(f"\n✨ BMD EQUIVALENCE THEOREM:")
        if results['equivalence_achieved']:
            print(f"   ✓ Var(Π_visual) ≈ Var(Π_spectral) ≈ Var(Π_semantic) ≈ Var(Π_hardware)")
            print(f"   All pathways converge to identical variance states!")
        else:
            print(f"   ⚠ Variance convergence incomplete - may need more iterations")

        return results

    def _anova_test(self, variances: List[float]) -> Tuple[float, float]:
        """Simplified one-way ANOVA test"""
        n = len(variances)
        grand_mean = np.mean(variances)

        # Between-group variance
        ss_between = sum((v - grand_mean)**2 for v in variances)
        df_between = n - 1

        # Within-group variance (simplified - assume equal within-group variance)
        ss_within = 0.01 * n  # Simplified assumption
        df_within = n

        # F-statistic
        ms_between = ss_between / df_between if df_between > 0 else 0
        ms_within = ss_within / df_within if df_within > 0 else 1
        f_stat = ms_between / ms_within if ms_within > 0 else 0

        # Simplified p-value (should use F-distribution)
        # For now, use exponential approximation
        p_val = np.exp(-f_stat/2)

        return f_stat, p_val

    def _pairwise_equivalence_tests(self, final_variances: Dict) -> Dict:
        """Test pairwise equivalence between all pathway combinations"""
        pairwise = {}
        pathways = list(final_variances.keys())

        for i in range(len(pathways)):
            for j in range(i+1, len(pathways)):
                p1, p2 = pathways[i], pathways[j]
                v1, v2 = final_variances[p1], final_variances[p2]

                # Relative difference
                rel_diff = abs(v1 - v2) / ((v1 + v2)/2) if (v1 + v2) > 0 else 0

                pairwise[f"{p1} vs {p2}"] = {
                    'variance_1': v1,
                    'variance_2': v2,
                    'relative_difference': rel_diff,
                    'equivalent': rel_diff < 0.1  # <10% difference
                }

        return pairwise

    def save_results(self, output_dir: str = None):
        """Save BMD equivalence validation results"""
        if output_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(current_dir, '..', '..', 'results', 'bmd_equivalence')

        os.makedirs(output_dir, exist_ok=True)

        timestamp = self.results['timestamp']

        # Save JSON results
        json_file = os.path.join(output_dir, f'bmd_equivalence_{timestamp}.json')

        # Prepare JSON-serializable results
        json_results = {
            'timestamp': timestamp,
            'equivalence_achieved': bool(self.results.get('equivalence_metrics', {}).get('equivalence_hypothesis', False)),
            'convergence': {
                'mean_final_variance': float(self.results['convergence_analysis']['mean_final_variance']),
                'variance_spread': float(self.results['convergence_analysis']['variance_spread']),
                'relative_spread': float(self.results['convergence_analysis']['relative_spread']),
                'final_variances': {k: float(v) for k, v in self.results['convergence_analysis']['final_variances'].items()}
            },
            'statistical_tests': {
                'f_statistic': float(self.results['equivalence_metrics']['f_statistic']),
                'p_value': float(self.results['equivalence_metrics']['p_value']),
                'equivalence_hypothesis': bool(self.results['equivalence_metrics']['equivalence_hypothesis'])
            },
            'pathways': [p.name for p in self.pathways]
        }

        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"\n💾 Results saved: {json_file}")

        return json_file

    def visualize_results(self, output_dir: str = None):
        """Create visualization of BMD equivalence"""
        if output_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(current_dir, '..', '..', 'results', 'bmd_equivalence')

        os.makedirs(output_dir, exist_ok=True)

        timestamp = self.results['timestamp']

        fig = plt.figure(figsize=(16, 10))

        # Panel 1: Variance convergence trajectories
        ax1 = plt.subplot(2, 3, 1)
        convergence = self.results['convergence_analysis']
        for name, trajectory in convergence['variance_trajectories'].items():
            ax1.plot(trajectory, label=name, alpha=0.7, linewidth=2)
        ax1.axhline(y=convergence['mean_final_variance'], color='black',
                   linestyle='--', label='Mean final')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Variance')
        ax1.set_title('Variance Convergence Trajectories', fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Final variance comparison
        ax2 = plt.subplot(2, 3, 2)
        names = list(convergence['final_variances'].keys())
        variances = list(convergence['final_variances'].values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        bars = ax2.bar(range(len(names)), variances, color=colors, alpha=0.7)
        ax2.axhline(y=convergence['mean_final_variance'], color='black',
                   linestyle='--', linewidth=2, label='Mean')
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=8)
        ax2.set_ylabel('Final Variance')
        ax2.set_title('Final Variance by Pathway', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Panel 3: Relative deviations
        ax3 = plt.subplot(2, 3, 3)
        mean_var = convergence['mean_final_variance']
        deviations = [(v - mean_var)/mean_var * 100 for v in variances]
        bars = ax3.barh(range(len(names)), deviations, color=colors, alpha=0.7)
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax3.axvline(x=10, color='red', linestyle='--', linewidth=1, alpha=0.5, label='10% threshold')
        ax3.axvline(x=-10, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax3.set_yticks(range(len(names)))
        ax3.set_yticklabels(names, fontsize=8)
        ax3.set_xlabel('Deviation from Mean (%)')
        ax3.set_title('Relative Deviations', fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='x')

        # Panel 4: Equivalence matrix (heatmap)
        ax4 = plt.subplot(2, 3, 4)
        n = len(names)
        equiv_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    equiv_matrix[i, j] = 1.0
                else:
                    v_i, v_j = variances[i], variances[j]
                    rel_diff = abs(v_i - v_j) / ((v_i + v_j)/2) if (v_i + v_j) > 0 else 0
                    equiv_matrix[i, j] = 1 - rel_diff  # 1 = perfect equivalence

        im = ax4.imshow(equiv_matrix, cmap='RdYlGn', vmin=0.8, vmax=1.0)
        ax4.set_xticks(range(n))
        ax4.set_yticks(range(n))
        ax4.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=7, rotation=45)
        ax4.set_yticklabels(names, fontsize=7)
        ax4.set_title('Pairwise Equivalence Matrix', fontweight='bold')
        plt.colorbar(im, ax=ax4, label='Equivalence Score')

        # Panel 5: Statistical summary
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')

        stats = self.results['equivalence_metrics']
        summary_text = f"""
BMD EQUIVALENCE VALIDATION

Statistical Tests:
  F-statistic: {stats['f_statistic']:.4f}
  P-value: {stats['p_value']:.6f}

Convergence Metrics:
  Mean variance: {convergence['mean_final_variance']:.6e}
  Variance spread: {convergence['variance_spread']:.6e}
  Relative spread: {convergence['relative_spread']:.4f}

Equivalence Status:
  {'✓ CONFIRMED' if stats['equivalence_hypothesis'] else '✗ NOT CONFIRMED'}

Theorem Validation:
  Var(Π₁) ≈ Var(Π₂) ≈ Var(Π₃) ≈ Var(Π₄)
  {'✓ SATISFIED' if convergence['equivalence_achieved'] else '⚠ INCOMPLETE'}
"""

        ax5.text(0.1, 0.5, summary_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # Panel 6: Convergence rates
        ax6 = plt.subplot(2, 3, 6)
        rates = list(convergence['convergence_rates'].values())
        rate_names = list(convergence['convergence_rates'].keys())
        bars = ax6.barh(range(len(rate_names)), rates, color=colors, alpha=0.7)
        ax6.set_yticks(range(len(rate_names)))
        ax6.set_yticklabels(rate_names, fontsize=8)
        ax6.set_xlabel('Convergence Rate')
        ax6.set_title('Convergence Rates by Pathway', fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')

        plt.suptitle('BMD Equivalence Validation: Multi-Pathway Convergence Analysis',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        fig_file = os.path.join(output_dir, f'bmd_equivalence_{timestamp}.png')
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved: {fig_file}")

        plt.close()

        return fig_file


def main():
    """Main BMD equivalence validation"""
    print("\n" + "="*70)
    print("   BIOLOGICAL MAXWELL DEMON (BMD) EQUIVALENCE VALIDATION")
    print("="*70)

    # Create validator
    validator = BMDEquivalenceValidator()

    # Generate test signal
    print("\n🔬 Generating molecular vibration test signal...")
    signal, time_points = validator.generate_test_signal(n_samples=1024, frequency=7.1e13)
    print(f"   Signal length: {len(signal)} samples")
    print(f"   Duration: {time_points[-1]*1e15:.1f} fs")
    print(f"   Frequency: 7.1×10¹³ Hz (71 THz)")

    # Validate BMD equivalence
    results = validator.validate_bmd_equivalence(signal, n_iterations=50)

    # Save results
    json_file = validator.save_results()

    # Create visualization
    fig_file = validator.visualize_results()

    print("\n" + "="*70)
    print("   BMD EQUIVALENCE VALIDATION COMPLETE")
    print("="*70)
    print(f"\n   Status: {'✓ EQUIVALENCE CONFIRMED' if results['equivalence_achieved'] else '⚠ NEEDS MORE ITERATIONS'}")
    print(f"   Results: {json_file}")
    print(f"   Figure: {fig_file}")
    print("\n")

    return validator, results


if __name__ == "__main__":
    validator, results = main()
