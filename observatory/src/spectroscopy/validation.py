#!/usr/bin/env python3
"""
Hardware-Based Spectroscopy Validation Framework
===============================================

Validates our virtual LED spectroscopy system against real instrument data
by comparing spectral features, peak detection, and molecular identification accuracy.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, interpolate, stats
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import warnings
warnings.filterwarnings('ignore')

from led_spectroscopy import LEDSpectroscopySystem
from spectral_analysis_algorithm import SpectralAnalyzer
from rgb_chemical_mapping import RGBChemicalMapper

class SpectroscopyValidator:
    """
    Comprehensive validation framework comparing hardware-based virtual spectroscopy
    with real instrument measurements
    """

    def __init__(self):
        self.led_system = LEDSpectroscopySystem()
        self.spectral_analyzer = SpectralAnalyzer()
        self.rgb_mapper = RGBChemicalMapper()
        self.real_spectra_data = {}
        self.virtual_spectra_data = {}

    def load_real_spectra(self, spectra_dir):
        """Load real instrument spectra from CSV files"""
        print("📊 Loading real instrument spectra...")

        csv_files = [f for f in os.listdir(spectra_dir) if f.endswith('.csv')]

        for filename in csv_files:
            filepath = os.path.join(spectra_dir, filename)
            try:
                # Try different CSV formats
                df = pd.read_csv(filepath)

                # Detect format and extract wavelength/intensity data
                if 'Wavelength (nm)' in df.columns and 'Absorbance' in df.columns[1]:
                    wavelengths = df['Wavelength (nm)'].values
                    intensities = df.iloc[:, 1].values
                    spectrum_type = 'UV-Vis_Absorbance'

                elif 'Time (min)' in df.columns:
                    # Chromatography data - convert time to pseudo-wavelength
                    times = df['Time (min)'].values
                    intensities = df.iloc[:, 2].values if len(df.columns) > 2 else df.iloc[:, 1].values
                    # Map time to wavelength range (200-800 nm)
                    wavelengths = 200 + (times - times.min()) / (times.max() - times.min()) * 600
                    spectrum_type = 'Chromatography'

                elif len(df.columns) == 2:
                    # Generic two-column format
                    wavelengths = df.iloc[:, 0].values
                    intensities = df.iloc[:, 1].values
                    spectrum_type = 'Generic'

                else:
                    print(f"⚠️ Unknown format for {filename}")
                    continue

                # Clean and validate data
                valid_mask = ~(np.isnan(wavelengths) | np.isnan(intensities))
                wavelengths = wavelengths[valid_mask]
                intensities = intensities[valid_mask]

                if len(wavelengths) > 10:  # Minimum data points
                    self.real_spectra_data[filename] = {
                        'wavelengths': wavelengths,
                        'intensities': intensities,
                        'type': spectrum_type,
                        'filename': filename
                    }
                    print(f"✅ Loaded {filename}: {len(wavelengths)} points ({spectrum_type})")

            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")

        print(f"📈 Total real spectra loaded: {len(self.real_spectra_data)}")
        return self.real_spectra_data

    def generate_virtual_spectra(self, molecular_patterns):
        """Generate virtual spectra using our hardware-based system"""
        print("🔬 Generating virtual spectra...")

        for i, pattern in enumerate(molecular_patterns):
            virtual_spectrum = {}

            # LED spectroscopy analysis
            for led_color, wavelength in self.led_system.led_wavelengths.items():
                led_analysis = self.led_system.analyze_molecular_fluorescence(pattern, wavelength)
                virtual_spectrum[f'led_{led_color}'] = led_analysis

            # Spectral analysis algorithm
            spectral_analysis = self.spectral_analyzer.analyze_spectrum(pattern)
            virtual_spectrum['spectral_analysis'] = spectral_analysis

            # RGB mapping
            rgb_mapping = self.rgb_mapper.map_pattern_to_rgb(pattern)
            virtual_spectrum['rgb_mapping'] = rgb_mapping

            # Combine LED responses with spectral analysis for enhanced virtual spectrum
            combined_wavelengths = spectral_analysis['wavelengths']
            combined_intensities = spectral_analysis['intensities'].copy()

            # Enhance with LED fluorescence data
            for led_color, led_data in virtual_spectrum.items():
                if led_color.startswith('led_'):
                    emission_spectrum = led_data['emission_spectrum']
                    led_wavelengths = np.array(emission_spectrum['wavelengths'])
                    led_intensities = np.array(emission_spectrum['intensities'])

                    # Add LED contribution to combined spectrum
                    for j, wl in enumerate(combined_wavelengths):
                        # Find closest LED wavelength
                        closest_idx = np.argmin(np.abs(led_wavelengths - wl))
                        if closest_idx < len(led_intensities):
                            combined_intensities[j] += led_intensities[closest_idx] * 0.3

            self.virtual_spectra_data[f'pattern_{i}'] = {
                'pattern': pattern,
                'virtual_spectrum': virtual_spectrum,
                'combined_wavelengths': combined_wavelengths,
                'combined_intensities': combined_intensities
            }

        print(f"🧪 Generated {len(self.virtual_spectra_data)} virtual spectra")
        return self.virtual_spectra_data

    def align_spectra(self, real_wl, real_int, virtual_wl, virtual_int):
        """Align real and virtual spectra for comparison"""
        # Find common wavelength range
        min_wl = max(real_wl.min(), virtual_wl.min())
        max_wl = min(real_wl.max(), virtual_wl.max())

        # Create common wavelength grid
        common_wl = np.linspace(min_wl, max_wl, 200)

        # Interpolate both spectra to common grid
        real_interp = interpolate.interp1d(real_wl, real_int, kind='linear',
                                         bounds_error=False, fill_value=0)
        virtual_interp = interpolate.interp1d(virtual_wl, virtual_int, kind='linear',
                                            bounds_error=False, fill_value=0)

        real_aligned = real_interp(common_wl)
        virtual_aligned = virtual_interp(common_wl)

        return common_wl, real_aligned, virtual_aligned

    def compare_peak_detection(self, real_spectrum, virtual_spectrum):
        """Compare peak detection between real and virtual spectra"""
        real_wl, real_int = real_spectrum['wavelengths'], real_spectrum['intensities']
        virtual_wl, virtual_int = virtual_spectrum['combined_wavelengths'], virtual_spectrum['combined_intensities']

        # Align spectra
        common_wl, real_aligned, virtual_aligned = self.align_spectra(
            real_wl, real_int, virtual_wl, virtual_int
        )

        # Detect peaks in both spectra
        real_peaks, real_props = signal.find_peaks(real_aligned, height=np.max(real_aligned)*0.1)
        virtual_peaks, virtual_props = signal.find_peaks(virtual_aligned, height=np.max(virtual_aligned)*0.1)

        # Calculate peak matching
        peak_matches = 0
        tolerance = 10  # wavelength tolerance for peak matching

        for real_peak in real_peaks:
            real_peak_wl = common_wl[real_peak]
            for virtual_peak in virtual_peaks:
                virtual_peak_wl = common_wl[virtual_peak]
                if abs(real_peak_wl - virtual_peak_wl) <= tolerance:
                    peak_matches += 1
                    break

        peak_precision = peak_matches / len(virtual_peaks) if len(virtual_peaks) > 0 else 0
        peak_recall = peak_matches / len(real_peaks) if len(real_peaks) > 0 else 0
        peak_f1 = 2 * (peak_precision * peak_recall) / (peak_precision + peak_recall) if (peak_precision + peak_recall) > 0 else 0

        return {
            'real_peaks': len(real_peaks),
            'virtual_peaks': len(virtual_peaks),
            'matched_peaks': peak_matches,
            'peak_precision': peak_precision,
            'peak_recall': peak_recall,
            'peak_f1_score': peak_f1,
            'common_wavelengths': common_wl,
            'real_aligned': real_aligned,
            'virtual_aligned': virtual_aligned
        }

    def calculate_spectral_similarity(self, real_spectrum, virtual_spectrum):
        """Calculate various similarity metrics between spectra"""
        real_wl, real_int = real_spectrum['wavelengths'], real_spectrum['intensities']
        virtual_wl, virtual_int = virtual_spectrum['combined_wavelengths'], virtual_spectrum['combined_intensities']

        # Align spectra
        common_wl, real_aligned, virtual_aligned = self.align_spectra(
            real_wl, real_int, virtual_wl, virtual_int
        )

        # Normalize intensities
        real_norm = (real_aligned - np.min(real_aligned)) / (np.max(real_aligned) - np.min(real_aligned) + 1e-10)
        virtual_norm = (virtual_aligned - np.min(virtual_aligned)) / (np.max(virtual_aligned) - np.min(virtual_aligned) + 1e-10)

        # Calculate similarity metrics
        mse = mean_squared_error(real_norm, virtual_norm)
        rmse = np.sqrt(mse)
        r2 = r2_score(real_norm, virtual_norm) if np.var(real_norm) > 1e-10 else 0

        # Pearson correlation
        correlation, p_value = stats.pearsonr(real_norm, virtual_norm)

        # Cosine similarity
        dot_product = np.dot(real_norm, virtual_norm)
        norm_product = np.linalg.norm(real_norm) * np.linalg.norm(virtual_norm)
        cosine_similarity = dot_product / (norm_product + 1e-10)

        # Spectral angle mapper (SAM)
        sam_angle = np.arccos(np.clip(cosine_similarity, -1, 1))

        return {
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'pearson_correlation': correlation,
            'correlation_p_value': p_value,
            'cosine_similarity': cosine_similarity,
            'spectral_angle': sam_angle,
            'common_wavelengths': common_wl,
            'real_normalized': real_norm,
            'virtual_normalized': virtual_norm
        }

    def validate_led_wavelength_response(self):
        """Validate LED wavelength-specific responses"""
        print("💡 Validating LED wavelength responses...")

        led_validation = {}

        for led_color, led_wavelength in self.led_system.led_wavelengths.items():
            responses = []

            # Test LED response across different molecular patterns
            test_patterns = ['c1ccccc1', 'CCO', 'CC(=O)O', 'C1=CC=C(C=C1)O']

            for pattern in test_patterns:
                analysis = self.led_system.analyze_molecular_fluorescence(pattern, led_wavelength)
                responses.append(analysis['fluorescence_intensity'])

            led_validation[led_color] = {
                'wavelength': led_wavelength,
                'responses': responses,
                'mean_response': np.mean(responses),
                'std_response': np.std(responses),
                'dynamic_range': np.max(responses) - np.min(responses)
            }

        return led_validation

    def comprehensive_validation(self, spectra_dir, molecular_patterns):
        """Run comprehensive validation comparing all systems"""
        print("🔬 Starting Comprehensive Hardware-Based Spectroscopy Validation")
        print("=" * 70)

        # Load data
        real_spectra = self.load_real_spectra(spectra_dir)
        virtual_spectra = self.generate_virtual_spectra(molecular_patterns)

        if not real_spectra or not virtual_spectra:
            print("❌ Insufficient data for validation")
            return None

        validation_results = {
            'peak_detection_results': [],
            'spectral_similarity_results': [],
            'led_validation': self.validate_led_wavelength_response(),
            'overall_metrics': {}
        }

        # Compare each virtual spectrum with each real spectrum
        print("\n📊 Comparing virtual vs real spectra...")

        for real_name, real_spectrum in real_spectra.items():
            for virtual_name, virtual_spectrum in virtual_spectra.items():

                # Peak detection comparison
                peak_results = self.compare_peak_detection(real_spectrum, virtual_spectrum)
                peak_results['real_spectrum'] = real_name
                peak_results['virtual_spectrum'] = virtual_name
                validation_results['peak_detection_results'].append(peak_results)

                # Spectral similarity comparison
                similarity_results = self.calculate_spectral_similarity(real_spectrum, virtual_spectrum)
                similarity_results['real_spectrum'] = real_name
                similarity_results['virtual_spectrum'] = virtual_name
                validation_results['spectral_similarity_results'].append(similarity_results)

        # Calculate overall metrics
        if validation_results['peak_detection_results']:
            peak_f1_scores = [r['peak_f1_score'] for r in validation_results['peak_detection_results']]
            correlations = [r['pearson_correlation'] for r in validation_results['spectral_similarity_results']]
            rmse_values = [r['rmse'] for r in validation_results['spectral_similarity_results']]

            validation_results['overall_metrics'] = {
                'mean_peak_f1_score': np.mean(peak_f1_scores),
                'std_peak_f1_score': np.std(peak_f1_scores),
                'mean_correlation': np.mean(correlations),
                'std_correlation': np.std(correlations),
                'mean_rmse': np.mean(rmse_values),
                'std_rmse': np.std(rmse_values),
                'validation_success_rate': sum(1 for c in correlations if c > 0.5) / len(correlations)
            }

        return validation_results

    def create_validation_visualizations(self, validation_results, save_dir):
        """Create comprehensive validation visualizations"""
        if not validation_results:
            return

        fig = plt.figure(figsize=(20, 15))

        # 1. Peak Detection Performance
        ax1 = plt.subplot(3, 4, 1)
        peak_f1_scores = [r['peak_f1_score'] for r in validation_results['peak_detection_results']]
        plt.hist(peak_f1_scores, bins=15, alpha=0.7, color='blue')
        plt.xlabel('Peak Detection F1 Score')
        plt.ylabel('Frequency')
        plt.title('Peak Detection Performance')
        plt.axvline(np.mean(peak_f1_scores), color='red', linestyle='--',
                   label=f'Mean: {np.mean(peak_f1_scores):.3f}')
        plt.legend()

        # 2. Spectral Correlation Distribution
        ax2 = plt.subplot(3, 4, 2)
        correlations = [r['pearson_correlation'] for r in validation_results['spectral_similarity_results']]
        plt.hist(correlations, bins=15, alpha=0.7, color='green')
        plt.xlabel('Pearson Correlation')
        plt.ylabel('Frequency')
        plt.title('Spectral Correlation Distribution')
        plt.axvline(np.mean(correlations), color='red', linestyle='--',
                   label=f'Mean: {np.mean(correlations):.3f}')
        plt.legend()

        # 3. RMSE Distribution
        ax3 = plt.subplot(3, 4, 3)
        rmse_values = [r['rmse'] for r in validation_results['spectral_similarity_results']]
        plt.hist(rmse_values, bins=15, alpha=0.7, color='orange')
        plt.xlabel('RMSE')
        plt.ylabel('Frequency')
        plt.title('RMSE Distribution')
        plt.axvline(np.mean(rmse_values), color='red', linestyle='--',
                   label=f'Mean: {np.mean(rmse_values):.3f}')
        plt.legend()

        # 4. LED Response Validation
        ax4 = plt.subplot(3, 4, 4)
        led_colors = list(validation_results['led_validation'].keys())
        led_responses = [validation_results['led_validation'][color]['mean_response']
                        for color in led_colors]
        colors = ['blue', 'green', 'red']
        plt.bar(led_colors, led_responses, color=colors, alpha=0.7)
        plt.xlabel('LED Color')
        plt.ylabel('Mean Response')
        plt.title('LED Wavelength Response')

        # 5-8. Example Spectral Comparisons
        for i in range(4):
            ax = plt.subplot(3, 4, 5 + i)
            if i < len(validation_results['spectral_similarity_results']):
                result = validation_results['spectral_similarity_results'][i]
                wl = result['common_wavelengths']
                real = result['real_normalized']
                virtual = result['virtual_normalized']

                plt.plot(wl, real, 'b-', label='Real', alpha=0.7)
                plt.plot(wl, virtual, 'r--', label='Virtual', alpha=0.7)
                plt.xlabel('Wavelength (nm)')
                plt.ylabel('Normalized Intensity')
                plt.title(f'Comparison {i+1}\nCorr: {result["pearson_correlation"]:.3f}')
                plt.legend()
                plt.grid(True, alpha=0.3)

        # 9. Peak Detection Scatter
        ax9 = plt.subplot(3, 4, 9)
        real_peaks = [r['real_peaks'] for r in validation_results['peak_detection_results']]
        virtual_peaks = [r['virtual_peaks'] for r in validation_results['peak_detection_results']]
        plt.scatter(real_peaks, virtual_peaks, alpha=0.6)
        plt.xlabel('Real Peaks')
        plt.ylabel('Virtual Peaks')
        plt.title('Peak Count Comparison')
        # Add diagonal line
        max_peaks = max(max(real_peaks), max(virtual_peaks))
        plt.plot([0, max_peaks], [0, max_peaks], 'r--', alpha=0.5)

        # 10. Correlation vs RMSE
        ax10 = plt.subplot(3, 4, 10)
        plt.scatter(correlations, rmse_values, alpha=0.6, color='purple')
        plt.xlabel('Pearson Correlation')
        plt.ylabel('RMSE')
        plt.title('Correlation vs RMSE')

        # 11. Overall Performance Metrics
        ax11 = plt.subplot(3, 4, 11)
        metrics = validation_results['overall_metrics']
        metric_names = ['Peak F1', 'Correlation', 'Success Rate']
        metric_values = [metrics['mean_peak_f1_score'],
                        metrics['mean_correlation'],
                        metrics['validation_success_rate']]
        bars = plt.bar(metric_names, metric_values, color=['blue', 'green', 'orange'], alpha=0.7)
        plt.ylabel('Score')
        plt.title('Overall Performance')
        plt.ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        # 12. Validation Summary
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        summary_text = f"""
        Validation Summary
        ==================

        Real Spectra: {len(validation_results['spectral_similarity_results']) // len(validation_results['peak_detection_results']) * len(validation_results['peak_detection_results'])}
        Virtual Spectra: {len(validation_results['peak_detection_results']) // len(validation_results['spectral_similarity_results']) * len(validation_results['spectral_similarity_results'])}

        Mean Peak F1: {metrics['mean_peak_f1_score']:.3f}
        Mean Correlation: {metrics['mean_correlation']:.3f}
        Mean RMSE: {metrics['mean_rmse']:.3f}
        Success Rate: {metrics['validation_success_rate']:.1%}

        LED Validation:
        Blue: {validation_results['led_validation']['blue']['mean_response']:.3f}
        Green: {validation_results['led_validation']['green']['mean_response']:.3f}
        Red: {validation_results['led_validation']['red']['mean_response']:.3f}
        """
        ax12.text(0.1, 0.9, summary_text, transform=ax12.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'comprehensive_validation.png'), dpi=300, bbox_inches='tight')
        plt.show()

        return fig

def load_molecular_patterns():
    """Load molecular patterns for validation"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, '..', '..')

    # Try to load from SMARTS files
    patterns = []
    smarts_files = [
        'agrafiotis-smarts-tar/agrafiotis.smarts',
        'ahmed-smarts-tar/ahmed.smarts',
        'hann-smarts-tar/hann.smarts'
    ]

    for smarts_file in smarts_files:
        filepath = os.path.join(base_dir, 'public', smarts_file)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if line.strip() and not line.startswith('#'):
                            parts = line.split()
                            if parts:
                                patterns.append(parts[0])
                                if len(patterns) >= 10:  # Limit for validation
                                    break
                if len(patterns) >= 10:
                    break
            except Exception as e:
                print(f"Error loading {smarts_file}: {e}")

    # Fallback to synthetic patterns
    if not patterns:
        patterns = [
            'c1ccccc1',           # Benzene
            'CCO',                # Ethanol
            'CC(=O)O',            # Acetic acid
            'C1=CC=C(C=C1)O',     # Phenol
            'CC(C)O',             # Isopropanol
            'c1ccc2ccccc2c1',     # Naphthalene
            'CC(=O)OC1=CC=CC=C1', # Aspirin
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', # Caffeine
            'CC(C)(C)OC(=O)NC1=CC=C(C=C1)O', # Boc-Tyrosine
            'C1=CC=C(C=C1)C(=O)O' # Benzoic acid
        ]

    return patterns[:10]  # Return first 10 for validation

def main():
    """Main validation execution"""
    print("🔬 Hardware-Based Spectroscopy Validation Framework")
    print("=" * 60)

    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, '..', '..')
    spectra_dir = os.path.join(base_dir, 'public', 'spectra')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Initialize validator
    validator = SpectroscopyValidator()

    # Load molecular patterns
    molecular_patterns = load_molecular_patterns()
    print(f"🧪 Loaded {len(molecular_patterns)} molecular patterns for validation")

    # Run comprehensive validation
    validation_results = validator.comprehensive_validation(spectra_dir, molecular_patterns)

    if validation_results:
        # Create visualizations
        validator.create_validation_visualizations(validation_results, results_dir)

        # Save results
        with open(os.path.join(results_dir, 'validation_results.json'), 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)

        # Print summary
        metrics = validation_results['overall_metrics']
        print(f"\n📊 Validation Results Summary:")
        print(f"   Mean Peak Detection F1: {metrics['mean_peak_f1_score']:.3f} ± {metrics['std_peak_f1_score']:.3f}")
        print(f"   Mean Spectral Correlation: {metrics['mean_correlation']:.3f} ± {metrics['std_correlation']:.3f}")
        print(f"   Mean RMSE: {metrics['mean_rmse']:.3f} ± {metrics['std_rmse']:.3f}")
        print(f"   Validation Success Rate: {metrics['validation_success_rate']:.1%}")

        # Validation assessment
        if metrics['mean_correlation'] > 0.6:
            print("✅ Strong correlation with real spectra achieved!")
        elif metrics['mean_correlation'] > 0.3:
            print("⚠️ Moderate correlation - system shows promise but needs improvement")
        else:
            print("❌ Low correlation - significant improvements needed")

        if metrics['validation_success_rate'] > 0.7:
            print("✅ High validation success rate!")
        elif metrics['validation_success_rate'] > 0.4:
            print("⚠️ Moderate success rate")
        else:
            print("❌ Low success rate")

        print(f"\n💾 Results saved to: {results_dir}")
        print("🏁 Validation complete!")

    else:
        print("❌ Validation failed - insufficient data")

if __name__ == "__main__":
    main()
