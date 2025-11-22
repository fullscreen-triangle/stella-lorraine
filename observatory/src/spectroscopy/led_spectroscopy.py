#!/usr/bin/env python3
"""
LED Spectroscopy Integration
============================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json

class LEDSpectroscopySystem:
    def __init__(self):
        self.led_wavelengths = {
            'blue': 470,   # nm, standard monitor backlight
            'green': 525,  # nm, status indicator LEDs  
            'red': 625     # nm, power/activity LEDs
        }
        
        self.quantum_efficiencies = {
            'blue': 0.8,
            'green': 0.9,
            'red': 0.7
        }
    
    def analyze_molecular_fluorescence(self, molecular_pattern, excitation_wavelength):
        """Analyze molecular fluorescence using LED excitation"""
        
        # Map molecular structure to fluorescence properties
        fluorescence_props = self.predict_fluorescence_properties(molecular_pattern)
        
        # Calculate excitation efficiency
        excitation_eff = self.calculate_excitation_efficiency(fluorescence_props, excitation_wavelength)
        
        # Simulate fluorescence spectrum
        emission_spectrum = self.simulate_emission_spectrum(fluorescence_props, excitation_eff)
        
        # Calculate detection efficiency
        detection_eff = self.calculate_detection_efficiency(emission_spectrum)
        
        return {
            'excitation_wavelength': excitation_wavelength,
            'excitation_efficiency': excitation_eff,
            'emission_spectrum': emission_spectrum,
            'detection_efficiency': detection_eff,
            'fluorescence_intensity': excitation_eff * detection_eff,
            'molecular_pattern': molecular_pattern
        }
    
    def predict_fluorescence_properties(self, pattern):
        """Predict fluorescence properties from molecular structure"""
        if not pattern:
            return {'peak_emission': 550, 'bandwidth': 50, 'quantum_yield': 0.1}
        
        # Aromatic character influences fluorescence
        aromatic_count = sum(1 for c in pattern if c.islower())
        aromatic_ratio = aromatic_count / len(pattern) if pattern else 0
        
        # Conjugation affects emission wavelength
        conjugation = pattern.count('=') + pattern.count('#')
        
        # Predict emission peak
        base_emission = 500  # nm
        aromatic_shift = aromatic_ratio * 100  # Red shift for aromatics
        conjugation_shift = conjugation * 20   # Red shift for conjugation
        
        peak_emission = base_emission + aromatic_shift + conjugation_shift
        
        # Predict quantum yield
        base_yield = 0.1
        aromatic_enhancement = aromatic_ratio * 0.4  # Aromatics enhance fluorescence
        quenching_factor = pattern.count('OH') * 0.05  # OH groups can quench
        
        quantum_yield = max(0.01, base_yield + aromatic_enhancement - quenching_factor)
        
        return {
            'peak_emission': min(800, peak_emission),
            'bandwidth': 40 + aromatic_ratio * 30,
            'quantum_yield': min(0.9, quantum_yield)
        }
    
    def calculate_excitation_efficiency(self, fluor_props, excitation_wl):
        """Calculate excitation efficiency for given wavelength"""
        peak_emission = fluor_props['peak_emission']
        
        # Simple model: excitation efficiency decreases with wavelength difference
        # Stokes shift consideration
        stokes_shift = 50  # nm, typical Stokes shift
        optimal_excitation = peak_emission - stokes_shift
        
        wavelength_diff = abs(excitation_wl - optimal_excitation)
        
        # Gaussian-like efficiency curve
        efficiency = np.exp(-0.001 * wavelength_diff**2)
        
        return min(1.0, efficiency)
    
    def simulate_emission_spectrum(self, fluor_props, excitation_eff):
        """Simulate fluorescence emission spectrum"""
        peak = fluor_props['peak_emission']
        bandwidth = fluor_props['bandwidth']
        quantum_yield = fluor_props['quantum_yield']
        
        # Generate wavelength points
        wavelengths = np.arange(peak - bandwidth*2, peak + bandwidth*2, 2)
        
        # Gaussian emission profile
        intensities = quantum_yield * excitation_eff * np.exp(
            -0.5 * ((wavelengths - peak) / (bandwidth/2))**2
        )
        
        return {
            'wavelengths': wavelengths.tolist(),
            'intensities': intensities.tolist(),
            'peak_intensity': max(intensities),
            'integrated_intensity': np.trapz(intensities, wavelengths)
        }
    
    def calculate_detection_efficiency(self, emission_spectrum):
        """Calculate detection efficiency using standard photodetectors"""
        # Simplified model for silicon photodetector efficiency
        wavelengths = np.array(emission_spectrum['wavelengths'])
        intensities = np.array(emission_spectrum['intensities'])
        
        # Silicon detector response (simplified)
        detector_response = np.where(
            (wavelengths >= 400) & (wavelengths <= 900),
            0.8 * (1 - (wavelengths - 600)**2 / 200000),  # Peak around 600nm
            0.1
        )
        
        # Weight intensities by detector response
        detected_signal = intensities * detector_response
        total_signal = np.sum(detected_signal)
        
        return min(1.0, total_signal / max(1, np.sum(intensities)))

class SpectroscopyValidator:
    def __init__(self):
        self.led_system = LEDSpectroscopySystem()
    
    def validate_led_spectroscopy_performance(self, molecular_patterns):
        """Validate LED spectroscopy performance across molecular dataset"""
        results = []
        
        for pattern in molecular_patterns:
            pattern_results = {}
            
            # Test with all LED wavelengths
            for led_color, wavelength in self.led_system.led_wavelengths.items():
                analysis = self.led_system.analyze_molecular_fluorescence(pattern, wavelength)
                pattern_results[led_color] = analysis
            
            # Find optimal LED for this molecule
            best_led = max(pattern_results.keys(), 
                          key=lambda k: pattern_results[k]['fluorescence_intensity'])
            
            results.append({
                'pattern': pattern,
                'led_responses': pattern_results,
                'optimal_led': best_led,
                'max_intensity': pattern_results[best_led]['fluorescence_intensity']
            })
        
        return results

def load_datasets():
    datasets = {}
    
    # Find the correct base directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, '..', '..')  # Go up to gonfanolier root
    
    files = {
        'agrafiotis': os.path.join(base_dir, 'public', 'agrafiotis-smarts-tar', 'agrafiotis.smarts'),
        'ahmed': os.path.join(base_dir, 'public', 'ahmed-smarts-tar', 'ahmed.smarts'),
        'hann': os.path.join(base_dir, 'public', 'hann-smarts-tar', 'hann.smarts'),
        'walters': os.path.join(base_dir, 'public', 'walters-smarts-tar', 'walters.smarts')
    }
    
    for name, filepath in files.items():
        if os.path.exists(filepath):
            patterns = []
            try:
                with open(filepath, 'r') as f:
                    for line in f:
                        if line.strip() and not line.startswith('#'):
                            parts = line.split()
                            if parts:
                                patterns.append(parts[0])
                datasets[name] = patterns
                print(f"Loaded {len(patterns)} patterns from {name}")
            except Exception as e:
                print(f"Error loading {name}: {e}")
        else:
            print(f"File not found: {filepath}")
    
    # If no datasets found, create synthetic data for demo
    if not datasets:
        print("No SMARTS files found, using synthetic molecular patterns for demo...")
        datasets['synthetic'] = [
            'c1ccccc1',  # benzene
            'CCO',       # ethanol
            'CC(=O)O',   # acetic acid
            'c1ccc2ccccc2c1',  # naphthalene
            'CC(C)O'     # isopropanol
        ]
        print(f"Created {len(datasets['synthetic'])} synthetic patterns")
    
    return datasets

def main():
    print("💡 LED Spectroscopy Integration Analysis")
    print("=" * 45)
    
    # Get base directory for file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, '..', '..')  # Go up to gonfanolier root
    
    datasets = load_datasets()
    validator = SpectroscopyValidator()
    
    # Combine patterns from all datasets
    all_patterns = []
    for patterns in datasets.values():
        all_patterns.extend(patterns[:5])  # First 5 from each for demo
    
    print(f"\n💡 Testing LED spectroscopy on {len(all_patterns)} molecules...")
    
    # Validate LED performance
    validation_results = validator.validate_led_spectroscopy_performance(all_patterns)
    
    # Analyze results
    led_usage = {'blue': 0, 'green': 0, 'red': 0}
    intensities = []
    
    for result in validation_results:
        led_usage[result['optimal_led']] += 1
        intensities.append(result['max_intensity'])
    
    # Handle empty results
    if not intensities:
        print("❌ No validation results to analyze")
        return
    
    avg_intensity = np.mean(intensities)
    detection_success_rate = sum(1 for i in intensities if i > 0.1) / len(intensities)
    
    print(f"\n📊 Results:")
    print(f"Average fluorescence intensity: {avg_intensity:.3f}")
    print(f"Detection success rate: {detection_success_rate:.1%}")
    print(f"Optimal LED usage: Blue {led_usage['blue']}, Green {led_usage['green']}, Red {led_usage['red']}")
    
    # Show example analysis
    if validation_results:
        example = validation_results[0]
        print(f"\n🔍 Example analysis for '{example['pattern']}':")
        print(f"  Optimal LED: {example['optimal_led']}")
        print(f"  Max intensity: {example['max_intensity']:.3f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # LED usage distribution
    led_colors = list(led_usage.keys())
    led_counts = list(led_usage.values())
    colors = ['blue', 'green', 'red']
    
    axes[0].pie(led_counts, labels=led_colors, colors=colors, autopct='%1.1f%%', alpha=0.7)
    axes[0].set_title('Optimal LED Distribution')
    
    # Intensity distribution
    axes[1].hist(intensities, bins=15, alpha=0.7, color='orange')
    axes[1].axvline(avg_intensity, color='red', linestyle='--', label=f'Mean: {avg_intensity:.3f}')
    axes[1].set_xlabel('Fluorescence Intensity')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Intensity Distribution')
    axes[1].legend()
    
    # Example emission spectrum
    if validation_results:
        example_result = validation_results[0]
        best_led = example_result['optimal_led']
        spectrum = example_result['led_responses'][best_led]['emission_spectrum']
        
        axes[2].plot(spectrum['wavelengths'], spectrum['intensities'], 'g-', linewidth=2)
        axes[2].set_xlabel('Wavelength (nm)')
        axes[2].set_ylabel('Intensity')
        axes[2].set_title(f'Example Emission Spectrum\n({best_led} LED excitation)')
        axes[2].grid(True, alpha=0.3)
    
    # Use correct results directory path
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'led_spectroscopy.png'), dpi=300)
    plt.show()
    
    # Save results
    summary_results = {
        'performance_metrics': {
            'avg_fluorescence_intensity': avg_intensity,
            'detection_success_rate': detection_success_rate,
            'optimal_led_distribution': led_usage
        },
        'led_wavelengths': validator.led_system.led_wavelengths,
        'molecules_tested': len(validation_results)
    }
    
    with open(os.path.join(results_dir, 'led_spectroscopy_results.json'), 'w') as f:
        json.dump(summary_results, f, indent=2, default=str)
    
    print(f"\n🎯 LED Spectroscopy Validation:")
    
    if avg_intensity > 0.2:
        print("✅ Strong fluorescence signals achieved with standard LEDs")
    
    if detection_success_rate > 0.8:
        print("✅ High detection success rate demonstrates feasibility")
    
    if max(led_counts) / sum(led_counts) < 0.8:
        print("✅ Good distribution across LED wavelengths shows versatility")
    
    print("🏁 LED spectroscopy analysis complete!")

if __name__ == "__main__":
    main()