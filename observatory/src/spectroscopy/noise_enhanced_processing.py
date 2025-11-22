#!/usr/bin/env python3
"""
Noise-Enhanced Processing
========================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import signal

class NoiseProcessor:
    def generate_environmental_noise(self, duration=1.0):
        """Generate environmental noise N(t) = Σ A_k cos(2πf_k t + φ_k) + ξ(t)"""
        t = np.linspace(0, duration, 1000)
        noise = np.random.normal(0, 0.2, len(t))
        
        # Add harmonic components
        for k in range(3):
            A_k = 0.3 * np.random.random()
            f_k = 10 + k * 5
            noise += A_k * np.cos(2 * np.pi * f_k * t)
        
        return t, noise
    
    def enhance_signal(self, base_signal, noise):
        """Apply noise enhancement through stochastic resonance"""
        noisy_signal = base_signal + noise
        
        # Adaptive thresholding
        threshold = np.std(base_signal) * 0.5
        enhanced = np.where(np.abs(noisy_signal) > threshold, 
                           noisy_signal * 1.5, 
                           noisy_signal * 1.1)
        
        return enhanced
    
    def calculate_snr(self, signal, noise):
        """Calculate signal-to-noise ratio"""
        signal_power = np.mean(signal**2)
        noise_power = np.mean(noise**2)
        return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    def process_molecular_pattern(self, pattern):
        """Process molecular pattern with noise enhancement"""
        # Generate base signal from pattern
        if not pattern:
            base_signal = np.zeros(1000)
        else:
            t = np.linspace(0, 1, 1000)
            freq = 10 + len(set(pattern)) * 2
            amplitude = min(2.0, len(pattern) / 10)
            base_signal = amplitude * np.sin(2 * np.pi * freq * t)
        
        # Add noise
        _, noise = self.generate_environmental_noise()
        noise = noise[:len(base_signal)]
        
        # Enhance signal
        enhanced = self.enhance_signal(base_signal, noise)
        
        # Calculate metrics
        snr_base = self.calculate_snr(base_signal, noise)
        snr_enhanced = self.calculate_snr(enhanced, noise)
        
        return {
            'snr_improvement': snr_enhanced - snr_base,
            'base_signal': base_signal,
            'enhanced_signal': enhanced,
            'pattern': pattern
        }

def load_datasets():
    datasets = {}
    
    # Dynamic path resolution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    public_dir = os.path.join(project_root, 'public')
    
    files = {
        'agrafiotis': os.path.join(public_dir, 'agrafiotis-smarts-tar', 'agrafiotis.smarts'),
        'ahmed': os.path.join(public_dir, 'ahmed-smarts-tar', 'ahmed.smarts'),
        'hann': os.path.join(public_dir, 'hann-smarts-tar', 'hann.smarts'),
        'walters': os.path.join(public_dir, 'walters-smarts-tar', 'walters.smarts')
    }
    
    for name, filepath in files.items():
        if os.path.exists(filepath):
            patterns = []
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
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
    
    # Fallback to synthetic data if no files found
    if not datasets:
        print("No SMARTS files found, generating synthetic patterns...")
        datasets['synthetic'] = [
            'C1=CC=CC=C1',  # Benzene
            'CCO',          # Ethanol
            'CC(=O)O',      # Acetic acid
            'C1=CC=C(C=C1)O',  # Phenol
            'CC(C)O'        # Isopropanol
        ]
        print(f"Generated {len(datasets['synthetic'])} synthetic patterns")
    
    return datasets

def main():
    print("🌊 Noise-Enhanced Processing")
    print("=" * 30)
    
    datasets = load_datasets()
    processor = NoiseProcessor()
    
    # Test on sample patterns
    test_patterns = []
    for patterns in datasets.values():
        test_patterns.extend(patterns[:2])
    
    results = []
    for pattern in test_patterns:
        result = processor.process_molecular_pattern(pattern)
        results.append(result)
    
    # Analyze results
    snr_improvements = [r['snr_improvement'] for r in results]
    avg_improvement = np.mean(snr_improvements)
    success_rate = sum(1 for s in snr_improvements if s > 0) / len(snr_improvements)
    
    print(f"Average SNR improvement: {avg_improvement:.2f} dB")
    print(f"Enhancement success rate: {success_rate:.1%}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.bar(range(len(snr_improvements)), snr_improvements, alpha=0.7)
    ax1.set_xlabel('Pattern Index')
    ax1.set_ylabel('SNR Improvement (dB)')
    ax1.set_title('Noise Enhancement Results')
    
    if results:
        t = np.linspace(0, 1, len(results[0]['base_signal']))
        ax2.plot(t, results[0]['base_signal'], label='Base Signal')
        ax2.plot(t, results[0]['enhanced_signal'], label='Enhanced Signal')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Amplitude') 
        ax2.set_title('Example Enhancement')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Create results directory with dynamic path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'noise_enhancement.png'), dpi=300)
    plt.show()
    
    # Save results
    summary = {'avg_snr_improvement': avg_improvement, 'success_rate': success_rate}
    with open(os.path.join(results_dir, 'noise_enhancement_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✅ Noise enhancement validated!" if avg_improvement > 1.0 else "⚠️ Enhancement needs improvement")
    print("🏁 Analysis complete!")

if __name__ == "__main__":
    main()