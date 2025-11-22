#!/usr/bin/env python3
"""
Molecule-to-Drip Algorithm - Computer Vision for Chemical Analysis
================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json

class MoleculeToDripConverter:
    def convert_smarts_to_drip(self, pattern):
        """Convert SMARTS pattern to droplet impact visualization"""
        # Calculate molecular properties
        complexity = len(set(pattern)) / len(pattern) if pattern else 0
        size_factor = np.log(len(pattern) + 1)
        reactivity = pattern.count('=') + pattern.count('#')
        
        # Map to droplet parameters
        velocity = 2.0 + complexity * 3.0
        radius = 0.5 + size_factor * 0.5
        impact_strength = velocity * radius * (1 + reactivity * 0.1)
        
        # Generate wave pattern
        grid_size = 200
        center = grid_size // 2
        y, x = np.ogrid[:grid_size, :grid_size]
        distance = np.sqrt((x - center)**2 + (y - center)**2)
        
        # Create concentric waves
        wave_pattern = impact_strength * np.exp(-distance / (radius * 20)) * \
                      np.cos(2 * np.pi * distance / (radius * 10))
        
        return {
            'wave_pattern': wave_pattern,
            'velocity': velocity,
            'radius': radius,
            'impact_strength': impact_strength,
            'pattern': pattern
        }

def load_datasets():
    """Load SMARTS patterns"""
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
                        line = line.strip()
                        if line and not line.startswith('#'):
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
    print("💧 Molecule-to-Drip Computer Vision Analysis")
    print("=" * 50)
    
    # Load data
    datasets = load_datasets()
    converter = MoleculeToDripConverter()
    
    # Create results directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, '..', '..')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Convert molecules to drip patterns
    all_results = {}
    
    for dataset_name, patterns in datasets.items():
        print(f"\n🔍 Processing {dataset_name} dataset...")
        
        # Process first 5 patterns as examples
        sample_patterns = patterns[:5]
        drip_results = []
        
        for i, pattern in enumerate(sample_patterns):
            result = converter.convert_smarts_to_drip(pattern)
            drip_results.append(result)
            
            # Visualize first pattern
            if i == 0:
                plt.figure(figsize=(10, 8))
                plt.subplot(2, 2, 1)
                plt.imshow(result['wave_pattern'], cmap='viridis')
                plt.title(f'{dataset_name}: {pattern}')
                plt.colorbar()
                
                # Radial profile
                center = 100
                distances = np.arange(0, 100, 2)
                radial_profile = []
                for r in distances:
                    mask = (np.abs(np.sqrt((np.arange(200)[:, None] - center)**2 + 
                                         (np.arange(200) - center)**2) - r) < 1)
                    radial_profile.append(np.mean(result['wave_pattern'][mask]))
                
                plt.subplot(2, 2, 2)
                plt.plot(distances, radial_profile)
                plt.title('Radial Wave Profile')
                plt.xlabel('Distance from center')
                plt.ylabel('Amplitude')
                
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f'{dataset_name}_drip_example.png'), dpi=300)
                plt.show()
        
        # Calculate dataset statistics
        avg_velocity = np.mean([r['velocity'] for r in drip_results])
        avg_radius = np.mean([r['radius'] for r in drip_results])
        avg_impact = np.mean([r['impact_strength'] for r in drip_results])
        
        all_results[dataset_name] = {
            'pattern_count': len(drip_results),
            'avg_velocity': avg_velocity,
            'avg_radius': avg_radius,
            'avg_impact_strength': avg_impact
        }
        
        print(f"  Avg velocity: {avg_velocity:.2f}")
        print(f"  Avg radius: {avg_radius:.2f}")
        print(f"  Avg impact: {avg_impact:.2f}")
    
    # Save results
    with open(os.path.join(results_dir, 'molecule_to_drip_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n🎯 Converted {sum(r['pattern_count'] for r in all_results.values())} molecules to drip patterns")
    print("🏁 Computer vision chemical analysis ready!")

if __name__ == "__main__":
    main()
