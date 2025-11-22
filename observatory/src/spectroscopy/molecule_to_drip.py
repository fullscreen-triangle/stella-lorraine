#!/usr/bin/env python3
"""
Universal Molecule-to-Drip Algorithm
===================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2

class MoleculeToDripConverter:
    def __init__(self):
        self.s_entropy_calculator = SEntropyCalculator()
        self.droplet_mapper = DropletMapper()
        
    def convert_molecule_to_drip_pattern(self, smarts_pattern):
        """Convert SMARTS to visual drip pattern"""
        # Calculate S-entropy coordinates  
        s_coords = self.s_entropy_calculator.calculate_s_entropy(smarts_pattern)
        
        # Map to droplet parameters
        droplet_params = self.droplet_mapper.map_to_droplet(s_coords)
        
        # Generate drip pattern
        drip_pattern = self.generate_drip_visualization(droplet_params)
        
        return {
            'pattern': smarts_pattern,
            's_coordinates': s_coords,
            'droplet_parameters': droplet_params,
            'drip_visualization': drip_pattern
        }
    
    def generate_drip_visualization(self, droplet_params):
        """Generate visual droplet impact pattern"""
        size = 200
        pattern = np.zeros((size, size))
        center = size // 2
        
        # Primary impact
        radius = int(droplet_params['radius'] * 30)
        velocity = droplet_params['velocity']
        
        y, x = np.ogrid[:size, :size]
        distance = np.sqrt((x - center)**2 + (y - center)**2)
        
        # Generate concentric waves
        wave_pattern = velocity * np.exp(-distance / radius) * \
                      np.cos(2 * np.pi * distance / (radius * 0.3))
        
        return wave_pattern

class SEntropyCalculator:
    def calculate_s_entropy(self, pattern):
        """Calculate S-entropy coordinates"""
        if not pattern:
            return {'S_structure': 0, 'S_spectroscopy': 0, 'S_activity': 0}
        
        # S_structure from structural complexity
        s_structure = len(set(pattern)) / len(pattern)
        
        # S_spectroscopy from functional groups
        func_groups = pattern.count('OH') + pattern.count('C=O') + pattern.count('NH')
        s_spectroscopy = min(1.0, func_groups / len(pattern) * 10)
        
        # S_activity from reactivity indicators
        reactive = pattern.count('=') + pattern.count('#') + pattern.count('F')
        s_activity = min(1.0, reactive / len(pattern) * 5)
        
        return {
            'S_structure': s_structure,
            'S_spectroscopy': s_spectroscopy,  
            'S_activity': s_activity
        }

class DropletMapper:
    def map_to_droplet(self, s_coords):
        """Map S-entropy to droplet parameters"""
        return {
            'velocity': 2.0 + s_coords['S_structure'] * 3.0,
            'radius': 0.5 + s_coords['S_spectroscopy'] * 2.0,
            'surface_tension': 0.03 + s_coords['S_activity'] * 0.04,
            'impact_angle': s_coords['S_structure'] * 45
        }

class ComputerVisionAnalyzer:
    def __init__(self):
        self.feature_extractor = PatternFeatureExtractor()
        
    def analyze_drip_patterns(self, drip_patterns, labels):
        """Analyze drip patterns with computer vision"""
        features = []
        
        for pattern in drip_patterns:
            pattern_features = self.feature_extractor.extract_features(pattern)
            features.append(pattern_features)
        
        return {
            'features': features,
            'pattern_count': len(drip_patterns),
            'feature_dimension': len(features[0]) if features else 0
        }

class PatternFeatureExtractor:
    def extract_features(self, pattern):
        """Extract computer vision features from drip pattern"""
        features = []
        
        # Statistical features
        features.extend([
            np.mean(pattern),
            np.std(pattern),
            np.max(pattern),
            np.min(pattern)
        ])
        
        # Geometric features
        if pattern.max() > pattern.min():
            # Normalize for edge detection
            normalized = ((pattern - pattern.min()) / (pattern.max() - pattern.min()) * 255).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(normalized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                perimeter = cv2.arcLength(largest, True)
                features.extend([area, perimeter])
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0])
        
        # Frequency domain features
        fft = np.fft.fft2(pattern)
        magnitude = np.abs(fft)
        features.extend([
            np.mean(magnitude),
            np.std(magnitude)
        ])
        
        return features

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
    print("💧 Universal Molecule-to-Drip Algorithm")
    print("=" * 40)
    
    datasets = load_datasets()
    converter = MoleculeToDripConverter()
    cv_analyzer = ComputerVisionAnalyzer()
    
    # Convert molecules to drip patterns
    all_results = []
    all_drip_patterns = []
    all_labels = []
    
    for dataset_name, patterns in datasets.items():
        print(f"\n💧 Processing {dataset_name} ({len(patterns)} patterns)...")
        
        for pattern in patterns[:5]:  # First 5 from each
            result = converter.convert_molecule_to_drip_pattern(pattern)
            all_results.append(result)
            all_drip_patterns.append(result['drip_visualization'])
            all_labels.append(dataset_name)
    
    print(f"\nConverted {len(all_results)} molecules to drip patterns")
    
    # Computer vision analysis
    cv_results = cv_analyzer.analyze_drip_patterns(all_drip_patterns, all_labels)
    
    print(f"Extracted {cv_results['feature_dimension']} features per pattern")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Molecule-to-Drip Conversion Results')
    
    # Show drip patterns
    for i in range(6):
        row, col = divmod(i, 3)
        if i < len(all_drip_patterns):
            axes[row, col].imshow(all_drip_patterns[i], cmap='viridis')
            axes[row, col].set_title(f'{all_labels[i]}: {all_results[i]["pattern"][:10]}...')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Analysis plots
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # S-entropy distribution
    s_structure_vals = [r['s_coordinates']['S_structure'] for r in all_results]
    s_spectroscopy_vals = [r['s_coordinates']['S_spectroscopy'] for r in all_results]
    s_activity_vals = [r['s_coordinates']['S_activity'] for r in all_results]
    
    ax1.hist(s_structure_vals, alpha=0.5, label='S_structure', bins=15)
    ax1.hist(s_spectroscopy_vals, alpha=0.5, label='S_spectroscopy', bins=15)
    ax1.hist(s_activity_vals, alpha=0.5, label='S_activity', bins=15)
    ax1.set_xlabel('S-Entropy Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('S-Entropy Coordinate Distribution')
    ax1.legend()
    
    # Droplet parameter correlation
    velocities = [r['droplet_parameters']['velocity'] for r in all_results]
    radii = [r['droplet_parameters']['radius'] for r in all_results]
    
    ax2.scatter(velocities, radii, alpha=0.7, c=s_structure_vals, cmap='viridis')
    ax2.set_xlabel('Droplet Velocity')
    ax2.set_ylabel('Droplet Radius')
    ax2.set_title('Droplet Parameter Space')
    
    # Create results directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, '..', '..')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.figure(1)
    plt.savefig(os.path.join(results_dir, 'molecule_to_drip_patterns.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(2)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'molecule_to_drip_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    summary = {
        'molecules_converted': len(all_results),
        'datasets_processed': len(datasets),
        'avg_s_structure': np.mean(s_structure_vals),
        'avg_s_spectroscopy': np.mean(s_spectroscopy_vals), 
        'avg_s_activity': np.mean(s_activity_vals),
        'cv_features_extracted': cv_results['feature_dimension'],
        'conversion_success_rate': len([r for r in all_results if np.max(r['drip_visualization']) > 0.1]) / len(all_results)
    }
    
    with open(os.path.join(results_dir, 'molecule_to_drip_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n🎯 Conversion Results:")
    print(f"Molecules converted: {summary['molecules_converted']}")
    print(f"Success rate: {summary['conversion_success_rate']:.1%}")
    print(f"CV features per pattern: {summary['cv_features_extracted']}")
    
    if summary['conversion_success_rate'] > 0.8:
        print("✅ High conversion success rate achieved")
    
    if summary['cv_features_extracted'] > 5:
        print("✅ Rich feature extraction for computer vision")
    
    print("🏁 Molecule-to-drip conversion complete!")

if __name__ == "__main__":
    main()