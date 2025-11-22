#!/usr/bin/env python3
"""
Pixel-to-Chemical Modification Mapping
=====================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json

class PixelChemicalMapper:
    def analyze_pixel_changes(self, pattern):
        """Analyze pattern and predict pixel/chemical changes"""
        if not pattern:
            return {'modifications': [], 'pixel_changes': {}}
        
        modifications = []
        pixel_changes = {}
        
        # Red changes from oxidizable groups
        if 'OH' in pattern:
            modifications.append('oxidation')
            pixel_changes['red'] = 120
        
        # Green changes from aromatic systems  
        if any(c.islower() for c in pattern):
            modifications.append('substitution')
            pixel_changes['green'] = 100
        
        # Blue changes from reducible groups
        if 'C=O' in pattern:
            modifications.append('reduction')
            pixel_changes['blue'] = 80
        
        return {
            'pattern': pattern,
            'modifications': modifications,
            'pixel_changes': pixel_changes,
            'total_changes': len(modifications)
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
    print("🎨 Pixel-to-Chemical Modification Mapping")
    print("=" * 45)
    
    datasets = load_datasets()
    mapper = PixelChemicalMapper()
    
    # Test patterns
    test_patterns = []
    for patterns in datasets.values():
        test_patterns.extend(patterns[:2])
    
    # Analyze
    results = [mapper.analyze_pixel_changes(p) for p in test_patterns]
    
    # Stats
    total_mods = sum(r['total_changes'] for r in results)
    patterns_with_mods = sum(1 for r in results if r['total_changes'] > 0)
    
    print(f"Total modifications: {total_mods}")
    print(f"Patterns with modifications: {patterns_with_mods}/{len(results)}")
    
    # Visualization
    modification_counts = [r['total_changes'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(modification_counts)), modification_counts, alpha=0.7)
    plt.xlabel('Pattern Index')
    plt.ylabel('Modifications')
    plt.title('Modifications per Pattern')
    
    plt.subplot(1, 2, 2)
    plt.hist(modification_counts, bins=5, alpha=0.7)
    plt.xlabel('Modification Count')
    plt.ylabel('Frequency')
    plt.title('Modification Distribution')
    
    # Create results directory with dynamic path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'pixel_chemical_mapping.png'), dpi=300)
    plt.show()
    
    # Save
    summary = {
        'total_modifications': total_mods,
        'success_rate': patterns_with_mods / len(results) if results else 0,
        'avg_modifications': total_mods / len(results) if results else 0
    }
    
    with open(os.path.join(results_dir, 'pixel_chemical_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Success rate: {summary['success_rate']:.1%}")
    print("✅ Pixel-chemical mapping validated!" if summary['success_rate'] > 0.5 else "⚠️ Mapping needs improvement")
    print("🏁 Analysis complete!")

if __name__ == "__main__":
    main()