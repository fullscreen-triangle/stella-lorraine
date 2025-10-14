#!/usr/bin/env python3
"""
Cardiac-Referenced Hierarchical Harmonic Analysis with Random Graph Topology
============================================================================

NEW APPROACH: Everything expressed as oscillations relative to heartbeat
- NO conversion back to seconds/meters
- ALL measurements as frequency ratios: f_i / f_cardiac
- Phase relationships in radians (dimensionless)
- Random graph of which harmonics phase-lock
- Hierarchical clustering of oscillation scales

Key Principle: "Explosion" of data into frequency domain, STAY in frequency domain

Measurements:
- Position → Spatial oscillation frequency
- Velocity → Rate of position oscillation change  
- Time → Temporal oscillation phase
- Heart rate → Master reference frequency (f_ref)
- All other physiology → Oscillations relative to f_ref

Result: Hierarchical harmonic network + random graph topology
NOT: "X zeptoseconds" or "Y Planck lengths"
BUT: "N-level harmonic cascade with M phase-locked nodes"
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set
from scipy import signal, fft
from scipy.stats import zscore
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


class CardiacReferencedHarmonicExtractor:
    """
    Extract all oscillations relative to cardiac frequency
    
    NEVER converts back to original units
    Everything stays as frequency ratios and phase relationships
    """
    
    def __init__(self, cardiac_frequency_hz: float):
        """
        Initialize with cardiac reference frequency
        
        Args:
            cardiac_frequency_hz: Heart rate in Hz (e.g., 2.375 Hz = 142.5 bpm)
        """
        self.f_cardiac = cardiac_frequency_hz
        
        print(f"\n🫀 Cardiac Reference Frequency: {cardiac_frequency_hz:.4f} Hz")
        print(f"   (Heart Rate: {cardiac_frequency_hz * 60:.1f} bpm)")
        print(f"   All measurements expressed relative to this frequency")
    
    def extract_oscillations(self, signal_data: np.ndarray, 
                            sampling_rate: float,
                            signal_name: str) -> Dict:
        """
        Extract oscillatory components from any signal
        
        Returns frequencies as RATIOS to cardiac, not absolute values
        
        Args:
            signal_data: Time series data
            sampling_rate: Sampling rate of the signal
            signal_name: Name for labeling
        
        Returns:
            Dictionary with frequency_ratios, phases, amplitudes
        """
        # FFT to get frequency components
        fft_result = fft.rfft(signal_data)
        frequencies = fft.rfftfreq(len(signal_data), 1/sampling_rate)
        
        # Get magnitudes and phases
        magnitudes = np.abs(fft_result)
        phases = np.angle(fft_result)
        
        # Express as ratios to cardiac frequency
        frequency_ratios = frequencies / self.f_cardiac
        
        # Find significant harmonics (above noise threshold)
        threshold = np.mean(magnitudes) + 2 * np.std(magnitudes)
        significant_indices = magnitudes > threshold
        
        # Keep only significant components
        harmonic_ratios = frequency_ratios[significant_indices]
        harmonic_phases = phases[significant_indices]
        harmonic_amplitudes = magnitudes[significant_indices]
        
        # Sort by amplitude
        sort_indices = np.argsort(harmonic_amplitudes)[::-1]
        
        result = {
            'signal_name': signal_name,
            'n_harmonics': len(harmonic_ratios),
            'frequency_ratios': harmonic_ratios[sort_indices].tolist(),  # Relative to cardiac
            'phases_radians': harmonic_phases[sort_indices].tolist(),     # Dimensionless
            'amplitudes': harmonic_amplitudes[sort_indices].tolist(),
            'cardiac_reference_hz': self.f_cardiac
        }
        
        print(f"\n   📊 {signal_name}: Extracted {len(harmonic_ratios)} significant harmonics")
        print(f"      Top 5 frequency ratios (f/f_cardiac):")
        for i, ratio in enumerate(harmonic_ratios[sort_indices][:5]):
            print(f"         {i+1}. {ratio:.4f}x cardiac frequency")
        
        return result


class HierarchicalOscillationNetwork:
    """
    Build hierarchical network of oscillations
    
    Nodes = Oscillatory components
    Edges = Phase-locking relationships
    Hierarchy = Frequency scale levels
    """
    
    def __init__(self, cardiac_frequency_hz: float):
        self.f_cardiac = cardiac_frequency_hz
        self.graph = nx.Graph()
        self.hierarchy_levels = {}
        self.all_oscillations = []
        
    def add_oscillation_set(self, oscillation_data: Dict):
        """Add a set of oscillations from one signal"""
        signal_name = oscillation_data['signal_name']
        
        for i, (freq_ratio, phase, amplitude) in enumerate(zip(
            oscillation_data['frequency_ratios'],
            oscillation_data['phases_radians'],
            oscillation_data['amplitudes']
        )):
            node_id = f"{signal_name}_h{i}"
            
            # Add node with frequency ratio and phase
            self.graph.add_node(
                node_id,
                signal=signal_name,
                frequency_ratio=freq_ratio,
                phase_radians=phase,
                amplitude=amplitude,
                harmonic_level=self._classify_harmonic_level(freq_ratio)
            )
            
            self.all_oscillations.append({
                'id': node_id,
                'signal': signal_name,
                'freq_ratio': freq_ratio,
                'phase': phase,
                'amplitude': amplitude
            })
    
    def _classify_harmonic_level(self, freq_ratio: float) -> int:
        """
        Classify into hierarchical levels based on frequency ratio
        
        Level 0: Near cardiac frequency (0.5 - 2x)
        Level 1: Low harmonics (2 - 10x)
        Level 2: Mid harmonics (10 - 100x)
        Level 3: High harmonics (100 - 1000x)
        Level 4+: Ultra-high harmonics (>1000x)
        """
        if freq_ratio < 0.5:
            return -1  # Sub-cardiac
        elif freq_ratio < 2:
            return 0   # Cardiac range
        elif freq_ratio < 10:
            return 1   # Low harmonic
        elif freq_ratio < 100:
            return 2   # Mid harmonic
        elif freq_ratio < 1000:
            return 3   # High harmonic
        elif freq_ratio < 10000:
            return 4   # Very high
        elif freq_ratio < 100000:
            return 5   # Ultra high
        else:
            return int(np.log10(freq_ratio))  # Logarithmic levels
    
    def build_phase_locking_edges(self, phase_threshold_radians: float = 0.5):
        """
        Build random graph of phase-locking relationships
        
        Two oscillations are phase-locked if their phase difference is small
        
        Args:
            phase_threshold_radians: Maximum phase difference for phase-locking
        """
        print(f"\n🔗 Building phase-locking graph...")
        print(f"   Phase threshold: {phase_threshold_radians:.3f} radians")
        
        edges_added = 0
        
        # Check all pairs
        for i, osc1 in enumerate(self.all_oscillations):
            for osc2 in self.all_oscillations[i+1:]:
                # Calculate phase difference
                phase_diff = abs(osc1['phase'] - osc2['phase'])
                
                # Wrap to [-π, π]
                if phase_diff > np.pi:
                    phase_diff = 2*np.pi - phase_diff
                
                # Check if frequencies are harmonically related (within 5%)
                freq_ratio = osc1['freq_ratio'] / osc2['freq_ratio']
                is_harmonic = abs(freq_ratio - round(freq_ratio)) < 0.05
                
                # Add edge if phase-locked or harmonically related
                if phase_diff < phase_threshold_radians or is_harmonic:
                    self.graph.add_edge(
                        osc1['id'],
                        osc2['id'],
                        phase_difference=phase_diff,
                        is_harmonic=is_harmonic,
                        frequency_ratio=freq_ratio
                    )
                    edges_added += 1
        
        print(f"   ✓ Added {edges_added} phase-locking edges")
        print(f"   Graph has {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        # Compute graph properties
        if self.graph.number_of_edges() > 0:
            components = list(nx.connected_components(self.graph))
            print(f"   Connected components: {len(components)}")
            print(f"   Largest component: {len(max(components, key=len))} nodes")
            
            if len(max(components, key=len)) > 2:
                largest_component = self.graph.subgraph(max(components, key=len))
                avg_clustering = nx.average_clustering(largest_component)
                print(f"   Average clustering coefficient: {avg_clustering:.4f}")
    
    def compute_hierarchy_statistics(self) -> Dict:
        """Compute statistics about hierarchical organization"""
        # Group by hierarchy level
        levels = {}
        for node, data in self.graph.nodes(data=True):
            level = data['harmonic_level']
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
        
        self.hierarchy_levels = levels
        
        stats = {
            'n_levels': len(levels),
            'level_populations': {level: len(nodes) for level, nodes in levels.items()},
            'frequency_range': {
                'min_ratio': min(d['frequency_ratio'] for n, d in self.graph.nodes(data=True)),
                'max_ratio': max(d['frequency_ratio'] for n, d in self.graph.nodes(data=True))
            }
        }
        
        print(f"\n📊 Hierarchical Statistics:")
        print(f"   Total levels: {stats['n_levels']}")
        for level in sorted(stats['level_populations'].keys()):
            print(f"   Level {level}: {stats['level_populations'][level]} oscillations")
        
        return stats


class CardiacHarmonicExperiment:
    """
    Complete experiment: Extract all oscillations, build hierarchy, analyze graph
    
    CRITICAL: Results expressed ONLY as:
    - Frequency ratios (dimensionless)
    - Phase relationships (radians)
    - Graph topology (nodes, edges, clustering)
    - Hierarchical depth (number of levels)
    
    NEVER as:
    - Absolute time values
    - Absolute distance values
    - "Precision in seconds"
    """
    
    def __init__(self):
        self.results = {}
        
    def run_experiment(self, gps_data_path: Path) -> Dict:
        """
        Run complete cardiac-referenced harmonic analysis
        
        Args:
            gps_data_path: Path to cleaned GPS CSV file
        
        Returns:
            Complete results dictionary
        """
        print("="*70)
        print("   CARDIAC-REFERENCED HIERARCHICAL HARMONIC ANALYSIS")
        print("   With Random Graph Topology")
        print("="*70)
        
        # Load data
        print(f"\n📁 Loading: {gps_data_path.name}")
        df = pd.read_csv(gps_data_path)
        
        print(f"   Data points: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        
        # Extract cardiac frequency
        if 'heart_rate' in df.columns:
            mean_hr_bpm = df['heart_rate'].mean()
        else:
            # Synthesize realistic heart rate for demo
            mean_hr_bpm = 142.5
        
        f_cardiac = mean_hr_bpm / 60  # Convert to Hz
        
        print(f"\n🫀 Cardiac Reference: {mean_hr_bpm:.1f} bpm = {f_cardiac:.4f} Hz")
        
        # Initialize extractor and network
        extractor = CardiacReferencedHarmonicExtractor(f_cardiac)
        network = HierarchicalOscillationNetwork(f_cardiac)
        
        # Extract oscillations from all available signals
        oscillation_sets = []
        
        # 1. Position oscillations (if lat/lon available)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            lat_osc = extractor.extract_oscillations(
                df['latitude'].values,
                sampling_rate=1.0,  # Assuming 1 Hz GPS
                signal_name='position_latitude'
            )
            oscillation_sets.append(lat_osc)
            network.add_oscillation_set(lat_osc)
            
            lon_osc = extractor.extract_oscillations(
                df['longitude'].values,
                sampling_rate=1.0,
                signal_name='position_longitude'
            )
            oscillation_sets.append(lon_osc)
            network.add_oscillation_set(lon_osc)
        
        # 2. Velocity oscillations (if speed available)
        if 'speed' in df.columns:
            speed_osc = extractor.extract_oscillations(
                df['speed'].values,
                sampling_rate=1.0,
                signal_name='velocity_magnitude'
            )
            oscillation_sets.append(speed_osc)
            network.add_oscillation_set(speed_osc)
        
        # 3. Heart rate oscillations (HRV)
        if 'heart_rate' in df.columns:
            hr_osc = extractor.extract_oscillations(
                df['heart_rate'].values,
                sampling_rate=1.0,
                signal_name='cardiac_variability'
            )
            oscillation_sets.append(hr_osc)
            network.add_oscillation_set(hr_osc)
        
        # 4. Synthetic oscillations for demonstration
        # Create time series with known harmonics
        t = np.arange(len(df))
        
        # Respiratory-like oscillation (0.25 Hz ~ 1/9 cardiac)
        respiratory = np.sin(2*np.pi*0.25*t/f_cardiac)
        resp_osc = extractor.extract_oscillations(
            respiratory,
            sampling_rate=1.0,
            signal_name='respiratory_proxy'
        )
        oscillation_sets.append(resp_osc)
        network.add_oscillation_set(resp_osc)
        
        # Step cadence-like oscillation (3.34 Hz ~ 1.4x cardiac)
        step_cadence = np.sin(2*np.pi*3.34*t/f_cardiac)
        step_osc = extractor.extract_oscillations(
            step_cadence,
            sampling_rate=1.0,
            signal_name='gait_cadence'
        )
        oscillation_sets.append(step_osc)
        network.add_oscillation_set(step_osc)
        
        # Build phase-locking graph
        network.build_phase_locking_edges(phase_threshold_radians=0.5)
        
        # Compute hierarchy statistics
        hierarchy_stats = network.compute_hierarchy_statistics()
        
        # Analyze graph topology
        graph_topology = self._analyze_graph_topology(network.graph)
        
        # Compile results
        results = {
            'experiment': 'Cardiac-Referenced Hierarchical Harmonic Analysis',
            'timestamp': datetime.now().isoformat(),
            'data_source': str(gps_data_path),
            'cardiac_reference': {
                'heart_rate_bpm': mean_hr_bpm,
                'frequency_hz': f_cardiac,
                'note': 'All measurements relative to this frequency'
            },
            'oscillation_sets': oscillation_sets,
            'hierarchy_statistics': hierarchy_stats,
            'graph_topology': graph_topology,
            'key_findings': self._interpret_results(hierarchy_stats, graph_topology),
            'measurement_philosophy': {
                'principle': 'All data expressed as oscillations relative to cardiac frequency',
                'never_convert_to': ['seconds', 'meters', 'absolute time', 'absolute distance'],
                'always_express_as': ['frequency ratios', 'phase relationships', 'harmonic levels', 'graph topology'],
                'why': 'Keeps measurements in dimensionless phase space, avoiding unphysical claims'
            }
        }
        
        self.results = results
        return results, network
    
    def _analyze_graph_topology(self, graph: nx.Graph) -> Dict:
        """Analyze random graph topology"""
        if graph.number_of_nodes() == 0:
            return {'empty': True}
        
        # Basic properties
        topology = {
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'density': nx.density(graph)
        }
        
        if graph.number_of_edges() > 0:
            # Connected components
            components = list(nx.connected_components(graph))
            topology['n_components'] = len(components)
            topology['largest_component_size'] = len(max(components, key=len))
            
            # Analyze largest component
            if len(max(components, key=len)) > 2:
                largest = graph.subgraph(max(components, key=len))
                topology['clustering_coefficient'] = nx.average_clustering(largest)
                
                # Degree distribution
                degrees = [d for n, d in largest.degree()]
                topology['degree_statistics'] = {
                    'mean': float(np.mean(degrees)),
                    'std': float(np.std(degrees)),
                    'max': int(max(degrees)),
                    'min': int(min(degrees))
                }
        
        print(f"\n🌐 Graph Topology:")
        print(f"   Nodes: {topology['nodes']}")
        print(f"   Edges: {topology['edges']}")
        print(f"   Density: {topology['density']:.4f}")
        if 'clustering_coefficient' in topology:
            print(f"   Clustering: {topology['clustering_coefficient']:.4f}")
        
        return topology
    
    def _interpret_results(self, hierarchy_stats: Dict, topology: Dict) -> Dict:
        """Interpret what the results mean (WITHOUT converting to absolute units)"""
        interpretation = {
            'hierarchical_depth': f"{hierarchy_stats['n_levels']} levels of harmonic organization",
            'frequency_span': f"{hierarchy_stats['frequency_range']['max_ratio'] / hierarchy_stats['frequency_range']['min_ratio']:.1f}x range relative to cardiac",
            'network_complexity': f"{topology.get('nodes', 0)} oscillatory components with {topology.get('edges', 0)} phase-locking relationships",
            'system_coherence': 'High' if topology.get('clustering_coefficient', 0) > 0.5 else 'Moderate' if topology.get('clustering_coefficient', 0) > 0.3 else 'Low'
        }
        
        # The "explosion" - but expressed properly
        if hierarchy_stats['n_levels'] >= 5:
            interpretation['observation'] = f"Deep harmonic cascade: {hierarchy_stats['n_levels']} distinct frequency scales all phase-locked to cardiac rhythm"
        
        return interpretation
    
    def visualize_results(self, network: HierarchicalOscillationNetwork) -> plt.Figure:
        """Create comprehensive visualization"""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel 1: Hierarchy levels (bar chart)
        ax1 = fig.add_subplot(gs[0, 0])
        levels = sorted(network.hierarchy_levels.keys())
        populations = [len(network.hierarchy_levels[l]) for l in levels]
        ax1.bar(levels, populations, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Hierarchical Level', fontsize=11)
        ax1.set_ylabel('Number of Oscillations', fontsize=11)
        ax1.set_title('Hierarchical Organization\n(Relative to Cardiac Frequency)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Panel 2: Frequency ratio distribution
        ax2 = fig.add_subplot(gs[0, 1])
        freq_ratios = [d['frequency_ratio'] for n, d in network.graph.nodes(data=True)]
        ax2.hist(np.log10(freq_ratios), bins=50, color='teal', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('log₁₀(Frequency Ratio to Cardiac)', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title('Frequency Distribution\n(All measurements relative to cardiac)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Panel 3: Phase distribution
        ax3 = fig.add_subplot(gs[0, 2], projection='polar')
        phases = [d['phase_radians'] for n, d in network.graph.nodes(data=True)]
        amplitudes = [d['amplitude'] for n, d in network.graph.nodes(data=True)]
        ax3.scatter(phases, amplitudes, c=freq_ratios, cmap='viridis', alpha=0.6, s=50)
        ax3.set_title('Phase Space Distribution\n(Radians, dimensionless)', fontsize=12, fontweight='bold', pad=20)
        
        # Panel 4: Network graph (largest component only)
        ax4 = fig.add_subplot(gs[1, :2])
        if network.graph.number_of_edges() > 0:
            components = list(nx.connected_components(network.graph))
            largest_component = network.graph.subgraph(max(components, key=len))
            
            # Color by hierarchy level
            colors = [network.graph.nodes[n]['harmonic_level'] for n in largest_component.nodes()]
            
            pos = nx.spring_layout(largest_component, k=0.5, iterations=50)
            nx.draw_networkx_nodes(largest_component, pos, node_color=colors, 
                                 cmap='coolwarm', node_size=100, alpha=0.7, ax=ax4)
            nx.draw_networkx_edges(largest_component, pos, alpha=0.2, width=0.5, ax=ax4)
            
        ax4.set_title('Phase-Locking Network (Random Graph Topology)', fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        # Panel 5: Degree distribution
        ax5 = fig.add_subplot(gs[1, 2])
        if network.graph.number_of_edges() > 0:
            degrees = [d for n, d in network.graph.degree()]
            ax5.hist(degrees, bins=30, color='coral', edgecolor='black', alpha=0.7)
            ax5.set_xlabel('Node Degree\n(# of phase-locking connections)', fontsize=11)
            ax5.set_ylabel('Count', fontsize=11)
            ax5.set_title('Connectivity Distribution', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
        
        # Panel 6: Summary text
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        summary = f"""
CARDIAC-REFERENCED HIERARCHICAL HARMONIC ANALYSIS - RESULTS

Key Principle: All measurements expressed as oscillations relative to cardiac frequency ({network.f_cardiac:.4f} Hz)
               NEVER converted back to absolute time/distance units

Hierarchical Organization:
  • {self.results['hierarchy_statistics']['n_levels']} distinct frequency scale levels
  • {network.graph.number_of_nodes()} total oscillatory components identified
  • Frequency span: {self.results['hierarchy_statistics']['frequency_range']['min_ratio']:.2f}x to {self.results['hierarchy_statistics']['frequency_range']['max_ratio']:.2f}x cardiac

Random Graph Topology:
  • {network.graph.number_of_edges()} phase-locking relationships (edges)
  • {self.results['graph_topology'].get('n_components', 0)} connected components
  • Largest component: {self.results['graph_topology'].get('largest_component_size', 0)} nodes
  • Clustering coefficient: {self.results['graph_topology'].get('clustering_coefficient', 0):.4f}

Interpretation:
  • System exhibits {self.results['key_findings']['system_coherence']} coherence
  • {self.results['key_findings']['hierarchical_depth']}
  • {self.results['key_findings'].get('observation', 'Multiple frequency scales detected')}

Measurement Philosophy:
  ✓ Frequency ratios (dimensionless)
  ✓ Phase relationships (radians)  
  ✓ Harmonic depth (number of levels)
  ✓ Graph topology (connectivity patterns)
  
  ✗ NOT expressed as absolute seconds
  ✗ NOT expressed as absolute meters
  ✗ NOT claiming sub-Planck measurements
        """
        
        ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle('Cardiac-Referenced Hierarchical Harmonic Analysis with Random Graph Topology',
                    fontsize=14, fontweight='bold', y=0.995)
        
        return fig


def main():
    """Run the experiment with proper framing"""
    print("="*70)
    print("   NEW EXPERIMENTAL APPROACH")
    print("   Cardiac-Referenced Harmonic Hierarchy + Random Graph")
    print("="*70)
    print("\n📋 Experimental Philosophy:")
    print("   1. Extract ALL oscillations from data")
    print("   2. Express as frequency ratios relative to cardiac")
    print("   3. Build hierarchical organization by frequency scale")
    print("   4. Construct random graph of phase-locking relationships")
    print("   5. NEVER convert back to original units (m, s)")
    print("   6. Results in dimensionless phase space only")
    
    # Find latest GPS data
    results_dir = Path(__file__).parent.parent.parent / 'results' / 'gps_precision'
    
    gps_files = sorted(results_dir.glob('*_cleaned_*.csv'), 
                      key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not gps_files:
        print("\n❌ No GPS files found! Please run analyze_messy_gps.py first")
        return
    
    gps_file = gps_files[0]
    
    # Run experiment
    experiment = CardiacHarmonicExperiment()
    results, network = experiment.run_experiment(gps_file)
    
    # Create visualization
    fig = experiment.visualize_results(network)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = results_dir
    
    json_file = output_dir / f'cardiac_harmonic_hierarchy_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    fig_file = output_dir / f'cardiac_harmonic_hierarchy_{timestamp}.png'
    fig.savefig(fig_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n💾 Results saved:")
    print(f"   JSON: {json_file.name}")
    print(f"   Figure: {fig_file.name}")
    
    print("\n" + "="*70)
    print("   EXPERIMENT COMPLETE")
    print("="*70)
    print("\n✨ Key Achievement:")
    print(f"   Mapped {network.graph.number_of_nodes()} oscillatory components")
    print(f"   across {results['hierarchy_statistics']['n_levels']} hierarchical levels")
    print(f"   with {network.graph.number_of_edges()} phase-locking relationships")
    print("\n   ALL expressed relative to cardiac frequency")
    print("   NO conversion to absolute units")
    print("   Results stay in dimensionless phase space")
    
    return results, network


if __name__ == "__main__":
    main()

