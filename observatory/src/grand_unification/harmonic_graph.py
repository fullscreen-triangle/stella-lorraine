"""
Harmonic Network Graph - 240-Component System
==============================================

Builds and navigates the 240-component harmonic network graph that connects
all aircraft/system components through harmonic coincidences. Enables O(1)
navigation through S-entropy space vs. O(log N) for traditional graph traversal.

Purpose:
--------
The harmonic graph enables:
1. Finding harmonic coincidences between components (within 0.1 Hz)
2. Building complete network of all system oscillations
3. O(1) direct navigation via S-entropy coordinates
4. Identifying hubs (highly connected nodes)
5. Hierarchical frequency multiplication analysis

Example 240-Component Aircraft System:
--------------------------------------
- Human physiological (12): breathing, heart rate, tremor, etc.
- Control surfaces (27): stick, pedals, buttons, actuators
- Thermal (23): heat exchangers, engine temps, cooling
- Structural (67): wing modes, fuselage, spars, ribs
- Aerodynamic (38): boundary layers, vortices, pressure points
- Propulsion (42): engines, pistons, propellers, fuel pumps
- Electromagnetic (31): KLA solenoids, EBL, sensors, cascade
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class HarmonicNode:
    """
    Node in the harmonic network graph
    
    Attributes:
        component_id: Unique identifier for component
        S_coords: S-entropy coordinates
        frequencies: Characteristic frequencies (Hz)
        amplitudes: Amplitudes for each frequency
        domain: Measurement domain
        timestamp: When measured
    """
    component_id: str
    S_coords: np.ndarray
    frequencies: np.ndarray
    amplitudes: np.ndarray
    domain: str
    timestamp: float


@dataclass
class HarmonicEdge:
    """
    Edge connecting two nodes via harmonic coincidence
    
    Attributes:
        node_A: First node ID
        node_B: Second node ID
        frequency: Coinciding frequency (Hz)
        strength: Coupling strength (based on amplitudes)
        harmonic_order_A: Harmonic order for node A
        harmonic_order_B: Harmonic order for node B
    """
    node_A: str
    node_B: str
    frequency: float
    strength: float
    harmonic_order_A: int
    harmonic_order_B: int


class HarmonicNetworkGraph:
    """
    Complete harmonic network graph for multi-component systems
    
    Implements O(1) navigation through S-entropy coordinates while
    maintaining traditional graph structure for analysis.
    """
    
    def __init__(self, max_harmonic_order: int = 10):
        """
        Initialize harmonic network graph
        
        Args:
            max_harmonic_order: Maximum harmonic order to consider
        """
        self.max_harmonic_order = max_harmonic_order
        
        # Graph structure (NetworkX for traditional operations)
        self.graph = nx.Graph()
        
        # Node storage
        self.nodes: Dict[str, HarmonicNode] = {}
        
        # Edge storage (for quick access)
        self.edges: List[HarmonicEdge] = []
        
        # S-coordinate mapping (for O(1) navigation)
        self.S_coords_map: Dict[str, np.ndarray] = {}
        
        # Statistics
        self.n_edges_created = 0
        self.n_harmonic_coincidences = 0
        
    def add_node(self,
                component_id: str,
                S_coords: np.ndarray,
                frequencies: np.ndarray,
                amplitudes: np.ndarray,
                domain: str,
                timestamp: float) -> HarmonicNode:
        """
        Add node to harmonic graph
        
        Args:
            component_id: Unique component identifier
            S_coords: S-entropy coordinates
            frequencies: Characteristic frequencies
            amplitudes: Amplitudes
            domain: Measurement domain
            timestamp: Trans-Planckian timestamp
            
        Returns:
            HarmonicNode created
        """
        node = HarmonicNode(
            component_id=component_id,
            S_coords=np.array(S_coords),
            frequencies=np.array(frequencies),
            amplitudes=np.array(amplitudes),
            domain=domain,
            timestamp=timestamp
        )
        
        # Store node
        self.nodes[component_id] = node
        self.S_coords_map[component_id] = node.S_coords
        
        # Add to NetworkX graph
        self.graph.add_node(component_id, **{
            'S_coords': node.S_coords,
            'domain': domain,
            'n_frequencies': len(frequencies)
        })
        
        return node
        
    def find_harmonic_coincidences(self,
                                  node_A_id: str,
                                  node_B_id: str,
                                  tolerance_hz: float = 0.1) -> List[HarmonicEdge]:
        """
        Find harmonic coincidences between two nodes
        
        Checks all harmonic multiples up to max_harmonic_order.
        
        Args:
            node_A_id: First node
            node_B_id: Second node
            tolerance_hz: Frequency tolerance for coincidence
            
        Returns:
            List of harmonic edges found
        """
        if node_A_id not in self.nodes or node_B_id not in self.nodes:
            return []
            
        node_A = self.nodes[node_A_id]
        node_B = self.nodes[node_B_id]
        
        coincidences = []
        
        # Check all frequency pairs
        for i, f_A in enumerate(node_A.frequencies):
            amp_A = node_A.amplitudes[i]
            
            for j, f_B in enumerate(node_B.frequencies):
                amp_B = node_B.amplitudes[j]
                
                # Check all harmonic combinations
                for n_A in range(1, self.max_harmonic_order + 1):
                    for n_B in range(1, self.max_harmonic_order + 1):
                        harm_A = n_A * f_A
                        harm_B = n_B * f_B
                        
                        # Check coincidence
                        if abs(harm_A - harm_B) < tolerance_hz:
                            # Calculate coupling strength
                            strength = min(amp_A / n_A, amp_B / n_B)
                            
                            edge = HarmonicEdge(
                                node_A=node_A_id,
                                node_B=node_B_id,
                                frequency=harm_A,
                                strength=strength,
                                harmonic_order_A=n_A,
                                harmonic_order_B=n_B
                            )
                            
                            coincidences.append(edge)
                            self.n_harmonic_coincidences += 1
                            
        return coincidences
        
    def build_complete_graph(self, tolerance_hz: float = 0.1):
        """
        Build complete harmonic network graph
        
        Finds all harmonic coincidences between all node pairs.
        
        Args:
            tolerance_hz: Frequency tolerance
        """
        print(f"Building harmonic graph for {len(self.nodes)} nodes...")
        
        node_ids = list(self.nodes.keys())
        
        # Check all pairs
        for i, node_A_id in enumerate(node_ids):
            for node_B_id in node_ids[i+1:]:
                # Find coincidences
                coincidences = self.find_harmonic_coincidences(
                    node_A_id, node_B_id, tolerance_hz
                )
                
                # Add edges to graph
                for edge in coincidences:
                    self.add_edge(edge)
                    
        print(f"✓ Graph built: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
              
    def add_edge(self, edge: HarmonicEdge):
        """Add edge to graph"""
        self.edges.append(edge)
        
        # Add to NetworkX graph (accumulate strength if edge exists)
        if self.graph.has_edge(edge.node_A, edge.node_B):
            # Edge exists - add to strength
            self.graph[edge.node_A][edge.node_B]['strength'] += edge.strength
            self.graph[edge.node_A][edge.node_B]['n_coincidences'] += 1
        else:
            # New edge
            self.graph.add_edge(
                edge.node_A,
                edge.node_B,
                frequency=edge.frequency,
                strength=edge.strength,
                n_coincidences=1
            )
            self.n_edges_created += 1
            
    def shortest_path(self,
                     source: str,
                     target: str) -> List[str]:
        """
        Find shortest path using traditional graph traversal
        
        Complexity: O(log N) for typical graphs
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            Path as list of node IDs
        """
        try:
            path = nx.shortest_path(self.graph, source, target)
            return path
        except nx.NetworkXNoPath:
            return []
            
    def direct_navigation_S_entropy(self,
                                   source: str,
                                   target: str) -> Dict[str, Any]:
        """
        Navigate directly via S-entropy coordinates
        
        Complexity: O(1) - constant time!
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            Navigation result
        """
        if source not in self.S_coords_map or target not in self.S_coords_map:
            return {'error': 'Node not found'}
            
        S_source = self.S_coords_map[source]
        S_target = self.S_coords_map[target]
        
        # Direct vector in S-space
        delta_S = S_target - S_source
        S_distance = np.linalg.norm(delta_S)
        
        # Direct jump (no intermediate steps)
        return {
            'source': source,
            'target': target,
            'S_distance': S_distance,
            'delta_S': delta_S,
            'complexity': 'O(1)',
            'method': 'direct_S_entropy_jump'
        }
        
    def find_hubs(self, min_degree: int = 30) -> List[Tuple[str, int, float]]:
        """
        Find hub nodes (highly connected)
        
        Args:
            min_degree: Minimum degree to be considered a hub
            
        Returns:
            List of (node_id, degree, avg_frequency) tuples
        """
        hubs = []
        
        for node_id in self.nodes:
            degree = self.graph.degree(node_id)
            
            if degree >= min_degree:
                # Calculate average frequency
                node = self.nodes[node_id]
                avg_freq = np.mean(node.frequencies)
                
                hubs.append((node_id, degree, avg_freq))
                
        # Sort by degree
        hubs.sort(key=lambda x: x[1], reverse=True)
        
        return hubs
        
    def find_multiplication_chain(self,
                                 start: str,
                                 end: str) -> List[Dict[str, Any]]:
        """
        Find hierarchical frequency multiplication chain
        
        Args:
            start: Starting component (low frequency)
            end: Ending component (high frequency)
            
        Returns:
            Chain of frequency multiplications
        """
        # Get path through graph
        path = self.shortest_path(start, end)
        
        if not path:
            return []
            
        # Build multiplication chain
        chain = []
        
        for node_id in path:
            node = self.nodes[node_id]
            dominant_freq = node.frequencies[np.argmax(node.amplitudes)]
            
            chain.append({
                'name': node_id,
                'frequency': dominant_freq,
                'domain': node.domain
            })
            
        return chain
        
    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute complete graph statistics
        
        Returns:
            Dictionary of statistics
        """
        if len(self.nodes) == 0:
            return {'n_nodes': 0}
            
        # Basic stats
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        
        # Degree statistics
        degrees = [self.graph.degree(n) for n in self.graph.nodes()]
        avg_degree = np.mean(degrees)
        max_degree = np.max(degrees) if degrees else 0
        max_degree_node = max(self.graph.nodes(), key=self.graph.degree) if degrees else None
        
        # Network topology
        try:
            clustering = nx.average_clustering(self.graph)
        except:
            clustering = 0.0
            
        # Path length (only if connected)
        if nx.is_connected(self.graph):
            avg_path_length = nx.average_shortest_path_length(self.graph)
            diameter = nx.diameter(self.graph)
            n_components = 1
        else:
            avg_path_length = float('inf')
            diameter = float('inf')
            n_components = nx.number_connected_components(self.graph)
            
        # Frequency range
        all_freqs = np.concatenate([node.frequencies for node in self.nodes.values()])
        freq_range = (np.min(all_freqs), np.max(all_freqs))
        
        # Domains
        domains = list(set(node.domain for node in self.nodes.values()))
        
        return {
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'avg_degree': avg_degree,
            'max_degree': max_degree,
            'max_degree_node': max_degree_node,
            'clustering': clustering,
            'avg_path_length': avg_path_length,
            'diameter': diameter,
            'n_components': n_components,
            'frequency_range': freq_range,
            'domains': domains,
            'n_harmonic_coincidences': self.n_harmonic_coincidences
        }
        
    def export_to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary for persistence"""
        return {
            'nodes': {
                node_id: {
                    'S_coords': node.S_coords.tolist(),
                    'frequencies': node.frequencies.tolist(),
                    'amplitudes': node.amplitudes.tolist(),
                    'domain': node.domain,
                    'timestamp': node.timestamp
                }
                for node_id, node in self.nodes.items()
            },
            'edges': [
                {
                    'node_A': edge.node_A,
                    'node_B': edge.node_B,
                    'frequency': edge.frequency,
                    'strength': edge.strength,
                    'harmonic_order_A': edge.harmonic_order_A,
                    'harmonic_order_B': edge.harmonic_order_B
                }
                for edge in self.edges
            ],
            'statistics': self.compute_statistics()
        }
        
    def __repr__(self) -> str:
        stats = self.compute_statistics()
        return (
            f"HarmonicNetworkGraph(nodes={stats['n_nodes']}, "
            f"edges={stats['n_edges']}, "
            f"avg_degree={stats['avg_degree']:.1f})"
        )