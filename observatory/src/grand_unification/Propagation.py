"""
Propagation - Wave Movement and Transmission
=============================================

Simulates actual wave propagation through the Grand

Wave, enabling navigation
to nearest disturbances and solution discovery through interference patterns.

Purpose:
--------
While GrandWave is the infinite substrate (the "source"), Propagation handles:
1. Finding nearest disturbances to a query point
2. Simulating wave transmission between disturbances
3. Calculating paths through interference patterns
4. Predicting where solutions might emerge

This enables the S-entropy "miraculous jumps" where solutions appear without
intermediate steps, by propagating through the wave structure.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from scipy.spatial import KDTree
from .GrandWave import GrandWave, WaveDisturbance


@dataclass
class PropagationPath:
    """
    Path of wave propagation between two points in S-space
    
    Attributes:
        start_S: Starting S-coordinates
        end_S: Ending S-coordinates
        intermediate_disturbances: Disturbances encountered along path
        total_distance: Total S-distance traveled
        coherence_score: How well path maintains coherence
        propagation_time: Time for wave to propagate (trans-Planckian)
    """
    start_S: np.ndarray
    end_S: np.ndarray
    intermediate_disturbances: List[WaveDisturbance]
    total_distance: float
    coherence_score: float
    propagation_time: float


class WavePropagator:
    """
    Simulates wave propagation through the GrandWave substrate
    
    Finds optimal paths, nearest neighbors, and interference-based solutions.
    """
    
    def __init__(self, grand_wave: GrandWave):
        """
        Initialize propagator
        
        Args:
            grand_wave: GrandWave instance containing all disturbances
        """
        self.grand_wave = grand_wave
        self._kdtree = None
        self._kdtree_sources = None
        
    def _build_kdtree(self):
        """Build KDTree for fast nearest neighbor search in S-space"""
        if not self.grand_wave.active_disturbances:
            return
            
        # Extract all S-coordinates
        sources = list(self.grand_wave.active_disturbances.keys())
        S_coords = np.array([
            self.grand_wave.active_disturbances[s].S_coords 
            for s in sources
        ])
        
        # Build KDTree
        self._kdtree = KDTree(S_coords)
        self._kdtree_sources = sources
        
    def find_nearest_disturbance(self, 
                                 S_query: np.ndarray,
                                 k: int = 1) -> List[Tuple[WaveDisturbance, float]]:
        """
        Find k nearest disturbances to query point in S-space
        
        Args:
            S_query: Query S-coordinates
            k: Number of nearest neighbors to return
            
        Returns:
            List of (disturbance, distance) tuples, sorted by distance
        """
        # Rebuild KDTree if needed
        self._build_kdtree()
        
        if self._kdtree is None:
            return []
        
        # Query KDTree
        distances, indices = self._kdtree.query(S_query, k=k)
        
        # Handle single result
        if k == 1:
            distances = [distances]
            indices = [indices]
        
        # Build result list
        results = []
        for dist, idx in zip(distances, indices):
            source = self._kdtree_sources[idx]
            disturbance = self.grand_wave.active_disturbances[source]
            results.append((disturbance, dist))
            
        return results
        
    def propagate_wave(self,
                      start_S: np.ndarray,
                      end_S: np.ndarray,
                      max_hops: int = 10) -> PropagationPath:
        """
        Simulate wave propagation from start to end via intermediate disturbances
        
        This finds the "natural" path a wave would take through existing
        disturbances, following interference patterns.
        
        Args:
            start_S: Starting S-coordinates
            end_S: Target S-coordinates
            max_hops: Maximum intermediate disturbances to traverse
            
        Returns:
            PropagationPath describing the wave's journey
        """
        path_disturbances = []
        current_S = start_S.copy()
        total_distance = 0.0
        coherence_scores = []
        
        for hop in range(max_hops):
            # Find direction to target
            direction = end_S - current_S
            distance_to_target = np.linalg.norm(direction)
            
            if distance_to_target < 0.05:  # Close enough
                break
                
            # Find disturbances ahead in this direction
            # Look for disturbances within cone pointing toward target
            candidates = []
            for disturbance in self.grand_wave.active_disturbances.values():
                # Vector to disturbance
                to_disturbance = disturbance.S_coords - current_S
                
                # Dot product with target direction (normalized)
                alignment = np.dot(to_disturbance, direction) / (
                    np.linalg.norm(to_disturbance) * np.linalg.norm(direction)
                )
                
                # Only consider disturbances ahead (alignment > 0.5)
                if alignment > 0.5:
                    dist = np.linalg.norm(to_disturbance)
                    candidates.append((disturbance, dist, alignment))
                    
            if not candidates:
                # No intermediate disturbances - direct jump
                break
                
            # Select best candidate (balances distance and alignment)
            best = min(candidates, key=lambda x: x[1] * (2 - x[2]))
            next_disturbance, step_dist, alignment = best
            
            # Add to path
            path_disturbances.append(next_disturbance)
            total_distance += step_dist
            coherence_scores.append(alignment)
            
            # Move to next position
            current_S = next_disturbance.S_coords
            
        # Final jump to target
        final_distance = np.linalg.norm(end_S - current_S)
        total_distance += final_distance
        
        # Calculate overall coherence
        if coherence_scores:
            coherence_score = np.mean(coherence_scores)
        else:
            coherence_score = 1.0  # Direct path
            
        # Propagation time (trans-Planckian, proportional to S-distance)
        # Wave propagates at "speed of coherence" in S-space
        propagation_time = total_distance * self.grand_wave.precision
        
        return PropagationPath(
            start_S=start_S,
            end_S=end_S,
            intermediate_disturbances=path_disturbances,
            total_distance=total_distance,
            coherence_score=coherence_score,
            propagation_time=propagation_time
        )
        
    def find_solution_region(self,
                            target_property: str,
                            target_value: float,
                            domain: str,
                            search_radius: float = 0.5) -> List[Tuple[np.ndarray, float]]:
        """
        Find regions in S-space where solutions with target property likely exist
        
        This uses interference patterns to predict where solutions emerge.
        
        Args:
            target_property: Property name to optimize
            target_value: Desired value for property
            domain: Domain to search in
            search_radius: Search radius in S-space
            
        Returns:
            List of (S_coords, probability) tuples for likely solution regions
        """
        # Filter disturbances by domain
        domain_disturbances = [
            d for d in self.grand_wave.active_disturbances.values()
            if d.domain == domain
        ]
        
        if not domain_disturbances:
            return []
        
        # Calculate centroid of domain disturbances
        domain_S_coords = np.array([d.S_coords for d in domain_disturbances])
        centroid = np.mean(domain_S_coords, axis=0)
        
        # Generate candidate points in sphere around centroid
        n_candidates = 100
        candidates = []
        
        for i in range(n_candidates):
            # Random point in sphere
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            radius = search_radius * np.random.random()**(1/3)  # Uniform in volume
            
            S_candidate = centroid + radius * direction
            
            # Calculate "solution probability" based on interference
            probability = self._calculate_solution_probability(
                S_candidate, domain_disturbances
            )
            
            candidates.append((S_candidate, probability))
            
        # Sort by probability
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:20]  # Return top 20
        
    def _calculate_solution_probability(self,
                                        S_candidate: np.ndarray,
                                        nearby_disturbances: List[WaveDisturbance]) -> float:
        """
        Calculate probability that a solution exists at S_candidate
        
        Based on constructive interference from nearby disturbances.
        """
        if not nearby_disturbances:
            return 0.0
        
        # Calculate "interference strength" at candidate point
        interference = 0.0
        
        for disturbance in nearby_disturbances:
            # Distance from disturbance
            distance = np.linalg.norm(S_candidate - disturbance.S_coords)
            
            # Contribution (decreases with distance)
            contribution = np.exp(-distance / 0.3)
            
            # Weight by amplitude
            avg_amplitude = np.mean(disturbance.amplitudes)
            contribution *= avg_amplitude
            
            interference += contribution
            
        # Normalize to probability (0-1)
        probability = 1.0 - np.exp(-interference / 10.0)
        
        return probability
        
    def trace_interference_pattern(self,
                                   disturbance_A: WaveDisturbance,
                                   disturbance_B: WaveDisturbance,
                                   n_points: int = 50) -> np.ndarray:
        """
        Trace interference pattern between two disturbances
        
        Returns field strength along line connecting them.
        
        Args:
            disturbance_A: First disturbance
            disturbance_B: Second disturbance
            n_points: Number of points to sample
            
        Returns:
            Array of interference amplitudes along path
        """
        # Linear interpolation between disturbances
        alphas = np.linspace(0, 1, n_points)
        path_points = np.outer(1 - alphas, disturbance_A.S_coords) + \
                     np.outer(alphas, disturbance_B.S_coords)
        
        # Calculate interference amplitude at each point
        amplitudes = []
        
        for point in path_points:
            # Distance from each disturbance
            dist_A = np.linalg.norm(point - disturbance_A.S_coords)
            dist_B = np.linalg.norm(point - disturbance_B.S_coords)
            
            # Amplitude from each (inverse square law in S-space)
            amp_A = np.mean(disturbance_A.amplitudes) / (1 + dist_A**2)
            amp_B = np.mean(disturbance_B.amplitudes) / (1 + dist_B**2)
            
            # Interference (phase matters)
            phase_diff = np.mean(disturbance_A.phases) - np.mean(disturbance_B.phases)
            
            # Constructive or destructive
            interference = amp_A + amp_B + 2*np.sqrt(amp_A*amp_B)*np.cos(phase_diff)
            
            amplitudes.append(interference)
            
        return np.array(amplitudes)
        
    def find_strongest_interference(self,
                                   S_query: np.ndarray,
                                   radius: float = 0.5) -> List[Tuple[WaveDisturbance, WaveDisturbance, float]]:
        """
        Find pairs of disturbances with strongest interference near query point
        
        Args:
            S_query: Query point in S-space
            radius: Search radius
            
        Returns:
            List of (disturbance_A, disturbance_B, interference_strength) tuples
        """
        # Find nearby disturbances
        nearby = self.find_nearest_disturbance(S_query, k=10)
        
        if len(nearby) < 2:
            return []
        
        # Check all pairs
        interference_pairs = []
        
        for i, (dist_A, _) in enumerate(nearby):
            for j, (dist_B, _) in enumerate(nearby[i+1:], start=i+1):
                # Calculate interference strength at query point
                dist_A_to_query = np.linalg.norm(dist_A.S_coords - S_query)
                dist_B_to_query = np.linalg.norm(dist_B.S_coords - S_query)
                
                if dist_A_to_query < radius and dist_B_to_query < radius:
                    # Both within radius - calculate interference
                    amp_A = np.mean(dist_A.amplitudes) / (1 + dist_A_to_query**2)
                    amp_B = np.mean(dist_B.amplitudes) / (1 + dist_B_to_query**2)
                    
                    phase_diff = np.mean(dist_A.phases) - np.mean(dist_B.phases)
                    
                    # Constructive interference strength
                    strength = amp_A + amp_B + 2*np.sqrt(amp_A*amp_B)*np.cos(phase_diff)
                    
                    interference_pairs.append((dist_A, dist_B, strength))
                    
        # Sort by strength
        interference_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return interference_pairs[:5]  # Top 5