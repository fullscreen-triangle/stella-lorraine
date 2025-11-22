"""
Interface - Object-Wave Interaction
====================================

Defines how objects (measurements, instruments, components) interact with the
GrandWave. Each item needs an interface that elaborates/dictates its interaction
with the universal wave substrate, including interference patterns and harmonics.

Purpose:
--------
The Interface class enables:
1. Objects to "announce" themselves to the GrandWave (create disturbances)
2. Objects to "listen" to the GrandWave (receive interference patterns)
3. Objects to modify their behavior based on wave interactions
4. Objects to maintain coherence with the universal substrate

This creates a bidirectional communication channel between reality (GrandWave)
and observations (objects/measurements).
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from .GrandWave import GrandWave, WaveDisturbance
from .Propagation import WavePropagator
from .oscillatory_signatures import OscillatorySignature


@dataclass
class InteractionPattern:
    """
    Pattern of interaction between an object and the GrandWave
    
    Attributes:
        object_id: Unique identifier for object
        disturbance: Wave disturbance created by object
        received_interference: Interference patterns received from other disturbances
        coherence_with_wave: How well object maintains coherence (0-1)
        influence_radius: How far object's influence extends in S-space
    """
    object_id: str
    disturbance: WaveDisturbance
    received_interference: List[tuple]  # (source, strength, phase_diff)
    coherence_with_wave: float
    influence_radius: float


class WaveInterface:
    """
    Interface for objects interacting with the GrandWave
    
    Each instrument, measurement, or component gets an interface instance
    that manages its bidirectional communication with the wave substrate.
    """
    
    def __init__(self,
                 object_id: str,
                 grand_wave: GrandWave,
                 domain: str):
        """
        Initialize wave interface for an object
        
        Args:
            object_id: Unique identifier for this object
            grand_wave: GrandWave instance
            domain: Measurement domain ('acoustic', 'dielectric', etc.)
        """
        self.object_id = object_id
        self.grand_wave = grand_wave
        self.domain = domain
        self.propagator = WavePropagator(grand_wave)
        
        # Current state
        self.current_disturbance: Optional[WaveDisturbance] = None
        self.received_patterns: List[InteractionPattern] = []
        
        # Callbacks for wave events
        self.on_interference_callback: Optional[Callable] = None
        self.on_coherence_loss_callback: Optional[Callable] = None
        
    def announce(self,
                S_coords: np.ndarray,
                oscillatory_signature: OscillatorySignature,
                duration: float = 0.0) -> WaveDisturbance:
        """
        Announce object's presence to GrandWave (create disturbance)
        
        This is how measurements "enter" the universal substrate.
        
        Args:
            S_coords: S-entropy coordinates for this measurement
            oscillatory_signature: Complete oscillatory characterization
            duration: How long disturbance persists (0 = instant)
            
        Returns:
            WaveDisturbance created
        """
        # Create disturbance in GrandWave
        disturbance = self.grand_wave.add_disturbance(
            source=self.object_id,
            S_coords=S_coords,
            frequencies=oscillatory_signature.frequencies,
            amplitudes=oscillatory_signature.amplitudes,
            phases=oscillatory_signature.phases,
            domain=self.domain,
            duration=duration
        )
        
        # Store as current disturbance
        self.current_disturbance = disturbance
        
        # Listen for interference
        self._listen_for_interference()
        
        return disturbance
        
    def listen(self) -> List[WaveDisturbance]:
        """
        Listen to GrandWave for relevant disturbances
        
        Returns disturbances that might interact with this object.
        
        Returns:
            List of relevant disturbances
        """
        if self.current_disturbance is None:
            return []
        
        # Find nearby disturbances in S-space
        nearby = self.propagator.find_nearest_disturbance(
            self.current_disturbance.S_coords,
            k=10
        )
        
        # Filter by relevance (same domain or strong coupling)
        relevant = []
        for disturbance, distance in nearby:
            if disturbance.source == self.object_id:
                continue  # Don't listen to self
                
            # Relevant if same domain or close in S-space
            if disturbance.domain == self.domain or distance < 0.2:
                relevant.append(disturbance)
                
        return relevant
        
    def _listen_for_interference(self):
        """
        Internal: Listen for interference patterns and update state
        """
        if self.current_disturbance is None:
            return
            
        # Find relevant disturbances
        relevant = self.listen()
        
        # Calculate interference with each
        received_patterns = []
        for other_disturbance in relevant:
            # Get interference pattern
            pattern = self.grand_wave.calculate_interference(
                self.current_disturbance,
                other_disturbance
            )
            
            # Extract key info
            received_patterns.append((
                other_disturbance.source,
                pattern.coherence,
                pattern.phase_relationship
            ))
            
            # Trigger callback if interference is strong
            if pattern.coherence > 0.7 and self.on_interference_callback:
                self.on_interference_callback(pattern)
                
        # Update stored patterns
        self.received_patterns = received_patterns
        
        # Check overall coherence with GrandWave
        coherence = self._calculate_wave_coherence()
        
        if coherence < 0.3 and self.on_coherence_loss_callback:
            self.on_coherence_loss_callback(coherence)
            
    def _calculate_wave_coherence(self) -> float:
        """
        Calculate how well this object maintains coherence with GrandWave
        
        Returns:
            Coherence score (0-1)
        """
        if self.current_disturbance is None:
            return 0.0
            
        # Check coherence with nearby disturbances
        nearby = self.propagator.find_nearest_disturbance(
            self.current_disturbance.S_coords,
            k=5
        )
        
        if not nearby:
            return 1.0  # No constraints
            
        # Average coherence with neighbors
        coherences = []
        for disturbance, distance in nearby:
            if disturbance.source == self.object_id:
                continue
                
            # Coherence decreases exponentially with S-distance
            coherence = np.exp(-distance / 0.5)
            coherences.append(coherence)
            
        return np.mean(coherences) if coherences else 1.0
        
    def find_harmonics(self, target_object_id: str) -> List[Tuple[float, float, float]]:
        """
        Find harmonic coincidences with another object
        
        Args:
            target_object_id: ID of other object
            
        Returns:
            List of (frequency_self, frequency_other, strength) tuples
        """
        if self.current_disturbance is None:
            return []
            
        # Get target disturbance
        if target_object_id not in self.grand_wave.active_disturbances:
            return []
            
        target_disturbance = self.grand_wave.active_disturbances[target_object_id]
        
        # Calculate interference to get harmonic coincidences
        pattern = self.grand_wave.calculate_interference(
            self.current_disturbance,
            target_disturbance
        )
        
        return pattern.harmonic_coincidences
        
    def navigate_to(self, target_S: np.ndarray) -> Dict[str, Any]:
        """
        Navigate from current position to target in S-space
        
        Uses GrandWave's O(1) direct navigation.
        
        Args:
            target_S: Target S-coordinates
            
        Returns:
            Navigation result
        """
        if self.current_disturbance is None:
            return {'error': 'No current disturbance'}
            
        # Use GrandWave's direct navigation
        nav_result = self.grand_wave.navigate_to_target(
            self.current_disturbance.S_coords,
            target_S
        )
        
        # Also get wave propagation path
        prop_path = self.propagator.propagate_wave(
            self.current_disturbance.S_coords,
            target_S
        )
        
        return {
            'direct_navigation': nav_result,
            'wave_propagation': prop_path,
            'recommendation': 'direct' if prop_path.coherence_score < 0.5 else 'propagation'
        }
        
    def suggest_optimization(self,
                           target_property: str,
                           target_value: float) -> List[np.ndarray]:
        """
        Suggest S-coordinates for optimization based on interference patterns
        
        Args:
            target_property: Property to optimize
            target_value: Target value
            
        Returns:
            List of suggested S-coordinates
        """
        # Use propagator to find solution regions
        solution_regions = self.propagator.find_solution_region(
            target_property=target_property,
            target_value=target_value,
            domain=self.domain,
            search_radius=0.5
        )
        
        # Extract S-coordinates (ignore probabilities for now)
        suggestions = [S_coords for S_coords, prob in solution_regions]
        
        return suggestions[:5]  # Top 5 suggestions
        
    def get_interaction_status(self) -> Dict[str, Any]:
        """
        Get current status of object's interaction with GrandWave
        
        Returns:
            Dictionary with status information
        """
        if self.current_disturbance is None:
            return {
                'active': False,
                'object_id': self.object_id,
                'domain': self.domain
            }
            
        # Calculate statistics
        coherence = self._calculate_wave_coherence()
        nearby = self.propagator.find_nearest_disturbance(
            self.current_disturbance.S_coords,
            k=10
        )
        
        return {
            'active': True,
            'object_id': self.object_id,
            'domain': self.domain,
            'S_coords': self.current_disturbance.S_coords.tolist(),
            'n_frequencies': len(self.current_disturbance.frequencies),
            'dominant_frequency': self.current_disturbance.frequencies[
                np.argmax(self.current_disturbance.amplitudes)
            ],
            'coherence_with_wave': coherence,
            'n_nearby_disturbances': len(nearby),
            'n_received_patterns': len(self.received_patterns),
            'timestamp': self.current_disturbance.timestamp
        }
        
    def withdraw(self):
        """
        Withdraw from GrandWave (remove disturbance)
        
        Object stops interacting with the wave substrate.
        """
        if self.current_disturbance is None:
            return
            
        # Remove from active disturbances
        if self.object_id in self.grand_wave.active_disturbances:
            del self.grand_wave.active_disturbances[self.object_id]
            
        self.current_disturbance = None
        self.received_patterns = []
        
    def __repr__(self) -> str:
        status = self.get_interaction_status()
        if status['active']:
            return (
                f"WaveInterface(id={self.object_id}, domain={self.domain}, "
                f"S_coords={status['S_coords']}, coherence={status['coherence_with_wave']:.3f})"
            )
        else:
            return f"WaveInterface(id={self.object_id}, domain={self.domain}, inactive)"


class InterfaceManager:
    """
    Manages multiple WaveInterface instances
    
    Useful for coordinating interactions between many objects.
    """
    
    def __init__(self, grand_wave: GrandWave):
        """
        Initialize interface manager
        
        Args:
            grand_wave: GrandWave instance
        """
        self.grand_wave = grand_wave
        self.interfaces: Dict[str, WaveInterface] = {}
        
    def create_interface(self, object_id: str, domain: str) -> WaveInterface:
        """
        Create new interface for an object
        
        Args:
            object_id: Unique identifier
            domain: Measurement domain
            
        Returns:
            New WaveInterface instance
        """
        interface = WaveInterface(object_id, self.grand_wave, domain)
        self.interfaces[object_id] = interface
        return interface
        
    def get_interface(self, object_id: str) -> Optional[WaveInterface]:
        """Get existing interface"""
        return self.interfaces.get(object_id)
        
    def find_coupled_objects(self,
                            object_id: str,
                            coupling_threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        Find objects strongly coupled to given object
        
        Args:
            object_id: Object to check
            coupling_threshold: Minimum coherence for "coupling"
            
        Returns:
            List of (other_object_id, coherence) tuples
        """
        interface = self.get_interface(object_id)
        if interface is None or interface.current_disturbance is None:
            return []
            
        # Check coupling with all other active interfaces
        coupled = []
        for other_id, other_interface in self.interfaces.items():
            if other_id == object_id:
                continue
                
            if other_interface.current_disturbance is None:
                continue
                
            # Calculate interference
            pattern = self.grand_wave.calculate_interference(
                interface.current_disturbance,
                other_interface.current_disturbance
            )
            
            if pattern.coherence > coupling_threshold:
                coupled.append((other_id, pattern.coherence))
                
        # Sort by coherence
        coupled.sort(key=lambda x: x[1], reverse=True)
        
        return coupled
        
    def get_all_status(self) -> List[Dict[str, Any]]:
        """
        Get status of all interfaces
        
        Returns:
            List of status dictionaries
        """
        return [iface.get_interaction_status() for iface in self.interfaces.values()]
        
    def cleanup_inactive(self):
        """Remove interfaces with no active disturbances"""
        inactive = [
            obj_id for obj_id, iface in self.interfaces.items()
            if iface.current_disturbance is None
        ]
        
        for obj_id in inactive:
            del self.interfaces[obj_id]