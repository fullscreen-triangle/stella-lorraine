"""
Observer Oscillation Hierarchy System

Implements observation process initiation and hierarchical oscillatory management
for S-Entropy alignment across multiple temporal and spatial scales.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import math


class ObservationMode(Enum):
    """Modes of observation process"""
    PASSIVE = "passive_observation"
    ACTIVE = "active_observation"
    RESONANT = "resonant_observation"
    TRANSCENDENT = "transcendent_observation"


class HierarchyScale(Enum):
    """Hierarchical scales for observation"""
    PLANCK = "planck_scale"           # 10^-35 m, 10^-44 s
    QUANTUM = "quantum_scale"         # 10^-15 m, 10^-21 s
    ATOMIC = "atomic_scale"           # 10^-10 m, 10^-16 s
    MOLECULAR = "molecular_scale"     # 10^-9 m, 10^-12 s
    CELLULAR = "cellular_scale"       # 10^-6 m, 10^-3 s
    TISSUE = "tissue_scale"           # 10^-3 m, 10^0 s
    ORGAN = "organ_scale"             # 10^-2 m, 10^1 s
    ORGANISM = "organism_scale"       # 10^0 m, 10^2 s
    ECOSYSTEM = "ecosystem_scale"     # 10^3 m, 10^5 s
    PLANETARY = "planetary_scale"     # 10^7 m, 10^8 s
    COSMIC = "cosmic_scale"           # 10^15 m, 10^15 s


@dataclass
class OscillatoryProperties:
    """Oscillatory properties for hierarchical levels"""
    frequency: float  # Base frequency (Hz)
    amplitude: float  # Oscillation amplitude
    phase: float  # Phase offset (radians)
    stability: float  # Frequency stability (0-1)
    coherence: float  # Oscillatory coherence (0-1)
    coupling_strength: float  # Inter-level coupling

    def calculate_oscillatory_signature(self, time_point: float) -> complex:
        """Calculate oscillatory signature at specific time"""
        # Include stability and coherence effects
        effective_frequency = self.frequency * (1.0 + (1.0 - self.stability) * np.random.normal(0, 0.01))
        effective_amplitude = self.amplitude * self.coherence

        # Complex oscillatory signature
        signature = effective_amplitude * np.exp(1j * (2 * np.pi * effective_frequency * time_point + self.phase))
        return signature


@dataclass
class ObservationProcess:
    """Active observation process with hierarchical context"""
    process_id: str
    observer_id: str
    target_scale: HierarchyScale
    observation_mode: ObservationMode
    oscillatory_properties: OscillatoryProperties
    start_time: float
    duration: float
    information_collected: Dict = field(default_factory=dict)
    observation_quality: float = 0.0
    active: bool = True

    def update_observation_quality(self, new_quality: float):
        """Update observation quality based on oscillatory alignment"""
        self.observation_quality = min(1.0, max(0.0, new_quality))


@dataclass
class HierarchicalObserver:
    """Observer operating across hierarchical scales"""
    observer_id: str
    supported_scales: List[HierarchyScale]
    current_scale: Optional[HierarchyScale] = None
    processing_capacity: float = 1.0
    observation_efficiency: float = 0.85
    active_processes: List[str] = field(default_factory=list)
    observation_history: List[Dict] = field(default_factory=list)

    def can_observe_scale(self, scale: HierarchyScale) -> bool:
        """Check if observer can observe specific scale"""
        return scale in self.supported_scales and self.processing_capacity > 0.1


class ObserverOscillationHierarchy:
    """
    Observer oscillation hierarchy management system

    Initiates and manages observation processes across multiple hierarchical scales
    with oscillatory synchronization for S-Entropy alignment optimization.
    """

    def __init__(self):
        self.scale_properties: Dict[HierarchyScale, OscillatoryProperties] = {}
        self.observers: Dict[str, HierarchicalObserver] = {}
        self.active_processes: Dict[str, ObservationProcess] = {}
        self.observation_history: List[Dict] = []
        self.system_coherence: float = 0.95

        # Initialize hierarchical scale properties
        self._initialize_scale_properties()

        # Gear ratio cache for inter-scale navigation
        self.gear_ratios: Dict[Tuple[HierarchyScale, HierarchyScale], float] = {}
        self._calculate_inter_scale_gear_ratios()

    def _initialize_scale_properties(self):
        """Initialize oscillatory properties for each hierarchical scale"""
        scale_configs = [
            (HierarchyScale.PLANCK, 1e44, 1.0, 0.0, 0.99, 0.95, 0.8),
            (HierarchyScale.QUANTUM, 1e21, 0.8, 0.1, 0.95, 0.90, 0.85),
            (HierarchyScale.ATOMIC, 1e16, 0.7, 0.2, 0.92, 0.88, 0.90),
            (HierarchyScale.MOLECULAR, 1e12, 0.6, 0.3, 0.90, 0.85, 0.85),
            (HierarchyScale.CELLULAR, 1e3, 0.5, 0.4, 0.88, 0.82, 0.80),
            (HierarchyScale.TISSUE, 1e0, 0.4, 0.5, 0.85, 0.80, 0.75),
            (HierarchyScale.ORGAN, 1e-1, 0.35, 0.6, 0.82, 0.78, 0.70),
            (HierarchyScale.ORGANISM, 1e-2, 0.3, 0.7, 0.80, 0.75, 0.65),
            (HierarchyScale.ECOSYSTEM, 1e-5, 0.25, 0.8, 0.75, 0.70, 0.60),
            (HierarchyScale.PLANETARY, 1e-8, 0.2, 0.9, 0.70, 0.65, 0.55),
            (HierarchyScale.COSMIC, 1e-15, 0.15, 1.0, 0.65, 0.60, 0.50)
        ]

        for scale, freq, amp, phase, stability, coherence, coupling in scale_configs:
            self.scale_properties[scale] = OscillatoryProperties(
                frequency=freq,
                amplitude=amp,
                phase=phase,
                stability=stability,
                coherence=coherence,
                coupling_strength=coupling
            )

    def _calculate_inter_scale_gear_ratios(self):
        """Calculate gear ratios between all hierarchical scales"""
        scales = list(self.scale_properties.keys())

        for source_scale in scales:
            for target_scale in scales:
                if source_scale != target_scale:
                    source_freq = self.scale_properties[source_scale].frequency
                    target_freq = self.scale_properties[target_scale].frequency

                    if target_freq != 0:
                        gear_ratio = source_freq / target_freq
                        self.gear_ratios[(source_scale, target_scale)] = gear_ratio

    def create_hierarchical_observer(self,
                                   observer_id: str,
                                   supported_scales: List[HierarchyScale],
                                   processing_capacity: float = 1.0) -> HierarchicalObserver:
        """Create observer capable of operating across specified scales"""
        observer = HierarchicalObserver(
            observer_id=observer_id,
            supported_scales=supported_scales,
            processing_capacity=processing_capacity,
            observation_efficiency=0.85 + np.random.normal(0, 0.05)  # Individual variation
        )

        self.observers[observer_id] = observer
        return observer

    def initiate_observation_process(self,
                                   observer_id: str,
                                   target_scale: HierarchyScale,
                                   observation_mode: ObservationMode = ObservationMode.ACTIVE,
                                   duration: float = 1.0) -> str:
        """
        Initiate observation process at specific hierarchical scale

        Returns process_id for tracking
        """
        if observer_id not in self.observers:
            raise ValueError(f"Observer {observer_id} not found")

        observer = self.observers[observer_id]
        if not observer.can_observe_scale(target_scale):
            raise ValueError(f"Observer {observer_id} cannot observe scale {target_scale}")

        # Generate unique process ID
        process_id = f"{observer_id}_{target_scale.value}_{int(time.time() * 1000)}"

        # Get oscillatory properties for target scale
        oscillatory_props = self.scale_properties[target_scale]

        # Create observation process
        process = ObservationProcess(
            process_id=process_id,
            observer_id=observer_id,
            target_scale=target_scale,
            observation_mode=observation_mode,
            oscillatory_properties=oscillatory_props,
            start_time=time.time(),
            duration=duration
        )

        # Register process
        self.active_processes[process_id] = process
        observer.active_processes.append(process_id)
        observer.current_scale = target_scale

        # Initialize observation quality based on oscillatory alignment
        initial_quality = self._calculate_observation_quality(observer, process)
        process.update_observation_quality(initial_quality)

        return process_id

    def _calculate_observation_quality(self,
                                     observer: HierarchicalObserver,
                                     process: ObservationProcess) -> float:
        """Calculate observation quality based on oscillatory alignment"""
        # Base quality from observer efficiency
        base_quality = observer.observation_efficiency

        # Oscillatory alignment factor
        props = process.oscillatory_properties
        alignment_factor = props.stability * props.coherence * props.coupling_strength

        # Mode-specific modifiers
        mode_modifiers = {
            ObservationMode.PASSIVE: 0.8,
            ObservationMode.ACTIVE: 1.0,
            ObservationMode.RESONANT: 1.2,
            ObservationMode.TRANSCENDENT: 1.5
        }

        mode_modifier = mode_modifiers.get(process.observation_mode, 1.0)

        # Processing capacity effect
        capacity_factor = min(1.0, observer.processing_capacity)

        # Calculate final quality
        quality = base_quality * alignment_factor * mode_modifier * capacity_factor
        return min(1.0, max(0.0, quality))

    def update_observation_process(self, process_id: str) -> Dict:
        """Update observation process and collect information"""
        if process_id not in self.active_processes:
            return {'error': f'Process {process_id} not found'}

        process = self.active_processes[process_id]
        observer = self.observers[process.observer_id]

        # Check if process should still be active
        elapsed_time = time.time() - process.start_time
        if elapsed_time >= process.duration:
            return self._complete_observation_process(process_id)

        # Calculate current oscillatory signature
        current_time = time.time()
        oscillatory_signature = process.oscillatory_properties.calculate_oscillatory_signature(current_time)

        # Update information collection
        information_update = {
            'timestamp': current_time,
            'elapsed_time': elapsed_time,
            'oscillatory_signature': {
                'magnitude': abs(oscillatory_signature),
                'phase': np.angle(oscillatory_signature),
                'frequency': process.oscillatory_properties.frequency
            },
            'observation_quality': process.observation_quality,
            'scale': process.target_scale.value
        }

        # Store information
        process.information_collected[f'update_{len(process.information_collected)}'] = information_update

        # Update observation quality based on current conditions
        current_quality = self._calculate_observation_quality(observer, process)
        process.update_observation_quality(current_quality)

        return {
            'process_id': process_id,
            'status': 'active',
            'elapsed_time': elapsed_time,
            'remaining_time': process.duration - elapsed_time,
            'current_quality': process.observation_quality,
            'information_collected': len(process.information_collected),
            'oscillatory_signature': information_update['oscillatory_signature']
        }

    def _complete_observation_process(self, process_id: str) -> Dict:
        """Complete observation process and archive results"""
        if process_id not in self.active_processes:
            return {'error': f'Process {process_id} not found'}

        process = self.active_processes[process_id]
        observer = self.observers[process.observer_id]

        # Mark process as inactive
        process.active = False

        # Calculate final metrics
        total_information = len(process.information_collected)
        average_quality = np.mean([info['observation_quality']
                                 for info in process.information_collected.values()])

        # Archive to observer history
        observation_record = {
            'process_id': process_id,
            'target_scale': process.target_scale.value,
            'observation_mode': process.observation_mode.value,
            'duration': process.duration,
            'total_information_collected': total_information,
            'average_quality': average_quality,
            'completion_time': time.time()
        }

        observer.observation_history.append(observation_record)
        self.observation_history.append(observation_record)

        # Remove from active processes
        del self.active_processes[process_id]
        observer.active_processes.remove(process_id)

        # Reset observer scale if no other active processes
        if not observer.active_processes:
            observer.current_scale = None

        return {
            'process_id': process_id,
            'status': 'completed',
            'total_information_collected': total_information,
            'average_quality': average_quality,
            'observation_record': observation_record
        }

    def navigate_between_scales(self,
                               observer_id: str,
                               source_scale: HierarchyScale,
                               target_scale: HierarchyScale) -> Dict:
        """Navigate observer between hierarchical scales using gear ratios"""
        if observer_id not in self.observers:
            return {'error': f'Observer {observer_id} not found'}

        observer = self.observers[observer_id]
        if not observer.can_observe_scale(target_scale):
            return {'error': f'Observer cannot observe target scale {target_scale}'}

        # Get gear ratio
        gear_ratio_key = (source_scale, target_scale)
        if gear_ratio_key not in self.gear_ratios:
            return {'error': f'No gear ratio available for {source_scale} → {target_scale}'}

        gear_ratio = self.gear_ratios[gear_ratio_key]

        # Calculate navigation efficiency
        navigation_efficiency = min(1.0, 1.0 / abs(gear_ratio) if gear_ratio != 0 else 0.0)

        # Apply navigation
        observer.current_scale = target_scale

        # Update processing capacity based on scale transition
        scale_adjustment = 1.0 - (abs(math.log10(abs(gear_ratio))) * 0.05) if gear_ratio != 0 else 1.0
        observer.processing_capacity *= max(0.5, scale_adjustment)

        return {
            'observer_id': observer_id,
            'source_scale': source_scale.value,
            'target_scale': target_scale.value,
            'gear_ratio': gear_ratio,
            'navigation_efficiency': navigation_efficiency,
            'new_processing_capacity': observer.processing_capacity,
            'navigation_successful': True
        }

    def create_transcendent_observation_network(self,
                                              network_id: str,
                                              observer_ids: List[str],
                                              coordination_mode: str = 'synchronized') -> Dict:
        """Create network of observers for transcendent observation"""
        network_observers = []

        for observer_id in observer_ids:
            if observer_id in self.observers:
                network_observers.append(self.observers[observer_id])

        if not network_observers:
            return {'error': 'No valid observers found for network'}

        # Calculate network properties
        total_capacity = sum(obs.processing_capacity for obs in network_observers)
        avg_efficiency = np.mean([obs.observation_efficiency for obs in network_observers])

        # Determine optimal scale distribution
        all_scales = set()
        for observer in network_observers:
            all_scales.update(observer.supported_scales)

        network_info = {
            'network_id': network_id,
            'observer_count': len(network_observers),
            'total_processing_capacity': total_capacity,
            'average_efficiency': avg_efficiency,
            'supported_scales': list(all_scales),
            'coordination_mode': coordination_mode,
            'network_coherence': self._calculate_network_coherence(network_observers)
        }

        return network_info

    def _calculate_network_coherence(self, observers: List[HierarchicalObserver]) -> float:
        """Calculate coherence of observer network"""
        if not observers:
            return 0.0

        # Calculate efficiency spread
        efficiencies = [obs.observation_efficiency for obs in observers]
        efficiency_variance = np.var(efficiencies)
        efficiency_coherence = 1.0 / (1.0 + efficiency_variance)

        # Calculate capacity balance
        capacities = [obs.processing_capacity for obs in observers]
        capacity_balance = 1.0 - (np.std(capacities) / np.mean(capacities)) if np.mean(capacities) > 0 else 0.0

        # Overall network coherence
        network_coherence = (efficiency_coherence + capacity_balance) / 2.0 * self.system_coherence

        return min(1.0, max(0.0, network_coherence))

    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        active_observer_count = sum(1 for obs in self.observers.values() if obs.active_processes)
        total_active_processes = len(self.active_processes)

        # Scale usage statistics
        scale_usage = {}
        for process in self.active_processes.values():
            scale = process.target_scale.value
            scale_usage[scale] = scale_usage.get(scale, 0) + 1

        # Average observation quality
        if self.active_processes:
            avg_quality = np.mean([process.observation_quality for process in self.active_processes.values()])
        else:
            avg_quality = 0.0

        return {
            'total_observers': len(self.observers),
            'active_observers': active_observer_count,
            'total_active_processes': total_active_processes,
            'supported_scales': len(self.scale_properties),
            'gear_ratios_available': len(self.gear_ratios),
            'average_observation_quality': avg_quality,
            'scale_usage_distribution': scale_usage,
            'system_coherence': self.system_coherence,
            'observation_history_length': len(self.observation_history)
        }

    def get_observer_performance(self, observer_id: str) -> Dict:
        """Get performance metrics for specific observer"""
        if observer_id not in self.observers:
            return {'error': f'Observer {observer_id} not found'}

        observer = self.observers[observer_id]

        # Calculate performance metrics from history
        if observer.observation_history:
            avg_quality = np.mean([record['average_quality'] for record in observer.observation_history])
            total_observations = len(observer.observation_history)
            avg_duration = np.mean([record['duration'] for record in observer.observation_history])
        else:
            avg_quality = 0.0
            total_observations = 0
            avg_duration = 0.0

        return {
            'observer_id': observer_id,
            'supported_scales': [scale.value for scale in observer.supported_scales],
            'current_scale': observer.current_scale.value if observer.current_scale else None,
            'processing_capacity': observer.processing_capacity,
            'observation_efficiency': observer.observation_efficiency,
            'active_processes': len(observer.active_processes),
            'performance_metrics': {
                'total_observations_completed': total_observations,
                'average_observation_quality': avg_quality,
                'average_observation_duration': avg_duration
            },
            'recent_observations': observer.observation_history[-5:] if len(observer.observation_history) >= 5 else observer.observation_history
        }


def create_observer_hierarchy_system() -> ObserverOscillationHierarchy:
    """Create observer oscillation hierarchy system for S-Entropy alignment"""
    return ObserverOscillationHierarchy()
