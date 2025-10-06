"""
Empty Dictionary - Memoryless Hierarchical Navigation System

Implements O(1) hierarchical navigation through gear ratio calculations
and transcendent observer coordination for memoryless state transitions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import math
import time


class ObserverType(Enum):
    """Types of observers in the hierarchical system"""
    FINITE = "finite_observer"
    TRANSCENDENT = "transcendent_observer"
    COMPOUND = "compound_observer"


class HierarchicalLevel(Enum):
    """Predefined hierarchical levels with frequency relationships"""
    QUANTUM = "quantum_level"          # ω₁ = ω₀
    MOLECULAR = "molecular_level"      # ω₂ = 2ω₀
    CELLULAR = "cellular_level"        # ω₃ = 4ω₀
    TISSUE = "tissue_level"            # ω₄ = 8ω₀
    ORGAN = "organ_level"              # ω₅ = 16ω₀
    ORGANISM = "organism_level"        # ω₆ = 32ω₀
    SYSTEM = "system_level"            # ω₇ = 64ω₀


@dataclass
class OscillatoryHierarchy:
    """Hierarchical level with oscillatory properties"""
    level: HierarchicalLevel
    frequency: float  # ωᵢ
    scaling_factor: float  # αᵢ
    base_frequency: float  # ω₀
    information_capacity: float = 0.0
    processing_load: float = 0.0

    def calculate_gear_ratio_to(self, target_level: 'OscillatoryHierarchy') -> float:
        """Calculate gear ratio R_{i→j} = ωᵢ/ωⱼ"""
        if target_level.frequency == 0:
            raise ValueError("Target level frequency cannot be zero")
        return self.frequency / target_level.frequency


@dataclass
class ObserverState:
    """State of a finite observer"""
    observer_id: str
    observer_type: ObserverType
    current_level: Optional[HierarchicalLevel] = None
    information_acquired: Dict = field(default_factory=dict)
    observation_duration: float = 0.0
    processing_capacity: float = 1.0
    active: bool = True

    def can_observe_level(self, level: HierarchicalLevel) -> bool:
        """Check if observer can observe specific level"""
        return self.active and self.processing_capacity > 0.1


@dataclass
class TranscendentObserverState:
    """State of transcendent observer coordinating finite observers"""
    observer_id: str
    observed_observers: List[str] = field(default_factory=list)
    gear_ratios: Dict[Tuple[HierarchicalLevel, HierarchicalLevel], float] = field(default_factory=dict)
    navigation_state: Dict = field(default_factory=dict)
    coordination_efficiency: float = 0.95

    def add_observed_observer(self, observer_id: str):
        """Add finite observer to coordination"""
        if observer_id not in self.observed_observers:
            self.observed_observers.append(observer_id)


class EmptyDictionaryNavigator:
    """
    Memoryless hierarchical navigation system using gear ratios

    Implements O(1) navigation complexity through pre-computed frequency relationships
    enabling direct transitions without path traversal or memory requirements.
    """

    def __init__(self, base_frequency: float = 1.0):
        self.base_frequency = base_frequency
        self.hierarchical_levels: Dict[HierarchicalLevel, OscillatoryHierarchy] = {}
        self.finite_observers: Dict[str, ObserverState] = {}
        self.transcendent_observers: Dict[str, TranscendentObserverState] = {}
        self.gear_ratio_cache: Dict[Tuple[HierarchicalLevel, HierarchicalLevel], float] = {}
        self.navigation_history: List[Dict] = []

        # Initialize standard hierarchical levels
        self._initialize_hierarchical_levels()
        self._precompute_gear_ratios()

    def _initialize_hierarchical_levels(self):
        """Initialize hierarchical levels with frequency relationships"""
        level_configs = [
            (HierarchicalLevel.QUANTUM, 1.0),      # ω₁ = ω₀
            (HierarchicalLevel.MOLECULAR, 2.0),    # ω₂ = 2ω₀
            (HierarchicalLevel.CELLULAR, 4.0),     # ω₃ = 4ω₀
            (HierarchicalLevel.TISSUE, 8.0),       # ω₄ = 8ω₀
            (HierarchicalLevel.ORGAN, 16.0),       # ω₅ = 16ω₀
            (HierarchicalLevel.ORGANISM, 32.0),    # ω₆ = 32ω₀
            (HierarchicalLevel.SYSTEM, 64.0)       # ω₇ = 64ω₀
        ]

        for level, scaling_factor in level_configs:
            frequency = scaling_factor * self.base_frequency
            hierarchy = OscillatoryHierarchy(
                level=level,
                frequency=frequency,
                scaling_factor=scaling_factor,
                base_frequency=self.base_frequency,
                information_capacity=100.0 * scaling_factor,
                processing_load=10.0 / scaling_factor  # Higher levels have lower processing load
            )
            self.hierarchical_levels[level] = hierarchy

    def _precompute_gear_ratios(self):
        """Pre-compute all gear ratios for O(1) navigation"""
        levels = list(self.hierarchical_levels.keys())

        for source_level in levels:
            for target_level in levels:
                if source_level != target_level:
                    source_hierarchy = self.hierarchical_levels[source_level]
                    target_hierarchy = self.hierarchical_levels[target_level]
                    gear_ratio = source_hierarchy.calculate_gear_ratio_to(target_hierarchy)
                    self.gear_ratio_cache[(source_level, target_level)] = gear_ratio

    def create_finite_observer(self,
                             observer_id: str,
                             initial_level: Optional[HierarchicalLevel] = None,
                             processing_capacity: float = 1.0) -> ObserverState:
        """Create a finite observer that can observe one level at a time"""
        observer = ObserverState(
            observer_id=observer_id,
            observer_type=ObserverType.FINITE,
            current_level=initial_level,
            processing_capacity=processing_capacity
        )

        self.finite_observers[observer_id] = observer
        return observer

    def create_transcendent_observer(self, observer_id: str) -> TranscendentObserverState:
        """Create transcendent observer that coordinates finite observers"""
        transcendent = TranscendentObserverState(
            observer_id=observer_id,
            gear_ratios=self.gear_ratio_cache.copy()
        )

        self.transcendent_observers[observer_id] = transcendent
        return transcendent

    def navigate_direct(self,
                       source_level: HierarchicalLevel,
                       target_level: HierarchicalLevel,
                       observer_id: Optional[str] = None) -> Dict:
        """
        Direct navigation with O(1) complexity using gear ratios

        Implements: Navigate(Ls → Lt) = ApplyGearRatio(Rs→t) = O(1)
        """
        start_time = time.time()

        # Phase 1: Lookup gear ratio - O(1)
        gear_ratio_key = (source_level, target_level)
        if gear_ratio_key not in self.gear_ratio_cache:
            return {
                'success': False,
                'error': f'No gear ratio available for {source_level} → {target_level}',
                'execution_time': time.time() - start_time
            }

        gear_ratio = self.gear_ratio_cache[gear_ratio_key]

        # Phase 2: Apply gear ratio transformation - O(1)
        source_hierarchy = self.hierarchical_levels[source_level]
        target_hierarchy = self.hierarchical_levels[target_level]

        # Calculate state transformation
        information_transfer_efficiency = min(1.0, 1.0 / abs(gear_ratio))
        processing_overhead = abs(math.log(gear_ratio)) * 0.1

        # Phase 3: Update observer state if specified
        if observer_id and observer_id in self.finite_observers:
            observer = self.finite_observers[observer_id]
            if observer.can_observe_level(target_level):
                observer.current_level = target_level
                observer.observation_duration = processing_overhead
                observer.information_acquired[target_level.value] = {
                    'frequency': target_hierarchy.frequency,
                    'information_capacity': target_hierarchy.information_capacity,
                    'gear_ratio_used': gear_ratio,
                    'transfer_efficiency': information_transfer_efficiency
                }

        execution_time = time.time() - start_time

        # Record navigation
        navigation_record = {
            'timestamp': time.time(),
            'source_level': source_level.value,
            'target_level': target_level.value,
            'gear_ratio': gear_ratio,
            'execution_time': execution_time,
            'transfer_efficiency': information_transfer_efficiency,
            'processing_overhead': processing_overhead,
            'observer_id': observer_id
        }
        self.navigation_history.append(navigation_record)

        return {
            'success': True,
            'source_level': source_level,
            'target_level': target_level,
            'gear_ratio': gear_ratio,
            'execution_time': execution_time,
            'transfer_efficiency': information_transfer_efficiency,
            'processing_overhead': processing_overhead,
            'complexity': 'O(1)',
            'navigation_record': navigation_record
        }

    def transcendent_navigate(self,
                            transcendent_observer_id: str,
                            navigation_sequence: List[Tuple[HierarchicalLevel, HierarchicalLevel]]) -> Dict:
        """
        Transcendent observer coordinates multiple navigation operations
        """
        if transcendent_observer_id not in self.transcendent_observers:
            return {'error': 'Transcendent observer not found'}

        transcendent = self.transcendent_observers[transcendent_observer_id]
        start_time = time.time()

        navigation_results = []
        total_gear_ratio = 1.0

        # Execute navigation sequence
        for source_level, target_level in navigation_sequence:
            nav_result = self.navigate_direct(source_level, target_level)
            navigation_results.append(nav_result)

            if nav_result['success']:
                total_gear_ratio *= nav_result['gear_ratio']

        # Update transcendent observer state
        transcendent.navigation_state.update({
            'last_sequence': navigation_sequence,
            'total_gear_ratio': total_gear_ratio,
            'sequence_length': len(navigation_sequence),
            'success_rate': sum(1 for r in navigation_results if r['success']) / len(navigation_results)
        })

        execution_time = time.time() - start_time

        return {
            'transcendent_observer_id': transcendent_observer_id,
            'navigation_sequence': navigation_sequence,
            'individual_results': navigation_results,
            'total_gear_ratio': total_gear_ratio,
            'total_execution_time': execution_time,
            'average_complexity': 'O(1) per operation',
            'coordination_efficiency': transcendent.coordination_efficiency
        }

    def memoryless_transition(self,
                            current_context: Dict,
                            target_specification: Dict) -> Dict:
        """
        Perform memoryless state transition using empty dictionary synthesis

        No stored path information required - synthesizes optimal route
        """
        # Extract levels from context and specification
        current_level = current_context.get('level')
        target_level = target_specification.get('level')

        if not current_level or not target_level:
            return {'error': 'Invalid level specification'}

        # Convert string levels to enum if needed
        if isinstance(current_level, str):
            current_level = HierarchicalLevel(current_level)
        if isinstance(target_level, str):
            target_level = HierarchicalLevel(target_level)

        # Synthesize optimal transition (memoryless)
        transition_result = self.navigate_direct(current_level, target_level)

        # Add memoryless properties
        transition_result.update({
            'memoryless': True,
            'context_dependency': False,
            'path_storage_required': False,
            'synthesis_method': 'empty_dictionary_gear_ratio'
        })

        return transition_result

    def optimize_observer_network(self,
                                transcendent_observer_id: str,
                                finite_observer_ids: List[str],
                                optimization_target: str = 'information_throughput') -> Dict:
        """
        Optimize network of finite observers coordinated by transcendent observer
        """
        if transcendent_observer_id not in self.transcendent_observers:
            return {'error': 'Transcendent observer not found'}

        transcendent = self.transcendent_observers[transcendent_observer_id]

        # Add finite observers to transcendent coordination
        for observer_id in finite_observer_ids:
            transcendent.add_observed_observer(observer_id)

        # Optimize based on target
        if optimization_target == 'information_throughput':
            optimization_result = self._optimize_for_throughput(transcendent, finite_observer_ids)
        elif optimization_target == 'processing_efficiency':
            optimization_result = self._optimize_for_efficiency(transcendent, finite_observer_ids)
        else:
            optimization_result = self._optimize_balanced(transcendent, finite_observer_ids)

        return {
            'transcendent_observer_id': transcendent_observer_id,
            'coordinated_observers': len(transcendent.observed_observers),
            'optimization_target': optimization_target,
            'optimization_result': optimization_result,
            'network_coherence': self._calculate_network_coherence(transcendent)
        }

    def _optimize_for_throughput(self, transcendent: TranscendentObserverState, observer_ids: List[str]) -> Dict:
        """Optimize network for maximum information throughput"""
        # Assign observers to levels with highest information capacity
        level_assignments = {}

        # Sort levels by information capacity (descending)
        sorted_levels = sorted(self.hierarchical_levels.items(),
                             key=lambda x: x[1].information_capacity, reverse=True)

        for i, observer_id in enumerate(observer_ids):
            if observer_id in self.finite_observers:
                level = sorted_levels[i % len(sorted_levels)][0]
                level_assignments[observer_id] = level
                self.finite_observers[observer_id].current_level = level

        return {
            'optimization_type': 'throughput',
            'level_assignments': level_assignments,
            'expected_throughput_improvement': len(observer_ids) * 0.8  # Simplified metric
        }

    def _optimize_for_efficiency(self, transcendent: TranscendentObserverState, observer_ids: List[str]) -> Dict:
        """Optimize network for processing efficiency"""
        # Assign observers to levels with lowest processing load
        level_assignments = {}

        # Sort levels by processing load (ascending)
        sorted_levels = sorted(self.hierarchical_levels.items(),
                             key=lambda x: x[1].processing_load)

        for i, observer_id in enumerate(observer_ids):
            if observer_id in self.finite_observers:
                level = sorted_levels[i % len(sorted_levels)][0]
                level_assignments[observer_id] = level
                self.finite_observers[observer_id].current_level = level

        return {
            'optimization_type': 'efficiency',
            'level_assignments': level_assignments,
            'expected_efficiency_improvement': len(observer_ids) * 0.6
        }

    def _optimize_balanced(self, transcendent: TranscendentObserverState, observer_ids: List[str]) -> Dict:
        """Optimize network for balanced performance"""
        # Balance information capacity and processing efficiency
        level_assignments = {}

        # Calculate balance score for each level
        level_scores = {}
        for level, hierarchy in self.hierarchical_levels.items():
            # Balance score = information_capacity / processing_load
            balance_score = hierarchy.information_capacity / max(hierarchy.processing_load, 0.1)
            level_scores[level] = balance_score

        # Sort by balance score
        sorted_levels = sorted(level_scores.items(), key=lambda x: x[1], reverse=True)

        for i, observer_id in enumerate(observer_ids):
            if observer_id in self.finite_observers:
                level = sorted_levels[i % len(sorted_levels)][0]
                level_assignments[observer_id] = level
                self.finite_observers[observer_id].current_level = level

        return {
            'optimization_type': 'balanced',
            'level_assignments': level_assignments,
            'expected_balance_improvement': len(observer_ids) * 0.7
        }

    def _calculate_network_coherence(self, transcendent: TranscendentObserverState) -> float:
        """Calculate coherence of observer network"""
        if not transcendent.observed_observers:
            return 0.0

        # Calculate coherence based on level distribution and coordination efficiency
        active_observers = [obs for obs in transcendent.observed_observers
                          if obs in self.finite_observers and self.finite_observers[obs].active]

        if not active_observers:
            return 0.0

        # Level distribution coherence
        level_distribution = {}
        for obs_id in active_observers:
            observer = self.finite_observers[obs_id]
            if observer.current_level:
                level_distribution[observer.current_level] = level_distribution.get(observer.current_level, 0) + 1

        # Calculate distribution entropy (lower = more coherent)
        total_observers = len(active_observers)
        distribution_entropy = 0.0
        for count in level_distribution.values():
            prob = count / total_observers
            distribution_entropy -= prob * math.log2(prob) if prob > 0 else 0

        # Normalize and invert (higher coherence = lower entropy)
        max_entropy = math.log2(len(self.hierarchical_levels))
        coherence = 1.0 - (distribution_entropy / max_entropy) if max_entropy > 0 else 1.0

        # Factor in coordination efficiency
        network_coherence = coherence * transcendent.coordination_efficiency

        return min(1.0, max(0.0, network_coherence))

    def get_navigation_statistics(self) -> Dict:
        """Get comprehensive navigation performance statistics"""
        if not self.navigation_history:
            return {'no_history': True}

        # Calculate statistics
        total_navigations = len(self.navigation_history)
        avg_execution_time = np.mean([record['execution_time'] for record in self.navigation_history])
        avg_transfer_efficiency = np.mean([record['transfer_efficiency'] for record in self.navigation_history])
        avg_processing_overhead = np.mean([record['processing_overhead'] for record in self.navigation_history])

        # Level usage distribution
        level_usage = {}
        for record in self.navigation_history:
            source = record['source_level']
            target = record['target_level']
            level_usage[source] = level_usage.get(source, 0) + 1
            level_usage[target] = level_usage.get(target, 0) + 1

        return {
            'total_navigations': total_navigations,
            'average_execution_time': avg_execution_time,
            'average_transfer_efficiency': avg_transfer_efficiency,
            'average_processing_overhead': avg_processing_overhead,
            'complexity_class': 'O(1)',
            'level_usage_distribution': level_usage,
            'finite_observers': len(self.finite_observers),
            'transcendent_observers': len(self.transcendent_observers),
            'recent_navigations': self.navigation_history[-10:] if len(self.navigation_history) >= 10 else self.navigation_history
        }

    def get_system_state(self) -> Dict:
        """Get current system state"""
        return {
            'hierarchical_levels': len(self.hierarchical_levels),
            'gear_ratios_cached': len(self.gear_ratio_cache),
            'finite_observers': len(self.finite_observers),
            'transcendent_observers': len(self.transcendent_observers),
            'navigation_history_length': len(self.navigation_history),
            'base_frequency': self.base_frequency,
            'system_coherence': self._calculate_system_coherence()
        }

    def _calculate_system_coherence(self) -> float:
        """Calculate overall system coherence"""
        if not self.transcendent_observers:
            return 0.8  # Base coherence without transcendent coordination

        # Average coherence across all transcendent observers
        coherences = [self._calculate_network_coherence(transcendent)
                     for transcendent in self.transcendent_observers.values()]

        return np.mean(coherences) if coherences else 0.8


def create_memoryless_navigation_system(base_frequency: float = 1.0) -> EmptyDictionaryNavigator:
    """Create memoryless navigation system for S-Entropy alignment"""
    return EmptyDictionaryNavigator(base_frequency=base_frequency)
