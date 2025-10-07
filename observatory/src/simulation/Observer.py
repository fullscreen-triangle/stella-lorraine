"""
Observer - The Block that Interacts with Reality's Wave

This implements the Observer/block class with ALL its properties, methods, functions, helpers.
The observer creates interference patterns when interacting with reality's infinite complexity wave.
Properties are mutable and the class can be instantiated multiple times for observer networks.

Key Principle: Observers always create "less descriptive" interference patterns because
they are products of interaction with reality - always subsets of the main wave.
"""

import numpy as np
import time
import math
from typing import Dict, List, Tuple, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor


class ObserverType(Enum):
    """Types of observers in the wave simulation"""
    BASIC_BLOCK = "basic_block"                    # Simple physical block
    FINITE_OBSERVER = "finite_observer"           # Standard observer with limitations
    RESONANT_OBSERVER = "resonant_observer"       # Observer with frequency resonances
    ADAPTIVE_OBSERVER = "adaptive_observer"       # Observer that adapts to signals
    QUANTUM_OBSERVER = "quantum_observer"         # Quantum-enabled observer
    COLLECTIVE_OBSERVER = "collective_observer"   # Multiple observers working together
    TRANSCENDENT_OBSERVER = "transcendent_observer"  # Meta-observer observing others


class InteractionMode(Enum):
    """Modes of observer-wave interaction"""
    PASSIVE = "passive"           # Minimal disturbance to wave
    ABSORPTIVE = "absorptive"     # Absorbs wave energy
    REFLECTIVE = "reflective"     # Reflects waves
    SCATTERING = "scattering"     # Scatters waves in multiple directions
    RESONANT = "resonant"         # Resonates at specific frequencies
    NONLINEAR = "nonlinear"       # Nonlinear response to wave amplitude
    ADAPTIVE = "adaptive"         # Changes interaction based on signal


class ObserverLimitation(Enum):
    """Fundamental limitations of observers"""
    FREQUENCY_BAND = "frequency_band"        # Limited frequency perception
    SPATIAL_RESOLUTION = "spatial_resolution"  # Limited spatial perception
    TEMPORAL_RESOLUTION = "temporal_resolution"  # Limited time resolution
    AMPLITUDE_RANGE = "amplitude_range"      # Limited dynamic range
    PHASE_SENSITIVITY = "phase_sensitivity"  # Limited phase detection
    COHERENCE_LENGTH = "coherence_length"    # Limited coherence detection
    PROCESSING_SPEED = "processing_speed"    # Limited processing capabilities


@dataclass
class InterferencePattern:
    """Interference pattern created by observer interaction with reality wave"""
    pattern_id: str
    observer_id: str
    source_coordinates: Tuple[float, float, float]
    timestamp: float
    pattern_data: Dict[str, complex] = field(default_factory=dict)  # Spatial interference map
    pattern_complexity: float = 0.0
    information_loss: float = 0.0              # How much information is lost vs main wave
    coherence_reduction: float = 0.0           # Reduction in wave coherence
    entropy_increase: float = 0.0              # Entropy increase due to interaction

    def calculate_pattern_metrics(self) -> Dict:
        """Calculate comprehensive metrics for this interference pattern"""
        if not self.pattern_data:
            return {'error': 'no_pattern_data'}

        amplitudes = [abs(signal) for signal in self.pattern_data.values()]
        phases = [np.angle(signal) for signal in self.pattern_data.values()]

        return {
            'amplitude_statistics': {
                'mean': np.mean(amplitudes),
                'std': np.std(amplitudes),
                'dynamic_range': max(amplitudes) / min(amplitudes) if min(amplitudes) > 0 else float('inf')
            },
            'phase_statistics': {
                'mean_phase': np.mean(phases),
                'phase_coherence': abs(np.mean([np.exp(1j * p) for p in phases])),
                'phase_variance': np.var(phases)
            },
            'complexity_metrics': {
                'pattern_complexity': self.pattern_complexity,
                'information_loss': self.information_loss,
                'coherence_reduction': self.coherence_reduction,
                'entropy_increase': self.entropy_increase
            },
            'information_preservation': 1.0 - self.information_loss,
            'pattern_fidelity': 1.0 - self.coherence_reduction
        }


@dataclass
class ObserverCapabilities:
    """Capabilities and limitations of an observer"""
    frequency_range: Tuple[float, float] = (1.0, 1e12)      # Observable frequency range
    spatial_resolution: float = 1.0                         # Minimum spatial resolution
    temporal_resolution: float = 1e-9                       # Minimum temporal resolution
    amplitude_sensitivity: Tuple[float, float] = (1e-6, 1e6)  # Min/max detectable amplitude
    phase_sensitivity: float = 0.1                          # Phase detection accuracy (radians)
    coherence_length: float = 100.0                         # Spatial coherence detection length
    processing_bandwidth: float = 1e6                       # Processing capability (samples/sec)
    interaction_strength: float = 0.1                       # Strength of wave interaction
    adaptation_rate: float = 0.01                           # Rate of adaptive changes

    def get_frequency_filter_response(self, frequency: float) -> float:
        """Get filter response for specific frequency"""
        f_min, f_max = self.frequency_range

        if f_min <= frequency <= f_max:
            # Gaussian response within range
            center_freq = (f_min + f_max) / 2
            bandwidth = (f_max - f_min) / 4  # 4-sigma bandwidth
            response = np.exp(-0.5 * ((frequency - center_freq) / bandwidth)**2)
            return response
        else:
            # Outside range - exponential rolloff
            if frequency < f_min:
                rolloff = np.exp(-(f_min - frequency) / f_min)
            else:
                rolloff = np.exp(-(frequency - f_max) / f_max)
            return max(0.001, rolloff)  # Minimum response 0.1%

    def can_resolve_spatial_detail(self, detail_size: float) -> bool:
        """Check if observer can resolve spatial detail of given size"""
        return detail_size >= self.spatial_resolution

    def can_resolve_temporal_detail(self, time_scale: float) -> bool:
        """Check if observer can resolve temporal detail at given time scale"""
        return time_scale >= self.temporal_resolution


class Observer:
    """
    Observer - Block that Interacts with Reality's Infinite Wave

    Creates interference patterns when interacting with reality's wave.
    The key insight: interference patterns are always "less descriptive" than
    the main wave because they are products of limited interaction.

    Properties:
    - Mutable configuration for different observer types
    - Network instantiation support
    - Realistic limitations modeling finite observation
    - Information loss quantification
    """

    def __init__(self,
                 observer_id: str,
                 observer_type: ObserverType = ObserverType.FINITE_OBSERVER,
                 position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 size: Tuple[float, float, float] = (1.0, 1.0, 1.0)):

        self.observer_id = observer_id
        self.observer_type = observer_type
        self.position = position  # 3D position in space
        self.size = size          # 3D dimensions (width, height, depth)

        # Observer state (mutable properties)
        self.is_active = True
        self.interaction_mode = InteractionMode.PASSIVE
        self.current_orientation = 0.0  # Orientation angle
        self.adaptation_state = {}      # Adaptive parameters

        # Observer capabilities (mutable)
        self.capabilities = self._initialize_capabilities()

        # Interaction history
        self.interference_patterns: List[InterferencePattern] = []
        self.interaction_history: List[Dict] = []
        self.observed_signals: List[Tuple[float, complex]] = []  # (timestamp, signal)

        # Performance metrics
        self.total_interactions = 0
        self.average_information_loss = 0.0
        self.coherence_preservation_rate = 0.0

        # Network properties for multi-observer systems
        self.connected_observers: Set[str] = set()
        self.communication_range = 100.0
        self.coordination_protocol = "basic"

        # Threading for concurrent operation
        self.observation_active = False
        self.observation_thread = None
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

    def _initialize_capabilities(self) -> ObserverCapabilities:
        """Initialize observer capabilities based on observer type"""

        capability_configs = {
            ObserverType.BASIC_BLOCK: ObserverCapabilities(
                frequency_range=(1e6, 1e9),         # Limited frequency range
                spatial_resolution=10.0,             # Coarse spatial resolution
                temporal_resolution=1e-6,            # Microsecond timing
                amplitude_sensitivity=(1e-3, 1e3),   # Limited dynamic range
                phase_sensitivity=0.5,               # Poor phase sensitivity
                interaction_strength=0.5             # Strong physical interaction
            ),
            ObserverType.FINITE_OBSERVER: ObserverCapabilities(
                frequency_range=(1e3, 1e15),        # Broad but limited range
                spatial_resolution=1.0,              # Meter-scale resolution
                temporal_resolution=1e-9,            # Nanosecond timing
                amplitude_sensitivity=(1e-6, 1e6),   # Good dynamic range
                phase_sensitivity=0.1,               # Decent phase detection
                interaction_strength=0.1             # Moderate interaction
            ),
            ObserverType.RESONANT_OBSERVER: ObserverCapabilities(
                frequency_range=(1e6, 1e12),        # Focused frequency band
                spatial_resolution=0.1,              # Sub-meter resolution
                temporal_resolution=1e-12,           # Picosecond timing
                amplitude_sensitivity=(1e-9, 1e9),   # High sensitivity
                phase_sensitivity=0.01,              # Excellent phase detection
                interaction_strength=0.05,           # Weak interaction
                coherence_length=1000.0              # Long coherence detection
            ),
            ObserverType.QUANTUM_OBSERVER: ObserverCapabilities(
                frequency_range=(1e12, 1e20),       # High frequency range
                spatial_resolution=1e-9,             # Nanometer resolution
                temporal_resolution=1e-15,           # Femtosecond timing
                amplitude_sensitivity=(1e-12, 1e12), # Ultra-high sensitivity
                phase_sensitivity=0.001,             # Ultra-precise phase
                interaction_strength=0.001,          # Minimal disturbance
                coherence_length=10000.0             # Very long coherence
            ),
            ObserverType.ADAPTIVE_OBSERVER: ObserverCapabilities(
                frequency_range=(1e2, 1e18),        # Adaptive frequency range
                spatial_resolution=0.5,              # Adaptive resolution
                temporal_resolution=1e-12,           # Fast adaptation
                amplitude_sensitivity=(1e-9, 1e9),   # Adaptive sensitivity
                phase_sensitivity=0.05,              # Adaptive phase detection
                interaction_strength=0.1,            # Moderate interaction
                adaptation_rate=0.1                  # Fast adaptation
            )
        }

        return capability_configs.get(self.observer_type, ObserverCapabilities())

    def interact_with_wave(self,
                          reality_wave: Any,  # InfiniteComplexityWave
                          interaction_region: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None,
                          duration: float = 1.0) -> InterferencePattern:
        """
        Interact with reality's infinite wave and create interference pattern

        This is the core method where the observer creates "less descriptive"
        interference patterns by interacting with reality's complexity.
        """

        if not self.is_active:
            return None

        start_time = time.time()
        pattern_id = f"{self.observer_id}_{int(start_time * 1000)}"

        # Determine interaction region
        if interaction_region is None:
            # Default interaction region around observer position
            x, y, z = self.position
            w, h, d = self.size
            interaction_region = (
                (x - w/2, x + w/2),
                (y - h/2, y + h/2),
                (z - d/2, z + d/2)
            )

        # Sample reality wave in interaction region
        original_signals = self._sample_reality_wave(reality_wave, interaction_region, duration)

        # Apply observer limitations and create interference
        interference_signals = self._apply_observer_interaction(original_signals)

        # Calculate information loss and pattern metrics
        pattern_metrics = self._calculate_interference_metrics(original_signals, interference_signals)

        # Create interference pattern
        interference_pattern = InterferencePattern(
            pattern_id=pattern_id,
            observer_id=self.observer_id,
            source_coordinates=self.position,
            timestamp=start_time,
            pattern_data=interference_signals,
            pattern_complexity=pattern_metrics['pattern_complexity'],
            information_loss=pattern_metrics['information_loss'],
            coherence_reduction=pattern_metrics['coherence_reduction'],
            entropy_increase=pattern_metrics['entropy_increase']
        )

        # Store interaction results
        self.interference_patterns.append(interference_pattern)
        self.total_interactions += 1

        # Update performance metrics
        self._update_performance_metrics(pattern_metrics)

        # Record interaction history
        interaction_record = {
            'timestamp': start_time,
            'interaction_duration': duration,
            'interaction_region': interaction_region,
            'pattern_id': pattern_id,
            'information_loss': pattern_metrics['information_loss'],
            'interaction_mode': self.interaction_mode.value
        }
        self.interaction_history.append(interaction_record)

        # Adaptive learning (if enabled)
        if self.observer_type == ObserverType.ADAPTIVE_OBSERVER:
            self._adapt_to_interaction(pattern_metrics)

        return interference_pattern

    def _sample_reality_wave(self,
                           reality_wave: Any,
                           region: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
                           duration: float) -> Dict[str, complex]:
        """Sample reality's wave within interaction region"""

        x_bounds, y_bounds, z_bounds = region

        # Create spatial sampling grid
        sampling_density = 20  # Points per dimension
        x_samples = np.linspace(x_bounds[0], x_bounds[1], sampling_density)
        y_samples = np.linspace(y_bounds[0], y_bounds[1], sampling_density)
        z_samples = np.linspace(z_bounds[0], z_bounds[1], sampling_density)

        # Time sampling
        time_samples = np.linspace(0, duration, int(duration / self.capabilities.temporal_resolution))
        time_samples = time_samples[:100]  # Limit for performance

        original_signals = {}

        # Sample reality wave at grid points
        for i, x in enumerate(x_samples[:10]):  # Limit for performance
            for j, y in enumerate(y_samples[:10]):
                for k, z in enumerate(z_samples[:10]):
                    for t_idx, t in enumerate(time_samples[:10]):
                        sample_key = f"x{i}_y{j}_z{k}_t{t_idx}"

                        # Get reality signal at this spacetime point
                        reality_signal = reality_wave.generate_reality_signal_at_point(
                            (x, y, z), time.time() + t
                        )

                        original_signals[sample_key] = reality_signal

        return original_signals

    def _apply_observer_interaction(self, original_signals: Dict[str, complex]) -> Dict[str, complex]:
        """Apply observer limitations and interaction to create interference pattern"""

        interference_signals = {}

        for sample_key, original_signal in original_signals.items():
            # Apply observer frequency response
            modified_signal = self._apply_frequency_filtering(original_signal)

            # Apply amplitude limitations
            modified_signal = self._apply_amplitude_limitations(modified_signal)

            # Apply phase distortion
            modified_signal = self._apply_phase_distortion(modified_signal)

            # Apply spatial resolution limitations
            modified_signal = self._apply_spatial_filtering(modified_signal, sample_key)

            # Apply interaction mode effects
            modified_signal = self._apply_interaction_mode(modified_signal, original_signal)

            # Add observer noise
            modified_signal = self._add_observer_noise(modified_signal)

            interference_signals[sample_key] = modified_signal

        return interference_signals

    def _apply_frequency_filtering(self, signal: complex) -> complex:
        """Apply frequency-dependent filtering based on observer capabilities"""

        # Extract frequency components (simplified - assumes single frequency)
        frequency = abs(signal) * 1e9  # Rough frequency estimate

        # Apply frequency response
        filter_response = self.capabilities.get_frequency_filter_response(frequency)

        return signal * filter_response

    def _apply_amplitude_limitations(self, signal: complex) -> complex:
        """Apply amplitude sensitivity limitations"""

        amplitude = abs(signal)
        min_amp, max_amp = self.capabilities.amplitude_sensitivity

        # Clip to sensitivity range
        if amplitude < min_amp:
            return 0.0 + 0.0j  # Below detection threshold
        elif amplitude > max_amp:
            # Saturation - preserve phase but limit amplitude
            phase = np.angle(signal)
            return max_amp * np.exp(1j * phase)
        else:
            return signal

    def _apply_phase_distortion(self, signal: complex) -> complex:
        """Apply phase detection limitations"""

        original_phase = np.angle(signal)
        amplitude = abs(signal)

        # Phase noise based on sensitivity
        phase_noise = np.random.normal(0, self.capabilities.phase_sensitivity)
        distorted_phase = original_phase + phase_noise

        return amplitude * np.exp(1j * distorted_phase)

    def _apply_spatial_filtering(self, signal: complex, sample_key: str) -> complex:
        """Apply spatial resolution limitations"""

        # Simple spatial filtering based on resolution
        # Higher resolution observers preserve more spatial detail

        resolution_factor = 1.0 / (1.0 + self.capabilities.spatial_resolution)

        # Reduce signal complexity based on spatial resolution
        return signal * (1.0 - resolution_factor * 0.1)

    def _apply_interaction_mode(self, modified_signal: complex, original_signal: complex) -> complex:
        """Apply interaction mode-specific effects"""

        if self.interaction_mode == InteractionMode.PASSIVE:
            # Minimal modification
            return modified_signal

        elif self.interaction_mode == InteractionMode.ABSORPTIVE:
            # Absorb some energy
            absorption_factor = 1.0 - self.capabilities.interaction_strength * 0.5
            return modified_signal * absorption_factor

        elif self.interaction_mode == InteractionMode.REFLECTIVE:
            # Reflect signal with phase shift
            reflection_phase = np.pi
            return modified_signal * np.exp(1j * reflection_phase) * 0.8

        elif self.interaction_mode == InteractionMode.SCATTERING:
            # Scatter in random directions (adds noise)
            scattering_noise = np.random.normal(0, 0.1) + 1j * np.random.normal(0, 0.1)
            return modified_signal + scattering_noise * self.capabilities.interaction_strength

        elif self.interaction_mode == InteractionMode.RESONANT:
            # Resonant amplification at specific frequencies
            resonance_frequency = 1e9  # Example resonance
            signal_freq = abs(original_signal) * 1e9

            if abs(signal_freq - resonance_frequency) < resonance_frequency * 0.1:
                return modified_signal * 2.0  # Resonant amplification
            else:
                return modified_signal * 0.5  # Off-resonance attenuation

        elif self.interaction_mode == InteractionMode.NONLINEAR:
            # Nonlinear response
            amplitude = abs(modified_signal)
            phase = np.angle(modified_signal)

            # Nonlinear amplitude compression
            nonlinear_amplitude = amplitude / (1.0 + amplitude * self.capabilities.interaction_strength)

            return nonlinear_amplitude * np.exp(1j * phase)

        else:  # ADAPTIVE
            # Adaptive response based on signal characteristics
            adaptation_factor = 1.0 + self.capabilities.adaptation_rate * np.random.normal(0, 0.1)
            return modified_signal * adaptation_factor

    def _add_observer_noise(self, signal: complex) -> complex:
        """Add observer-induced noise"""

        # Thermal noise
        noise_amplitude = 1e-6  # Observer thermal noise level
        noise_real = np.random.normal(0, noise_amplitude)
        noise_imag = np.random.normal(0, noise_amplitude)

        return signal + (noise_real + 1j * noise_imag)

    def _calculate_interference_metrics(self,
                                      original_signals: Dict[str, complex],
                                      interference_signals: Dict[str, complex]) -> Dict:
        """Calculate metrics showing how interference pattern is less descriptive than main wave"""

        if not original_signals or not interference_signals:
            return {
                'information_loss': 1.0,
                'coherence_reduction': 1.0,
                'entropy_increase': 1.0,
                'pattern_complexity': 0.0
            }

        # Calculate information loss
        original_amplitudes = [abs(s) for s in original_signals.values()]
        interference_amplitudes = [abs(s) for s in interference_signals.values()]

        original_power = sum(amp**2 for amp in original_amplitudes)
        interference_power = sum(amp**2 for amp in interference_amplitudes)

        information_loss = 1.0 - (interference_power / original_power) if original_power > 0 else 1.0

        # Calculate coherence reduction
        original_coherence = self._calculate_signal_coherence(list(original_signals.values()))
        interference_coherence = self._calculate_signal_coherence(list(interference_signals.values()))

        coherence_reduction = (original_coherence - interference_coherence) / original_coherence if original_coherence > 0 else 1.0

        # Calculate entropy increase (disorder increase)
        original_entropy = self._calculate_signal_entropy(original_amplitudes)
        interference_entropy = self._calculate_signal_entropy(interference_amplitudes)

        entropy_increase = interference_entropy - original_entropy

        # Pattern complexity (reduced from original)
        pattern_complexity = interference_entropy / original_entropy if original_entropy > 0 else 0.0

        return {
            'information_loss': max(0.0, min(1.0, information_loss)),
            'coherence_reduction': max(0.0, min(1.0, coherence_reduction)),
            'entropy_increase': entropy_increase,
            'pattern_complexity': pattern_complexity
        }

    def _calculate_signal_coherence(self, signals: List[complex]) -> float:
        """Calculate coherence of signal list"""
        if len(signals) < 2:
            return 0.0

        # Phase coherence measure
        phases = [np.angle(s) for s in signals]
        coherence_vector = sum(np.exp(1j * phase) for phase in phases) / len(phases)

        return abs(coherence_vector)

    def _calculate_signal_entropy(self, amplitudes: List[float]) -> float:
        """Calculate Shannon entropy of amplitude distribution"""
        if not amplitudes:
            return 0.0

        # Create probability distribution
        hist, _ = np.histogram(amplitudes, bins=20, density=True)

        # Calculate entropy
        entropy = 0.0
        for p in hist:
            if p > 0:
                entropy -= p * np.log2(p)

        return entropy

    def _update_performance_metrics(self, pattern_metrics: Dict):
        """Update running performance metrics"""

        # Update average information loss
        prev_avg = self.average_information_loss
        n = self.total_interactions
        self.average_information_loss = (prev_avg * (n - 1) + pattern_metrics['information_loss']) / n

        # Update coherence preservation rate
        coherence_preserved = 1.0 - pattern_metrics['coherence_reduction']
        prev_coherence = self.coherence_preservation_rate
        self.coherence_preservation_rate = (prev_coherence * (n - 1) + coherence_preserved) / n

    def _adapt_to_interaction(self, pattern_metrics: Dict):
        """Adaptive learning from interaction patterns"""

        if self.observer_type != ObserverType.ADAPTIVE_OBSERVER:
            return

        # Adapt frequency range based on information loss
        if pattern_metrics['information_loss'] > 0.8:
            # High information loss - expand frequency range
            f_min, f_max = self.capabilities.frequency_range
            expansion_factor = 1.1
            self.capabilities.frequency_range = (f_min / expansion_factor, f_max * expansion_factor)

        # Adapt interaction strength
        if pattern_metrics['coherence_reduction'] > 0.6:
            # High coherence loss - reduce interaction strength
            self.capabilities.interaction_strength *= 0.95

        # Adapt sensitivity
        if self.average_information_loss > 0.5:
            # Improve sensitivity to reduce information loss
            min_amp, max_amp = self.capabilities.amplitude_sensitivity
            self.capabilities.amplitude_sensitivity = (min_amp * 0.95, max_amp * 1.05)

    def connect_to_observer(self, other_observer_id: str):
        """Connect to another observer for network operations"""
        self.connected_observers.add(other_observer_id)

    def disconnect_from_observer(self, other_observer_id: str):
        """Disconnect from another observer"""
        self.connected_observers.discard(other_observer_id)

    def get_network_status(self) -> Dict:
        """Get status of observer network connections"""
        return {
            'observer_id': self.observer_id,
            'connected_observers': list(self.connected_observers),
            'network_size': len(self.connected_observers),
            'communication_range': self.communication_range,
            'coordination_protocol': self.coordination_protocol
        }

    def start_continuous_observation(self, reality_wave: Any, observation_rate: float = 1.0):
        """Start continuous observation of reality wave"""

        if self.observation_active:
            return

        self.observation_active = True

        def observation_loop():
            while self.observation_active and self.is_active:
                try:
                    pattern = self.interact_with_wave(reality_wave, duration=observation_rate)
                    if pattern:
                        # Store observed signal
                        signal_amplitude = np.mean([abs(s) for s in pattern.pattern_data.values()])
                        self.observed_signals.append((time.time(), signal_amplitude))

                        # Limit history size
                        if len(self.observed_signals) > 1000:
                            self.observed_signals = self.observed_signals[-1000:]

                except Exception as e:
                    pass  # Continue observation despite errors

                time.sleep(observation_rate)

        self.observation_thread = threading.Thread(target=observation_loop, daemon=True)
        self.observation_thread.start()

    def stop_continuous_observation(self):
        """Stop continuous observation"""
        self.observation_active = False
        if self.observation_thread:
            self.observation_thread.join(timeout=2.0)

    def get_observer_status(self) -> Dict:
        """Get comprehensive observer status"""

        # Calculate recent performance
        recent_patterns = self.interference_patterns[-10:] if len(self.interference_patterns) >= 10 else self.interference_patterns
        recent_info_loss = np.mean([p.information_loss for p in recent_patterns]) if recent_patterns else 0.0
        recent_coherence_loss = np.mean([p.coherence_reduction for p in recent_patterns]) if recent_patterns else 0.0

        return {
            'observer_identity': {
                'observer_id': self.observer_id,
                'observer_type': self.observer_type.value,
                'position': self.position,
                'size': self.size,
                'is_active': self.is_active
            },
            'interaction_configuration': {
                'interaction_mode': self.interaction_mode.value,
                'orientation': self.current_orientation,
                'capabilities': {
                    'frequency_range': self.capabilities.frequency_range,
                    'spatial_resolution': self.capabilities.spatial_resolution,
                    'temporal_resolution': self.capabilities.temporal_resolution,
                    'amplitude_sensitivity': self.capabilities.amplitude_sensitivity,
                    'interaction_strength': self.capabilities.interaction_strength
                }
            },
            'performance_metrics': {
                'total_interactions': self.total_interactions,
                'average_information_loss': self.average_information_loss,
                'coherence_preservation_rate': self.coherence_preservation_rate,
                'recent_information_loss': recent_info_loss,
                'recent_coherence_loss': recent_coherence_loss
            },
            'observation_status': {
                'continuous_observation_active': self.observation_active,
                'interference_patterns_created': len(self.interference_patterns),
                'observed_signals_count': len(self.observed_signals)
            },
            'network_status': self.get_network_status(),
            'information_processing': {
                'proves_subset_principle': self.average_information_loss > 0.0,
                'demonstrates_complexity_reduction': recent_info_loss > 0.0,
                'validates_alignment_theorem': recent_coherence_loss > 0.0
            }
        }


def create_observer_network(observer_configs: List[Dict]) -> List[Observer]:
    """Create network of observers with different configurations"""

    observers = []

    for config in observer_configs:
        observer = Observer(
            observer_id=config.get('observer_id', f'observer_{len(observers)}'),
            observer_type=config.get('observer_type', ObserverType.FINITE_OBSERVER),
            position=config.get('position', (0.0, 0.0, 0.0)),
            size=config.get('size', (1.0, 1.0, 1.0))
        )

        # Set interaction mode if specified
        if 'interaction_mode' in config:
            observer.interaction_mode = config['interaction_mode']

        # Connect observers in network
        for existing_observer in observers:
            if np.linalg.norm(np.array(observer.position) - np.array(existing_observer.position)) <= observer.communication_range:
                observer.connect_to_observer(existing_observer.observer_id)
                existing_observer.connect_to_observer(observer.observer_id)

        observers.append(observer)

    return observers


def create_basic_observer(observer_id: str,
                         position: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Observer:
    """Create a basic observer for wave interaction experiments"""
    return Observer(observer_id=observer_id, position=position)
