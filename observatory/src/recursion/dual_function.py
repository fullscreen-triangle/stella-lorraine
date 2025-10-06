"""
Dual Function Molecular System Implementation

Implements the mathematical relationship:
```
Molecule(i) = Processor(i) ⊗ Oscillator(i)
```

Where:
- `Processor(i)` = computational capacity of molecule i
- `Oscillator(i)` = temporal frequency reference of molecule i
- `⊗` = tensor product representing dual functionality
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class MoleculeType(Enum):
    """Types of molecules with different dual function characteristics"""
    N2 = "nitrogen"
    O2 = "oxygen"
    H2O = "water"
    TRACE = "trace_gas"


@dataclass
class ProcessorCapacity:
    """Computational capacity characteristics of a molecule"""
    processing_rate: float  # Operations per second
    memory_capacity: float  # Information storage capacity
    bandwidth: float  # Information transfer rate
    efficiency: float  # Processing efficiency factor (0-1)

    def compute_operations(self, duration: float) -> float:
        """Calculate total operations possible in given duration"""
        return self.processing_rate * duration * self.efficiency


@dataclass
class OscillatorProperties:
    """Temporal frequency reference characteristics of a molecule"""
    fundamental_frequency: float  # Base oscillation frequency (Hz)
    harmonic_frequencies: List[float]  # Harmonic oscillation modes
    phase: float  # Current phase angle (radians)
    amplitude: float  # Oscillation amplitude
    stability: float  # Frequency stability factor (0-1)

    def get_frequency_at_time(self, time: float) -> float:
        """Get instantaneous frequency considering stability"""
        frequency_drift = (1.0 - self.stability) * np.sin(time * 0.1)
        return self.fundamental_frequency * (1.0 + frequency_drift)

    def get_phase_at_time(self, time: float) -> float:
        """Get phase at specific time"""
        return (self.phase + 2 * np.pi * self.fundamental_frequency * time) % (2 * np.pi)


class DualFunctionMolecule:
    """
    Implementation of dual-function molecule combining processing and oscillatory capabilities

    Mathematical representation: Molecule(i) = Processor(i) ⊗ Oscillator(i)
    """

    def __init__(self,
                 molecule_id: int,
                 molecule_type: MoleculeType,
                 processor: ProcessorCapacity,
                 oscillator: OscillatorProperties,
                 position: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.molecule_id = molecule_id
        self.molecule_type = molecule_type
        self.processor = processor
        self.oscillator = oscillator
        self.position = position
        self.current_time = 0.0
        self.processing_history: List[Dict] = []
        self.oscillation_history: List[Dict] = []

    def tensor_product_capability(self) -> float:
        """
        Calculate the tensor product ⊗ representing dual functionality
        Returns combined processing-oscillatory capability
        """
        processing_component = self.processor.processing_rate * self.processor.efficiency
        oscillatory_component = self.oscillator.fundamental_frequency * self.oscillator.stability

        # Tensor product as geometric mean with interaction term
        base_capability = np.sqrt(processing_component * oscillatory_component)

        # Interaction enhancement: better processors can utilize oscillatory precision better
        interaction_factor = 1.0 + (self.processor.efficiency * self.oscillator.stability)

        return base_capability * interaction_factor

    def process_information(self, information_load: float, duration: float) -> Dict:
        """
        Process information using computational capacity

        Args:
            information_load: Amount of information to process
            duration: Processing time duration

        Returns:
            Dictionary with processing results
        """
        max_operations = self.processor.compute_operations(duration)
        operations_needed = information_load / self.processor.bandwidth

        if operations_needed <= max_operations:
            processing_success = True
            processing_efficiency = 1.0
            processed_information = information_load
        else:
            processing_success = False
            processing_efficiency = max_operations / operations_needed
            processed_information = information_load * processing_efficiency

        result = {
            'success': processing_success,
            'processed_information': processed_information,
            'efficiency': processing_efficiency,
            'operations_used': min(operations_needed, max_operations),
            'processing_time': duration,
            'timestamp': self.current_time
        }

        self.processing_history.append(result)
        return result

    def generate_oscillatory_signal(self, duration: float, sampling_rate: float = 1e6) -> np.ndarray:
        """
        Generate oscillatory signal for temporal reference

        Args:
            duration: Signal duration in seconds
            sampling_rate: Sampling rate in Hz

        Returns:
            Numpy array with oscillatory signal
        """
        time_points = np.linspace(0, duration, int(duration * sampling_rate))
        signal = np.zeros_like(time_points)

        # Fundamental frequency contribution
        freq = self.oscillator.get_frequency_at_time(self.current_time)
        phase = self.oscillator.get_phase_at_time(self.current_time)
        signal += self.oscillator.amplitude * np.sin(2 * np.pi * freq * time_points + phase)

        # Harmonic contributions
        for i, harmonic_freq in enumerate(self.oscillator.harmonic_frequencies):
            harmonic_amplitude = self.oscillator.amplitude / (i + 2)  # Decreasing amplitude
            signal += harmonic_amplitude * np.sin(2 * np.pi * harmonic_freq * time_points + phase)

        # Record oscillation data
        oscillation_record = {
            'duration': duration,
            'frequency': freq,
            'phase': phase,
            'amplitude': self.oscillator.amplitude,
            'timestamp': self.current_time,
            'signal_power': np.mean(signal**2)
        }
        self.oscillation_history.append(oscillation_record)

        return signal

    def dual_function_operation(self,
                              information_load: float,
                              duration: float,
                              temporal_precision_required: float) -> Dict:
        """
        Perform dual-function operation: simultaneous processing and oscillatory timing

        Args:
            information_load: Information to process
            duration: Operation duration
            temporal_precision_required: Required timing precision

        Returns:
            Combined processing and oscillatory results
        """
        # Processing component
        processing_result = self.process_information(information_load, duration)

        # Oscillatory component
        oscillatory_signal = self.generate_oscillatory_signal(duration)

        # Calculate achieved temporal precision
        frequency_stability = self.oscillator.stability
        achieved_precision = 1.0 / (self.oscillator.fundamental_frequency * frequency_stability)

        # Dual function synergy: processing quality affects temporal precision
        synergy_factor = processing_result['efficiency'] * frequency_stability
        enhanced_precision = achieved_precision / (1.0 + synergy_factor)

        # Check if precision requirements are met
        precision_met = enhanced_precision <= temporal_precision_required

        dual_result = {
            'processing': processing_result,
            'oscillatory': {
                'signal_length': len(oscillatory_signal),
                'achieved_precision': enhanced_precision,
                'precision_met': precision_met,
                'frequency_stability': frequency_stability
            },
            'dual_function': {
                'tensor_product_capability': self.tensor_product_capability(),
                'synergy_factor': synergy_factor,
                'combined_efficiency': processing_result['efficiency'] * frequency_stability
            },
            'timestamp': self.current_time
        }

        self.current_time += duration
        return dual_result

    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics for the dual-function molecule"""
        if not self.processing_history or not self.oscillation_history:
            return {'error': 'No operation history available'}

        # Processing metrics
        processing_efficiencies = [record['efficiency'] for record in self.processing_history]
        avg_processing_efficiency = np.mean(processing_efficiencies)

        # Oscillatory metrics
        frequencies = [record['frequency'] for record in self.oscillation_history]
        frequency_stability = 1.0 - (np.std(frequencies) / np.mean(frequencies))

        # Dual function metrics
        tensor_capability = self.tensor_product_capability()

        return {
            'molecule_id': self.molecule_id,
            'molecule_type': self.molecule_type.value,
            'processing_metrics': {
                'average_efficiency': avg_processing_efficiency,
                'total_operations': len(self.processing_history),
                'processing_rate': self.processor.processing_rate
            },
            'oscillatory_metrics': {
                'frequency_stability': frequency_stability,
                'fundamental_frequency': self.oscillator.fundamental_frequency,
                'total_oscillations': len(self.oscillation_history)
            },
            'dual_function_metrics': {
                'tensor_product_capability': tensor_capability,
                'combined_performance': tensor_capability * frequency_stability
            }
        }


class MolecularDualFunctionNetwork:
    """Network of dual-function molecules for collective operations"""

    def __init__(self):
        self.molecules: Dict[int, DualFunctionMolecule] = {}
        self.network_history: List[Dict] = []

    def add_molecule(self, molecule: DualFunctionMolecule):
        """Add a dual-function molecule to the network"""
        self.molecules[molecule.molecule_id] = molecule

    def create_standard_molecule(self,
                               molecule_id: int,
                               molecule_type: MoleculeType,
                               position: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> DualFunctionMolecule:
        """Create a molecule with standard parameters based on type"""

        if molecule_type == MoleculeType.N2:
            processor = ProcessorCapacity(
                processing_rate=1e12,  # 1 THz equivalent
                memory_capacity=1e6,
                bandwidth=1e9,
                efficiency=0.85
            )
            oscillator = OscillatorProperties(
                fundamental_frequency=2.36e14,  # N2 vibrational frequency
                harmonic_frequencies=[4.72e14, 7.08e14],
                phase=0.0,
                amplitude=1.0,
                stability=0.92
            )

        elif molecule_type == MoleculeType.O2:
            processor = ProcessorCapacity(
                processing_rate=1.2e12,
                memory_capacity=8e5,
                bandwidth=1.1e9,
                efficiency=0.88
            )
            oscillator = OscillatorProperties(
                fundamental_frequency=4.74e14,  # O2 vibrational frequency
                harmonic_frequencies=[9.48e14, 1.42e15],
                phase=0.0,
                amplitude=1.0,
                stability=0.90
            )

        elif molecule_type == MoleculeType.H2O:
            processor = ProcessorCapacity(
                processing_rate=8e11,
                memory_capacity=1.2e6,
                bandwidth=9e8,
                efficiency=0.82
            )
            oscillator = OscillatorProperties(
                fundamental_frequency=1e12,  # H2O rotational frequency (average)
                harmonic_frequencies=[2e12, 3e12, 5e12],
                phase=0.0,
                amplitude=1.2,
                stability=0.88
            )

        else:  # TRACE
            processor = ProcessorCapacity(
                processing_rate=5e11,
                memory_capacity=5e5,
                bandwidth=5e8,
                efficiency=0.75
            )
            oscillator = OscillatorProperties(
                fundamental_frequency=1e13,
                harmonic_frequencies=[2e13, 3e13],
                phase=0.0,
                amplitude=0.8,
                stability=0.85
            )

        molecule = DualFunctionMolecule(molecule_id, molecule_type, processor, oscillator, position)
        self.add_molecule(molecule)
        return molecule

    def collective_dual_operation(self,
                                information_loads: List[float],
                                duration: float,
                                precision_requirement: float) -> Dict:
        """Perform collective dual-function operation across network"""

        if len(information_loads) != len(self.molecules):
            raise ValueError("Information load list must match number of molecules")

        results = {}
        total_tensor_capability = 0.0
        successful_operations = 0

        for i, (mol_id, molecule) in enumerate(self.molecules.items()):
            mol_result = molecule.dual_function_operation(
                information_loads[i], duration, precision_requirement
            )
            results[mol_id] = mol_result
            total_tensor_capability += mol_result['dual_function']['tensor_product_capability']

            if mol_result['oscillatory']['precision_met']:
                successful_operations += 1

        collective_result = {
            'individual_results': results,
            'collective_metrics': {
                'total_tensor_capability': total_tensor_capability,
                'average_tensor_capability': total_tensor_capability / len(self.molecules),
                'precision_success_rate': successful_operations / len(self.molecules),
                'network_size': len(self.molecules)
            },
            'timestamp': duration
        }

        self.network_history.append(collective_result)
        return collective_result

    def get_network_performance(self) -> Dict:
        """Get comprehensive network performance metrics"""
        if not self.network_history:
            return {'error': 'No network operation history'}

        # Aggregate individual molecule metrics
        individual_metrics = [mol.get_performance_metrics() for mol in self.molecules.values()]

        # Network-level metrics
        total_tensor_capability = sum([
            metrics['dual_function_metrics']['tensor_product_capability']
            for metrics in individual_metrics
        ])

        avg_processing_efficiency = np.mean([
            metrics['processing_metrics']['average_efficiency']
            for metrics in individual_metrics
        ])

        avg_frequency_stability = np.mean([
            metrics['oscillatory_metrics']['frequency_stability']
            for metrics in individual_metrics
        ])

        return {
            'network_size': len(self.molecules),
            'molecule_composition': {
                mol_type.value: sum(1 for mol in self.molecules.values() if mol.molecule_type == mol_type)
                for mol_type in MoleculeType
            },
            'collective_performance': {
                'total_tensor_capability': total_tensor_capability,
                'average_processing_efficiency': avg_processing_efficiency,
                'average_frequency_stability': avg_frequency_stability,
                'network_coherence': avg_processing_efficiency * avg_frequency_stability
            },
            'individual_molecule_metrics': individual_metrics
        }
