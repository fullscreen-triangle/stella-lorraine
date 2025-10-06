"""
Atmospheric Molecular Network Extension System

Implements atmospheric molecular density distribution and harvesting efficiency for
precision enhancement through Earth's entire atmospheric molecular network.

Based on:
- Earth's atmosphere: ~10^44 molecules distributed across altitude layers
- Molecular harvesting efficiency modeling
- Multi-layer atmospheric processing capabilities
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import math


class AtmosphericLayer(Enum):
    """Atmospheric layer classifications"""
    TROPOSPHERE = "troposphere"  # 0-12km
    STRATOSPHERE = "stratosphere"  # 12-50km
    MESOSPHERE = "mesosphere"  # 50-80km
    THERMOSPHERE = "thermosphere"  # 80-600km
    EXOSPHERE = "exosphere"  # 600km+


class MolecularGasType(Enum):
    """Types of atmospheric gas molecules"""
    N2 = "nitrogen"
    O2 = "oxygen"
    H2O = "water_vapor"
    AR = "argon"
    CO2 = "carbon_dioxide"
    TRACE = "trace_gases"


@dataclass
class AtmosphericComposition:
    """Atmospheric composition data for a specific layer"""
    layer: AtmosphericLayer
    altitude_range: Tuple[float, float]  # km
    total_molecules: float  # Total molecules in layer
    gas_concentrations: Dict[MolecularGasType, float]  # Molecules by gas type
    temperature: float  # Average temperature (K)
    pressure: float  # Average pressure (Pa)
    density: float  # Air density (kg/m³)

    def get_gas_count(self, gas_type: MolecularGasType) -> float:
        """Get molecule count for specific gas type"""
        return self.gas_concentrations.get(gas_type, 0.0)

    def get_layer_volume(self) -> float:
        """Calculate approximate layer volume (m³)"""
        earth_radius = 6.371e6  # Earth radius in meters
        lower_alt, upper_alt = self.altitude_range

        # Volume of spherical shell
        r1 = earth_radius + lower_alt * 1000  # Convert km to m
        r2 = earth_radius + upper_alt * 1000

        volume = (4/3) * math.pi * (r2**3 - r1**3)
        return volume


@dataclass
class MolecularHarvestingParameters:
    """Parameters for molecular harvesting efficiency calculation"""
    sensing_precision: float = 0.85  # Sensor precision factor (0-1)
    processing_efficiency: float = 0.92  # Processing efficiency (0-1)
    frequency_accuracy: float = 0.88  # Oscillatory frequency accuracy (0-1)
    atmospheric_interference: float = 0.15  # Atmospheric interference factor (0-1)
    altitude_accessibility: float = 0.75  # Altitude-dependent accessibility (0-1)

    def calculate_efficiency(self, layer: AtmosphericLayer, gas_type: MolecularGasType) -> float:
        """Calculate harvesting efficiency for specific layer and gas type"""
        # Base efficiency from sensing and processing
        base_efficiency = self.sensing_precision * self.processing_efficiency * self.frequency_accuracy

        # Layer-specific adjustments
        layer_factor = self._get_layer_accessibility_factor(layer)

        # Gas-specific adjustments
        gas_factor = self._get_gas_harvesting_factor(gas_type)

        # Atmospheric interference
        interference_factor = 1.0 - self.atmospheric_interference

        total_efficiency = base_efficiency * layer_factor * gas_factor * interference_factor
        return min(1.0, max(0.0, total_efficiency))

    def _get_layer_accessibility_factor(self, layer: AtmosphericLayer) -> float:
        """Get accessibility factor based on atmospheric layer"""
        factors = {
            AtmosphericLayer.TROPOSPHERE: 0.95,  # Most accessible
            AtmosphericLayer.STRATOSPHERE: 0.80,
            AtmosphericLayer.MESOSPHERE: 0.60,
            AtmosphericLayer.THERMOSPHERE: 0.40,
            AtmosphericLayer.EXOSPHERE: 0.20  # Least accessible
        }
        return factors.get(layer, 0.5) * self.altitude_accessibility

    def _get_gas_harvesting_factor(self, gas_type: MolecularGasType) -> float:
        """Get harvesting factor based on gas molecular properties"""
        factors = {
            MolecularGasType.N2: 0.92,  # High stability, good harvesting
            MolecularGasType.O2: 0.88,  # Good harvesting
            MolecularGasType.H2O: 0.85,  # Variable due to phase changes
            MolecularGasType.AR: 0.90,  # Inert, consistent
            MolecularGasType.CO2: 0.82,  # More complex molecule
            MolecularGasType.TRACE: 0.70  # Variable trace gases
        }
        return factors.get(gas_type, 0.75)


class AtmosphericMolecularNetwork:
    """
    Implementation of Earth's atmospheric molecular network for precision enhancement

    Manages ~10^44 atmospheric molecules across multiple layers for collective
    processing and oscillatory timing applications.
    """

    def __init__(self):
        self.atmospheric_layers: Dict[AtmosphericLayer, AtmosphericComposition] = {}
        self.harvesting_params = MolecularHarvestingParameters()
        self.harvesting_history: List[Dict] = []
        self._initialize_standard_atmosphere()

    def _initialize_standard_atmosphere(self):
        """Initialize standard atmospheric composition data"""

        # Troposphere (0-12km) - Most dense layer
        troposphere = AtmosphericComposition(
            layer=AtmosphericLayer.TROPOSPHERE,
            altitude_range=(0.0, 12.0),
            total_molecules=1.54e44,  # ~78% of total atmosphere
            gas_concentrations={
                MolecularGasType.N2: 1.20e44,    # 78% by volume
                MolecularGasType.O2: 3.23e43,    # 21% by volume
                MolecularGasType.AR: 1.43e42,    # 0.93% by volume
                MolecularGasType.CO2: 6.31e40,   # 0.04% by volume
                MolecularGasType.H2O: 7.70e41,   # Variable, ~0.5% average
                MolecularGasType.TRACE: 1.54e41  # Other trace gases
            },
            temperature=255.0,  # Average temp (K)
            pressure=50000.0,   # Average pressure (Pa)
            density=0.36        # Average density (kg/m³)
        )

        # Stratosphere (12-50km)
        stratosphere = AtmosphericComposition(
            layer=AtmosphericLayer.STRATOSPHERE,
            altitude_range=(12.0, 50.0),
            total_molecules=2.31e43,  # ~15% of total atmosphere
            gas_concentrations={
                MolecularGasType.N2: 1.80e43,
                MolecularGasType.O2: 4.85e42,
                MolecularGasType.AR: 2.14e41,
                MolecularGasType.CO2: 9.24e39,
                MolecularGasType.H2O: 2.31e40,   # Much lower water vapor
                MolecularGasType.TRACE: 2.31e40  # Including O3 (ozone)
            },
            temperature=220.0,
            pressure=5500.0,
            density=0.088
        )

        # Mesosphere (50-80km)
        mesosphere = AtmosphericComposition(
            layer=AtmosphericLayer.MESOSPHERE,
            altitude_range=(50.0, 80.0),
            total_molecules=7.70e42,  # ~5% of total atmosphere
            gas_concentrations={
                MolecularGasType.N2: 6.01e42,
                MolecularGasType.O2: 1.62e42,
                MolecularGasType.AR: 7.15e40,
                MolecularGasType.CO2: 3.08e39,
                MolecularGasType.H2O: 7.70e39,
                MolecularGasType.TRACE: 7.70e40
            },
            temperature=180.0,
            pressure=200.0,
            density=0.0020
        )

        # Thermosphere (80-600km)
        thermosphere = AtmosphericComposition(
            layer=AtmosphericLayer.THERMOSPHERE,
            altitude_range=(80.0, 600.0),
            total_molecules=1.54e42,  # ~1% of total atmosphere
            gas_concentrations={
                MolecularGasType.N2: 9.24e41,
                MolecularGasType.O2: 4.62e41,    # More atomic O than O2
                MolecularGasType.AR: 1.54e40,
                MolecularGasType.CO2: 1.54e39,
                MolecularGasType.H2O: 1.54e39,
                MolecularGasType.TRACE: 7.70e40  # He, atomic O, etc.
            },
            temperature=1000.0,  # Highly variable, can reach 2500K
            pressure=0.1,
            density=1e-6
        )

        # Exosphere (600km+)
        exosphere = AtmosphericComposition(
            layer=AtmosphericLayer.EXOSPHERE,
            altitude_range=(600.0, 10000.0),  # Extends to ~10,000km
            total_molecules=1.54e41,  # <1% of total atmosphere
            gas_concentrations={
                MolecularGasType.N2: 4.62e40,
                MolecularGasType.O2: 3.08e40,
                MolecularGasType.AR: 1.54e39,
                MolecularGasType.CO2: 1.54e38,
                MolecularGasType.H2O: 1.54e38,
                MolecularGasType.TRACE: 6.16e40  # Mostly H and He
            },
            temperature=1500.0,  # Highly variable
            pressure=1e-6,
            density=1e-12
        )

        # Store all layers
        self.atmospheric_layers = {
            AtmosphericLayer.TROPOSPHERE: troposphere,
            AtmosphericLayer.STRATOSPHERE: stratosphere,
            AtmosphericLayer.MESOSPHERE: mesosphere,
            AtmosphericLayer.THERMOSPHERE: thermosphere,
            AtmosphericLayer.EXOSPHERE: exosphere
        }

    def get_total_atmospheric_molecules(self) -> float:
        """Get total number of atmospheric molecules"""
        return sum(layer.total_molecules for layer in self.atmospheric_layers.values())

    def get_layer_composition(self, layer: AtmosphericLayer) -> Optional[AtmosphericComposition]:
        """Get atmospheric composition for specific layer"""
        return self.atmospheric_layers.get(layer)

    def calculate_harvesting_efficiency(self,
                                      target_molecules: float,
                                      layer: AtmosphericLayer,
                                      gas_type: MolecularGasType) -> Dict:
        """
        Calculate molecular harvesting efficiency for specific parameters

        Implementation of: η = (Nsensed / Ntotal) × (Pprocessing / Pmax) × (Foscillation / Fmax)
        """
        layer_data = self.atmospheric_layers.get(layer)
        if not layer_data:
            return {'error': f'Layer {layer.value} not found'}

        # Get available molecules of requested type
        available_molecules = layer_data.get_gas_count(gas_type)

        if available_molecules == 0:
            return {'error': f'No {gas_type.value} molecules in {layer.value}'}

        # Calculate harvesting ratios
        n_sensed_ratio = min(1.0, target_molecules / available_molecules)

        # Get processing and frequency ratios from harvesting efficiency
        efficiency = self.harvesting_params.calculate_efficiency(layer, gas_type)

        # Molecular frequency characteristics
        frequency_characteristics = self._get_molecular_frequency_characteristics(gas_type)
        max_frequency = max(frequency_characteristics.values())
        fundamental_frequency = frequency_characteristics['fundamental']
        f_oscillation_ratio = fundamental_frequency / max_frequency

        # Calculate total harvesting efficiency
        total_efficiency = (n_sensed_ratio *
                          efficiency *
                          f_oscillation_ratio)

        # Calculate actual harvested molecules
        harvested_molecules = available_molecules * total_efficiency

        # Calculate processing capacity harvested
        processing_capacity = self._calculate_processing_capacity(harvested_molecules, gas_type)

        result = {
            'layer': layer.value,
            'gas_type': gas_type.value,
            'target_molecules': target_molecules,
            'available_molecules': available_molecules,
            'harvested_molecules': harvested_molecules,
            'harvesting_efficiency': total_efficiency,
            'ratios': {
                'n_sensed_ratio': n_sensed_ratio,
                'processing_efficiency': efficiency,
                'frequency_ratio': f_oscillation_ratio
            },
            'processing_capacity': processing_capacity,
            'frequency_characteristics': frequency_characteristics
        }

        self.harvesting_history.append(result)
        return result

    def _get_molecular_frequency_characteristics(self, gas_type: MolecularGasType) -> Dict[str, float]:
        """Get molecular frequency characteristics for different gas types"""
        characteristics = {
            MolecularGasType.N2: {
                'fundamental': 2.36e14,  # N₂ vibrational frequency
                'vibrational': 2.36e14,
                'rotational': 4.0e12,
                'max_frequency': 2.36e14
            },
            MolecularGasType.O2: {
                'fundamental': 4.74e14,  # O₂ vibrational frequency
                'vibrational': 4.74e14,
                'rotational': 4.3e12,
                'max_frequency': 4.74e14
            },
            MolecularGasType.H2O: {
                'fundamental': 1.0e12,   # H₂O rotational (average)
                'vibrational': 1.0e14,
                'rotational': 1.0e12,
                'max_frequency': 1.0e14
            },
            MolecularGasType.AR: {
                'fundamental': 1.0e13,   # Atomic vibrations
                'vibrational': 1.0e13,
                'rotational': 0.0,       # Monatomic
                'max_frequency': 1.0e13
            },
            MolecularGasType.CO2: {
                'fundamental': 6.4e13,   # CO₂ vibrational modes
                'vibrational': 6.4e13,
                'rotational': 3.9e11,
                'max_frequency': 6.4e13
            },
            MolecularGasType.TRACE: {
                'fundamental': 5.0e13,   # Average for trace gases
                'vibrational': 5.0e13,
                'rotational': 1.0e12,
                'max_frequency': 5.0e13
            }
        }

        return characteristics.get(gas_type, {
            'fundamental': 1.0e13,
            'vibrational': 1.0e13,
            'rotational': 1.0e12,
            'max_frequency': 1.0e13
        })

    def _calculate_processing_capacity(self, molecule_count: float, gas_type: MolecularGasType) -> Dict:
        """Calculate total processing capacity from harvested molecules"""
        # Processing capacity per molecule (operations per second)
        capacity_per_molecule = {
            MolecularGasType.N2: 1.2e12,
            MolecularGasType.O2: 1.1e12,
            MolecularGasType.H2O: 8.5e11,
            MolecularGasType.AR: 5.0e11,    # Lower for monatomic
            MolecularGasType.CO2: 9.0e11,
            MolecularGasType.TRACE: 6.0e11
        }

        base_capacity = capacity_per_molecule.get(gas_type, 5.0e11)
        total_capacity = molecule_count * base_capacity

        return {
            'total_processing_capacity': total_capacity,
            'capacity_per_molecule': base_capacity,
            'equivalent_processors': total_capacity / 3e9,  # Equivalent 3GHz processors
            'quantum_time_processors': total_capacity / 1e30  # Equivalent quantum-time processors
        }

    def harvest_atmospheric_network(self,
                                  harvesting_targets: Dict[AtmosphericLayer, Dict[MolecularGasType, float]]) -> Dict:
        """
        Harvest atmospheric molecular network across multiple layers and gas types

        Args:
            harvesting_targets: Dictionary mapping layers to gas types and target molecule counts

        Returns:
            Comprehensive harvesting results
        """
        harvest_results = {}
        total_harvested = 0.0
        total_processing_capacity = 0.0

        for layer, gas_targets in harvesting_targets.items():
            layer_results = {}

            for gas_type, target_count in gas_targets.items():
                result = self.calculate_harvesting_efficiency(target_count, layer, gas_type)

                if 'error' not in result:
                    layer_results[gas_type.value] = result
                    total_harvested += result['harvested_molecules']
                    total_processing_capacity += result['processing_capacity']['total_processing_capacity']

            harvest_results[layer.value] = layer_results

        # Calculate network-wide metrics
        total_available = self.get_total_atmospheric_molecules()
        network_efficiency = total_harvested / total_available if total_available > 0 else 0.0

        summary = {
            'harvest_results': harvest_results,
            'network_summary': {
                'total_molecules_available': total_available,
                'total_molecules_harvested': total_harvested,
                'network_harvesting_efficiency': network_efficiency,
                'total_processing_capacity': total_processing_capacity,
                'equivalent_3ghz_processors': total_processing_capacity / 3e9,
                'equivalent_quantum_processors': total_processing_capacity / 1e30
            }
        }

        return summary

    def optimize_harvesting_strategy(self, target_processing_capacity: float) -> Dict:
        """
        Optimize harvesting strategy to achieve target processing capacity

        Args:
            target_processing_capacity: Desired total processing capacity (ops/sec)

        Returns:
            Optimized harvesting strategy
        """
        strategy = {}
        remaining_capacity = target_processing_capacity

        # Prioritize by accessibility and efficiency
        layer_priority = [
            AtmosphericLayer.TROPOSPHERE,
            AtmosphericLayer.STRATOSPHERE,
            AtmosphericLayer.MESOSPHERE,
            AtmosphericLayer.THERMOSPHERE,
            AtmosphericLayer.EXOSPHERE
        ]

        gas_priority = [
            MolecularGasType.N2,
            MolecularGasType.O2,
            MolecularGasType.AR,
            MolecularGasType.CO2,
            MolecularGasType.H2O,
            MolecularGasType.TRACE
        ]

        for layer in layer_priority:
            if remaining_capacity <= 0:
                break

            layer_strategy = {}
            layer_data = self.atmospheric_layers[layer]

            for gas_type in gas_priority:
                if remaining_capacity <= 0:
                    break

                available = layer_data.get_gas_count(gas_type)
                if available == 0:
                    continue

                # Calculate how many molecules needed for remaining capacity
                capacity_per_molecule = self._calculate_processing_capacity(1.0, gas_type)
                per_molecule_capacity = capacity_per_molecule['capacity_per_molecule']

                molecules_needed = min(available, remaining_capacity / per_molecule_capacity)

                if molecules_needed > 0:
                    layer_strategy[gas_type] = molecules_needed
                    remaining_capacity -= molecules_needed * per_molecule_capacity

            if layer_strategy:
                strategy[layer] = layer_strategy

        # Test the strategy
        harvest_result = self.harvest_atmospheric_network(strategy)

        return {
            'strategy': strategy,
            'predicted_capacity': target_processing_capacity - remaining_capacity,
            'capacity_shortfall': remaining_capacity,
            'harvest_simulation': harvest_result
        }

    def get_network_statistics(self) -> Dict:
        """Get comprehensive atmospheric network statistics"""
        stats = {
            'total_molecules': self.get_total_atmospheric_molecules(),
            'layer_breakdown': {},
            'gas_type_totals': {gas_type.value: 0.0 for gas_type in MolecularGasType},
            'harvesting_history_summary': self._get_harvesting_history_summary()
        }

        # Layer breakdown
        for layer, composition in self.atmospheric_layers.items():
            layer_stats = {
                'total_molecules': composition.total_molecules,
                'altitude_range': composition.altitude_range,
                'gas_breakdown': {gas_type.value: count
                                for gas_type, count in composition.gas_concentrations.items()},
                'physical_properties': {
                    'temperature': composition.temperature,
                    'pressure': composition.pressure,
                    'density': composition.density,
                    'volume': composition.get_layer_volume()
                }
            }
            stats['layer_breakdown'][layer.value] = layer_stats

            # Add to gas type totals
            for gas_type, count in composition.gas_concentrations.items():
                stats['gas_type_totals'][gas_type.value] += count

        return stats

    def _get_harvesting_history_summary(self) -> Dict:
        """Get summary of harvesting history"""
        if not self.harvesting_history:
            return {'no_history': True}

        total_harvested = sum(record['harvested_molecules'] for record in self.harvesting_history)
        avg_efficiency = np.mean([record['harvesting_efficiency'] for record in self.harvesting_history])

        return {
            'total_harvesting_operations': len(self.harvesting_history),
            'total_molecules_harvested': total_harvested,
            'average_efficiency': avg_efficiency,
            'recent_operations': self.harvesting_history[-5:] if len(self.harvesting_history) >= 5 else self.harvesting_history
        }
