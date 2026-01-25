"""
Dual-Membrane as Electrical Circuit
====================================

The dual-membrane pixel demon can be represented as an electrical circuit where:
- Front face = Observable circuit components (resistors, capacitors, etc.)
- Back face = Hidden conjugate components
- The circuit is ALWAYS complete and balanced (Kirchhoff's laws satisfied)
- You can only measure one set of components at a time

Key Concepts:
- Voltage (V) = S-entropy potential difference
- Current (I) = Information flow rate
- Resistance (R) = Categorical impedance
- Capacitance (C) = Categorical memory/storage
- Inductance (L) = Categorical inertia

The circuit is balanced through complementary components:
- Front resistor R_f ←→ Back resistor R_b (conjugate relationship)
- Front voltage V_f ←→ Back voltage V_b (conjugate)
- Current is conserved (Kirchhoff's current law)
- Voltage drops sum to zero around loops (Kirchhoff's voltage law)

Author: Kundai Sachikonye
Date: 2024
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import logging

from dual_membrane_pixel_demon import (
    DualMembranePixelDemon,
    MembraneFace,
    ConjugateTransform,
    SEntropyCoordinates
)

logger = logging.getLogger(__name__)


class CircuitElement(Enum):
    """Types of circuit elements"""
    RESISTOR = "resistor"
    CAPACITOR = "capacitor"
    INDUCTOR = "inductor"
    VOLTAGE_SOURCE = "voltage_source"
    CURRENT_SOURCE = "current_source"


@dataclass
class CircuitComponent:
    """
    A circuit component with front and back (conjugate) values

    The circuit is always balanced: front and back components
    are conjugate pairs that ensure Kirchhoff's laws are satisfied.
    """
    element_type: CircuitElement
    node_a: str  # Connection point A
    node_b: str  # Connection point B

    # Front face values (observable)
    front_value: float  # R, C, L, V, or I
    front_voltage: float = 0.0  # Voltage across component
    front_current: float = 0.0  # Current through component

    # Back face values (hidden, conjugate)
    back_value: float = 0.0
    back_voltage: float = 0.0
    back_current: float = 0.0

    # Component metadata
    s_state: Optional[SEntropyCoordinates] = None

    def __post_init__(self):
        """Initialize back values as conjugate of front"""
        if self.back_value == 0.0:
            # For phase conjugate: back = -front
            # For resistors/impedances: back = 1/front (dual)
            if self.element_type in [CircuitElement.RESISTOR, CircuitElement.CAPACITOR, CircuitElement.INDUCTOR]:
                self.back_value = 1.0 / self.front_value if self.front_value != 0 else np.inf
            else:
                self.back_value = -self.front_value

    def apply_kirchhoff_current_law(self, node_currents: Dict[str, float]) -> bool:
        """
        Verify Kirchhoff's current law at nodes
        Sum of currents entering node = Sum of currents leaving node
        """
        # Current conservation at node_a
        node_a_sum = node_currents.get(self.node_a, 0.0)
        # Current conservation at node_b
        node_b_sum = node_currents.get(self.node_b, 0.0)

        # For complete circuit with front and back, total current is conserved
        total_current = self.front_current + self.back_current

        return abs(total_current) < 1e-6  # Should sum to zero in balanced circuit

    def apply_kirchhoff_voltage_law(self, loop_voltages: List[float]) -> bool:
        """
        Verify Kirchhoff's voltage law around loop
        Sum of voltage drops around closed loop = 0
        """
        # In complete circuit, front and back voltages must balance
        total_voltage = self.front_voltage + self.back_voltage

        return abs(total_voltage) < 1e-6  # Should sum to zero

    def get_observable_values(self, face: MembraneFace) -> Dict[str, float]:
        """Get observable values for specified face"""
        if face == MembraneFace.FRONT:
            return {
                'value': self.front_value,
                'voltage': self.front_voltage,
                'current': self.front_current
            }
        else:
            return {
                'value': self.back_value,
                'voltage': self.back_voltage,
                'current': self.back_current
            }

    def to_dict(self) -> Dict[str, Any]:
        return {
            'element_type': self.element_type.value,
            'nodes': [self.node_a, self.node_b],
            'front': {
                'value': self.front_value,
                'voltage': self.front_voltage,
                'current': self.front_current
            },
            'back': {
                'value': self.back_value,
                'voltage': self.back_voltage,
                'current': self.back_current
            }
        }


class DualMembraneCircuit:
    """
    Dual-membrane represented as an electrical circuit

    The circuit has two faces:
    - Front face: Observable components
    - Back face: Hidden conjugate components

    The circuit is ALWAYS complete and balanced, but you can only
    observe one face at a time (complementarity).
    """

    def __init__(
        self,
        pixel_demon: Optional[DualMembranePixelDemon] = None,
        name: str = "dual_membrane_circuit"
    ):
        """
        Initialize circuit from dual-membrane pixel demon

        Each molecular demon → circuit component
        S-entropy state → voltage/current state
        """
        self.name = name
        self.pixel_demon = pixel_demon

        # Circuit topology
        self.components: List[CircuitComponent] = []
        self.nodes: Dict[str, Dict[str, float]] = {}  # node_id → {voltage, current}

        # Observable face
        self.observable_face = MembraneFace.FRONT

        # Circuit state
        self.ground_node = "ground"
        self.is_balanced = False

        if pixel_demon:
            self._construct_circuit_from_pixel_demon()

        logger.info(f"Created DualMembraneCircuit '{name}' with {len(self.components)} components")

    def _construct_circuit_from_pixel_demon(self):
        """
        Construct circuit from pixel demon's molecular demon lattice

        Each molecular demon becomes a circuit component:
        - Number density → Resistance (1/density)
        - S_k (knowledge entropy) → Voltage
        - S_t (temporal entropy) → Current
        - S_e (evolution entropy) → Capacitance
        """
        if not self.pixel_demon:
            return

        # Create ground node
        self.nodes[self.ground_node] = {'voltage': 0.0, 'current': 0.0}

        # Create component for each molecular demon
        for i, (molecule, demon) in enumerate(self.pixel_demon.front_demons.items()):
            # Node names
            node_a = self.ground_node
            node_b = f"node_{molecule}_{i}"

            # Create node
            self.nodes[node_b] = {
                'voltage': demon.s_state.S_k,  # S_k → Voltage
                'current': demon.s_state.S_t   # S_t → Current
            }

            # Resistance from number density
            resistance = 1.0 / demon.number_density if demon.number_density > 0 else np.inf
            resistance *= 1e20  # Scale to reasonable values

            # Create resistor component
            component = CircuitComponent(
                element_type=CircuitElement.RESISTOR,
                node_a=node_a,
                node_b=node_b,
                front_value=resistance,
                front_voltage=demon.s_state.S_k,
                front_current=demon.s_state.S_t,
                s_state=demon.s_state
            )

            # Set back values from back demon
            if molecule in self.pixel_demon.back_demons:
                back_demon = self.pixel_demon.back_demons[molecule]
                back_resistance = 1.0 / back_demon.number_density if back_demon.number_density > 0 else np.inf
                back_resistance *= 1e20

                component.back_value = back_resistance
                component.back_voltage = back_demon.s_state.S_k
                component.back_current = back_demon.s_state.S_t

            self.components.append(component)

        logger.info(f"Constructed circuit with {len(self.components)} components from pixel demon")

    def measure_observable_circuit(self) -> Dict[str, Any]:
        """
        Measure the observable face of the circuit

        CRITICAL: Like ammeter vs voltmeter, you can only DIRECTLY measure
        one face. The other face must be DERIVED from the conjugate relationship.

        - Observable face = Direct measurement (like ammeter measuring current)
        - Hidden face = Cannot be directly measured (like voltage when ammeter is connected)
        - You can CALCULATE the hidden face, but not MEASURE it

        Attempting to measure both simultaneously would be like putting an
        ammeter and voltmeter in series - the measurement apparatus itself
        determines what you can observe.
        """
        measurements = {
            'observable_face': self.observable_face.value,
            'measurement_type': 'DIRECT',  # This is a direct measurement
            'num_components': len(self.components),
            'components': [],
            'total_resistance': 0.0,
            'total_voltage': 0.0,
            'total_current': 0.0,
            'is_balanced': self.is_balanced,
            'hidden_face_accessible': False,  # Cannot directly measure hidden face
            'hidden_face_note': 'Use derive_hidden_face() to calculate from conjugate relationship'
        }

        for component in self.components:
            obs_values = component.get_observable_values(self.observable_face)
            measurements['components'].append({
                'type': component.element_type.value,
                'nodes': [component.node_a, component.node_b],
                'value': obs_values['value'],
                'voltage': obs_values['voltage'],
                'current': obs_values['current']
            })

            if component.element_type == CircuitElement.RESISTOR:
                measurements['total_resistance'] += obs_values['value']
            measurements['total_voltage'] += obs_values['voltage']
            measurements['total_current'] += obs_values['current']

        return measurements

    def derive_hidden_face(self) -> Dict[str, Any]:
        """
        DERIVE (not measure) the hidden face from the observable face

        Like using Ohm's law (V = IR) to calculate voltage when you've
        measured current with an ammeter:
        - You MEASURED current (observable face)
        - You CALCULATE voltage (hidden face) from V = IR
        - You didn't MEASURE voltage directly

        The hidden face is calculated from the conjugate transformation,
        just as voltage is calculated from current and resistance.
        """
        if self.observable_face == MembraneFace.FRONT:
            hidden_face = MembraneFace.BACK
        else:
            hidden_face = MembraneFace.FRONT

        derived = {
            'hidden_face': hidden_face.value,
            'measurement_type': 'DERIVED',  # This is calculated, not measured
            'derived_from': self.observable_face.value,
            'components': [],
            'total_resistance': 0.0,
            'total_voltage': 0.0,
            'total_current': 0.0,
            'derivation_note': 'Calculated from conjugate relationship (like V=IR)'
        }

        for component in self.components:
            hidden_values = component.get_observable_values(hidden_face)
            derived['components'].append({
                'type': component.element_type.value,
                'nodes': [component.node_a, component.node_b],
                'value': hidden_values['value'],
                'voltage': hidden_values['voltage'],
                'current': hidden_values['current'],
                'note': 'DERIVED from observable face using conjugate transform'
            })

            if component.element_type == CircuitElement.RESISTOR:
                derived['total_resistance'] += hidden_values['value']
            derived['total_voltage'] += hidden_values['voltage']
            derived['total_current'] += hidden_values['current']

        logger.warning(
            f"Deriving hidden face '{hidden_face.value}' from observable face '{self.observable_face.value}'. "
            f"Like calculating V from I*R - this is calculation, not direct measurement."
        )

        return derived

    def verify_kirchhoff_laws(self) -> Dict[str, Any]:
        """
        Verify that circuit satisfies Kirchhoff's laws

        Even though we can only observe one face, the complete circuit
        (front + back) must satisfy:
        - Kirchhoff's Current Law (KCL): ΣI = 0 at each node
        - Kirchhoff's Voltage Law (KVL): ΣV = 0 around each loop
        """
        # Calculate node currents
        node_currents = {}
        for node in self.nodes:
            node_currents[node] = 0.0

        for component in self.components:
            # Current entering node_a
            node_currents[component.node_a] -= (component.front_current + component.back_current)
            # Current leaving node_b
            node_currents[component.node_b] += (component.front_current + component.back_current)

        # Check KCL at each node
        kcl_satisfied = all(abs(current) < 1e-6 for current in node_currents.values())

        # Calculate loop voltages (simple series circuit)
        loop_voltages = [comp.front_voltage + comp.back_voltage for comp in self.components]
        total_loop_voltage = sum(loop_voltages)

        # Check KVL
        kvl_satisfied = abs(total_loop_voltage) < 1e-6

        self.is_balanced = kcl_satisfied and kvl_satisfied

        return {
            'kcl_satisfied': kcl_satisfied,
            'kvl_satisfied': kvl_satisfied,
            'is_balanced': self.is_balanced,
            'node_currents': node_currents,
            'total_loop_voltage': total_loop_voltage,
            'max_current_imbalance': max(abs(i) for i in node_currents.values()),
            'voltage_imbalance': abs(total_loop_voltage)
        }

    def attempt_simultaneous_measurement(self) -> Dict[str, Any]:
        """
        Attempt to measure BOTH faces simultaneously

        THIS SHOULD FAIL - just like trying to use ammeter and voltmeter in series.

        The measurement apparatus (which face you're observing) determines
        what you can measure. You cannot directly measure both.

        Returns error/warning indicating measurement incompatibility.
        """
        logger.error(
            "MEASUREMENT ERROR: Cannot measure both faces simultaneously!\n"
            "This is like trying to connect ammeter and voltmeter in series:\n"
            "  - Ammeter (low impedance, series) measures CURRENT\n"
            "  - Voltmeter (high impedance, parallel) measures VOLTAGE\n"
            "  - You can only have ONE measurement apparatus connected\n"
            "  - Observable face = Your measurement apparatus\n"
            "  - To 'see' the other face, you must switch apparatus (change measurement)"
        )

        return {
            'error': 'MEASUREMENT_INCOMPATIBILITY',
            'message': 'Cannot directly measure both faces simultaneously',
            'observable_face': self.observable_face.value,
            'hidden_face': 'BACK' if self.observable_face == MembraneFace.FRONT else 'FRONT',
            'resolution': 'Use measure_observable_circuit() for direct measurement, derive_hidden_face() for calculation',
            'analogy': {
                'observable_face': 'Like ammeter measuring current (direct)',
                'hidden_face': 'Like voltage (must calculate from V=IR)',
                'constraint': 'Measurement apparatus determines observable'
            }
        }

    def switch_observable_face(self):
        """
        Switch which face of the circuit is observable

        Like switching from ammeter to voltmeter:
        - Remove ammeter (stop measuring current/front face)
        - Connect voltmeter (start measuring voltage/back face)

        The circuit remains balanced, but your MEASUREMENT APPARATUS
        has changed, so you now measure different quantities.
        """
        old_face = self.observable_face
        self.observable_face = (
            MembraneFace.BACK if self.observable_face == MembraneFace.FRONT
            else MembraneFace.FRONT
        )

        logger.info(
            f"Switched measurement apparatus: {old_face.value} → {self.observable_face.value}\n"
            f"(Like switching from ammeter to voltmeter)"
        )

    def calculate_power_dissipation(self) -> Dict[str, float]:
        """
        Calculate power dissipation in circuit

        P = I²R for resistors
        P = VI for sources

        Total power from both faces must balance (energy conservation).
        """
        front_power = 0.0
        back_power = 0.0

        for component in self.components:
            if component.element_type == CircuitElement.RESISTOR:
                # P = I²R
                front_power += component.front_current**2 * component.front_value
                back_power += component.back_current**2 * component.back_value
            else:
                # P = VI
                front_power += component.front_voltage * component.front_current
                back_power += component.back_voltage * component.back_current

        return {
            'front_power': front_power,
            'back_power': back_power,
            'total_power': front_power + back_power,
            'power_balanced': abs(front_power + back_power) < 1e-6
        }

    def get_circuit_impedance(self, face: MembraneFace) -> complex:
        """
        Calculate total circuit impedance for specified face

        Z = R + jX where X depends on capacitors/inductors
        """
        total_impedance = 0.0 + 0.0j

        for component in self.components:
            if face == MembraneFace.FRONT:
                value = component.front_value
            else:
                value = component.back_value

            if component.element_type == CircuitElement.RESISTOR:
                total_impedance += value  # Real resistance
            elif component.element_type == CircuitElement.CAPACITOR:
                # Capacitive reactance (imaginary, negative)
                total_impedance += -1j * value
            elif component.element_type == CircuitElement.INDUCTOR:
                # Inductive reactance (imaginary, positive)
                total_impedance += 1j * value

        return total_impedance

    def to_circuit_diagram(self) -> str:
        """
        Generate ASCII circuit diagram

        Shows observable face only (hidden face indicated by [...])
        """
        lines = []
        lines.append(f"Circuit: {self.name} (Observable: {self.observable_face.value})")
        lines.append("=" * 60)

        for component in self.components:
            obs_vals = component.get_observable_values(self.observable_face)

            # Circuit element symbol
            if component.element_type == CircuitElement.RESISTOR:
                symbol = "---[R]---"
            elif component.element_type == CircuitElement.CAPACITOR:
                symbol = "---||-|---"
            elif component.element_type == CircuitElement.INDUCTOR:
                symbol = "---((()))---"
            elif component.element_type == CircuitElement.VOLTAGE_SOURCE:
                symbol = "---( + )---"
            else:
                symbol = "---[?]---"

            lines.append(f"{component.node_a} {symbol} {component.node_b}")
            lines.append(f"  Value: {obs_vals['value']:.2e}")
            lines.append(f"  V: {obs_vals['voltage']:.3f}, I: {obs_vals['current']:.3e}")
            lines.append(f"  [Hidden face: {...}]")
            lines.append("")

        lines.append("=" * 60)

        # Add Kirchhoff verification
        kirchhoff = self.verify_kirchhoff_laws()
        lines.append(f"KCL satisfied: {kirchhoff['kcl_satisfied']}")
        lines.append(f"KVL satisfied: {kirchhoff['kvl_satisfied']}")
        lines.append(f"Circuit balanced: {kirchhoff['is_balanced']}")

        return "\n".join(lines)

    def export_spice_netlist(self, face: MembraneFace) -> str:
        """
        Export circuit as SPICE netlist for specified face

        Can only export one face at a time (complementarity).
        """
        lines = []
        lines.append(f"* Dual-Membrane Circuit: {self.name}")
        lines.append(f"* Face: {face.value}")
        lines.append(f"* Generated: {np.datetime64('now')}")
        lines.append("")

        for i, component in enumerate(self.components):
            obs_vals = component.get_observable_values(face)

            # SPICE component line
            if component.element_type == CircuitElement.RESISTOR:
                lines.append(f"R{i} {component.node_a} {component.node_b} {obs_vals['value']:.6e}")
            elif component.element_type == CircuitElement.CAPACITOR:
                lines.append(f"C{i} {component.node_a} {component.node_b} {obs_vals['value']:.6e}")
            elif component.element_type == CircuitElement.INDUCTOR:
                lines.append(f"L{i} {component.node_a} {component.node_b} {obs_vals['value']:.6e}")
            elif component.element_type == CircuitElement.VOLTAGE_SOURCE:
                lines.append(f"V{i} {component.node_a} {component.node_b} DC {obs_vals['value']:.6e}")

        lines.append("")
        lines.append(".end")

        return "\n".join(lines)


def create_circuit_from_s_coordinates(
    s_k: float,
    s_t: float,
    s_e: float,
    name: str = "s_entropy_circuit"
) -> DualMembraneCircuit:
    """
    Create a simple circuit directly from S-entropy coordinates

    S_k → Voltage source
    S_t → Current source
    S_e → Capacitor (stores evolutionary state)
    """
    circuit = DualMembraneCircuit(name=name)

    # Voltage source from S_k
    v_source = CircuitComponent(
        element_type=CircuitElement.VOLTAGE_SOURCE,
        node_a=circuit.ground_node,
        node_b="node_sk",
        front_value=s_k,
        front_voltage=s_k,
        front_current=0.0
    )
    circuit.components.append(v_source)

    # Current source from S_t
    i_source = CircuitComponent(
        element_type=CircuitElement.CURRENT_SOURCE,
        node_a="node_sk",
        node_b="node_st",
        front_value=s_t,
        front_voltage=0.0,
        front_current=s_t
    )
    circuit.components.append(i_source)

    # Capacitor from S_e
    capacitor = CircuitComponent(
        element_type=CircuitElement.CAPACITOR,
        node_a="node_st",
        node_b=circuit.ground_node,
        front_value=s_e,
        front_voltage=s_k,  # Voltage across capacitor
        front_current=s_t   # Current charging capacitor
    )
    circuit.components.append(capacitor)

    return circuit
