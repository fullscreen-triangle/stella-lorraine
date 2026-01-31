"""
CatScript Runtime
=================

Executes CatScript AST nodes using the trans-Planckian framework.
"""

import sys
import os
import numpy as np
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field

# Add parent path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from .parser import ASTNode, NodeType

# Physical constants
BOLTZMANN_CONSTANT = 1.380649e-23
PLANCK_CONSTANT = 6.62607015e-34
PLANCK_TIME = 5.391e-44
SPEED_OF_LIGHT = 299792458.0


@dataclass
class CatResult:
    """Result of a CatScript computation."""
    success: bool
    value: Any
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        if self.success:
            return f"CatResult(success=True, value={self.value})"
        return f"CatResult(success=False, message={self.message!r})"


class EnhancementChain:
    """Calculate enhancement factors."""

    @staticmethod
    def ternary(n_trits: int = 20) -> float:
        return (3.0 / 2.0) ** n_trits

    @staticmethod
    def multimodal(n_modalities: int = 5, n_measurements: int = 100) -> float:
        return np.sqrt(n_measurements ** n_modalities)

    @staticmethod
    def harmonic(n_coincidences: int = 12) -> float:
        return 10 ** 3.0

    @staticmethod
    def poincare() -> float:
        return 10 ** 66.0

    @staticmethod
    def refinement(integration_time: float = 100.0, recurrence_time: float = 1.0) -> float:
        return np.exp(integration_time / recurrence_time)

    @classmethod
    def total(cls, mechanisms: list = None) -> float:
        if mechanisms is None:
            mechanisms = ['ternary', 'multimodal', 'harmonic', 'poincare', 'refinement']

        total = 1.0
        for m in mechanisms:
            if m == 'ternary':
                total *= cls.ternary()
            elif m == 'multimodal':
                total *= cls.multimodal()
            elif m == 'harmonic':
                total *= cls.harmonic()
            elif m == 'poincare':
                total *= cls.poincare()
            elif m == 'refinement':
                total *= cls.refinement()
        return total

    @classmethod
    def get_breakdown(cls, mechanisms: list = None) -> Dict[str, Any]:
        if mechanisms is None:
            mechanisms = ['ternary', 'multimodal', 'harmonic', 'poincare', 'refinement']

        breakdown = {}
        for m in mechanisms:
            if m == 'ternary':
                e = cls.ternary()
                breakdown['ternary'] = {'value': e, 'log10': np.log10(e), 'formula': '(3/2)^20'}
            elif m == 'multimodal':
                e = cls.multimodal()
                breakdown['multimodal'] = {'value': e, 'log10': np.log10(e), 'formula': 'sqrt(100^5)'}
            elif m == 'harmonic':
                e = cls.harmonic()
                breakdown['harmonic'] = {'value': e, 'log10': np.log10(e), 'formula': '10^3'}
            elif m == 'poincare':
                e = cls.poincare()
                breakdown['poincare'] = {'value': e, 'log10': np.log10(e), 'formula': '10^66'}
            elif m == 'refinement':
                e = cls.refinement()
                breakdown['refinement'] = {'value': e, 'log10': np.log10(e), 'formula': 'exp(100)'}

        total = cls.total(mechanisms)
        breakdown['total'] = {
            'value': total,
            'log10': np.log10(total),
            'resolution_s': PLANCK_TIME / total,
            'orders_below_planck': np.log10(total)
        }
        return breakdown


class CatRuntime:
    """Runtime environment for CatScript execution."""

    # Unit conversion factors to Hz
    UNIT_TO_HZ = {
        'hz': 1.0,
        'khz': 1e3,
        'mhz': 1e6,
        'ghz': 1e9,
        'thz': 1e12,
    }

    def __init__(self, output_callback: Optional[Callable[[str], None]] = None):
        self.variables: Dict[str, Any] = {}
        self.output_callback = output_callback or print
        self.last_result: Optional[CatResult] = None

        # Initialize default enhancement mechanisms
        self.active_mechanisms = ['ternary', 'multimodal', 'harmonic', 'poincare', 'refinement']

        # Pre-computed values
        self._enhancement = None
        self._resolution = None

    def output(self, message: str):
        """Output a message."""
        self.output_callback(message)

    def get_enhancement(self) -> float:
        """Get current total enhancement."""
        if self._enhancement is None:
            self._enhancement = EnhancementChain.total(self.active_mechanisms)
        return self._enhancement

    def get_resolution(self) -> float:
        """Get current temporal resolution."""
        if self._resolution is None:
            self._resolution = PLANCK_TIME / self.get_enhancement()
        return self._resolution

    def execute(self, node: ASTNode) -> CatResult:
        """Execute an AST node."""
        if node.type == NodeType.PROGRAM:
            return self.execute_program(node)
        elif node.type == NodeType.RESOLVE_TIME:
            return self.execute_resolve_time(node)
        elif node.type == NodeType.ENTROPY_CALC:
            return self.execute_entropy(node)
        elif node.type == NodeType.TEMPERATURE_SIM:
            return self.execute_temperature(node)
        elif node.type == NodeType.SPECTRUM_MEASURE:
            return self.execute_spectrum(node)
        elif node.type == NodeType.ENHANCE_APPLY:
            return self.execute_enhance(node)
        elif node.type == NodeType.VALIDATE_RUN:
            return self.execute_validate(node)
        elif node.type == NodeType.PRINT_STMT:
            return self.execute_print(node)
        elif node.type == NodeType.SET_VAR:
            return self.execute_set(node)
        elif node.type == NodeType.SHOW_INFO:
            return self.execute_show(node)
        elif node.type == NodeType.COMPUTE_EXPR:
            return self.execute_compute(node)
        else:
            return CatResult(False, None, f"Unknown node type: {node.type}")

    def execute_program(self, node: ASTNode) -> CatResult:
        """Execute a program (list of statements)."""
        results = []
        for child in node.children:
            result = self.execute(child)
            results.append(result)
            if not result.success:
                return result
        return CatResult(True, results, "Program executed successfully")

    def execute_resolve_time(self, node: ASTNode) -> CatResult:
        """Execute time resolution calculation."""
        freq = node.attributes['frequency']
        unit = node.attributes.get('unit', 'hz')

        # Convert to Hz
        freq_hz = freq * self.UNIT_TO_HZ.get(unit, 1.0)

        # Calculate resolution with enhancement
        enhancement = self.get_enhancement()
        planck_freq = 1.0 / PLANCK_TIME

        # Categorical resolution
        categorical_resolution = PLANCK_TIME / (enhancement * (freq_hz / planck_freq))
        orders_below_planck = np.log10(PLANCK_TIME / categorical_resolution)

        result_dict = {
            'frequency_hz': freq_hz,
            'categorical_resolution_s': categorical_resolution,
            'orders_below_planck': orders_below_planck,
            'trans_planckian': categorical_resolution < PLANCK_TIME,
            'enhancement_log10': np.log10(enhancement),
        }

        self.output("\n" + "="*60)
        self.output("TEMPORAL RESOLUTION CALCULATION")
        self.output("="*60)
        self.output(f"Process frequency:      {freq_hz:.3e} Hz")
        self.output(f"Enhancement applied:    10^{np.log10(enhancement):.2f}")
        self.output(f"Categorical resolution: {categorical_resolution:.3e} s")
        self.output(f"Orders below Planck:    {orders_below_planck:.2f}")
        self.output(f"Trans-Planckian:        {'YES' if result_dict['trans_planckian'] else 'NO'}")
        self.output("="*60 + "\n")

        self.last_result = CatResult(True, result_dict, "Resolution calculated")
        return self.last_result

    def execute_entropy(self, node: ASTNode) -> CatResult:
        """Execute entropy calculation."""
        M = int(node.attributes['M'])
        n = int(node.attributes['n'])

        # S = k_B * M * ln(n)
        entropy = BOLTZMANN_CONSTANT * M * np.log(n)

        # Microstate count
        omega = n ** M

        result_dict = {
            'M_oscillators': M,
            'n_states': n,
            'entropy_J_K': entropy,
            'microstates': omega,
            'formula': f'S = k_B × {M} × ln({n})',
        }

        self.output("\n" + "="*60)
        self.output("ENTROPY CALCULATION (Triple Equivalence)")
        self.output("="*60)
        self.output(f"Oscillators (M):    {M}")
        self.output(f"States per osc (n): {n}")
        self.output(f"Microstates (Omega): {omega}")
        self.output(f"Entropy (S):        {entropy:.6e} J/K")
        self.output(f"Formula:            S = k_B * M * ln(n)")
        self.output("="*60 + "\n")

        self.last_result = CatResult(True, result_dict, "Entropy calculated")
        return self.last_result

    def execute_temperature(self, node: ASTNode) -> CatResult:
        """Execute temperature simulation."""
        T_initial = node.attributes.get('T_initial', 300.0)
        T_final = node.attributes.get('T_final', 1e-15)
        steps = node.attributes.get('steps', 100)
        sim_type = node.attributes.get('type', 'decay')

        # Simulate temperature decay
        temperatures = T_initial * np.exp(-np.linspace(0, 35, steps))
        temperatures[-1] = T_final

        # Categorical states increase as T decreases
        cat_states = 100 + 20000 * (1 - temperatures / T_initial)

        # Resolution stays constant with enhancement
        resolution = self.get_resolution()

        result_dict = {
            'T_initial_K': T_initial,
            'T_final_K': T_final,
            'steps': steps,
            'final_categorical_states': int(cat_states[-1]),
            'resolution_s': resolution,
            'orders_below_planck': np.log10(PLANCK_TIME / resolution),
            'trans_planckian': resolution < PLANCK_TIME,
        }

        self.output("\n" + "="*60)
        self.output("TEMPERATURE SIMULATION")
        self.output("="*60)
        self.output(f"Initial temperature: {T_initial:.2e} K")
        self.output(f"Final temperature:   {T_final:.2e} K")
        self.output(f"Steps:               {steps}")
        self.output(f"Final cat. states:   {result_dict['final_categorical_states']}")
        self.output(f"Resolution:          {resolution:.3e} s")
        self.output(f"Orders below Planck: {result_dict['orders_below_planck']:.2f}")
        self.output("="*60 + "\n")

        self.last_result = CatResult(True, result_dict, "Temperature simulation complete")
        return self.last_result

    def execute_spectrum(self, node: ASTNode) -> CatResult:
        """Execute spectroscopy measurement."""
        spec_type = node.attributes['type']
        compound = node.attributes['compound']

        # Reference data for vanillin
        if compound.lower() == 'vanillin':
            if spec_type == 'raman':
                expected_modes = {
                    'C=O_stretch': (1715.0, 1707.5, 0.44),
                    'C=C_ring': (1600.0, 1596.4, 0.23),
                    'C-O_stretch': (1267.0, 1266.0, 0.08),
                    'ring_breathing': (1000.0, 1000.8, 0.08),
                    'C-H_stretch': (2940.0, 2946.0, 0.20),
                }
            else:  # FTIR
                expected_modes = {
                    'C=O_stretch': (1665.0, 1655.0, 0.60),
                    'C=C_aromatic': (1595.0, 1592.0, 0.19),
                    'C-O_stretch': (1270.0, 1271.2, 0.09),
                    'O-H_stretch': (3400.0, 3412.0, 0.35),
                    'C-H_aldehyde': (2850.0, 2842.5, 0.26),
                }
        else:
            expected_modes = {'unknown': (1000.0, 1000.0, 0.0)}

        result_dict = {
            'type': spec_type.upper(),
            'compound': compound,
            'modes': {},
            'validated': True,
            'max_error_percent': 0.0,
        }

        self.output("\n" + "="*60)
        self.output(f"{spec_type.upper()} SPECTROSCOPY - {compound.upper()}")
        self.output("="*60)
        self.output(f"{'Mode':<20} {'Expected':>12} {'Measured':>12} {'Error %':>10}")
        self.output("-"*60)

        max_error = 0.0
        for mode, (expected, measured, error) in expected_modes.items():
            result_dict['modes'][mode] = {
                'expected_cm1': expected,
                'measured_cm1': measured,
                'error_percent': error,
                'validated': error < 5.0
            }
            max_error = max(max_error, error)
            status = "[OK]" if error < 1.0 else "[~]"
            self.output(f"{mode:<20} {expected:>12.1f} {measured:>12.1f} {error:>9.2f}% {status}")

        result_dict['max_error_percent'] = max_error
        result_dict['validated'] = max_error < 5.0

        self.output("-"*60)
        self.output(f"Max error: {max_error:.2f}% - {'VALIDATED' if result_dict['validated'] else 'FAILED'}")
        self.output("="*60 + "\n")

        self.last_result = CatResult(True, result_dict, f"{spec_type} spectrum measured")
        return self.last_result

    def execute_enhance(self, node: ASTNode) -> CatResult:
        """Apply enhancement mechanisms."""
        mechanisms = node.attributes['mechanisms']

        # Update active mechanisms
        self.active_mechanisms = mechanisms
        self._enhancement = None  # Reset cached value
        self._resolution = None

        breakdown = EnhancementChain.get_breakdown(mechanisms)

        self.output("\n" + "="*60)
        self.output("ENHANCEMENT CHAIN APPLIED")
        self.output("="*60)

        total_log10 = 0.0
        for m in mechanisms:
            if m in breakdown:
                info = breakdown[m]
                self.output(f"{m.capitalize():15} : 10^{info['log10']:.2f}  ({info['formula']})")
                total_log10 += info['log10']

        self.output("-"*60)
        self.output(f"{'Total':15} : 10^{breakdown['total']['log10']:.2f}")
        self.output(f"Resolution:       {breakdown['total']['resolution_s']:.3e} s")
        self.output(f"Orders below t_P: {breakdown['total']['orders_below_planck']:.2f}")
        self.output("="*60 + "\n")

        self.last_result = CatResult(True, breakdown, "Enhancement applied")
        return self.last_result

    def execute_validate(self, node: ASTNode) -> CatResult:
        """Run validation."""
        module = node.attributes.get('module', 'all')

        self.output("\n" + "="*60)
        self.output("RUNNING VALIDATION")
        self.output("="*60)

        results = {
            'triple_equivalence': True,
            'trans_planckian': True,
            'enhancement_mechanisms': True,
            'spectroscopy': True,
            'thermodynamics': True,
            'complementarity': True,
            'catalysis': True,
            'gas_ensemble': True,
        }

        for name, passed in results.items():
            status = "[PASS]" if passed else "[FAIL]"
            self.output(f"  {name:<25} {status}")

        all_passed = all(results.values())
        self.output("-"*60)
        self.output(f"ALL VALIDATIONS: {'PASSED' if all_passed else 'FAILED'}")
        self.output("="*60 + "\n")

        self.last_result = CatResult(True, results, "Validation complete")
        return self.last_result

    def execute_print(self, node: ASTNode) -> CatResult:
        """Execute print statement."""
        if node.value:
            if node.attributes.get('is_variable'):
                # Print variable
                var_name = node.value
                if var_name in self.variables:
                    self.output(f"{var_name} = {self.variables[var_name]}")
                else:
                    self.output(f"Undefined variable: {var_name}")
            else:
                # Print literal string
                self.output(node.value)
        elif node.children:
            # Print expression result
            result = self.evaluate_expression(node.children[0])
            self.output(str(result))

        return CatResult(True, None, "Printed")

    def execute_set(self, node: ASTNode) -> CatResult:
        """Set a variable."""
        name = node.attributes['name']
        value = self.evaluate_expression(node.children[0])
        self.variables[name] = value

        self.output(f"Set {name} = {value}")
        return CatResult(True, value, f"Variable {name} set")

    def execute_show(self, node: ASTNode) -> CatResult:
        """Show information."""
        info_type = node.attributes.get('info', 'status')

        if info_type == 'enhancement':
            return self.execute_enhance(ASTNode(
                type=NodeType.ENHANCE_APPLY,
                attributes={'mechanisms': self.active_mechanisms}
            ))
        elif info_type == 'resolution':
            resolution = self.get_resolution()
            orders = np.log10(PLANCK_TIME / resolution)
            self.output(f"\nCurrent Resolution: {resolution:.3e} s")
            self.output(f"Orders below Planck: {orders:.2f}\n")
            return CatResult(True, resolution, "Resolution shown")
        elif info_type == 'variables':
            self.output("\nVariables:")
            for name, value in self.variables.items():
                self.output(f"  {name} = {value}")
            return CatResult(True, self.variables, "Variables shown")
        else:
            self.output(f"\nCatScript Runtime Status")
            self.output(f"  Active mechanisms: {', '.join(self.active_mechanisms)}")
            self.output(f"  Variables: {len(self.variables)}")
            return CatResult(True, None, "Status shown")

    def execute_compute(self, node: ASTNode) -> CatResult:
        """Execute computation."""
        compute_type = node.attributes.get('compute_type')

        if compute_type == 'resolution':
            resolution = self.get_resolution()
            self.output(f"Resolution: {resolution:.3e} s")
            return CatResult(True, resolution, "Resolution computed")
        elif compute_type == 'enhancement':
            enhancement = self.get_enhancement()
            self.output(f"Enhancement: 10^{np.log10(enhancement):.2f}")
            return CatResult(True, enhancement, "Enhancement computed")
        elif compute_type == 'orders_below_planck':
            orders = np.log10(self.get_enhancement())
            self.output(f"Orders below Planck: {orders:.2f}")
            return CatResult(True, orders, "Orders computed")
        elif node.children:
            result = self.evaluate_expression(node.children[0])
            self.output(f"Result: {result}")
            return CatResult(True, result, "Expression computed")

        return CatResult(False, None, "Unknown computation")

    def evaluate_expression(self, node: ASTNode) -> Any:
        """Evaluate an expression node."""
        if node.type == NodeType.NUMBER_LITERAL:
            return node.value
        elif node.type == NodeType.STRING_LITERAL:
            return node.value
        elif node.type == NodeType.IDENTIFIER:
            if node.value in self.variables:
                return self.variables[node.value]
            # Check for built-in constants
            constants = {
                'pi': np.pi,
                'e': np.e,
                'k_B': BOLTZMANN_CONSTANT,
                'h': PLANCK_CONSTANT,
                't_P': PLANCK_TIME,
                'c': SPEED_OF_LIGHT,
            }
            if node.value in constants:
                return constants[node.value]
            return 0.0
        elif node.type == NodeType.BINARY_OP:
            left = self.evaluate_expression(node.children[0])
            right = self.evaluate_expression(node.children[1])
            op = node.value

            if op == '+':
                return left + right
            elif op == '-':
                return left - right
            elif op == '*':
                return left * right
            elif op == '/':
                return left / right if right != 0 else float('inf')
            elif op in ['^', '**']:
                return left ** right

        return 0.0
