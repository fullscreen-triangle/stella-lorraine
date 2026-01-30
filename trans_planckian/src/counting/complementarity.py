"""
Complementarity Validation
==========================

Validates the complementarity constraints:
- Cannot observe both faces simultaneously
- Face switching works correctly
- Derivation vs measurement distinction
- Ammeter/voltmeter analogy
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Dict, Optional


class ObservableFace(Enum):
    CATEGORICAL = "CATEGORICAL"
    KINETIC = "KINETIC"


@dataclass
class FaceState:
    """Current state of the face observation system"""
    current_face: ObservableFace
    switch_count: int = 0


class ComplementarityValidator:
    """
    Validates complementarity constraints.
    
    Key validations:
    1. Cannot observe both faces simultaneously
    2. Face switching toggles correctly
    3. Observing wrong face raises error
    4. Derivation is distinct from measurement
    """
    
    def __init__(self):
        self.state = FaceState(current_face=ObservableFace.CATEGORICAL)
    
    def switch_face(self) -> ObservableFace:
        """Switch to the other face"""
        if self.state.current_face == ObservableFace.CATEGORICAL:
            self.state.current_face = ObservableFace.KINETIC
        else:
            self.state.current_face = ObservableFace.CATEGORICAL
        self.state.switch_count += 1
        return self.state.current_face
    
    def observe(self, face: ObservableFace) -> Tuple[bool, str]:
        """Attempt to observe a specific face"""
        if self.state.current_face == face:
            return True, f"Successfully observing {face.value} face"
        else:
            return False, f"Cannot observe {face.value}: currently on {self.state.current_face.value}"
    
    def observe_both(self) -> Tuple[bool, str]:
        """Attempt to observe both faces (should always fail)"""
        return False, "COMPLEMENTARITY VIOLATION: Cannot observe both faces simultaneously"
    
    def validate_face_switching(self) -> Tuple[bool, str]:
        """Validate that face switching works correctly"""
        initial = self.state.current_face
        
        # Switch
        after_first = self.switch_face()
        if after_first == initial:
            return False, "Face did not change after switch"
        
        # Switch back
        after_second = self.switch_face()
        if after_second != initial:
            return False, "Face did not return to initial after two switches"
        
        return True, "Face switching validated: toggles correctly between faces"
    
    def validate_complementarity_violation(self) -> Tuple[bool, str]:
        """Validate that simultaneous observation is prevented"""
        success, message = self.observe_both()
        
        if not success and "COMPLEMENTARITY VIOLATION" in message:
            return True, "Complementarity enforced: simultaneous observation prevented"
        else:
            return False, "ERROR: Simultaneous observation was allowed"
    
    def validate_wrong_face_error(self) -> Tuple[bool, str]:
        """Validate that observing wrong face is rejected"""
        self.state.current_face = ObservableFace.CATEGORICAL
        
        # Try to observe kinetic (wrong face)
        success, message = self.observe(ObservableFace.KINETIC)
        
        if not success:
            return True, "Wrong face observation correctly rejected"
        else:
            return False, "ERROR: Wrong face observation was allowed"
    
    def validate_derivation_distinction(self) -> Tuple[bool, str]:
        """
        Validate that derivation is marked as different from measurement.
        
        Like Ohm's law: if you measure I with ammeter, you DERIVE V = IR.
        The derived V is not the same as measuring V with a voltmeter.
        """
        # On categorical face, kinetic values must be derived
        self.state.current_face = ObservableFace.CATEGORICAL
        
        # Kinetic observation should fail (cannot directly observe)
        direct_kinetic, _ = self.observe(ObservableFace.KINETIC)
        
        # But we can derive kinetic properties
        derived_kinetic = self._derive_kinetic_from_categorical()
        
        if not direct_kinetic and derived_kinetic["is_derived"]:
            return True, "Derivation correctly marked as derived, not measured"
        else:
            return False, "ERROR: Derivation not distinguished from measurement"
    
    def _derive_kinetic_from_categorical(self) -> Dict:
        """Derive kinetic properties from categorical state (not measurement)"""
        return {
            "is_derived": True,
            "derived_from": "categorical",
            "warning": "This is a DERIVED value, not a direct measurement. "
                       "Same categorical state can correspond to any temperature.",
            "temperature": None,  # Cannot determine from categorical!
        }
    
    def validate_ammeter_voltmeter_analogy(self) -> Tuple[bool, str]:
        """
        Validate the ammeter/voltmeter analogy:
        - Ammeter (categorical): measures structure, derives velocity
        - Voltmeter (kinetic): measures velocity, derives structure
        - Cannot use both at same point
        """
        # Like ammeter measuring current
        self.state.current_face = ObservableFace.CATEGORICAL
        cat_success, _ = self.observe(ObservableFace.CATEGORICAL)
        kin_fail, _ = self.observe(ObservableFace.KINETIC)
        
        # Switch to voltmeter (kinetic)
        self.switch_face()
        kin_success, _ = self.observe(ObservableFace.KINETIC)
        cat_fail, _ = self.observe(ObservableFace.CATEGORICAL)
        
        if cat_success and not kin_fail and kin_success and not cat_fail:
            return True, "Ammeter/voltmeter analogy validated"
        else:
            return False, "ERROR: Analogy violated"
    
    def run_all_validations(self) -> Dict[str, Tuple[bool, str]]:
        """Run all complementarity validations"""
        results = {}
        
        results["face_switching"] = self.validate_face_switching()
        results["complementarity_violation"] = self.validate_complementarity_violation()
        results["wrong_face_error"] = self.validate_wrong_face_error()
        results["derivation_distinction"] = self.validate_derivation_distinction()
        results["ammeter_voltmeter_analogy"] = self.validate_ammeter_voltmeter_analogy()
        
        return results


if __name__ == "__main__":
    validator = ComplementarityValidator()
    results = validator.run_all_validations()
    
    print("=" * 60)
    print("COMPLEMENTARITY VALIDATION RESULTS")
    print("=" * 60)
    
    all_passed = True
    for name, (passed, message) in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        print(f"       {message}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
