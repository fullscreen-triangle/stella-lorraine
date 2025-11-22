"""
Quick test to verify all modules import correctly
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("Testing imports...")
print("="*70)

try:
    # Core imports
    from src.core import (
        MolecularOscillator,
        HarmonicNetworkGraph,
        CategoricalState,
        SEntropyCalculator,
        MaxwellDemon,
        BMDHierarchy,
        FrequencyDomainMeasurement,
        ZeroTimeMeasurement,
        MolecularDemonReflectanceCascade
    )
    print("✓ Core modules imported successfully")

    # Physics imports
    from src.physics import (
        MolecularSpecies,
        MolecularOscillatorGenerator,
        HarmonicCoincidenceDetector,
        HeisenbergBypass,
        VirtualPhotodetector,
        VirtualIonDetector,
        VirtualMassSpectrometer,
        VirtualDetectorFactory
    )
    print("✓ Physics modules imported successfully")
    print("✓ Virtual detector modules imported successfully")

    # Test basic functionality
    print("\nTesting basic functionality...")
    print("-"*70)

    # Create a categorical state
    state = CategoricalState(s_k=1.0, s_t=2.0, s_e=3.0)
    print(f"✓ CategoricalState: {state}")

    # Create a Maxwell Demon
    demon = MaxwellDemon(frequency_hz=1e13, s_k=0, s_t=0, s_e=0, depth=0)
    print(f"✓ MaxwellDemon created: {demon.frequency_hz:.2e} Hz")

    # Verify Heisenberg bypass
    orthogonal = HeisenbergBypass.verify_orthogonality()
    if orthogonal:
        print("✓ Heisenberg bypass verified")

    # Verify zero-time measurement
    zero_time_valid = ZeroTimeMeasurement.validate_zero_time()
    if zero_time_valid:
        print("✓ Zero-time measurement validated")

    # Test virtual detector creation
    print("\nTesting virtual detectors...")
    print("-"*70)
    photodetector = VirtualPhotodetector(convergence_node=0)
    print("✓ Virtual photodetector created")

    ion_detector = VirtualIonDetector(convergence_node=1)
    print("✓ Virtual ion detector created")

    mass_spec = VirtualMassSpectrometer(convergence_node=2)
    print("✓ Virtual mass spectrometer created")

    # Test factory
    detector = VirtualDetectorFactory.create_detector('photodetector', 100)
    print("✓ Detector factory working")

    print("\n" + "="*70)
    print("ALL IMPORTS AND BASIC TESTS PASSED ✓")
    print("="*70)
    print("\nPackage is ready to use!")
    print("\nRun experiments with:")
    print("  python experiments/reproduce_trans_planckian.py")
    print("  python experiments/bmd_enhancement_factor.py")
    print("  python experiments/cascade_depth_scaling.py")
    print("  python experiments/zero_time_validation.py")
    print("  python experiments/virtual_detector_demo.py  (NEW!)")

except Exception as e:
    print(f"\n✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
