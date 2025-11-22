#!/usr/bin/env python3
"""
Quick import check for validation framework

Verifies all modules can be imported without errors.
Run this first before running full validation suite.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def check_imports():
    """Check all validation module imports"""

    print("Checking validation framework imports...")
    print("=" * 60)

    errors = []

    # Categorical modules
    print("\n[1/3] Categorical Framework...")
    try:
        from categorical.categorical_state import (
            CategoricalState,
            CategoricalStateEstimator,
            EntropicCoordinates
        )
        print("  ✓ categorical_state.py")
    except Exception as e:
        print(f"  ✗ categorical_state.py: {e}")
        errors.append(('categorical_state', e))

    try:
        from categorical.oscillator_synchronization import (
            HydrogenOscillatorSync,
            MultiStationSync,
            OscillatorState
        )
        print("  ✓ oscillator_synchronization.py")
    except Exception as e:
        print(f"  ✗ oscillator_synchronization.py: {e}")
        errors.append(('oscillator_synchronization', e))

    # Interferometry modules
    print("\n[2/3] Trans-Planckian Interferometry...")
    try:
        from interferometry.angular_resolution import (
            AngularResolutionCalculator,
            TransPlanckianResolutionValidator,
            ResolutionMetrics
        )
        print("  ✓ angular_resolution.py")
    except Exception as e:
        print(f"  ✗ angular_resolution.py: {e}")
        errors.append(('angular_resolution', e))

    try:
        from interferometry.atmospheric_effects import (
            ConventionalAtmosphericDegradation,
            CategoricalAtmosphericImmunity,
            AtmosphericComparisonExperiment
        )
        print("  ✓ atmospheric_effects.py")
    except Exception as e:
        print(f"  ✗ atmospheric_effects.py: {e}")
        errors.append(('atmospheric_effects', e))

    try:
        from interferometry.baseline_coherence import (
            BaselineCoherenceAnalyzer,
            FringeVisibilityExperiment,
            CoherenceMetrics
        )
        print("  ✓ baseline_coherence.py")
    except Exception as e:
        print(f"  ✗ baseline_coherence.py: {e}")
        errors.append(('baseline_coherence', e))

    try:
        from interferometry.phase_correlation import (
            CategoricalPhaseAnalyzer,
            TransPlanckianInterferometer,
            PhaseCorrelation
        )
        print("  ✓ phase_correlation.py")
    except Exception as e:
        print(f"  ✗ phase_correlation.py: {e}")
        errors.append(('phase_correlation', e))

    # Thermometry modules
    print("\n[3/3] Categorical Quantum Thermometry...")
    try:
        from thermometry.temperature_extraction import (
            ThermometryAnalyzer,
            TimeOfFlightComparison
        )
        print("  ✓ temperature_extraction.py")
    except Exception as e:
        print(f"  ✗ temperature_extraction.py: {e}")
        errors.append(('temperature_extraction', e))

    try:
        from thermometry.momentum_recovery import (
            MomentumRecovery,
            QuantumBackactionAnalyzer
        )
        print("  ✓ momentum_recovery.py")
    except Exception as e:
        print(f"  ✗ momentum_recovery.py: {e}")
        errors.append(('momentum_recovery', e))

    try:
        from thermometry.real_time_monitor import (
            RealTimeThermometer,
            EvaporativeCoolingSimulator,
            TemperatureSnapshot
        )
        print("  ✓ real_time_monitor.py")
    except Exception as e:
        print(f"  ✗ real_time_monitor.py: {e}")
        errors.append(('real_time_monitor', e))

    try:
        from thermometry.comparison_tof import (
            CategoricalThermometryComparison,
            TimeOfFlightThermometry,
            ThermometryPerformance
        )
        print("  ✓ comparison_tof.py")
    except Exception as e:
        print(f"  ✗ comparison_tof.py: {e}")
        errors.append(('comparison_tof', e))

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"\n✗ Import check failed: {len(errors)} error(s)\n")
        for module, error in errors:
            print(f"  {module}: {error}")
        return 1
    else:
        print("\n✓ All imports successful!")
        print("\nYou can now run:")
        print("  python run_all_validations.py")
        return 0


if __name__ == "__main__":
    sys.exit(check_imports())
