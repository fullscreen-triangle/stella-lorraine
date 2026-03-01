"""
State Counting Framework for Mass Spectrometry
==============================================

This package implements a novel framework for understanding mass spectrometry
through the lens of hardware oscillator state counting.

The framework validates three theoretical claims:
1. Trans-Planckian: Phase space is bounded and discrete
2. CatScript: Categorical partition coordinates from oscillator counts
3. Categorical Cryogenics: T = 2E / (3k_B × M)

Key Identity: TIME = COUNTING = TEMPERATURE

Modules:
    TrappedIon: Hardware oscillator and ion trajectory tracking
    ThermodynamicRegimes: Five canonical regime classification
    Pipeline: Main orchestration and validation pipeline
    SpectraReader: mzML file parsing
    EntropyTransformation: S-entropy coordinate calculation
    IonisationPhysics: Ionization models (ESI, MALDI, EI)
    Dissociation: CID/HCD fragmentation as partition cascade
    Detector: Multi-modal detection modes
    StateCounting: Core state counting implementation
    IonJourney: Visualization tools

Author: Kundai Sachikonye
"""

# Version
__version__ = "0.1.0"

# Core hardware oscillator and ion tracking
from .TrappedIon import (
    HardwareOscillator,
    OscillatorState,
    PartitionCoordinates,
    IonState,
    IonTrajectory,
    StageTransition,
    JourneyStage,
    create_ion_trajectory,
)

# Thermodynamic regimes
from .ThermodynamicRegimes import (
    ThermodynamicRegime,
    ThermodynamicRegimeClassifier,
    ThermodynamicState,
    SEntropyCoordinates,
    DimensionlessParameters,
    UniversalEquationOfState,
    RegimeTransitionDetector,
    classify_ion_regime,
    calculate_categorical_temperature,
)

# Main pipeline
from .Pipeline import (
    StateCountingPipeline,
    ValidationPipeline,
    PipelineConfig,
    PipelineResults,
    IonRecord,
    process_mzml_file,
    validate_from_mzml,
)

# Spectral reading
from .SpectraReader import (
    extract_mzml,
    extract_spectra,
    ppm_window_bounds,
    find_peaks_in_window,
    get_xic_from_pl,
)

# Define public API
__all__ = [
    # Version
    "__version__",

    # Hardware oscillator
    "HardwareOscillator",
    "OscillatorState",
    "PartitionCoordinates",
    "IonState",
    "IonTrajectory",
    "StageTransition",
    "JourneyStage",
    "create_ion_trajectory",

    # Thermodynamic regimes
    "ThermodynamicRegime",
    "ThermodynamicRegimeClassifier",
    "ThermodynamicState",
    "SEntropyCoordinates",
    "DimensionlessParameters",
    "UniversalEquationOfState",
    "RegimeTransitionDetector",
    "classify_ion_regime",
    "calculate_categorical_temperature",

    # Pipeline
    "StateCountingPipeline",
    "ValidationPipeline",
    "PipelineConfig",
    "PipelineResults",
    "IonRecord",
    "process_mzml_file",
    "validate_from_mzml",

    # Spectral reading
    "extract_mzml",
    "extract_spectra",
    "ppm_window_bounds",
    "find_peaks_in_window",
    "get_xic_from_pl",
]


def quick_demo():
    """
    Quick demonstration of the state counting framework.

    Example:
        >>> from counting.src import quick_demo
        >>> quick_demo()
    """
    from .TrappedIon import demonstrate_state_counting
    from .ThermodynamicRegimes import demonstrate_regimes
    from .Pipeline import demonstrate_pipeline

    print("=" * 70)
    print("STATE COUNTING FRAMEWORK DEMONSTRATION")
    print("=" * 70)

    print("\n[1/3] Hardware Oscillator State Counting")
    print("-" * 40)
    demonstrate_state_counting()

    print("\n[2/3] Thermodynamic Regime Classification")
    print("-" * 40)
    demonstrate_regimes()

    print("\n[3/3] Complete Pipeline")
    print("-" * 40)
    demonstrate_pipeline()

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("""
Key Results Validated:
  1. Trans-Planckian: Phase space is bounded by C(n) = 2n²
  2. CatScript: Partition coordinates (n,l,m,s) from oscillator counts
  3. Categorical Cryogenics: T = 2E / (3k_B × M)

Fundamental Identity: TIME = COUNTING
  - Every measurement is counting oscillator cycles
  - More states → Lower effective temperature
  - Temperature suppression factor = 1/M
""")


# Convenience function for common use case
def process_ion(
    mz: float,
    intensity: float = 1000.0,
    charge: int = 1,
    energy_eV: float = 10.0,
    instrument: str = "orbitrap"
) -> dict:
    """
    Quick function to process a single ion through the framework.

    Args:
        mz: Mass-to-charge ratio
        intensity: Peak intensity
        charge: Charge state
        energy_eV: Kinetic energy
        instrument: Instrument type

    Returns:
        Dictionary with validation results

    Example:
        >>> from counting.src import process_ion
        >>> result = process_ion(500.25, intensity=10000, charge=2)
        >>> print(result['categorical_temperature'])
    """
    trajectory = create_ion_trajectory(
        mz=mz,
        intensity=intensity,
        charge=charge,
        energy_eV=energy_eV,
        instrument=instrument
    )

    trajectory.complete_ms1_journey()
    report = trajectory.get_validation_report()

    # Add categorical temperature
    final_state = trajectory.get_final_state()
    if final_state:
        report['categorical_temperature'] = final_state.categorical_temperature
        report['state_count'] = final_state.state_count
        report['temperature_suppression'] = final_state.temperature_suppression

    return report
