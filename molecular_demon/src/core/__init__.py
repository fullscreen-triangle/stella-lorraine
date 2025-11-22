"""
Core modules for Molecular Demon Reflectance Cascade

Provides the fundamental building blocks:
- Molecular network graph construction
- Categorical state theory (S-entropy coordinates)
- BMD recursive decomposition
- Frequency domain measurements
- Reflectance cascade algorithm
"""

from .molecular_network import (
    MolecularOscillator,
    HarmonicNetworkGraph
)

from .categorical_state import (
    CategoricalState,
    SEntropyCalculator,
    CategoryOrthogonality,
    navigate_categorical_space
)

from .bmd_decomposition import (
    MaxwellDemon,
    BMDHierarchy,
    verify_exponential_scaling
)

from .frequency_domain import (
    FrequencyDomainMeasurement,
    ZeroTimeMeasurement,
    calculate_trans_planckian_precision
)

from .reflectance_cascade import (
    ReflectionStep,
    SpectrometerState,
    MolecularDemonReflectanceCascade
)

__all__ = [
    # Network
    'MolecularOscillator',
    'HarmonicNetworkGraph',

    # Categorical state
    'CategoricalState',
    'SEntropyCalculator',
    'CategoryOrthogonality',
    'navigate_categorical_space',

    # BMD
    'MaxwellDemon',
    'BMDHierarchy',
    'verify_exponential_scaling',

    # Frequency domain
    'FrequencyDomainMeasurement',
    'ZeroTimeMeasurement',
    'calculate_trans_planckian_precision',

    # Cascade
    'ReflectionStep',
    'SpectrometerState',
    'MolecularDemonReflectanceCascade',
]
