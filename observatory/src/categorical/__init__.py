"""
Categorical State Framework

Core modules for categorical state representation and oscillator synchronization.
"""

from .categorical_state import (
    CategoricalState,
    CategoricalStateEstimator,
    EntropicCoordinates
)

from .oscillator_synchronization import (
    HydrogenOscillatorSync,
    MultiStationSync,
    OscillatorState
)

__all__ = [
    'CategoricalState',
    'CategoricalStateEstimator',
    'EntropicCoordinates',
    'HydrogenOscillatorSync',
    'MultiStationSync',
    'OscillatorState',
]
