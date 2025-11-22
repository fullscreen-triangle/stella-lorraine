"""
Oscillatory Muscle Modeling Module

Multi-scale oscillatory coupling framework for muscle and musculoskeletal modeling.

Extends classical Hill-type muscle models with:
- 10-scale oscillatory hierarchy (quantum membrane to allometric)
- Multi-scale coupling strength computation
- Gear ratio transformations between scales
- Tri-dimensional state space coordinates (knowledge, time, entropy)
- Coordinated body segment dynamics

Main classes:
- OscillatoryMuscleModel: Extended Hill muscle with oscillatory coupling
- OscillatoryKinematicChain: Body segments as coupled oscillators
- LowerLimbModel: Complete lower limb with integrated muscle-segment coupling

Examples:
    >>> from upward.muscle import OscillatoryMuscleModel
    >>> 
    >>> # Create muscle model
    >>> muscle = OscillatoryMuscleModel()
    >>> 
    >>> # Define excitation and length functions
    >>> def excitation(t):
    ...     return 1.0 if 0.5 <= t <= 2.0 else 0.01
    >>> 
    >>> def muscle_tendon_length(t):
    ...     return 0.31  # Isometric
    >>> 
    >>> # Simulate with oscillatory coupling
    >>> results = muscle.simulate_muscle_with_coupling(
    ...     excitation, muscle_tendon_length,
    ...     enable_coupling=True
    ... )
    >>> 
    >>> # Analyze performance
    >>> metrics = muscle.compute_performance_metrics(results)
"""

from .muscle_model import (
    OscillatoryMuscleModel,
    OscillatoryCouplingAnalyzer,
    GearRatioTransform,
    StateSpaceCoordinates,
    OscillatoryHierarchy,
    OscillatoryScale,
)

from .body_segmentation import (
    BodySegment,
    BodySegmentParameters,
    OscillatoryKinematicChain,
    LowerLimbModel,
)

__all__ = [
    # Muscle models
    'OscillatoryMuscleModel',
    'OscillatoryCouplingAnalyzer',
    'GearRatioTransform',
    'StateSpaceCoordinates',
    'OscillatoryHierarchy',
    'OscillatoryScale',
    
    # Body segmentation
    'BodySegment',
    'BodySegmentParameters',
    'OscillatoryKinematicChain',
    'LowerLimbModel',
]

__version__ = '0.1.0'
__author__ = 'Kundai Farai Sachikonye'

