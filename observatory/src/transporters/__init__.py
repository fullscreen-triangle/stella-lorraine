"""
Membrane Transporter Maxwell Demons
====================================

Complete framework for understanding membrane transporters as
categorical molecular Maxwell demons.

Modules:
--------
- categorical_coordinates: S-entropy coordinate mapping
- phase_locked_selection: Substrate selection via phase-locking
- transplanckian_observation: Zero-backaction observation

Key Concepts:
-------------
1. Transporters operate in dual coordinate systems (physical + categorical)
2. Substrate selection emerges from phase-locking in frequency space
3. ATP modulates binding site frequency (scans for substrates)
4. Trans-Planckian observation enables watching without disturbance
5. Maxwell Demon operation validated mechanistically

This framework extends:
- Flatt et al. (2023) Nature Comm. Phys. - ABC transporters as Maxwell Demons
- Stella Lorraine Observatory protein folding - Phase-locking framework
- Stella Lorraine Observatory atmospheric computation - Trans-Planckian measurement
"""

from .categorical_coordinates import (
    SEntropyCoordinates,
    TransporterState,
    ConformationalState,
    TransporterConformationalLandscape
)

from .phase_locked_selection import (
    SubstrateVibrationalProfile,
    PhaseLockingTransporter,
    create_example_substrates,
    validate_phase_locking
)

from .transplanckian_observation import (
    ObservationPoint,
    TransPlanckianObserver,
    validate_transplanckian_observation
)

from .ensemble_transporter_demon import (
    EnsembleStatistics,
    EnsembleTransporterDemon,
    validate_ensemble_demon
)

__all__ = [
    # Categorical coordinates
    'SEntropyCoordinates',
    'TransporterState',
    'ConformationalState',
    'TransporterConformationalLandscape',

    # Phase-locking
    'SubstrateVibrationalProfile',
    'PhaseLockingTransporter',
    'create_example_substrates',
    'validate_phase_locking',

    # Trans-Planckian observation
    'ObservationPoint',
    'TransPlanckianObserver',
    'validate_transplanckian_observation',

    # Ensemble collective behavior
    'EnsembleStatistics',
    'EnsembleTransporterDemon',
    'validate_ensemble_demon',
]

__version__ = '1.0.0'
__author__ = 'Stella Lorraine Observatory'
