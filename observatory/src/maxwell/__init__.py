"""
Maxwell Package: Pixel Maxwell Demon Framework
==============================================

Complete implementation of Pixel Maxwell Demons for:
- Multi-modal hypothesis validation (virtual detectors)
- Categorical rendering (graphics without ray tracing)
- Live cell microscopy (interaction-free imaging)
- Trans-Planckian precision (femtosecond to yoctosecond)

Key Components:
- pixel_maxwell_demon: Core PMD and molecular demon classes
- virtual_detectors: Multi-modal virtual instruments
- harmonic_coincidence: Frequency-based categorical queries
- reflectance_cascade: Quadratic information gain
- categorical_light_sources: Information-theoretic light
- three_dim_categorical_renderer: Ray-free 3D rendering
- temporal_dynamics: Trans-Planckian temporal precision
- live_cell_imaging: Hypothesis-validated microscopy

Author: Kundai Sachikonye
Date: 2024
"""

from .pixel_maxwell_demon import (
    PixelMaxwellDemon,
    PixelDemonGrid,
    MolecularDemon,
    SEntropyCoordinates,
    Hypothesis
)

from .virtual_detectors import (
    VirtualDetector,
    VirtualThermometer,
    VirtualBarometer,
    VirtualHygrometer,
    VirtualIRSpectrometer,
    VirtualRamanSpectrometer,
    VirtualMassSpectrometer,
    VirtualPhotodiode,
    VirtualInterferometer,
    VirtualDetectorFactory,
    ConsilienceEngine
)

from .harmonic_coincidence import (
    HarmonicCoincidenceNetwork,
    MolecularHarmonicNetwork,
    Oscillator,
    HarmonicCoincidence,
    build_atmospheric_harmonic_network
)

from .reflectance_cascade import (
    ReflectanceCascade,
    PixelDemonCascade,
    CascadeObservation
)

from .categorical_light_sources import (
    Color,
    CategoricalLight,
    CategoricalPointLight,
    CategoricalDirectionalLight,
    CategoricalSpotLight,
    CategoricalAreaLight,
    LightingEnvironment,
    create_standard_lighting_environments
)

from .three_dim_categorical_renderer import (
    CategoricalRenderer3D,
    CategoricalScene,
    Surface
)

from .temporal_dynamics import (
    TransPlanckianClock,
    MotionBlurEngine,
    TemporalMeasurement
)

from .live_cell_imaging import (
    LiveCellMicroscope,
    LiveCellSample,
    BiologicalMolecule,
    validate_with_real_data
)

__version__ = '1.0.0'

__all__ = [
    # Core
    'PixelMaxwellDemon',
    'PixelDemonGrid',
    'MolecularDemon',
    'SEntropyCoordinates',
    'Hypothesis',

    # Virtual Detectors
    'VirtualDetector',
    'VirtualDetectorFactory',
    'ConsilienceEngine',
    'VirtualThermometer',
    'VirtualBarometer',
    'VirtualHygrometer',
    'VirtualIRSpectrometer',
    'VirtualRamanSpectrometer',
    'VirtualMassSpectrometer',
    'VirtualPhotodiode',
    'VirtualInterferometer',

    # Harmonic Networks
    'HarmonicCoincidenceNetwork',
    'MolecularHarmonicNetwork',
    'Oscillator',
    'HarmonicCoincidence',
    'build_atmospheric_harmonic_network',

    # Cascade
    'ReflectanceCascade',
    'PixelDemonCascade',
    'CascadeObservation',

    # Rendering
    'Color',
    'CategoricalLight',
    'CategoricalPointLight',
    'CategoricalDirectionalLight',
    'CategoricalSpotLight',
    'CategoricalAreaLight',
    'LightingEnvironment',
    'create_standard_lighting_environments',
    'CategoricalRenderer3D',
    'CategoricalScene',
    'Surface',

    # Temporal
    'TransPlanckianClock',
    'MotionBlurEngine',
    'TemporalMeasurement',

    # Microscopy
    'LiveCellMicroscope',
    'LiveCellSample',
    'BiologicalMolecule',
    'validate_with_real_data',
]
