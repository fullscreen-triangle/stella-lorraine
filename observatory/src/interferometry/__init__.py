"""
Trans-Planckian Interferometry

Validation modules for ultra-high angular resolution interferometry
using categorical state propagation.
"""

from .angular_resolution import (
    AngularResolutionCalculator,
    TransPlanckianResolutionValidator,
    ResolutionMetrics
)

from .atmospheric_effects import (
    ConventionalAtmosphericDegradation,
    CategoricalAtmosphericImmunity,
    AtmosphericComparisonExperiment,
    AtmosphericParameters
)

from .baseline_coherence import (
    BaselineCoherenceAnalyzer,
    FringeVisibilityExperiment,
    CoherenceMetrics
)

from .phase_correlation import (
    CategoricalPhaseAnalyzer,
    TransPlanckianInterferometer,
    PhaseCorrelation
)

__all__ = [
    'AngularResolutionCalculator',
    'TransPlanckianResolutionValidator',
    'ResolutionMetrics',
    'ConventionalAtmosphericDegradation',
    'CategoricalAtmosphericImmunity',
    'AtmosphericComparisonExperiment',
    'AtmosphericParameters',
    'BaselineCoherenceAnalyzer',
    'FringeVisibilityExperiment',
    'CoherenceMetrics',
    'CategoricalPhaseAnalyzer',
    'TransPlanckianInterferometer',
    'PhaseCorrelation',
]
