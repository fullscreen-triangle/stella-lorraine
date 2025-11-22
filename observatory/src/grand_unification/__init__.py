"""
Grand Unification Framework - Complete Implementation
======================================================

All 9 core modules implemented and ready for use.

Core Modules:
-------------
1. GrandWave - Universal reality substrate
2. clock_synchronization - Trans-Planckian timing
3. oscillatory_signatures - Pathway 1 (FFT → harmonics)
4. Propagation - Wave navigation
5. Interface - Object-wave interaction
6. s_entropy - S-coordinate calculation
7. cross_validator - Dual validation engine
8. harmonic_graph - 240-node network
9. visual_pathway - Pathway 2 (droplet → CNN)
10. domain_transformer - Cross-domain solution transfer

Quick Start:
------------
```python
from grand_unification.GrandWave import GrandWave
from grand_unification.clock_synchronization import HardwareClockSync
from grand_unification.oscillatory_signatures import extract_oscillatory_signature
from grand_unification.s_entropy import SEntropyCalculator
from grand_unification.visual_pathway import VisualAnalysisEngine
from grand_unification.cross_validator import DualValidationEngine

# Initialize
grand_wave = GrandWave()
clock = HardwareClockSync()

# ... (see README.md for complete examples)
```
"""

__version__ = '1.0.0'
__author__ = 'Kundai Farai Sachikonye'

# Core modules
from .GrandWave import GrandWave, WaveDisturbance, InterferencePattern
from .clock_synchronization import HardwareClockSync, BeatFrequencyMethod
from .oscillatory_signatures import (
    OscillatorySignature,
    OscillatoryAnalysisEngine,
    extract_oscillatory_signature
)
from .Propagation import WavePropagator, PropagationPath
from .Interface import WaveInterface, InterfaceManager, InteractionPattern
from .s_entropy import SEntropyCalculator
from .cross_validator import DualValidationEngine, ValidationResult
from .harmonic_graph import HarmonicNetworkGraph, HarmonicNode, HarmonicEdge
from .visual_pathway import (
    VisualAnalysisEngine,
    WaterSurfacePhysics,
    DropletParameters,
    VisualFeatures
)
from .domain_transformer import (
    CrossDomainTransformer,
    DomainMapping,
    TransferredSolution
)

__all__ = [
    # Core substrate
    'GrandWave',
    'WaveDisturbance',
    'InterferencePattern',
    
    # Timing
    'HardwareClockSync',
    'BeatFrequencyMethod',
    
    # Pathway 1 (Oscillatory)
    'OscillatorySignature',
    'OscillatoryAnalysisEngine',
    'extract_oscillatory_signature',
    'SEntropyCalculator',
    
    # Pathway 2 (Visual)
    'VisualAnalysisEngine',
    'WaterSurfacePhysics',
    'DropletParameters',
    'VisualFeatures',
    
    # Validation
    'DualValidationEngine',
    'ValidationResult',
    
    # Navigation
    'WavePropagator',
    'PropagationPath',
    'WaveInterface',
    'InterfaceManager',
    'InteractionPattern',
    
    # Graph
    'HarmonicNetworkGraph',
    'HarmonicNode',
    'HarmonicEdge',
    
    # Cross-domain
    'CrossDomainTransformer',
    'DomainMapping',
    'TransferredSolution',
]

