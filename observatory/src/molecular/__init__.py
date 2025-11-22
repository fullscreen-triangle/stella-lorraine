"""
Molecular Structure Processing

Converts molecular structures (SMILES, SDF) into features for oscillatory
signature generation.

Modules:
- structure_encoder: Molecular structure → feature vectors
- bond_analyzer: Chemical bond analysis
- geometry_calculator: 3D geometry computation
- mass_properties: Mass, isotopes, molecular properties
"""

from .structure_encoder import MolecularStructureEncoder
from .bond_analyzer import BondAnalyzer
from .geometry_calculator import GeometryCalculator
from .mass_properties import MassPropertiesCalculator

__all__ = [
    'MolecularStructureEncoder',
    'BondAnalyzer',
    'GeometryCalculator',
    'MassPropertiesCalculator',
]

