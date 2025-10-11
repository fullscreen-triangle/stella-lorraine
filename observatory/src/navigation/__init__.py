"""
Stella-Lorraine Navigation Module
==================================
Multi-dimensional S-entropy navigation for trans-Planckian precision timing.

Note: Each module has its own main() function for standalone execution.
Import classes directly if needed for programmatic use.
"""

# Classes available for import, but each module runs independently via main()
__all__ = [
    'MolecularObserver',
    'RecursiveObserverLattice',
    'HarmonicNetworkGraph'
]

# Lazy imports - only import if explicitly requested
def __getattr__(name):
    if name == 'MolecularObserver':
        from .gas_molecule_lattice import MolecularObserver
        return MolecularObserver
    elif name == 'RecursiveObserverLattice':
        from .gas_molecule_lattice import RecursiveObserverLattice
        return RecursiveObserverLattice
    elif name == 'HarmonicNetworkGraph':
        from .harmonic_network_graph import HarmonicNetworkGraph
        return HarmonicNetworkGraph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
