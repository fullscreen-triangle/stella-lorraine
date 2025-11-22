"""
Gibbs' Paradox Resolution: Categorical State Theory

This package demonstrates the resolution of Gibbs' paradox through categorical
state theory and introduces two fundamental reformulations of entropy:

1. Entropy as Oscillatory Termination: S = -k_B log α
2. Entropy as Categorical Completion Rate: dS/dt = k_B dC/dt

These reformulations avoid the microstate counting ambiguities of classical
Boltzmann entropy while maintaining equivalence for equilibrium systems.

Modules:
--------
- seperate_containers: Initial separated state visualization
- mixing_process: Mixing process with A-B edge formation
- reseperation: Re-separated state showing categorical memory
- unpertubed: Control showing spatial vs categorical distinction
- rate_of_categorical_completion: Comprehensive entropy reformulation demo
- bmd_in_cytoplasm: THE BRIDGE TO LIFE - enzymes as BMDs in cytoplasm

Key Concepts:
-------------
- Categorical States: Ordered sequence of actualized configurations
- Categorical Irreversibility: Completed states cannot be re-occupied
- Phase-Lock Networks: Molecular coupling via Van der Waals forces
- Residual Edges: Phase correlations persisting after spatial separation

The Resolution:
---------------
Gibbs' paradox arises from assuming entropy depends only on spatial
configuration S(q,p). The categorical framework shows entropy depends on
both spatial AND categorical coordinates: S(q,p,C).

Two spatially identical configurations can have different categorical
positions and thus different entropies. This explains why mixing-separation
cycles increase entropy even when spatial configurations appear identical.

Usage:
------
Run individual visualization scripts:
    python seperate_containers.py
    python mixing_process.py
    python reseperation.py
    python unpertubed.py
    python rate_of_categorical_completion.py

Author: Kundai Farai Sachikonye
Institution: Technical University of Munich
Email: kundai.sachikonye@wzw.tum.de
"""

__version__ = "1.0.0"
__author__ = "Kundai Farai Sachikonye"
__email__ = "kundai.sachikonye@wzw.tum.de"

# Physical constants
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K

# Entropy reformulations
def entropy_boltzmann(Omega):
    """Classical Boltzmann entropy: S = k_B log Ω"""
    import numpy as np
    return BOLTZMANN_CONSTANT * np.log(Omega)

def entropy_oscillatory(alpha):
    """Oscillatory entropy: S = -k_B log α"""
    import numpy as np
    return -BOLTZMANN_CONSTANT * np.log(alpha)

def entropy_completion(C):
    """Completion rate entropy: S = k_B C"""
    return BOLTZMANN_CONSTANT * C

def entropy_production_rate(C_dot):
    """Entropy production rate: dS/dt = k_B dC/dt"""
    return BOLTZMANN_CONSTANT * C_dot

# Package metadata
__all__ = [
    'entropy_boltzmann',
    'entropy_oscillatory',
    'entropy_completion',
    'entropy_production_rate',
    'BOLTZMANN_CONSTANT'
]
