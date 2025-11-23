"""
Protein Folding Framework - Phase-Locked Proton Maxwell Demons.

Complete framework for protein folding through phase-locked hydrogen bond networks,
synchronized to GroEL cyclic resonance chamber and O₂ master clock.

Based on theoretical papers on cytoplasmic phase-locking dynamics.

## Key Components

1. **Proton Maxwell Demon** (`proton_maxwell_demon.py`)
   - H-bonds as proton oscillators with phase-locking capability
   - Coupling to GroEL cavity resonance
   - S-entropy coordinates for categorical state tracking

2. **Protein Folding Network** (`protein_folding_network.py`)
   - Protein as network of coupled proton oscillators
   - Phase-coherence cluster identification
   - Network stability from collective phase-locking

3. **GroEL Resonance Chamber** (`groel_resonance_chamber.py`)
   - ATP-driven cyclical cavity dynamics
   - Frequency scanning through harmonic series
   - Cyclic protein state testing

4. **Reverse Folding Algorithm** (`reverse_folding_algorithm.py`)
   - Cycle-by-cycle H-bond formation tracking
   - Dependency graph construction
   - Folding pathway discovery

## Key Insights from Papers

### Phase-Locking IS the Mechanism
- All biological processes operate through phase-locked oscillatory networks
- GroEL actively phase-locks with protein H-bond oscillations
- Cavity operates at frequencies synchronized with cytoplasmic O₂ clock

### Cyclical Operation
- GroEL operates in ATP-driven cycles (~1 Hz base frequency)
- Each cycle samples different frequency space (harmonic series)
- Protein folds through iterative refinement across cycles

### Reverse Folding Reveals Pathway
- Native state → systematic destabilization → formation order
- Bonds that lock in later cycles DEPEND on earlier bonds
- Folding pathway = causal sequence revealed by cycle dependencies

## Usage Example

```python
from protein_folding import (
    create_h_bond_oscillator,
    ProteinFoldingNetwork,
    GroELResonanceChamber,
    ReverseFoldingAlgorithm
)

# Create protein network
protein = ProteinFoldingNetwork("my_protein", temperature=310.0)

# Add H-bonds
bond = create_h_bond_oscillator(
    donor='N', acceptor='O',
    donor_res=10, acceptor_res=25,
    length=2.8, angle=175.0
)
protein.add_h_bond(bond)

# Simulate folding in GroEL
groel = GroELResonanceChamber(temperature=310.0)
result = groel.run_folding_simulation(protein, max_cycles=20)

# Discover folding pathway
algorithm = ReverseFoldingAlgorithm(temperature=310.0)
pathway = algorithm.discover_folding_pathway(protein)
```

## Physical Constants

- O₂ master clock: 10^13 Hz (cytoplasmic phase reference)
- H⁺ field: 4×10^13 Hz (reality substrate)
- GroEL base frequency: ~1 Hz (ATP cycle rate)
- Proton oscillation: ~10^14 Hz (typical H-bond)

## Validation

Run complete validation suite:
```bash
python validate_cycle_by_cycle_folding.py
```
"""

# Core classes
from proton_maxwell_demon import (
    ProtonMaxwellDemon,
    HBondOscillator,
    SEntropyCoordinates,
    create_h_bond_oscillator,
    O2_MASTER_CLOCK_HZ,
    HPLUS_FIELD_HZ,
    GROEL_BASE_HZ,
    PROTON_OSCILLATION_HZ
)

from protein_folding_network import (
    ProteinFoldingNetwork,
    PhaseCoherenceCluster
)

from groel_resonance_chamber import (
    GroELResonanceChamber,
    ATPCycleState
)

from reverse_folding_algorithm import (
    ReverseFoldingAlgorithm,
    HBondFormationEvent,
    FoldingPathwayNode
)

__all__ = [
    # Core classes
    'ProtonMaxwellDemon',
    'HBondOscillator',
    'SEntropyCoordinates',
    'ProteinFoldingNetwork',
    'PhaseCoherenceCluster',
    'GroELResonanceChamber',
    'ATPCycleState',
    'ReverseFoldingAlgorithm',
    'HBondFormationEvent',
    'FoldingPathwayNode',

    # Factory functions
    'create_h_bond_oscillator',

    # Constants
    'O2_MASTER_CLOCK_HZ',
    'HPLUS_FIELD_HZ',
    'GROEL_BASE_HZ',
    'PROTON_OSCILLATION_HZ',
]

__version__ = '2.0.0'  # Phase-locking version
__author__ = 'Stella Lorraine Observatory'
