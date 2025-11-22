"""
Chemical Bond Analyzer

Analyzes chemical bonds for oscillatory signature generation.
Bond properties determine vibrational frequencies and coupling.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


@dataclass
class BondProperties:
    """Properties of a chemical bond."""
    bond_idx: int
    atom1_idx: int
    atom2_idx: int
    atom1_symbol: str
    atom2_symbol: str
    bond_type: str  # 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'
    bond_length: Optional[float]  # Angstroms
    is_aromatic: bool
    is_conjugated: bool
    is_in_ring: bool
    
    # Bond strength estimates
    force_constant: float  # N/m (estimated)
    vibrational_frequency: float  # Hz (estimated)
    
    # Reduced mass
    reduced_mass: float  # amu
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert float values
        for key in ['bond_length', 'force_constant', 'vibrational_frequency', 'reduced_mass']:
            if d[key] is not None:
                d[key] = float(d[key])
        return d


class BondAnalyzer:
    """
    Analyzes chemical bonds in molecules.
    
    Provides:
    - Bond type identification
    - Vibrational frequency estimation
    - Force constant calculation
    - Conjugation and aromaticity analysis
    """
    
    # Bond force constants (N/m) - typical values
    FORCE_CONSTANTS = {
        ('C', 'C', 'SINGLE'): 450,
        ('C', 'C', 'DOUBLE'): 950,
        ('C', 'C', 'TRIPLE'): 1600,
        ('C', 'C', 'AROMATIC'): 700,
        ('C', 'H', 'SINGLE'): 500,
        ('C', 'O', 'SINGLE'): 360,
        ('C', 'O', 'DOUBLE'): 1200,
        ('C', 'N', 'SINGLE'): 305,
        ('C', 'N', 'DOUBLE'): 615,
        ('C', 'N', 'TRIPLE'): 1600,
        ('O', 'H', 'SINGLE'): 770,
        ('N', 'H', 'SINGLE'): 650,
    }
    
    # Atomic masses (amu)
    ATOMIC_MASSES = {
        'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
        'F': 18.998, 'P': 30.974, 'S': 32.065, 'Cl': 35.453,
        'Br': 79.904, 'I': 126.904
    }
    
    def __init__(self):
        """Initialize bond analyzer."""
        if not RDKIT_AVAILABLE:
            print("Warning: RDKit not available. Bond analysis will be limited.")
    
    def analyze_molecule(self, mol: 'Chem.Mol') -> List[BondProperties]:
        """
        Analyze all bonds in a molecule.
        
        Args:
            mol: RDKit Mol object (should have 3D coordinates if available)
            
        Returns:
            List of BondProperties for each bond
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit required for bond analysis")
        
        bond_properties_list = []
        
        for bond in mol.GetBonds():
            props = self._analyze_bond(mol, bond)
            bond_properties_list.append(props)
        
        return bond_properties_list
    
    def _analyze_bond(self, mol: 'Chem.Mol', bond: 'Chem.Bond') -> BondProperties:
        """Analyze a single bond."""
        # Get atoms
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        
        # Bond type
        bond_type_map = {
            Chem.BondType.SINGLE: 'SINGLE',
            Chem.BondType.DOUBLE: 'DOUBLE',
            Chem.BondType.TRIPLE: 'TRIPLE',
            Chem.BondType.AROMATIC: 'AROMATIC',
        }
        bond_type = bond_type_map.get(bond.GetBondType(), 'SINGLE')
        
        # Aromatic/conjugated
        is_aromatic = bond.GetIsAromatic()
        is_conjugated = bond.GetIsConjugated()
        is_in_ring = bond.IsInRing()
        
        # Atomic symbols
        sym1 = atom1.GetSymbol()
        sym2 = atom2.GetSymbol()
        
        # Bond length (if 3D coordinates available)
        bond_length = None
        if mol.GetNumConformers() > 0:
            bond_length = self._calculate_bond_length(mol, bond)
        
        # Calculate force constant and frequency
        force_constant = self._estimate_force_constant(sym1, sym2, bond_type)
        reduced_mass = self._calculate_reduced_mass(sym1, sym2)
        vibrational_frequency = self._calculate_vibrational_frequency(
            force_constant, reduced_mass
        )
        
        return BondProperties(
            bond_idx=bond.GetIdx(),
            atom1_idx=atom1.GetIdx(),
            atom2_idx=atom2.GetIdx(),
            atom1_symbol=sym1,
            atom2_symbol=sym2,
            bond_type=bond_type,
            bond_length=bond_length,
            is_aromatic=is_aromatic,
            is_conjugated=is_conjugated,
            is_in_ring=is_in_ring,
            force_constant=force_constant,
            vibrational_frequency=vibrational_frequency,
            reduced_mass=reduced_mass,
        )
    
    def _calculate_bond_length(self, mol: 'Chem.Mol', bond: 'Chem.Bond') -> float:
        """Calculate bond length from 3D coordinates."""
        conf = mol.GetConformer()
        pos1 = conf.GetAtomPosition(bond.GetBeginAtomIdx())
        pos2 = conf.GetAtomPosition(bond.GetEndAtomIdx())
        
        dx = pos2.x - pos1.x
        dy = pos2.y - pos1.y
        dz = pos2.z - pos1.z
        
        return np.sqrt(dx*dx + dy*dy + dz*dz)
    
    def _estimate_force_constant(self, sym1: str, sym2: str, bond_type: str) -> float:
        """
        Estimate bond force constant.
        
        Args:
            sym1, sym2: Atomic symbols
            bond_type: 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'
            
        Returns:
            Force constant in N/m
        """
        # Try both orderings
        key1 = (sym1, sym2, bond_type)
        key2 = (sym2, sym1, bond_type)
        
        if key1 in self.FORCE_CONSTANTS:
            return self.FORCE_CONSTANTS[key1]
        elif key2 in self.FORCE_CONSTANTS:
            return self.FORCE_CONSTANTS[key2]
        else:
            # Default estimate based on bond type
            defaults = {
                'SINGLE': 400,
                'DOUBLE': 800,
                'TRIPLE': 1400,
                'AROMATIC': 600,
            }
            return defaults.get(bond_type, 400)
    
    def _calculate_reduced_mass(self, sym1: str, sym2: str) -> float:
        """
        Calculate reduced mass of two atoms.
        
        μ = (m1 * m2) / (m1 + m2)
        
        Returns:
            Reduced mass in amu
        """
        m1 = self.ATOMIC_MASSES.get(sym1, 12.0)  # Default to carbon
        m2 = self.ATOMIC_MASSES.get(sym2, 12.0)
        
        return (m1 * m2) / (m1 + m2)
    
    def _calculate_vibrational_frequency(self, force_constant: float, reduced_mass: float) -> float:
        """
        Calculate vibrational frequency from force constant and reduced mass.
        
        ν = (1 / 2π) * sqrt(k / μ)
        
        Args:
            force_constant: Force constant in N/m
            reduced_mass: Reduced mass in amu
            
        Returns:
            Frequency in Hz
        """
        # Convert amu to kg
        reduced_mass_kg = reduced_mass * 1.66054e-27
        
        # Calculate frequency
        frequency = (1.0 / (2.0 * np.pi)) * np.sqrt(force_constant / reduced_mass_kg)
        
        return frequency
    
    def get_dominant_vibrational_modes(self, 
                                      bond_properties_list: List[BondProperties],
                                      n_modes: int = 5) -> List[Tuple[int, float]]:
        """
        Get dominant vibrational modes.
        
        Args:
            bond_properties_list: List of bond properties
            n_modes: Number of modes to return
            
        Returns:
            List of (bond_idx, frequency) tuples, sorted by frequency
        """
        frequencies = [(bp.bond_idx, bp.vibrational_frequency) 
                      for bp in bond_properties_list]
        
        # Sort by frequency (descending)
        frequencies.sort(key=lambda x: x[1], reverse=True)
        
        return frequencies[:n_modes]
    
    def calculate_average_vibrational_frequency(self, 
                                                bond_properties_list: List[BondProperties]) -> float:
        """Calculate average vibrational frequency across all bonds."""
        if not bond_properties_list:
            return 0.0
        
        total_freq = sum(bp.vibrational_frequency for bp in bond_properties_list)
        return total_freq / len(bond_properties_list)
    
    def count_conjugated_systems(self, bond_properties_list: List[BondProperties]) -> int:
        """Count number of conjugated bond systems."""
        # Count consecutive conjugated bonds
        conjugated_count = sum(1 for bp in bond_properties_list if bp.is_conjugated)
        return conjugated_count
    
    def calculate_bond_alternation(self, bond_properties_list: List[BondProperties]) -> float:
        """
        Calculate bond length alternation (for conjugated systems).
        
        Returns:
            Standard deviation of bond lengths
        """
        lengths = [bp.bond_length for bp in bond_properties_list 
                  if bp.bond_length is not None and bp.is_conjugated]
        
        if len(lengths) < 2:
            return 0.0
        
        return float(np.std(lengths))
    
    def save_bond_analysis(self,
                          bond_properties_list: List[BondProperties],
                          output_path: str,
                          smiles: Optional[str] = None,
                          metadata: Optional[Dict] = None) -> None:
        """
        Save bond analysis results to JSON file.
        
        Args:
            bond_properties_list: List of bond properties
            output_path: Path to output JSON file
            smiles: Optional SMILES string
            metadata: Optional metadata
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict
        bonds_data = [bp.to_dict() for bp in bond_properties_list]
        
        # Calculate summary statistics
        avg_freq = self.calculate_average_vibrational_frequency(bond_properties_list)
        n_conjugated = self.count_conjugated_systems(bond_properties_list)
        dominant_modes = self.get_dominant_vibrational_modes(bond_properties_list, n_modes=5)
        
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'smiles': smiles,
            'num_bonds': len(bond_properties_list),
            'bonds': bonds_data,
            'summary': {
                'average_vibrational_frequency': float(avg_freq),
                'num_conjugated_bonds': n_conjugated,
                'dominant_modes': [
                    {'bond_idx': int(idx), 'frequency': float(freq)}
                    for idx, freq in dominant_modes
                ]
            }
        }
        
        if metadata:
            output_data['metadata'] = metadata
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Saved bond analysis to {output_path}")


def demonstrate_bond_analysis():
    """Demonstrate bond analysis."""
    print("="*80)
    print("CHEMICAL BOND ANALYSIS DEMONSTRATION")
    print("="*80 + "\n")
    
    if not RDKIT_AVAILABLE:
        print("✗ RDKit not available. Install with: pip install rdkit")
        return
    
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    # Example molecule: Vanillin (aromatic with C=O)
    smiles = 'COc1cc(C=O)ccc1O'
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    analyzer = BondAnalyzer()
    
    print(f"Molecule: Vanillin")
    print(f"SMILES: {smiles}")
    print(f"Number of bonds: {mol.GetNumBonds()}\n")
    
    # Analyze all bonds
    bond_props = analyzer.analyze_molecule(mol)
    
    print("Bond Analysis:")
    print("-" * 80)
    for i, bp in enumerate(bond_props[:10]):  # Show first 10
        print(f"Bond {i}: {bp.atom1_symbol}-{bp.atom2_symbol} ({bp.bond_type})")
        print(f"  Force constant: {bp.force_constant:.1f} N/m")
        print(f"  Frequency: {bp.vibrational_frequency:.2e} Hz")
        print(f"  Reduced mass: {bp.reduced_mass:.3f} amu")
        if bp.bond_length:
            print(f"  Bond length: {bp.bond_length:.3f} Å")
        if bp.is_aromatic:
            print(f"  Aromatic: Yes")
        print()
    
    # Dominant modes
    print("Dominant Vibrational Modes:")
    print("-" * 80)
    dominant = analyzer.get_dominant_vibrational_modes(bond_props, n_modes=5)
    for bond_idx, freq in dominant:
        bp = bond_props[bond_idx]
        print(f"  {bp.atom1_symbol}-{bp.atom2_symbol}: {freq:.2e} Hz")
    
    avg_freq = analyzer.calculate_average_vibrational_frequency(bond_props)
    print(f"\nAverage frequency: {avg_freq:.2e} Hz")
    
    n_conjugated = analyzer.count_conjugated_systems(bond_props)
    print(f"Conjugated bonds: {n_conjugated}")
    
    # Save results
    print("\nSaving results...")
    print("-" * 80)
    
    output_dir = Path("results/bond_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analyzer.save_bond_analysis(
        bond_props,
        output_dir / "vanillin_bonds.json",
        smiles=smiles,
        metadata={'description': 'Bond analysis of vanillin molecule'}
    )
    
    print("\n✓ Demonstration complete!")


if __name__ == "__main__":
    demonstrate_bond_analysis()

