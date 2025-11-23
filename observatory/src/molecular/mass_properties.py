"""
Mass Properties Calculator

Calculates mass-related properties including isotope effects.
Critical for distinguishing oscillatory theory from mass-based theories.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


@dataclass
class MassProperties:
    """Mass-related properties of a molecule."""
    molecular_weight: float  # amu
    exact_mass: float  # amu (monoisotopic)
    heavy_atom_count: int

    # Element composition
    element_counts: Dict[str, int]

    # Isotope-specific
    has_deuterium: bool
    deuterium_count: int

    # Mass distribution
    reduced_mass: float  # Average reduced mass for vibrations
    mass_distribution_moment: float  # Second moment of mass distribution

    # Optional isotope pattern
    isotope_pattern: Optional[List[Tuple[float, float]]] = None  # [(mass, abundance)]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert float values
        for key in ['molecular_weight', 'exact_mass', 'reduced_mass', 'mass_distribution_moment']:
            if d[key] is not None:
                d[key] = float(d[key])
        return d


class MassPropertiesCalculator:
    """
    Calculates mass properties of molecules.

    Key capability: Isotope differentiation
    - Critical for validating oscillatory theory vs. mass-based theories
    - H/D substitution changes vibrational frequencies by factor of √2
    """

    # Atomic masses (most common isotope)
    ATOMIC_MASSES = {
        'H': 1.00783, 'D': 2.01410,  # Hydrogen and deuterium
        'C': 12.0000, 'N': 14.00307, 'O': 15.99491,
        'F': 18.99840, 'P': 30.97376, 'S': 31.97207,
        'Cl': 34.96885, 'Br': 78.91834, 'I': 126.90447
    }

    def __init__(self):
        """Initialize mass properties calculator."""
        if not RDKIT_AVAILABLE:
            print("Warning: RDKit not available. Mass calculations will be limited.")

    def calculate_properties(self, mol: 'Chem.Mol') -> MassProperties:
        """
        Calculate mass properties for a molecule.

        Args:
            mol: RDKit Mol object

        Returns:
            MassProperties object
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit required for mass calculations")

        # Basic mass properties
        molecular_weight = Descriptors.MolWt(mol)
        exact_mass = Descriptors.ExactMolWt(mol)
        heavy_atom_count = Descriptors.HeavyAtomCount(mol)

        # Element composition
        element_counts = self._count_elements(mol)

        # Check for deuterium
        has_deuterium, deuterium_count = self._check_deuterium(mol)

        # Calculate reduced mass
        reduced_mass = self._calculate_average_reduced_mass(mol)

        # Mass distribution moment
        mass_moment = self._calculate_mass_distribution_moment(mol)

        return MassProperties(
            molecular_weight=molecular_weight,
            exact_mass=exact_mass,
            heavy_atom_count=heavy_atom_count,
            element_counts=element_counts,
            has_deuterium=has_deuterium,
            deuterium_count=deuterium_count,
            reduced_mass=reduced_mass,
            mass_distribution_moment=mass_moment,
        )

    def _count_elements(self, mol: 'Chem.Mol') -> Dict[str, int]:
        """Count atoms of each element."""
        counts = {}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            counts[symbol] = counts.get(symbol, 0) + 1
        return counts

    def _check_deuterium(self, mol: 'Chem.Mol') -> Tuple[bool, int]:
        """Check for deuterium atoms."""
        d_count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'H' and atom.GetIsotope() == 2:
                d_count += 1

        return d_count > 0, d_count

    def _calculate_average_reduced_mass(self, mol: 'Chem.Mol') -> float:
        """
        Calculate average reduced mass for vibrational modes.

        Uses harmonic approximation for all bonds.
        """
        if mol.GetNumBonds() == 0:
            return 0.0

        total_reduced_mass = 0.0

        for bond in mol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()

            m1 = atom1.GetMass()
            m2 = atom2.GetMass()

            # Reduced mass: μ = (m1 * m2) / (m1 + m2)
            reduced_mass = (m1 * m2) / (m1 + m2)
            total_reduced_mass += reduced_mass

        return total_reduced_mass / mol.GetNumBonds()

    def _calculate_mass_distribution_moment(self, mol: 'Chem.Mol') -> float:
        """
        Calculate second moment of mass distribution.

        Measures how mass is distributed in space.
        """
        if mol.GetNumConformers() == 0:
            # Can't calculate without 3D coordinates
            return 0.0

        conf = mol.GetConformer()

        # Get center of mass
        total_mass = 0.0
        com = np.zeros(3)

        for i, atom in enumerate(mol.GetAtoms()):
            mass = atom.GetMass()
            pos = conf.GetAtomPosition(i)
            com += mass * np.array([pos.x, pos.y, pos.z])
            total_mass += mass

        com /= total_mass

        # Calculate second moment
        moment = 0.0
        for i, atom in enumerate(mol.GetAtoms()):
            mass = atom.GetMass()
            pos = conf.GetAtomPosition(i)
            r = np.array([pos.x, pos.y, pos.z]) - com
            moment += mass * np.dot(r, r)

        return float(moment)

    def substitute_hydrogens_with_deuterium(self, smiles: str) -> str:
        """
        Create deuterated version of molecule.

        Args:
            smiles: SMILES string

        Returns:
            SMILES string with all H replaced by D
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit required")

        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        # Replace all hydrogens with deuterium
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'H':
                atom.SetIsotope(2)  # Deuterium

        # Convert back to SMILES
        smiles_d = Chem.MolToSmiles(mol)
        return smiles_d

    def calculate_isotope_frequency_shift(self,
                                         mass_h: float,
                                         mass_d: float) -> float:
        """
        Calculate expected frequency shift due to isotope substitution.

        Frequency ratio: ν_D/ν_H = sqrt(μ_H/μ_D)

        For H→D: ratio ≈ 1/√2 ≈ 0.707

        Args:
            mass_h: Mass with hydrogen
            mass_d: Mass with deuterium

        Returns:
            Frequency ratio (ν_D/ν_H)
        """
        # Simplified: assume mass ratio directly affects frequency
        # Real calculation would need force constants
        return np.sqrt(mass_h / mass_d)

    def compare_isotope_masses(self,
                              mol_h: 'Chem.Mol',
                              mol_d: 'Chem.Mol') -> Dict[str, float]:
        """
        Compare mass properties of H vs D versions.

        Args:
            mol_h: Molecule with hydrogen
            mol_d: Molecule with deuterium

        Returns:
            Dict with comparison metrics
        """
        props_h = self.calculate_properties(mol_h)
        props_d = self.calculate_properties(mol_d)

        mass_ratio = props_d.molecular_weight / props_h.molecular_weight
        reduced_mass_ratio = props_d.reduced_mass / props_h.reduced_mass
        frequency_ratio = np.sqrt(props_h.reduced_mass / props_d.reduced_mass)

        return {
            'mass_h': props_h.molecular_weight,
            'mass_d': props_d.molecular_weight,
            'mass_ratio': mass_ratio,
            'reduced_mass_h': props_h.reduced_mass,
            'reduced_mass_d': props_d.reduced_mass,
            'reduced_mass_ratio': reduced_mass_ratio,
            'expected_frequency_ratio': frequency_ratio,
            'theoretical_frequency_ratio': 1.0 / np.sqrt(2.0),
        }

    def save_mass_properties(self,
                            properties: MassProperties,
                            output_path: str,
                            smiles: Optional[str] = None,
                            metadata: Optional[Dict] = None) -> None:
        """
        Save mass properties to JSON file.

        Args:
            properties: MassProperties object
            output_path: Path to output JSON file
            smiles: Optional SMILES string
            metadata: Optional metadata
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            'timestamp': datetime.now().isoformat(),
            'smiles': smiles,
            'properties': properties.to_dict()
        }

        if metadata:
            output_data['metadata'] = metadata

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"✓ Saved mass properties to {output_path}")

    def save_isotope_comparison(self,
                               comparison: Dict[str, float],
                               output_path: str,
                               smiles_h: Optional[str] = None,
                               smiles_d: Optional[str] = None,
                               metadata: Optional[Dict] = None) -> None:
        """
        Save isotope comparison results to JSON file.

        Args:
            comparison: Comparison dictionary
            output_path: Path to output JSON file
            smiles_h: SMILES for hydrogen version
            smiles_d: SMILES for deuterium version
            metadata: Optional metadata
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            'timestamp': datetime.now().isoformat(),
            'smiles_h': smiles_h,
            'smiles_d': smiles_d,
            'comparison': comparison,
            'theory_note': 'Oscillatory theory predicts different frequencies → different scents'
        }

        if metadata:
            output_data['metadata'] = metadata

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"✓ Saved isotope comparison to {output_path}")


def demonstrate_mass_properties():
    """Demonstrate mass property calculations."""
    print("="*80)
    print("MASS PROPERTIES DEMONSTRATION")
    print("="*80 + "\n")

    if not RDKIT_AVAILABLE:
        print("✗ RDKit not available. Install with: pip install rdkit")
        return

    from rdkit import Chem
    from rdkit.Chem import AllChem

    calculator = MassPropertiesCalculator()

    # Example: Benzene
    print("Molecule: Benzene (C₆H₆)")
    smiles_h = 'c1ccccc1'
    mol_h = Chem.MolFromSmiles(smiles_h)
    mol_h = Chem.AddHs(mol_h)

    props_h = calculator.calculate_properties(mol_h)

    print(f"  Molecular weight: {props_h.molecular_weight:.3f} amu")
    print(f"  Exact mass: {props_h.exact_mass:.3f} amu")
    print(f"  Heavy atoms: {props_h.heavy_atom_count}")
    print(f"  Element composition: {props_h.element_counts}")
    print(f"  Average reduced mass: {props_h.reduced_mass:.3f} amu")
    print()

    # Deuterated version
    print("Molecule: Deuterated Benzene (C₆D₆)")
    smiles_d = calculator.substitute_hydrogens_with_deuterium(smiles_h)
    mol_d = Chem.MolFromSmiles(smiles_d)
    mol_d = Chem.AddHs(mol_d)

    props_d = calculator.calculate_properties(mol_d)

    print(f"  Molecular weight: {props_d.molecular_weight:.3f} amu")
    print(f"  Exact mass: {props_d.exact_mass:.3f} amu")
    print(f"  Deuterium count: {props_d.deuterium_count}")
    print(f"  Average reduced mass: {props_d.reduced_mass:.3f} amu")
    print()

    # Compare
    print("Isotope Comparison (H vs D):")
    print("-" * 80)
    comparison = calculator.compare_isotope_masses(mol_h, mol_d)

    print(f"  Mass ratio (D/H): {comparison['mass_ratio']:.3f}")
    print(f"  Reduced mass ratio (D/H): {comparison['reduced_mass_ratio']:.3f}")
    print(f"  Expected frequency ratio (D/H): {comparison['expected_frequency_ratio']:.3f}")
    print(f"  Theoretical frequency ratio: {comparison['theoretical_frequency_ratio']:.3f}")
    print()

    print("CRITICAL TEST: Isotope Effect")
    print("-" * 80)
    print("Oscillatory theory predicts: Different frequencies → Different scents")
    print("Classical shape theory predicts: Same shape → Same scent")
    print()
    print(f"Frequency shift: {(1 - comparison['expected_frequency_ratio'])*100:.1f}%")
    print("This frequency shift should produce DIFFERENT scents!")
    print()

    # Save results
    print("Saving results...")
    print("-" * 80)

    output_dir = Path("results/mass_properties")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save hydrogen version properties
    calculator.save_mass_properties(
        props_h,
        output_dir / "benzene_h_mass.json",
        smiles=smiles_h,
        metadata={'description': 'Mass properties of benzene (C₆H₆)'}
    )

    # Save deuterium version properties
    calculator.save_mass_properties(
        props_d,
        output_dir / "benzene_d_mass.json",
        smiles=smiles_d,
        metadata={'description': 'Mass properties of deuterated benzene (C₆D₆)'}
    )

    # Save isotope comparison
    calculator.save_isotope_comparison(
        comparison,
        output_dir / "isotope_comparison.json",
        smiles_h=smiles_h,
        smiles_d=smiles_d,
        metadata={'description': 'Critical test of oscillatory vs shape theory'}
    )

    print("✓ Demonstration complete!")


if __name__ == "__main__":
    demonstrate_mass_properties()
