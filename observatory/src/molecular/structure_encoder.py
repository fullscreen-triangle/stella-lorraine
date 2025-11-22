"""
Molecular Structure Encoder

Converts molecular structures (SMILES, SDF) into feature vectors for
oscillatory signature generation.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Molecular structure encoding will be limited.")


@dataclass
class MolecularFeatures:
    """Container for molecular features."""
    n_atoms: int
    n_bonds: int
    n_rings: int
    molecular_weight: float
    n_rotatable_bonds: int
    n_h_bond_donors: int
    n_h_bond_acceptors: int
    topological_polar_surface_area: float
    n_aromatic_rings: int
    n_saturated_rings: int
    n_heteroatoms: int
    formal_charge: int
    n_stereocenters: int
    
    # Bond type counts
    n_single_bonds: int
    n_double_bonds: int
    n_triple_bonds: int
    n_aromatic_bonds: int
    
    # Atom type counts
    n_carbon: int
    n_hydrogen: int
    n_oxygen: int
    n_nitrogen: int
    n_other: int
    
    # Geometry features (if 3D)
    molecular_volume: Optional[float] = None
    asphericity: Optional[float] = None
    eccentricity: Optional[float] = None
    
    def to_vector(self) -> np.ndarray:
        """Convert features to numpy vector."""
        features = [
            self.n_atoms,
            self.n_bonds,
            self.n_rings,
            self.molecular_weight,
            self.n_rotatable_bonds,
            self.n_h_bond_donors,
            self.n_h_bond_acceptors,
            self.topological_polar_surface_area,
            self.n_aromatic_rings,
            self.n_saturated_rings,
            self.n_heteroatoms,
            self.formal_charge,
            self.n_stereocenters,
            self.n_single_bonds,
            self.n_double_bonds,
            self.n_triple_bonds,
            self.n_aromatic_bonds,
            self.n_carbon,
            self.n_hydrogen,
            self.n_oxygen,
            self.n_nitrogen,
            self.n_other,
        ]
        
        # Add geometry features if available
        if self.molecular_volume is not None:
            features.extend([
                self.molecular_volume,
                self.asphericity or 0.0,
                self.eccentricity or 0.0,
            ])
        
        return np.array(features, dtype=np.float64)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert float values
        for key in ['molecular_weight', 'topological_polar_surface_area', 
                    'molecular_volume', 'asphericity', 'eccentricity']:
            if d.get(key) is not None:
                d[key] = float(d[key])
        return d


class MolecularStructureEncoder:
    """
    Encodes molecular structures into feature vectors.
    
    Supports:
    - SMILES strings
    - RDKit Mol objects
    - 2D and 3D coordinates
    """
    
    def __init__(self, generate_3d: bool = True):
        """
        Args:
            generate_3d: If True, generate 3D coordinates for geometry features
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for molecular structure encoding")
        
        self.generate_3d = generate_3d
    
    def encode_smiles(self, smiles: str) -> MolecularFeatures:
        """
        Encode SMILES string into molecular features.
        
        Args:
            smiles: SMILES string representation
            
        Returns:
            MolecularFeatures object
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        return self.encode_mol(mol)
    
    def encode_mol(self, mol: 'Chem.Mol') -> MolecularFeatures:
        """
        Encode RDKit Mol object into molecular features.
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            MolecularFeatures object
        """
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates if requested
        if self.generate_3d:
            try:
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                pass  # 3D generation failed, continue with 2D
        
        # Extract features
        features = MolecularFeatures(
            n_atoms=mol.GetNumAtoms(),
            n_bonds=mol.GetNumBonds(),
            n_rings=Descriptors.RingCount(mol),
            molecular_weight=Descriptors.MolWt(mol),
            n_rotatable_bonds=Descriptors.NumRotatableBonds(mol),
            n_h_bond_donors=Descriptors.NumHDonors(mol),
            n_h_bond_acceptors=Descriptors.NumHAcceptors(mol),
            topological_polar_surface_area=Descriptors.TPSA(mol),
            n_aromatic_rings=Descriptors.NumAromaticRings(mol),
            n_saturated_rings=Descriptors.NumSaturatedRings(mol),
            n_heteroatoms=Descriptors.NumHeteroatoms(mol),
            formal_charge=Chem.GetFormalCharge(mol),
            n_stereocenters=len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
            
            # Bond type counts
            n_single_bonds=self._count_bond_type(mol, Chem.BondType.SINGLE),
            n_double_bonds=self._count_bond_type(mol, Chem.BondType.DOUBLE),
            n_triple_bonds=self._count_bond_type(mol, Chem.BondType.TRIPLE),
            n_aromatic_bonds=sum(1 for bond in mol.GetBonds() if bond.GetIsAromatic()),
            
            # Atom type counts
            n_carbon=self._count_atom_type(mol, 'C'),
            n_hydrogen=self._count_atom_type(mol, 'H'),
            n_oxygen=self._count_atom_type(mol, 'O'),
            n_nitrogen=self._count_atom_type(mol, 'N'),
            n_other=self._count_other_atoms(mol),
        )
        
        # Add geometry features if 3D is available
        if mol.GetNumConformers() > 0:
            features.molecular_volume = self._calculate_volume(mol)
            features.asphericity = self._calculate_asphericity(mol)
            features.eccentricity = self._calculate_eccentricity(mol)
        
        return features
    
    def _count_bond_type(self, mol: 'Chem.Mol', bond_type: 'Chem.BondType') -> int:
        """Count bonds of specific type."""
        return sum(1 for bond in mol.GetBonds() if bond.GetBondType() == bond_type)
    
    def _count_atom_type(self, mol: 'Chem.Mol', symbol: str) -> int:
        """Count atoms of specific type."""
        return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == symbol)
    
    def _count_other_atoms(self, mol: 'Chem.Mol') -> int:
        """Count atoms that are not C, H, O, N."""
        common = {'C', 'H', 'O', 'N'}
        return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() not in common)
    
    def _calculate_volume(self, mol: 'Chem.Mol') -> float:
        """
        Calculate molecular volume from 3D coordinates.
        
        Uses grid-based volume calculation.
        """
        try:
            # Use RDKit's built-in volume calculation
            return AllChem.ComputeMolVolume(mol)
        except:
            # Fallback: estimate from van der Waals radii
            return self._estimate_volume_from_vdw(mol)
    
    def _estimate_volume_from_vdw(self, mol: 'Chem.Mol') -> float:
        """Estimate volume from van der Waals radii."""
        # Van der Waals radii (Angstroms)
        vdw_radii = {'C': 1.70, 'H': 1.20, 'O': 1.52, 'N': 1.55, 'S': 1.80, 'P': 1.80}
        
        total_volume = 0.0
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            radius = vdw_radii.get(symbol, 1.70)  # Default to carbon
            # Volume of sphere: (4/3) * π * r³
            total_volume += (4.0/3.0) * np.pi * (radius ** 3)
        
        return total_volume
    
    def _calculate_asphericity(self, mol: 'Chem.Mol') -> float:
        """
        Calculate asphericity (deviation from spherical shape).
        
        Based on moments of inertia tensor.
        """
        if mol.GetNumConformers() == 0:
            return 0.0
        
        # Get 3D coordinates
        conf = mol.GetConformer()
        coords = []
        masses = []
        
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])
            masses.append(atom.GetMass())
        
        coords = np.array(coords)
        masses = np.array(masses)
        
        # Center of mass
        com = np.sum(coords * masses[:, np.newaxis], axis=0) / np.sum(masses)
        centered = coords - com
        
        # Moment of inertia tensor
        I = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                if i == j:
                    I[i, j] = np.sum(masses * (np.sum(centered**2, axis=1) - centered[:, i]**2))
                else:
                    I[i, j] = -np.sum(masses * centered[:, i] * centered[:, j])
        
        # Eigenvalues (principal moments)
        eigenvalues = np.linalg.eigvalsh(I)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
        
        # Asphericity = (λ1 - (λ2 + λ3)/2) / (λ1 + λ2 + λ3)
        if np.sum(eigenvalues) > 0:
            asphericity = (eigenvalues[0] - (eigenvalues[1] + eigenvalues[2])/2) / np.sum(eigenvalues)
        else:
            asphericity = 0.0
        
        return float(asphericity)
    
    def _calculate_eccentricity(self, mol: 'Chem.Mol') -> float:
        """
        Calculate eccentricity from principal moments of inertia.
        
        Eccentricity = sqrt(1 - (I_min/I_max))
        """
        if mol.GetNumConformers() == 0:
            return 0.0
        
        # Get 3D coordinates
        conf = mol.GetConformer()
        coords = []
        masses = []
        
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])
            masses.append(atom.GetMass())
        
        coords = np.array(coords)
        masses = np.array(masses)
        
        # Center of mass
        com = np.sum(coords * masses[:, np.newaxis], axis=0) / np.sum(masses)
        centered = coords - com
        
        # Moment of inertia tensor
        I = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                if i == j:
                    I[i, j] = np.sum(masses * (np.sum(centered**2, axis=1) - centered[:, i]**2))
                else:
                    I[i, j] = -np.sum(masses * centered[:, i] * centered[:, j])
        
        # Eigenvalues (principal moments)
        eigenvalues = np.linalg.eigvalsh(I)
        eigenvalues = eigenvalues[eigenvalues > 0]  # Filter out zeros
        
        if len(eigenvalues) < 2:
            return 0.0
        
        I_min = np.min(eigenvalues)
        I_max = np.max(eigenvalues)
        
        if I_max > 0:
            eccentricity = np.sqrt(1.0 - I_min / I_max)
        else:
            eccentricity = 0.0
        
        return float(eccentricity)
    
    def batch_encode_smiles(self, smiles_list: List[str]) -> List[MolecularFeatures]:
        """
        Encode multiple SMILES strings in batch.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of MolecularFeatures objects
        """
        features_list = []
        
        for smiles in smiles_list:
            try:
                features = self.encode_smiles(smiles)
                features_list.append(features)
            except Exception as e:
                print(f"Warning: Failed to encode {smiles}: {e}")
                # Append None or skip
                features_list.append(None)
        
        return features_list
    
    def save_features(self,
                     features: MolecularFeatures,
                     output_path: str,
                     smiles: Optional[str] = None,
                     metadata: Optional[Dict] = None) -> None:
        """
        Save molecular features to JSON file.
        
        Args:
            features: MolecularFeatures object
            output_path: Path to output JSON file
            smiles: Optional SMILES string
            metadata: Optional metadata
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'smiles': smiles,
            'features': features.to_dict(),
            'feature_vector_length': len(features.to_vector())
        }
        
        if metadata:
            output_data['metadata'] = metadata
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Saved molecular features to {output_path}")
    
    def save_batch_features(self,
                           features_dict: Dict[str, MolecularFeatures],
                           output_path: str,
                           metadata: Optional[Dict] = None) -> None:
        """
        Save batch of molecular features to JSON file.
        
        Args:
            features_dict: Dict[molecule_name] = MolecularFeatures
            output_path: Path to output JSON file
            metadata: Optional metadata
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        features_data = {
            name: feat.to_dict() 
            for name, feat in features_dict.items()
            if feat is not None
        }
        
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'num_molecules': len(features_data),
            'features': features_data
        }
        
        if metadata:
            output_data['metadata'] = metadata
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Saved batch features to {output_path}")


def demonstrate_encoding():
    """Demonstrate molecular structure encoding."""
    print("="*80)
    print("MOLECULAR STRUCTURE ENCODING DEMONSTRATION")
    print("="*80 + "\n")
    
    if not RDKIT_AVAILABLE:
        print("✗ RDKit not available. Install with: pip install rdkit")
        return
    
    # Example molecules
    molecules = {
        'Vanillin': 'COc1cc(C=O)ccc1O',
        'Benzene': 'c1ccccc1',
        'Ethanol': 'CCO',
        'Indole': 'c1ccc2c(c1)cc[nH]2',
    }
    
    encoder = MolecularStructureEncoder(generate_3d=True)
    
    features_dict = {}
    
    for name, smiles in molecules.items():
        print(f"Encoding: {name}")
        print(f"SMILES: {smiles}")
        
        features = encoder.encode_smiles(smiles)
        features_dict[name] = features
        
        print(f"  Atoms: {features.n_atoms}")
        print(f"  Bonds: {features.n_bonds}")
        print(f"  Rings: {features.n_rings}")
        print(f"  MW: {features.molecular_weight:.2f}")
        print(f"  Aromatic rings: {features.n_aromatic_rings}")
        
        if features.molecular_volume is not None:
            print(f"  Volume: {features.molecular_volume:.2f} Ų")
            print(f"  Asphericity: {features.asphericity:.3f}")
            print(f"  Eccentricity: {features.eccentricity:.3f}")
        
        vector = features.to_vector()
        print(f"  Feature vector length: {len(vector)}")
        print()
    
    # Save results
    print("Saving results...")
    print("-" * 80)
    
    output_dir = Path("results/structure_encoding")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save batch features
    encoder.save_batch_features(
        features_dict,
        output_dir / "molecular_features.json",
        metadata={'description': 'Structural features of example molecules'}
    )
    
    print("✓ Demonstration complete!")


if __name__ == "__main__":
    demonstrate_encoding()

