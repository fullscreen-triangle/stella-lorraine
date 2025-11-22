"""
Molecular Geometry Calculator

Calculates 3D geometric properties of molecules for oscillatory signatures.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


@dataclass
class GeometricProperties:
    """3D geometric properties of a molecule."""
    # Center of mass
    center_of_mass: np.ndarray  # [x, y, z]
    
    # Moments of inertia
    principal_moments: np.ndarray  # [I1, I2, I3] sorted descending
    principal_axes: np.ndarray  # [3, 3] rotation matrix
    
    # Shape descriptors
    asphericity: float  # 0 = spherical, 1 = linear
    eccentricity: float  # 0 = spherical, 1 = very elongated
    radius_of_gyration: float  # Angstroms
    
    # Spatial extent
    max_distance: float  # Maximum interatomic distance
    molecular_diameter: float  # Effective diameter
    molecular_volume: float  # Volume in Ų
    surface_area: float  # Surface area in Ų
    
    # Dipole moment (if calculated)
    dipole_moment: Optional[float] = None  # Debye
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'center_of_mass': self.center_of_mass.tolist(),
            'principal_moments': self.principal_moments.tolist(),
            'principal_axes': self.principal_axes.tolist(),
            'asphericity': float(self.asphericity),
            'eccentricity': float(self.eccentricity),
            'radius_of_gyration': float(self.radius_of_gyration),
            'max_distance': float(self.max_distance),
            'molecular_diameter': float(self.molecular_diameter),
            'molecular_volume': float(self.molecular_volume),
            'surface_area': float(self.surface_area),
            'dipole_moment': float(self.dipole_moment) if self.dipole_moment else None
        }


class GeometryCalculator:
    """
    Calculates 3D geometric properties of molecules.
    
    Provides:
    - Center of mass
    - Moments of inertia
    - Shape descriptors (asphericity, eccentricity)
    - Spatial extent metrics
    """
    
    # Van der Waals radii (Angstroms)
    VDW_RADII = {
        'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52,
        'F': 1.47, 'P': 1.80, 'S': 1.80, 'Cl': 1.75,
        'Br': 1.85, 'I': 1.98
    }
    
    def __init__(self):
        """Initialize geometry calculator."""
        if not RDKIT_AVAILABLE:
            print("Warning: RDKit not available. Geometry calculations will be limited.")
    
    def calculate_properties(self, mol: 'Chem.Mol') -> GeometricProperties:
        """
        Calculate all geometric properties for a molecule.
        
        Args:
            mol: RDKit Mol object with 3D coordinates
            
        Returns:
            GeometricProperties object
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit required for geometry calculations")
        
        # Ensure 3D coordinates exist
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
        
        # Get coordinates and masses
        coords, masses = self._get_coords_and_masses(mol)
        
        # Calculate center of mass
        com = self._calculate_center_of_mass(coords, masses)
        
        # Calculate moments of inertia
        principal_moments, principal_axes = self._calculate_inertia_tensor(coords, masses, com)
        
        # Calculate shape descriptors
        asphericity = self._calculate_asphericity(principal_moments)
        eccentricity = self._calculate_eccentricity(principal_moments)
        radius_of_gyration = self._calculate_radius_of_gyration(coords, masses, com)
        
        # Calculate spatial extent
        max_distance = self._calculate_max_distance(coords)
        molecular_diameter = self._calculate_molecular_diameter(coords, mol)
        
        # Calculate volume and surface area
        molecular_volume = self._calculate_volume(mol)
        surface_area = self._calculate_surface_area(mol)
        
        return GeometricProperties(
            center_of_mass=com,
            principal_moments=principal_moments,
            principal_axes=principal_axes,
            asphericity=asphericity,
            eccentricity=eccentricity,
            radius_of_gyration=radius_of_gyration,
            max_distance=max_distance,
            molecular_diameter=molecular_diameter,
            molecular_volume=molecular_volume,
            surface_area=surface_area,
        )
    
    def _get_coords_and_masses(self, mol: 'Chem.Mol') -> Tuple[np.ndarray, np.ndarray]:
        """Extract 3D coordinates and atomic masses."""
        conf = mol.GetConformer()
        
        coords = []
        masses = []
        
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])
            masses.append(atom.GetMass())
        
        return np.array(coords), np.array(masses)
    
    def _calculate_center_of_mass(self, coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """Calculate center of mass."""
        total_mass = np.sum(masses)
        com = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
        return com
    
    def _calculate_inertia_tensor(self, 
                                  coords: np.ndarray, 
                                  masses: np.ndarray, 
                                  com: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate moment of inertia tensor and principal axes.
        
        Returns:
            principal_moments: [I1, I2, I3] sorted descending
            principal_axes: [3, 3] rotation matrix
        """
        # Centered coordinates
        centered = coords - com
        
        # Build inertia tensor
        I = np.zeros((3, 3))
        
        for i in range(3):
            for j in range(3):
                if i == j:
                    # Diagonal elements
                    I[i, j] = np.sum(masses * (np.sum(centered**2, axis=1) - centered[:, i]**2))
                else:
                    # Off-diagonal elements
                    I[i, j] = -np.sum(masses * centered[:, i] * centered[:, j])
        
        # Diagonalize to get principal moments and axes
        eigenvalues, eigenvectors = np.linalg.eigh(I)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        principal_moments = eigenvalues[idx]
        principal_axes = eigenvectors[:, idx]
        
        return principal_moments, principal_axes
    
    def _calculate_asphericity(self, principal_moments: np.ndarray) -> float:
        """
        Calculate asphericity parameter.
        
        Asphericity = (I1 - (I2 + I3)/2) / (I1 + I2 + I3)
        
        Range: [0, 1] where 0 = sphere, 1 = rod
        """
        I1, I2, I3 = principal_moments
        total = I1 + I2 + I3
        
        if total > 0:
            asphericity = (I1 - (I2 + I3)/2) / total
        else:
            asphericity = 0.0
        
        return float(asphericity)
    
    def _calculate_eccentricity(self, principal_moments: np.ndarray) -> float:
        """
        Calculate eccentricity.
        
        Eccentricity = sqrt(1 - I_min/I_max)
        
        Range: [0, 1] where 0 = sphere, 1 = very elongated
        """
        I_max = np.max(principal_moments)
        I_min = np.min(principal_moments)
        
        if I_max > 0:
            eccentricity = np.sqrt(1.0 - I_min / I_max)
        else:
            eccentricity = 0.0
        
        return float(eccentricity)
    
    def _calculate_radius_of_gyration(self, 
                                     coords: np.ndarray, 
                                     masses: np.ndarray, 
                                     com: np.ndarray) -> float:
        """
        Calculate radius of gyration.
        
        R_g = sqrt(Σ m_i * r_i² / Σ m_i)
        """
        centered = coords - com
        distances_sq = np.sum(centered**2, axis=1)
        total_mass = np.sum(masses)
        
        R_g = np.sqrt(np.sum(masses * distances_sq) / total_mass)
        return float(R_g)
    
    def _calculate_max_distance(self, coords: np.ndarray) -> float:
        """Calculate maximum interatomic distance."""
        from scipy.spatial.distance import pdist
        
        distances = pdist(coords)
        return float(np.max(distances))
    
    def _calculate_molecular_diameter(self, coords: np.ndarray, mol: 'Chem.Mol') -> float:
        """
        Calculate effective molecular diameter including van der Waals radii.
        """
        # Maximum distance between atomic centers
        max_center_dist = self._calculate_max_distance(coords)
        
        # Add van der Waals radii of the two furthest atoms
        # Simplified: just add average VDW radius
        avg_vdw = np.mean([self.VDW_RADII.get(atom.GetSymbol(), 1.7) 
                          for atom in mol.GetAtoms()])
        
        return max_center_dist + 2 * avg_vdw
    
    def _calculate_volume(self, mol: 'Chem.Mol') -> float:
        """Calculate molecular volume."""
        try:
            return AllChem.ComputeMolVolume(mol)
        except:
            # Fallback: sum of atomic volumes
            total_volume = 0.0
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                radius = self.VDW_RADII.get(symbol, 1.70)
                # Volume of sphere
                volume = (4.0/3.0) * np.pi * (radius ** 3)
                total_volume += volume
            return total_volume
    
    def _calculate_surface_area(self, mol: 'Chem.Mol') -> float:
        """Calculate molecular surface area."""
        try:
            # Use RDKit's SASA (Solvent Accessible Surface Area)
            from rdkit.Chem import Descriptors3D
            return Descriptors3D.CalcLabuteASA(mol)
        except:
            # Fallback: sum of atomic surface areas
            total_area = 0.0
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                radius = self.VDW_RADII.get(symbol, 1.70)
                # Surface area of sphere
                area = 4.0 * np.pi * (radius ** 2)
                total_area += area
            return total_area * 0.75  # Correction factor for overlap
    
    def save_geometry(self,
                     properties: GeometricProperties,
                     output_path: str,
                     smiles: Optional[str] = None,
                     metadata: Optional[Dict] = None) -> None:
        """
        Save geometric properties to JSON file.
        
        Args:
            properties: GeometricProperties object
            output_path: Path to output JSON file
            smiles: Optional SMILES string
            metadata: Optional metadata
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'smiles': smiles,
            'geometry': properties.to_dict()
        }
        
        if metadata:
            output_data['metadata'] = metadata
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Saved geometry to {output_path}")
    
    def save_batch_geometries(self,
                             geometries_dict: Dict[str, GeometricProperties],
                             output_path: str,
                             metadata: Optional[Dict] = None) -> None:
        """
        Save batch of geometric properties to JSON file.
        
        Args:
            geometries_dict: Dict[molecule_name] = GeometricProperties
            output_path: Path to output JSON file
            metadata: Optional metadata
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        geometries_data = {
            name: geom.to_dict()
            for name, geom in geometries_dict.items()
        }
        
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'num_molecules': len(geometries_data),
            'geometries': geometries_data
        }
        
        if metadata:
            output_data['metadata'] = metadata
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Saved batch geometries to {output_path}")


def demonstrate_geometry_calculation():
    """Demonstrate geometry calculations."""
    print("="*80)
    print("MOLECULAR GEOMETRY CALCULATION DEMONSTRATION")
    print("="*80 + "\n")
    
    if not RDKIT_AVAILABLE:
        print("✗ RDKit not available. Install with: pip install rdkit")
        return
    
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    # Example molecules
    molecules = {
        'Methane (spherical)': 'C',
        'Benzene (planar)': 'c1ccccc1',
        'Octane (linear)': 'CCCCCCCC',
        'Vanillin': 'COc1cc(C=O)ccc1O',
    }
    
    calculator = GeometryCalculator()
    
    geometries_dict = {}
    
    for name, smiles in molecules.items():
        print(f"Molecule: {name}")
        print(f"SMILES: {smiles}")
        
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        
        # Generate 3D
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        props = calculator.calculate_properties(mol)
        geometries_dict[name] = props
        
        print(f"  Center of mass: [{props.center_of_mass[0]:.2f}, "
              f"{props.center_of_mass[1]:.2f}, {props.center_of_mass[2]:.2f}]")
        print(f"  Principal moments: [{props.principal_moments[0]:.2f}, "
              f"{props.principal_moments[1]:.2f}, {props.principal_moments[2]:.2f}]")
        print(f"  Asphericity: {props.asphericity:.3f} (0=sphere, 1=rod)")
        print(f"  Eccentricity: {props.eccentricity:.3f} (0=sphere, 1=elongated)")
        print(f"  Radius of gyration: {props.radius_of_gyration:.2f} Å")
        print(f"  Max distance: {props.max_distance:.2f} Å")
        print(f"  Molecular diameter: {props.molecular_diameter:.2f} Å")
        print(f"  Volume: {props.molecular_volume:.2f} Ų")
        print(f"  Surface area: {props.surface_area:.2f} Ų")
        print()
    
    # Save results
    print("Saving results...")
    print("-" * 80)
    
    output_dir = Path("results/geometry")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save batch geometries
    calculator.save_batch_geometries(
        geometries_dict,
        output_dir / "molecular_geometries.json",
        metadata={'description': '3D geometric properties of example molecules'}
    )
    
    print("✓ Demonstration complete!")


if __name__ == "__main__":
    demonstrate_geometry_calculation()

