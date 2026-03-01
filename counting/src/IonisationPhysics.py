"""
Ionization Physics: ESI, MALDI, EI as Partition Operations

Ionization is a partition operation that:
1. Changes the molecule's charge state (adds/removes electrons)
2. Transitions the molecule from solution/solid to gas phase
3. Assigns initial partition coordinates (n, ℓ, m, s)

The ionization process IS a categorical state transition:
    Neutral molecule → Ion with partition coordinates

Three ionization methods implemented:
1. Electrospray Ionization (ESI) - soft ionization for large biomolecules
2. Matrix-Assisted Laser Desorption/Ionization (MALDI) - soft ionization
3. Electron Ionization (EI) - hard ionization for small molecules
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Physical constants
K_B = 1.380649e-23  # Boltzmann constant (J/K)
E_CHARGE = 1.602176634e-19  # Elementary charge (C)
AMU = 1.66053906660e-27  # Atomic mass unit (kg)
HBAR = 1.054571817e-34  # Reduced Planck constant
EPSILON_0 = 8.854187817e-12  # Vacuum permittivity (F/m)
ELECTRON_MASS = 9.1093837015e-31  # Electron mass (kg)


class IonizationMethod(Enum):
    """Available ionization methods."""
    ESI = "electrospray"
    MALDI = "maldi"
    EI = "electron_ionization"
    APCI = "atmospheric_pressure_chemical"
    APPI = "atmospheric_pressure_photo"


class IonPolarity(Enum):
    """Ion polarity."""
    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass
class PartitionCoordinates:
    """
    Partition coordinates (n, ℓ, m, s) assigned during ionization.

    n: Principal depth (energy level) - n ≥ 1
    ℓ: Angular complexity (0 ≤ ℓ < n)
    m: Orientation (-ℓ ≤ m ≤ ℓ)
    s: Chirality/spin (±1/2)
    """
    n: int
    l: int
    m: int
    s: float

    @property
    def capacity(self) -> int:
        """Capacity at this depth: C(n) = 2n²."""
        return 2 * self.n * self.n

    def is_valid(self) -> bool:
        """Check if coordinates are valid."""
        if self.n < 1:
            return False
        if not 0 <= self.l < self.n:
            return False
        if not -self.l <= self.m <= self.l:
            return False
        if self.s not in [-0.5, 0.5]:
            return False
        return True


@dataclass
class IonizedSpecies:
    """Result of ionization process."""
    precursor_mass: float  # Neutral molecule mass (Da)
    ion_mass: float  # Ion mass (Da)
    charge: int  # Number of charges
    polarity: IonPolarity
    mz_ratio: float  # m/z
    partition_coords: PartitionCoordinates
    internal_energy_eV: float = 0.0  # Internal energy from ionization
    ionization_efficiency: float = 1.0  # Ionization efficiency (0-1)

    @property
    def is_protonated(self) -> bool:
        """Check if ion is protonated (positive mode)."""
        return self.polarity == IonPolarity.POSITIVE and self.charge > 0

    @property
    def is_deprotonated(self) -> bool:
        """Check if ion is deprotonated (negative mode)."""
        return self.polarity == IonPolarity.NEGATIVE and self.charge < 0


class ESIModel:
    """
    Electrospray Ionization (ESI) Physics Model.

    ESI is a soft ionization technique:
    1. Solution sprayed through charged capillary
    2. Droplets formed with excess charge
    3. Solvent evaporation concentrates charge
    4. Coulombic explosion produces charged droplets
    5. Ion evaporation model (IEM) or charged residue model (CRM)

    Categorical interpretation:
    - Droplet formation = initial partition assignment
    - Coulombic explosion = partition refinement
    - Desolvation = transition to gas-phase categorical state
    """

    PROTON_MASS = 1.007276  # Da
    ELECTRON_MASS_DA = 0.000548579909  # Da

    def __init__(
        self,
        spray_voltage: float = 3500,  # V
        capillary_temperature: float = 573,  # K (300°C)
        flow_rate_uL_min: float = 0.3
    ):
        self.spray_voltage = spray_voltage
        self.temperature = capillary_temperature
        self.flow_rate = flow_rate_uL_min

    def calculate_rayleigh_limit(
        self,
        droplet_radius_nm: float,
        surface_tension: float = 0.072  # N/m for water
    ) -> int:
        """
        Calculate Rayleigh charge limit for a droplet.

        q_R = 8π × √(ε₀ × γ × r³)

        Above this charge, droplet undergoes Coulombic explosion.
        """
        r = droplet_radius_nm * 1e-9  # Convert to meters
        q_R = 8 * np.pi * np.sqrt(EPSILON_0 * surface_tension * r**3)
        n_charges = int(q_R / E_CHARGE)
        return n_charges

    def predict_charge_state(
        self,
        molecular_mass: float,
        n_basic_sites: int = None,
        polarity: IonPolarity = IonPolarity.POSITIVE
    ) -> List[int]:
        """
        Predict likely charge states for a given molecule.

        For proteins: z_max ≈ M^0.5 (empirical)
        For small molecules: typically z = 1-3
        """
        if n_basic_sites is not None:
            # Charge limited by basic sites (positive mode)
            max_charge = n_basic_sites
        else:
            # Empirical relationship for proteins
            if molecular_mass > 5000:
                max_charge = int(np.sqrt(molecular_mass) / 10) + 1
            else:
                max_charge = min(3, int(molecular_mass / 500) + 1)

        # Return likely charge states
        charges = list(range(1, max_charge + 1))
        return charges

    def ionize(
        self,
        molecular_mass: float,
        n_basic_sites: int = None,
        polarity: IonPolarity = IonPolarity.POSITIVE,
        adduct: str = "H"  # H+, Na+, K+, etc.
    ) -> List[IonizedSpecies]:
        """
        Perform ESI ionization.

        Returns list of ionized species (different charge states).
        """
        charge_states = self.predict_charge_state(molecular_mass, n_basic_sites, polarity)
        species_list = []

        # Adduct masses
        adduct_masses = {
            "H": self.PROTON_MASS,
            "Na": 22.989769,
            "K": 38.963707,
            "NH4": 18.034374,
            "-H": -self.PROTON_MASS,  # Deprotonation
        }

        adduct_mass = adduct_masses.get(adduct, self.PROTON_MASS)

        for z in charge_states:
            # Calculate ion mass
            if polarity == IonPolarity.POSITIVE:
                ion_mass = molecular_mass + z * adduct_mass
                mz = ion_mass / z
            else:
                ion_mass = molecular_mass - z * self.PROTON_MASS
                mz = ion_mass / abs(z)
                z = -z

            # Assign partition coordinates based on mass and charge
            coords = self._assign_partition_coords(molecular_mass, abs(z))

            # Calculate ionization efficiency
            efficiency = self._calculate_efficiency(molecular_mass, z)

            # Internal energy (ESI is soft - low internal energy)
            internal_E = K_B * self.temperature / E_CHARGE  # ~0.05 eV

            species = IonizedSpecies(
                precursor_mass=molecular_mass,
                ion_mass=ion_mass,
                charge=z,
                polarity=polarity,
                mz_ratio=mz,
                partition_coords=coords,
                internal_energy_eV=internal_E,
                ionization_efficiency=efficiency
            )
            species_list.append(species)

        return species_list

    def _assign_partition_coords(self, mass: float, charge: int) -> PartitionCoordinates:
        """
        Assign partition coordinates based on molecular properties.

        The ionization process assigns initial (n, ℓ, m, s) coordinates.
        """
        # n (principal depth) from mass
        # Higher mass → higher n
        n = max(1, int(np.log10(mass + 1)) + 1)

        # ℓ (angular complexity) from charge state
        # Higher charge → higher angular complexity
        l = min(charge - 1, n - 1)

        # m (orientation) starts at 0 (can be refined by MS)
        m = 0

        # s (chirality/spin) from electron configuration
        s = 0.5  # Default

        return PartitionCoordinates(n=n, l=l, m=m, s=s)

    def _calculate_efficiency(self, mass: float, charge: int) -> float:
        """Calculate ionization efficiency."""
        # Empirical model: efficiency peaks around z = sqrt(M)/10
        optimal_charge = np.sqrt(mass) / 10
        deviation = abs(charge - optimal_charge) / optimal_charge if optimal_charge > 0 else 0
        efficiency = np.exp(-deviation)
        return min(1.0, efficiency)


class MALDIModel:
    """
    Matrix-Assisted Laser Desorption/Ionization (MALDI) Model.

    MALDI process:
    1. Analyte co-crystallized with matrix
    2. Laser pulse ablates crystal surface
    3. Matrix absorbs laser energy
    4. Energy transfer to analyte
    5. Proton transfer in gas-phase plume

    Categorical interpretation:
    - Crystallization = initial partition (solid state)
    - Laser ablation = partition excitation
    - Proton transfer = charge partition assignment
    """

    PROTON_MASS = 1.007276  # Da

    def __init__(
        self,
        laser_wavelength_nm: float = 337,  # N2 laser
        laser_fluence_J_cm2: float = 50,
        matrix: str = "CHCA"  # α-cyano-4-hydroxycinnamic acid
    ):
        self.wavelength = laser_wavelength_nm
        self.fluence = laser_fluence_J_cm2
        self.matrix = matrix

        # Matrix properties
        self.matrix_data = {
            "CHCA": {"mass": 189.04, "threshold_fluence": 30},
            "DHB": {"mass": 154.03, "threshold_fluence": 40},
            "SA": {"mass": 224.07, "threshold_fluence": 35},  # Sinapinic acid
        }

    def calculate_ion_yield(self, fluence: float) -> float:
        """
        Calculate ion yield as function of laser fluence.

        Above threshold: Y ∝ (F - F_th)^n
        """
        threshold = self.matrix_data.get(self.matrix, {}).get("threshold_fluence", 30)

        if fluence < threshold:
            return 0

        # Power law dependence
        n = 2  # Typical exponent
        yield_factor = ((fluence - threshold) / threshold) ** n
        return min(1.0, yield_factor)

    def ionize(
        self,
        molecular_mass: float,
        polarity: IonPolarity = IonPolarity.POSITIVE
    ) -> List[IonizedSpecies]:
        """
        Perform MALDI ionization.

        MALDI typically produces singly charged ions.
        """
        species_list = []

        # MALDI typically produces [M+H]+ or [M-H]-
        if polarity == IonPolarity.POSITIVE:
            ion_mass = molecular_mass + self.PROTON_MASS
            charge = 1
        else:
            ion_mass = molecular_mass - self.PROTON_MASS
            charge = -1

        mz = ion_mass / abs(charge)

        # Assign partition coordinates
        coords = self._assign_partition_coords(molecular_mass)

        # Ion yield
        yield_factor = self.calculate_ion_yield(self.fluence)

        # Internal energy (MALDI can deposit more energy than ESI)
        # Depends on fluence
        internal_E = 0.1 + 0.01 * (self.fluence - 30)  # eV

        species = IonizedSpecies(
            precursor_mass=molecular_mass,
            ion_mass=ion_mass,
            charge=charge,
            polarity=polarity,
            mz_ratio=mz,
            partition_coords=coords,
            internal_energy_eV=max(0, internal_E),
            ionization_efficiency=yield_factor
        )
        species_list.append(species)

        # Also consider matrix adducts
        matrix_mass = self.matrix_data.get(self.matrix, {}).get("mass", 189.04)
        adduct_mass = molecular_mass + matrix_mass + self.PROTON_MASS
        if polarity == IonPolarity.POSITIVE:
            species_list.append(IonizedSpecies(
                precursor_mass=molecular_mass,
                ion_mass=adduct_mass,
                charge=1,
                polarity=polarity,
                mz_ratio=adduct_mass,
                partition_coords=coords,
                internal_energy_eV=max(0, internal_E),
                ionization_efficiency=yield_factor * 0.1  # Less common
            ))

        return species_list

    def _assign_partition_coords(self, mass: float) -> PartitionCoordinates:
        """Assign partition coordinates."""
        n = max(1, int(np.log10(mass + 1)) + 1)
        l = 0  # MALDI typically gives ground state
        m = 0
        s = 0.5

        return PartitionCoordinates(n=n, l=l, m=m, s=s)


class EIModel:
    """
    Electron Ionization (EI) Model.

    EI process:
    1. Gaseous molecules enter ion source
    2. Collide with 70 eV electron beam
    3. Electron removed from molecule: M + e⁻ → M⁺• + 2e⁻
    4. Excess energy causes fragmentation

    Categorical interpretation:
    - Electron impact = violent partition transition
    - Fragmentation = partition cascade
    - Fragment pattern = partition fingerprint
    """

    ELECTRON_MASS_DA = 0.000548579909

    def __init__(
        self,
        electron_energy_eV: float = 70,  # Standard EI energy
        ion_source_temperature: float = 473  # K
    ):
        self.electron_energy = electron_energy_eV
        self.temperature = ion_source_temperature

    def calculate_ionization_cross_section(
        self,
        molecular_mass: float,
        n_electrons: int = None
    ) -> float:
        """
        Calculate EI cross-section.

        σ_EI ≈ A × n_electrons × E^(-1) for E > ionization potential
        """
        if n_electrons is None:
            # Estimate from mass (roughly 0.5 electrons per Da)
            n_electrons = int(molecular_mass * 0.5)

        # Simplified cross-section model
        A = 1e-20  # m² reference
        sigma = A * n_electrons / np.sqrt(self.electron_energy)

        return sigma

    def calculate_internal_energy(
        self,
        ionization_potential_eV: float = 10.0
    ) -> float:
        """
        Calculate internal energy deposited.

        E_internal = E_electron - IP + thermal energy
        """
        # Average internal energy (broad distribution)
        avg_internal = (self.electron_energy - ionization_potential_eV) * 0.5

        # Add thermal contribution
        avg_internal += K_B * self.temperature / E_CHARGE

        return max(0, avg_internal)

    def ionize(
        self,
        molecular_mass: float,
        ionization_potential_eV: float = 10.0
    ) -> List[IonizedSpecies]:
        """
        Perform EI ionization.

        EI produces radical cation M⁺• plus fragments.
        """
        species_list = []

        # Molecular ion M⁺•
        ion_mass = molecular_mass - self.ELECTRON_MASS_DA
        mz = ion_mass

        coords = self._assign_partition_coords(molecular_mass, excited=True)

        internal_E = self.calculate_internal_energy(ionization_potential_eV)
        cross_section = self.calculate_ionization_cross_section(molecular_mass)

        species = IonizedSpecies(
            precursor_mass=molecular_mass,
            ion_mass=ion_mass,
            charge=1,
            polarity=IonPolarity.POSITIVE,
            mz_ratio=mz,
            partition_coords=coords,
            internal_energy_eV=internal_E,
            ionization_efficiency=min(1.0, cross_section * 1e20)
        )
        species_list.append(species)

        # EI typically produces fragments (handled separately by CID module)

        return species_list

    def _assign_partition_coords(
        self,
        mass: float,
        excited: bool = False
    ) -> PartitionCoordinates:
        """
        Assign partition coordinates.

        EI deposits significant energy, so ion is often excited.
        """
        n = max(1, int(np.log10(mass + 1)) + 1)

        # EI produces excited states
        if excited:
            l = min(n - 1, 2)  # Excited angular state
        else:
            l = 0

        m = 0
        s = 0.5

        return PartitionCoordinates(n=n, l=l, m=m, s=s)


class IonizationEngine:
    """
    Unified ionization engine supporting multiple methods.

    Ionization is treated as a partition operation:
    Neutral molecule → Ionic species with (n, ℓ, m, s) coordinates

    This provides the bridge between neutral molecule and
    the partition-based analysis in mass spectrometry.
    """

    def __init__(self):
        self.esi = ESIModel()
        self.maldi = MALDIModel()
        self.ei = EIModel()

    def ionize(
        self,
        molecular_mass: float,
        method: IonizationMethod = IonizationMethod.ESI,
        polarity: IonPolarity = IonPolarity.POSITIVE,
        **kwargs
    ) -> List[IonizedSpecies]:
        """
        Ionize a molecule using specified method.

        Returns list of ionized species with partition coordinates assigned.
        """
        if method == IonizationMethod.ESI:
            return self.esi.ionize(
                molecular_mass,
                polarity=polarity,
                n_basic_sites=kwargs.get('n_basic_sites'),
                adduct=kwargs.get('adduct', 'H')
            )
        elif method == IonizationMethod.MALDI:
            return self.maldi.ionize(molecular_mass, polarity)
        elif method == IonizationMethod.EI:
            return self.ei.ionize(
                molecular_mass,
                ionization_potential_eV=kwargs.get('ionization_potential', 10.0)
            )
        else:
            raise ValueError(f"Unsupported ionization method: {method}")

    def compare_methods(
        self,
        molecular_mass: float,
        polarity: IonPolarity = IonPolarity.POSITIVE
    ) -> Dict[str, List[IonizedSpecies]]:
        """
        Compare ionization results from all methods.

        Shows how different methods assign different initial states
        but the partition coordinates provide a unified description.
        """
        results = {
            'ESI': self.ionize(molecular_mass, IonizationMethod.ESI, polarity),
            'MALDI': self.ionize(molecular_mass, IonizationMethod.MALDI, polarity),
            'EI': self.ionize(molecular_mass, IonizationMethod.EI, polarity)
        }

        return results

    def summarize_ionization(
        self,
        species: IonizedSpecies
    ) -> Dict[str, Any]:
        """
        Summarize ionization result.
        """
        return {
            'precursor_mass': species.precursor_mass,
            'ion_mass': species.ion_mass,
            'mz': species.mz_ratio,
            'charge': species.charge,
            'polarity': species.polarity.value,
            'partition_coordinates': {
                'n': species.partition_coords.n,
                'l': species.partition_coords.l,
                'm': species.partition_coords.m,
                's': species.partition_coords.s
            },
            'internal_energy_eV': species.internal_energy_eV,
            'efficiency': species.ionization_efficiency
        }


def ionize_molecule(
    molecular_mass: float,
    method: str = "ESI",
    polarity: str = "positive"
) -> List[Dict[str, Any]]:
    """
    Convenience function to ionize a molecule.

    Args:
        molecular_mass: Molecular mass in Da
        method: "ESI", "MALDI", or "EI"
        polarity: "positive" or "negative"

    Returns:
        List of ionization results with partition coordinates
    """
    engine = IonizationEngine()

    method_enum = IonizationMethod[method.upper()]
    polarity_enum = IonPolarity[polarity.upper()]

    species_list = engine.ionize(molecular_mass, method_enum, polarity_enum)

    return [engine.summarize_ionization(s) for s in species_list]
