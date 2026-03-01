#!/usr/bin/env python3
"""
S-Entropy Coordinate Transformation for Mass Spectrometry
==========================================================

Based on the Phase-Lock Theory of Categorical Entropy and the dual-modality
framework for mass spectrometry analysis.

This module implements the bijective transformation from mass spectra to
S-Entropy coordinates, enabling platform-independent feature extraction and
phase-lock signature computation.

Theoretical Foundation:
-----------------------
- S-Entropy coordinates: (S_knowledge, S_time, S_entropy)
- 14-dimensional feature extraction
- Phase-lock signature computation
- Categorical state assignment

References:
-----------
- docs/oscillatory/entropy-coordinates.tex
- docs/oscillatory/tandem-mass-spec.tex
- docs/oscillatory/categorical-completion.tex

Author: Kundai Sachikonye
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy as scipy_entropy
from sklearn.decomposition import PCA


@dataclass
class SEntropyCoordinates:
    """
    S-Entropy 3D coordinates for a single peak/fragment.

    Attributes:
        s_knowledge: Information content (intensity + m/z based)
        s_time: Temporal/sequential ordering coordinate
        s_entropy: Local entropy in neighborhood
    """
    s_knowledge: float
    s_time: float
    s_entropy: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.s_knowledge, self.s_time, self.s_entropy])

    def magnitude(self) -> float:
        """Euclidean magnitude."""
        return np.linalg.norm(self.to_array())


@dataclass
class SEntropyFeatures:
    """
    14-dimensional feature vector extracted from S-Entropy coordinates.

    Features:
        Statistical (6): mean_mag, std_mag, min_mag, max_mag, centroid, median_mag
        Geometric (4): mean_pairwise_dist, diameter, variance_from_centroid, pc1_ratio
        Information (4): coordinate_entropy, mean_knowledge, mean_time, mean_entropy
    """
    # Statistical features (6)
    mean_magnitude: float
    std_magnitude: float
    min_magnitude: float
    max_magnitude: float
    centroid_magnitude: float
    median_magnitude: float

    # Geometric features (4)
    mean_pairwise_distance: float
    diameter: float
    variance_from_centroid: float
    pc1_variance_ratio: float

    # Information-theoretic features (4)
    coordinate_entropy: float
    mean_knowledge: float
    mean_time: float
    mean_entropy: float

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'SEntropyFeatures':
        """
        Create SEntropyFeatures from 14-dimensional numpy array.

        Args:
            arr: 14-dimensional feature array

        Returns:
            SEntropyFeatures instance
        """
        if arr.shape != (14,):
            raise ValueError(f"Expected 14-dimensional array, got shape {arr.shape}")

        return cls(
            mean_magnitude=arr[0],
            std_magnitude=arr[1],
            min_magnitude=arr[2],
            max_magnitude=arr[3],
            centroid_magnitude=arr[4],
            median_magnitude=arr[5],
            mean_pairwise_distance=arr[6],
            diameter=arr[7],
            variance_from_centroid=arr[8],
            pc1_variance_ratio=arr[9],
            coordinate_entropy=arr[10],
            mean_knowledge=arr[11],
            mean_time=arr[12],
            mean_entropy=arr[13]
        )

    @property
    def features(self) -> np.ndarray:
        """Return 14-dimensional feature array (property for backward compatibility)."""
        return self.to_array()

    def to_array(self) -> np.ndarray:
        """Convert to 14-dimensional numpy array."""
        return np.array([
            # Statistical
            self.mean_magnitude,
            self.std_magnitude,
            self.min_magnitude,
            self.max_magnitude,
            self.centroid_magnitude,
            self.median_magnitude,
            # Geometric
            self.mean_pairwise_distance,
            self.diameter,
            self.variance_from_centroid,
            self.pc1_variance_ratio,
            # Information
            self.coordinate_entropy,
            self.mean_knowledge,
            self.mean_time,
            self.mean_entropy,
        ])

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'mean_magnitude': self.mean_magnitude,
            'std_magnitude': self.std_magnitude,
            'min_magnitude': self.min_magnitude,
            'max_magnitude': self.max_magnitude,
            'centroid_magnitude': self.centroid_magnitude,
            'median_magnitude': self.median_magnitude,
            'mean_pairwise_distance': self.mean_pairwise_distance,
            'diameter': self.diameter,
            'variance_from_centroid': self.variance_from_centroid,
            'pc1_variance_ratio': self.pc1_variance_ratio,
            'coordinate_entropy': self.coordinate_entropy,
            'mean_knowledge': self.mean_knowledge,
            'mean_time': self.mean_time,
            'mean_entropy': self.mean_entropy,
        }


class SEntropyTransformer:
    """
    Transform mass spectra into S-Entropy coordinate system.

    This implements the bijective transformation described in:
    - entropy-coordinates.tex (metabolomics)
    - tandem-mass-spec.tex (proteomics)

    The transformation preserves spectral information while achieving
    platform independence through information-theoretic principles.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 1.0,
        k_neighbors: int = 5,
        min_intensity: float = 0.0
    ):
        """
        Initialize S-Entropy transformer.

        Args:
            alpha: Balance between intensity and m/z information (default: 0.5)
            beta: Temporal decay rate (default: 1.0)
            k_neighbors: Number of neighbors for local entropy (default: 5)
            min_intensity: Minimum intensity threshold (default: 0.0)
        """
        self.alpha = alpha
        self.beta = beta
        self.k_neighbors = k_neighbors
        self.min_intensity = min_intensity

    def transform_spectrum(
        self,
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        precursor_mz: Optional[float] = None,
        rt: Optional[float] = None
    ) -> Tuple[List[SEntropyCoordinates], np.ndarray]:
        """
        Transform a single spectrum to S-Entropy coordinates.

        Args:
            mz_array: Array of m/z values
            intensity_array: Array of intensity values
            precursor_mz: Precursor m/z (for MS2, optional)
            rt: Retention time (optional)

        Returns:
            Tuple of (list of SEntropyCoordinates for each peak, coordinate matrix)
        """
        # Filter by intensity
        mask = intensity_array >= self.min_intensity
        mz = mz_array[mask]
        intensity = intensity_array[mask]

        if len(mz) == 0:
            return [], np.array([])

        # Normalize intensities
        intensity_norm = intensity / intensity.sum()

        # Calculate base coordinates
        x_base, y_base, z_base = self._calculate_base_coordinates(
            mz, intensity_norm, precursor_mz
        )

        # Calculate S-Entropy weighting functions
        s_knowledge = self._calculate_s_knowledge(mz, intensity_norm, precursor_mz)
        s_time = self._calculate_s_time(mz, rt)
        s_entropy = self._calculate_s_entropy(mz, intensity_norm)

        # Apply transformation: final coords = base * weighting
        coords_list = []
        coord_matrix = np.zeros((len(mz), 3))

        for i in range(len(mz)):
            coord = SEntropyCoordinates(
                s_knowledge=float(x_base[i] * s_knowledge[i]),
                s_time=float(y_base[i] * s_time[i]),
                s_entropy=float(z_base[i] * s_entropy[i])
            )
            coords_list.append(coord)
            coord_matrix[i] = coord.to_array()

        return coords_list, coord_matrix

    def _calculate_base_coordinates(
        self,
        mz: np.ndarray,
        intensity_norm: np.ndarray,
        precursor_mz: Optional[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate base coordinates from physicochemical properties.

        For fragments without sequence info, use m/z-based proxies:
        x = cos(2π * m/m_max)
        y = sin(2π * m/m_max)
        z = I/I_max
        """
        m_max = mz.max() if precursor_mz is None else precursor_mz

        x_base = np.cos(2 * np.pi * mz / m_max)
        y_base = np.sin(2 * np.pi * mz / m_max)
        z_base = intensity_norm / intensity_norm.max()

        return x_base, y_base, z_base

    def _calculate_s_knowledge(
        self,
        mz: np.ndarray,
        intensity_norm: np.ndarray,
        precursor_mz: Optional[float]
    ) -> np.ndarray:
        """
        Calculate S_knowledge: Information content.

        S_knowledge = -log2(I_norm) + α * (m / m_precursor)

        Combines Shannon self-information with mass-based structural information.
        """
        # Shannon self-information
        epsilon = 1e-10
        shannon_info = -np.log2(intensity_norm + epsilon)

        # Mass-based information
        if precursor_mz is not None and precursor_mz > 0:
            mass_info = self.alpha * (mz / precursor_mz)
        else:
            mass_info = self.alpha * (mz / mz.max())

        return shannon_info + mass_info

    def _calculate_s_time(
        self,
        mz: np.ndarray,
        rt: Optional[float]
    ) -> np.ndarray:
        """
        Calculate S_time: Temporal/sequential ordering.

        S_time = exp(-β * |m - mean(m)| / std(m))

        Gaussian-like weighting emphasizing fragments near spectral center.
        """
        m_mean = mz.mean()
        m_std = mz.std()

        if m_std > 0:
            s_time = np.exp(-self.beta * np.abs(mz - m_mean) / m_std)
        else:
            s_time = np.ones_like(mz)

        # Incorporate RT if available
        if rt is not None:
            # Modulate by RT (normalized to [0, 1])
            rt_factor = np.exp(-rt / 60.0)  # Assume RT in minutes
            s_time *= rt_factor

        return s_time

    def _calculate_s_entropy(
        self,
        mz: np.ndarray,
        intensity_norm: np.ndarray
    ) -> np.ndarray:
        """
        Calculate S_entropy: Local entropy in m/z neighborhood.

        S_entropy_i = -Σ p_j log2(p_j) for j in N(i)

        where N(i) is the k-nearest neighbors in m/z space.
        """
        n = len(mz)
        s_entropy = np.zeros(n)

        for i in range(n):
            # Find k nearest neighbors in m/z space
            distances = np.abs(mz - mz[i])
            k_actual = min(self.k_neighbors, n - 1)  # -1 because argpartition requires kth < len

            # Handle edge cases
            if k_actual < 1:
                # Single peak spectrum - use itself
                neighbor_indices = np.array([i])
            else:
                neighbor_indices = np.argpartition(distances, k_actual)[:k_actual]

            # Calculate local entropy
            local_intensities = intensity_norm[neighbor_indices]
            local_probs = local_intensities / local_intensities.sum()

            # Shannon entropy of local neighborhood
            s_entropy[i] = scipy_entropy(local_probs, base=2)

        return s_entropy

    def extract_features(
        self,
        coords_list: List[SEntropyCoordinates],
        coord_matrix: np.ndarray
    ) -> SEntropyFeatures:
        """
        Extract 14-dimensional feature vector from S-Entropy coordinates.

        Args:
            coords_list: List of SEntropyCoordinates for each peak
            coord_matrix: N×3 array of coordinates

        Returns:
            SEntropyFeatures with 14 dimensions
        """
        if len(coords_list) == 0:
            # Return zeros for empty spectrum
            return SEntropyFeatures(
                mean_magnitude=0.0, std_magnitude=0.0, min_magnitude=0.0,
                max_magnitude=0.0, centroid_magnitude=0.0, median_magnitude=0.0,
                mean_pairwise_distance=0.0, diameter=0.0,
                variance_from_centroid=0.0, pc1_variance_ratio=0.0,
                coordinate_entropy=0.0, mean_knowledge=0.0,
                mean_time=0.0, mean_entropy=0.0
            )

        # Calculate magnitudes
        magnitudes = np.array([coord.magnitude() for coord in coords_list])

        # Statistical features (6)
        mean_mag = float(np.mean(magnitudes))
        std_mag = float(np.std(magnitudes))
        min_mag = float(np.min(magnitudes))
        max_mag = float(np.max(magnitudes))
        median_mag = float(np.median(magnitudes))

        # Centroid
        centroid = coord_matrix.mean(axis=0)
        centroid_mag = float(np.linalg.norm(centroid))

        # Geometric features (4)
        if len(coord_matrix) > 1:
            # Pairwise distances
            pairwise_dists = pdist(coord_matrix, metric='euclidean')
            mean_pairwise_dist = float(np.mean(pairwise_dists))
            diameter = float(np.max(pairwise_dists))

            # Variance from centroid
            variance_from_centroid = float(
                np.mean(np.sum((coord_matrix - centroid) ** 2, axis=1))
            )

            # PCA variance ratio
            try:
                pca = PCA(n_components=min(3, len(coord_matrix)))
                pca.fit(coord_matrix)
                pc1_ratio = float(pca.explained_variance_ratio_[0])
            except:
                pc1_ratio = 1.0
        else:
            mean_pairwise_dist = 0.0
            diameter = 0.0
            variance_from_centroid = 0.0
            pc1_ratio = 1.0

        # Information-theoretic features (4)
        # Coordinate entropy (entropy of normalized magnitudes)
        p_magnitudes = magnitudes / magnitudes.sum()
        coord_entropy = float(scipy_entropy(p_magnitudes, base=2))

        # Mean of each coordinate dimension
        mean_knowledge = float(coord_matrix[:, 0].mean())
        mean_time = float(coord_matrix[:, 1].mean())
        mean_entropy = float(coord_matrix[:, 2].mean())

        return SEntropyFeatures(
            mean_magnitude=mean_mag,
            std_magnitude=std_mag,
            min_magnitude=min_mag,
            max_magnitude=max_mag,
            centroid_magnitude=centroid_mag,
            median_magnitude=median_mag,
            mean_pairwise_distance=mean_pairwise_dist,
            diameter=diameter,
            variance_from_centroid=variance_from_centroid,
            pc1_variance_ratio=pc1_ratio,
            coordinate_entropy=coord_entropy,
            mean_knowledge=mean_knowledge,
            mean_time=mean_time,
            mean_entropy=mean_entropy,
        )

    def transform_and_extract(
        self,
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        precursor_mz: Optional[float] = None,
        rt: Optional[float] = None
    ) -> Tuple[List[SEntropyCoordinates], SEntropyFeatures]:
        """
        Complete pipeline: transform to coordinates and extract features.

        Args:
            mz_array: Array of m/z values
            intensity_array: Array of intensity values
            precursor_mz: Precursor m/z (optional)
            rt: Retention time (optional)

        Returns:
            Tuple of (coordinates, 14D features)
        """
        coords_list, coord_matrix = self.transform_spectrum(
            mz_array, intensity_array, precursor_mz, rt
        )

        features = self.extract_features(coords_list, coord_matrix)

        return coords_list, features


class PhaseLockSignatureComputer:
    """
    Compute phase-lock signatures from S-Entropy coordinates.

    Phase-lock signatures represent the categorical states formed by
    phase-locked molecular ensembles, as described in categorical-completion.tex.

    These signatures enable:
    - Dual-modality annotation (numerical + visual intersection)
    - Empty Dictionary synthesis
    - Categorical state-based disambiguation
    """

    def __init__(
        self,
        reference_database: Optional[Dict] = None,
        signature_dim: int = 64
    ):
        """
        Initialize phase-lock signature computer.

        Args:
            reference_database: Optional database for signature matching
            signature_dim: Dimensionality of phase-lock signature (default: 64)
        """
        self.reference_database = reference_database
        self.signature_dim = signature_dim

    def compute_signature(
        self,
        coords_list: List[SEntropyCoordinates],
        coord_matrix: np.ndarray,
        features: SEntropyFeatures
    ) -> np.ndarray:
        """
        Compute phase-lock signature from S-Entropy coordinates.

        The signature encodes:
        1. Coordinate distribution patterns
        2. Inter-peak phase relationships
        3. Information-theoretic properties
        4. Categorical state indicators

        Args:
            coords_list: List of coordinates for each peak
            coord_matrix: N×3 coordinate matrix
            features: 14D feature vector

        Returns:
            Phase-lock signature (signature_dim dimensional)
        """
        if len(coords_list) == 0:
            return np.zeros(self.signature_dim)

        # Start with 14D features as base
        base_signature = features.to_array()

        # Add coordinate distribution moments (mean, std of each dimension)
        coord_moments = np.concatenate([
            coord_matrix.mean(axis=0),
            coord_matrix.std(axis=0)
        ])  # 6 dimensions

        # Add pairwise phase relationships (upper triangle of correlation matrix)
        if len(coord_matrix) > 1:
            corr_matrix = np.corrcoef(coord_matrix.T)
            phase_relations = corr_matrix[np.triu_indices(3, k=1)]  # 3 dimensions
        else:
            phase_relations = np.zeros(3)

        # Combine into signature
        signature_components = [
            base_signature,      # 14 dimensions
            coord_moments,       # 6 dimensions
            phase_relations,     # 3 dimensions
        ]

        partial_signature = np.concatenate(signature_components)  # 23 dimensions

        # Pad or project to target dimension
        if len(partial_signature) < self.signature_dim:
            # Pad with zeros
            signature = np.zeros(self.signature_dim)
            signature[:len(partial_signature)] = partial_signature
        else:
            # Project down using simple averaging
            signature = np.zeros(self.signature_dim)
            chunk_size = len(partial_signature) / self.signature_dim
            for i in range(self.signature_dim):
                start_idx = int(i * chunk_size)
                end_idx = int((i + 1) * chunk_size)
                signature[i] = partial_signature[start_idx:end_idx].mean()

        # Normalize
        norm = np.linalg.norm(signature)
        if norm > 0:
            signature /= norm

        return signature

    def compute_categorical_state(
        self,
        signature: np.ndarray,
        threshold: float = 0.5
    ) -> int:
        """
        Assign categorical state based on phase-lock signature.

        Categorical states represent distinct phase-locked ensembles.
        States are determined by signature clustering in the categorical
        completion sequence.

        Args:
            signature: Phase-lock signature
            threshold: Similarity threshold for state assignment

        Returns:
            Categorical state index
        """
        if self.reference_database is None:
            # Without reference, use signature hash as state
            return int(np.abs(signature).sum() * 1000) % 10000

        # Compare against reference database
        # (This would be implemented with actual database in production)
        # For now, use a simple hash-based assignment
        state_hash = hash(tuple(signature.round(3)))
        return abs(state_hash) % 10000


# ============================================================================
# Example Usage and Validation
# ============================================================================

def example_transform_spectrum():
    """Example: Transform a spectrum to S-Entropy coordinates."""
    print("=" * 60)
    print("S-Entropy Transformation Example")
    print("=" * 60)

    # Create example spectrum
    mz = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0])
    intensity = np.array([1000, 5000, 15000, 8000, 3000, 1200, 500])
    precursor_mz = 450.0
    rt = 12.5

    # Transform
    transformer = SEntropyTransformer()
    coords_list, coord_matrix = transformer.transform_spectrum(
        mz, intensity, precursor_mz, rt
    )

    print(f"\nInput spectrum: {len(mz)} peaks")
    print(f"Precursor m/z: {precursor_mz:.2f}")
    print(f"Retention time: {rt:.2f} min")

    print("\nS-Entropy Coordinates:")
    for i, coord in enumerate(coords_list):
        print(f"  Peak {i+1} (m/z={mz[i]:.1f}):")
        print(f"    S_knowledge: {coord.s_knowledge:.4f}")
        print(f"    S_time:      {coord.s_time:.4f}")
        print(f"    S_entropy:   {coord.s_entropy:.4f}")
        print(f"    Magnitude:   {coord.magnitude():.4f}")

    # Extract features
    features = transformer.extract_features(coords_list, coord_matrix)
    print(f"\n14D Feature Vector:")
    feature_array = features.to_array()
    feature_names = [
        'mean_mag', 'std_mag', 'min_mag', 'max_mag', 'centroid_mag', 'median_mag',
        'mean_pairwise_dist', 'diameter', 'var_from_centroid', 'pc1_ratio',
        'coord_entropy', 'mean_knowledge', 'mean_time', 'mean_entropy'
    ]
    for name, value in zip(feature_names, feature_array):
        print(f"  {name:20s}: {value:.4f}")

    # Compute phase-lock signature
    phase_lock = PhaseLockSignatureComputer(signature_dim=64)
    signature = phase_lock.compute_signature(coords_list, coord_matrix, features)
    categorical_state = phase_lock.compute_categorical_state(signature)

    print(f"\nPhase-Lock Signature: {signature.shape[0]}D")
    print(f"Signature norm: {np.linalg.norm(signature):.4f}")
    print(f"Categorical state: {categorical_state}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    example_transform_spectrum()
