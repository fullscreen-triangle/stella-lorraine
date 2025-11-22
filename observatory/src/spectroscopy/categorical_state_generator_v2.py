#!/usr/bin/env python3
"""
Categorical State Generator V2
==============================

Creates categorical states using INTRINSIC molecular properties
that produce LARGE categorical separations.

Based on actual molecular differences, not LED fluorescence.
"""

import numpy as np
import time

class MolecularCategoricalStateGenerator:
    """
    Generate categorical states from INTRINSIC molecular properties
    """

    def __init__(self):
        self.k_B = 1.380649e-23  # Boltzmann constant

    def create_categorical_state(self, smarts_pattern):
        """
        Create categorical state from molecular STRUCTURE

        Uses intrinsic properties that create REAL categorical differences:
        - Molecular complexity (# unique atoms/bonds)
        - Functional group count
        - Ring systems
        - Bond types
        - Aromatic character
        """
        if not smarts_pattern:
            return (0.0, 0.0, 0.0)

        # Extract molecular features
        features = self._extract_molecular_features(smarts_pattern)

        # Calculate S-entropy coordinates from features
        S_k = self._calculate_knowledge_entropy(features)
        S_t = self._calculate_temporal_entropy(features)
        S_e = self._calculate_structural_entropy(features)

        return (S_k, S_t, S_e)

    def _extract_molecular_features(self, pattern):
        """Extract intrinsic molecular features that define categorical position"""
        features = {
            # Basic complexity
            'length': len(pattern),
            'unique_chars': len(set(pattern)),
            'complexity': len(set(pattern)) / len(pattern) if pattern else 0,

            # Functional groups (create LARGE categorical differences)
            'hydroxyl': pattern.count('OH'),
            'carbonyl': pattern.count('C=O'),
            'carboxyl': pattern.count('(=O)O'),
            'amine': pattern.count('N'),
            'nitrile': pattern.count('C#N'),

            # Aromatic character
            'aromatic_carbons': sum(1 for c in pattern if c.islower()),
            'aromatic_rings': pattern.count('c1'),

            # Bond types
            'single_bonds': pattern.count('C') + pattern.count('O') + pattern.count('N'),
            'double_bonds': pattern.count('='),
            'triple_bonds': pattern.count('#'),

            # Ring systems
            'ring_count': sum(1 for c in pattern if c.isdigit()),
            'ring_size': self._estimate_ring_size(pattern),

            # Heteroatoms
            'oxygen_count': pattern.count('O'),
            'nitrogen_count': pattern.count('N'),
            'sulfur_count': pattern.count('S'),
            'halogen_count': pattern.count('F') + pattern.count('Cl') + pattern.count('Br')
        }

        return features

    def _estimate_ring_size(self, pattern):
        """Estimate largest ring size from pattern"""
        digits = [c for c in pattern if c.isdigit()]
        if digits:
            return max(int(d) for d in digits)
        return 0

    def _calculate_knowledge_entropy(self, features):
        """
        S_k: Knowledge coordinate

        Based on information content of molecular structure
        More complex molecules = higher S_k
        """
        # Complexity contribution
        complexity_term = features['complexity'] * 10

        # Functional group diversity (major contributor)
        functional_groups = [
            features['hydroxyl'],
            features['carbonyl'],
            features['carboxyl'],
            features['amine'],
            features['nitrile']
        ]
        fg_count = sum(1 for fg in functional_groups if fg > 0)
        fg_diversity = fg_count * 15  # Each functional group adds 15 to S_k

        # Aromatic contribution
        aromatic_term = features['aromatic_carbons'] * 2

        # Ring complexity
        ring_term = features['ring_count'] * 3

        S_k = complexity_term + fg_diversity + aromatic_term + ring_term

        return S_k

    def _calculate_temporal_entropy(self, features):
        """
        S_t: Temporal coordinate

        Based on molecular dynamics timescales
        More bonds/atoms = slower dynamics = higher S_t
        """
        # Mass/size contribution (larger molecules move slower)
        size_term = np.log(1 + features['length'])

        # Bond flexibility (more bonds = more vibrational modes)
        bond_term = np.log(1 + features['single_bonds'] +
                          2*features['double_bonds'] +
                          3*features['triple_bonds'])

        # Ring rigidity (rings slow down conformational changes)
        ring_term = features['ring_count'] * 0.5

        S_t = (size_term + bond_term + ring_term) * 5

        return S_t

    def _calculate_structural_entropy(self, features):
        """
        S_e: Structural entropy coordinate

        Based on configurational complexity and disorder
        More heteroatoms and functional groups = higher S_e
        """
        # Heteroatom diversity creates configurational entropy
        heteroatoms = (features['oxygen_count'] +
                      features['nitrogen_count'] +
                      features['sulfur_count'] +
                      features['halogen_count'])

        if heteroatoms == 0:
            return 0.0

        # Functional group contribution
        fg_entropy = (features['hydroxyl'] * 2.5 +  # OH groups create H-bonding possibilities
                     features['carbonyl'] * 2.0 +   # C=O creates dipole orientations
                     features['carboxyl'] * 3.0 +   # COOH can ionize
                     features['amine'] * 2.5 +      # NH can H-bond
                     features['nitrile'] * 1.5)     # CN is rigid

        # Aromatic entropy (delocalized electrons)
        aromatic_entropy = features['aromatic_carbons'] * 0.5

        S_e = fg_entropy + aromatic_entropy + np.log(1 + heteroatoms)

        return S_e

    def calculate_categorical_separation(self, C1, C2):
        """
        Calculate categorical separation ΔC between two states

        This is the FUNDAMENTAL quantity - the categorical analog of spatial distance
        """
        S_k1, S_t1, S_e1 = C1
        S_k2, S_t2, S_e2 = C2

        # Euclidean distance in S-entropy space
        delta_C = np.sqrt((S_k2 - S_k1)**2 +
                         (S_t2 - S_t1)**2 +
                         (S_e2 - S_e1)**2)

        return delta_C

class CategoricalPredictor:
    """
    Predict categorical states using phase-lock network theory from Gibbs' paradox paper
    """

    def __init__(self):
        # From Gibbs' paradox paper: phase-lock network densification
        self.phase_lock_coupling = 0.15  # Typical coupling strength
        self.k_B = 1.380649e-23

    def predict_categorical_state(self, C1, delta_C):
        """
        Predict C2 from C1 + ΔC using categorical completion theory

        From Gibbs' paradox paper:
        - Categorical states advance: C1 ≺ C2
        - Phase-lock edges increase: |E(C2)| > |E(C1)|
        - Entropy increases: S(C2) = S(C1) + k_B * ΔC

        This is a PURE MATHEMATICAL CALCULATION - should be near-instant
        """
        S_k1, S_t1, S_e1 = C1

        # Predict advancement using categorical completion formulas
        # These are based on phase-lock network topology from the paper

        # S_k increases with categorical completion (new information acquired)
        # ΔS_k ∝ √ΔC (sub-linear growth due to diminishing returns)
        S_k2 = S_k1 + np.sqrt(delta_C) * 0.5

        # S_t increases linearly with categorical progression
        # ΔS_t ∝ ΔC (time advances linearly)
        S_t2 = S_t1 + delta_C * 0.1

        # S_e increases with phase-lock densification
        # ΔS_e ∝ ΔC * coupling (more edges = more entropy)
        S_e2 = S_e1 + delta_C * self.phase_lock_coupling * 0.8

        return (S_k2, S_t2, S_e2)

    def calculate_prediction_error(self, C_predicted, C_actual):
        """Calculate error between predicted and actual states"""
        error = np.sqrt(sum((p - a)**2 for p, a in zip(C_predicted, C_actual)))
        return error

def test_categorical_state_generation():
    """Test that different molecules create LARGE categorical separations"""
    generator = MolecularCategoricalStateGenerator()

    test_molecules = [
        ('C', 'Methane (simplest)'),
        ('CCO', 'Ethanol (simple alcohol)'),
        ('c1ccccc1', 'Benzene (aromatic)'),
        ('CC(=O)O', 'Acetic acid (carboxylic acid)'),
        ('c1ccc(O)cc1', 'Phenol (aromatic + hydroxyl)'),
        ('c1ccc2ccccc2c1', 'Naphthalene (fused rings)'),
        ('c1ccc(C(=O)O)cc1', 'Benzoic acid (complex)')
    ]

    print("\n" + "="*70)
    print("TESTING CATEGORICAL STATE GENERATION")
    print("="*70)

    states = []
    for smarts, name in test_molecules:
        state = generator.create_categorical_state(smarts)
        states.append((smarts, name, state))
        print(f"\n{name}:")
        print(f"  SMARTS: {smarts}")
        print(f"  Categorical State: S_k={state[0]:.2f}, S_t={state[1]:.2f}, S_e={state[2]:.2f}")

    # Calculate all pairwise separations
    print("\n" + "="*70)
    print("PAIRWISE CATEGORICAL SEPARATIONS")
    print("="*70)

    for i in range(len(states)):
        for j in range(i+1, len(states)):
            smarts1, name1, C1 = states[i]
            smarts2, name2, C2 = states[j]
            delta_C = generator.calculate_categorical_separation(C1, C2)
            print(f"\n{name1} → {name2}")
            print(f"  ΔC = {delta_C:.2f}")

    print("\n" + "="*70)

def test_categorical_predictor():
    """Test that predictor is FAST and reasonable"""
    generator = MolecularCategoricalStateGenerator()
    predictor = CategoricalPredictor()

    print("\n" + "="*70)
    print("TESTING CATEGORICAL PREDICTOR")
    print("="*70)

    # Create two different states
    C1 = generator.create_categorical_state('CCO')  # Ethanol
    C2_actual = generator.create_categorical_state('c1ccccc1')  # Benzene

    delta_C_actual = generator.calculate_categorical_separation(C1, C2_actual)

    print(f"\nActual States:")
    print(f"  C1 (ethanol): {C1}")
    print(f"  C2 (benzene): {C2_actual}")
    print(f"  ΔC (actual): {delta_C_actual:.2f}")

    # Time the prediction (should be sub-nanosecond)
    t_start = time.perf_counter_ns()
    C2_predicted = predictor.predict_categorical_state(C1, delta_C_actual)
    t_end = time.perf_counter_ns()
    t_prediction = t_end - t_start

    error = predictor.calculate_prediction_error(C2_predicted, C2_actual)

    print(f"\nPrediction:")
    print(f"  C2 (predicted): {C2_predicted}")
    print(f"  Prediction time: {t_prediction} ns")
    print(f"  Prediction error: {error:.2f}")

    print("\n" + "="*70)

if __name__ == "__main__":
    test_categorical_state_generation()
    test_categorical_predictor()
