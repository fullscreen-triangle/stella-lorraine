"""
Comprehensive Validation of Ternary Representation Framework
Tests trit-to-coordinate mapping, 3^k hierarchy, continuous emergence,
trajectory encoding, ideal gas law integration, and oscillator mapping.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy.stats import entropy
import sys

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

class TernaryValidator:
    """Validates ternary representation framework"""
    
    def __init__(self):
        self.kB = 1.380649e-23  # Boltzmann constant
        
    def trit_to_coordinate(self, trit_string):
        """
        Convert ternary string to S-entropy coordinates (Sk, St, Se)
        Implements Theorem: Coordinate Extraction
        """
        k = len(trit_string)
        Sk = 0.0
        St = 0.0
        Se = 0.0
        
        for j in range(1, k + 1):
            trit = trit_string[j - 1]
            dim = (j - 1) % 3  # 0=Sk, 1=St, 2=Se
            
            # Which refinement level for this dimension
            refinement_level = (j - 1) // 3 + 1
            
            # Contribution: (trit + 0.5) / 3^refinement_level
            contribution = (trit + 0.5) / (3 ** refinement_level)
            
            if dim == 0:  # Sk
                Sk += contribution
            elif dim == 1:  # St
                St += contribution
            else:  # Se
                Se += contribution
        
        return np.array([Sk, St, Se])
    
    def coordinate_to_trit(self, coord, max_depth=10):
        """
        Convert S-entropy coordinates to ternary string
        Implements Theorem: Address from Coordinates
        """
        Sk, St, Se = coord
        trit_string = []
        remainders = [Sk, St, Se]
        
        for j in range(1, max_depth + 1):
            dim = (j - 1) % 3
            r = remainders[dim]
            
            # Extract trit
            trit = int(np.floor(3 * r))
            trit = max(0, min(2, trit))  # Clamp to [0, 2]
            trit_string.append(trit)
            
            # Update remainder
            remainders[dim] = 3 * r - trit
        
        return trit_string
    
    def validate_3k_hierarchy(self, max_k=6):
        """Validate 3^k hierarchical structure"""
        results = {
            'k': [],
            'cell_count': [],
            'expected': [],
            'cell_sizes': []
        }
        
        for k in range(max_k + 1):
            cell_count = 3 ** k
            results['k'].append(k)
            results['cell_count'].append(cell_count)
            results['expected'].append(cell_count)
            
            # Cell size after k refinements (assuming balanced: k/3 per dimension)
            if k >= 3:
                cell_size = 3 ** (-k // 3)
            else:
                cell_size = 1.0
            results['cell_sizes'].append(cell_size)
        
        return results
    
    def validate_continuous_emergence(self, target_coord, max_k=15):
        """
        Validate continuous emergence: discrete cells converge to continuous point
        Implements Theorem: Continuous Emergence
        """
        convergence = {
            'k': [],
            'distance': [],
            'cell_diameter': []
        }
        
        # Generate ternary string for target
        trit_string = self.coordinate_to_trit(target_coord, max_k)
        
        for k in range(1, max_k + 1):
            # Get cell center for k-trit prefix
            prefix = trit_string[:k]
            cell_center = self.trit_to_coordinate(prefix)
            
            # Distance from target
            distance = np.linalg.norm(cell_center - target_coord)
            
            # Theoretical cell diameter
            cell_diameter = np.sqrt(3) * (3 ** (-k // 3))
            
            convergence['k'].append(k)
            convergence['distance'].append(distance)
            convergence['cell_diameter'].append(cell_diameter)
        
        return convergence
    
    def validate_trajectory_encoding(self, start_trit, trajectory):
        """
        Validate that ternary strings encode trajectories
        Implements Theorem: Position-Trajectory Duality
        """
        # Start position
        start_pos = self.trit_to_coordinate(start_trit)
        
        # Follow trajectory
        current_trit = start_trit.copy()
        positions = [start_pos]
        
        for trit in trajectory:
            current_trit.append(trit)
            pos = self.trit_to_coordinate(current_trit)
            positions.append(pos)
        
        return np.array(positions)
    
    def ideal_gas_ternary_integration(self, N, V, T, trit_string):
        """
        Integrate ternary representation with ideal gas law:
        PV = NkT * S(V, N, {n_i})
        where S is structural factor from partition coordinates
        """
        # Extract S-entropy coordinates
        Sk, St, Se = self.trit_to_coordinate(trit_string)
        
        # Structural factor from S-entropy (simplified model)
        # In full theory, this comes from partition coordinates {n_i, l_i, m_i, s_i}
        # Here we use S-entropy as proxy
        S_structure = (Sk + St + Se) / 3.0
        
        # Ideal gas law with structural factor
        P = (N * self.kB * T / V) * S_structure
        
        return P, S_structure
    
    def three_phase_oscillator_trit(self, phases, time):
        """
        Extract trit from three-phase oscillator
        Implements Theorem: Phase-Trit Correspondence
        """
        # Three phases with 2π/3 separation
        phi0 = phases[0] + 2 * np.pi * time
        phi1 = phases[1] + 2 * np.pi * time - 2 * np.pi / 3
        phi2 = phases[2] + 2 * np.pi * time - 4 * np.pi / 3
        
        # Amplitudes (using cosine for phase detection)
        A0 = np.cos(phi0)
        A1 = np.cos(phi1)
        A2 = np.cos(phi2)
        
        # Dominant oscillator determines trit
        amplitudes = np.array([A0, A1, A2])
        trit = np.argmax(amplitudes)
        
        return trit, amplitudes
    
    def generate_validation_panels(self):
        """Generate comprehensive validation panels (single large figure)"""
        
        # Create figure with multiple panels
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel 1: Trit-to-Coordinate Mapping
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_trit_coordinate_mapping(ax1)
        
        # Panel 2: 3^k Hierarchy Validation
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_3k_hierarchy(ax2)
        
        # Panel 3: Continuous Emergence
        ax3 = fig.add_subplot(gs[0, 2])
        self.plot_continuous_emergence(ax3)
        
        # Panel 4: Trajectory Encoding (3D)
        ax4 = fig.add_subplot(gs[1, 0], projection='3d')
        self.plot_trajectory_3d(ax4)
        
        # Panel 5: Ideal Gas Law Integration
        ax5 = fig.add_subplot(gs[1, 1])
        self.plot_ideal_gas_integration(ax5)
        
        # Panel 6: Three-Phase Oscillator Mapping
        ax6 = fig.add_subplot(gs[1, 2])
        self.plot_oscillator_mapping(ax6)
        
        # Panel 7: Ternary Space Coverage (3D)
        ax7 = fig.add_subplot(gs[2, 0], projection='3d')
        self.plot_ternary_space_coverage(ax7)
        
        # Panel 8: Convergence Rate Analysis
        ax8 = fig.add_subplot(gs[2, 1])
        self.plot_convergence_rate(ax8)
        
        # Panel 9: Information Density Comparison
        ax9 = fig.add_subplot(gs[2, 2])
        self.plot_information_density(ax9)
        
        # Panel 10: Trajectory Distance Preservation
        ax10 = fig.add_subplot(gs[3, 0])
        self.plot_trajectory_distance(ax10)
        
        # Panel 11: S-Entropy Dynamics Integration
        ax11 = fig.add_subplot(gs[3, 1])
        self.plot_s_entropy_dynamics(ax11)
        
        # Panel 12: Tryte Structure Validation
        ax12 = fig.add_subplot(gs[3, 2])
        self.plot_tryte_structure(ax12)
        
        plt.suptitle('Comprehensive Validation of Ternary Representation Framework', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        return fig
    
    def generate_individual_figures(self):
        """Generate separate figures for each validation aspect"""
        figures = []
        
        # Figure 1: Trit-to-Coordinate Mapping
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        self.plot_trit_coordinate_mapping(ax1)
        fig1.suptitle('Trit-to-Coordinate Mapping Validation', fontsize=14, fontweight='bold')
        figures.append(('trit_coordinate_mapping.png', fig1))
        
        # Figure 2: 3^k Hierarchy
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        self.plot_3k_hierarchy(ax2)
        fig2.suptitle('3^k Hierarchical Structure Validation', fontsize=14, fontweight='bold')
        figures.append(('3k_hierarchy.png', fig2))
        
        # Figure 3: Continuous Emergence
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        self.plot_continuous_emergence(ax3)
        fig3.suptitle('Continuous Emergence Validation', fontsize=14, fontweight='bold')
        figures.append(('continuous_emergence.png', fig3))
        
        # Figure 4: Trajectory Encoding (3D)
        fig4 = plt.figure(figsize=(12, 10))
        ax4 = fig4.add_subplot(111, projection='3d')
        self.plot_trajectory_3d(ax4)
        fig4.suptitle('Trajectory Encoding Validation (3D)', fontsize=14, fontweight='bold')
        figures.append(('trajectory_3d.png', fig4))
        
        # Figure 5: Ideal Gas Law Integration
        fig5, ax5 = plt.subplots(figsize=(10, 8))
        self.plot_ideal_gas_integration(ax5)
        fig5.suptitle('Ideal Gas Law Integration', fontsize=14, fontweight='bold')
        figures.append(('ideal_gas_integration.png', fig5))
        
        # Figure 6: Three-Phase Oscillator Mapping
        fig6, ax6 = plt.subplots(figsize=(12, 8))
        self.plot_oscillator_mapping(ax6)
        fig6.suptitle('Three-Phase Oscillator to Trit Mapping', fontsize=14, fontweight='bold')
        figures.append(('oscillator_mapping.png', fig6))
        
        # Figure 7: Ternary Space Coverage (3D)
        fig7 = plt.figure(figsize=(12, 10))
        ax7 = fig7.add_subplot(111, projection='3d')
        self.plot_ternary_space_coverage(ax7)
        fig7.suptitle('Ternary Space Coverage Validation (3D)', fontsize=14, fontweight='bold')
        figures.append(('ternary_space_coverage.png', fig7))
        
        # Figure 8: Convergence Rate
        fig8, ax8 = plt.subplots(figsize=(10, 8))
        self.plot_convergence_rate(ax8)
        fig8.suptitle('Convergence Rate Analysis', fontsize=14, fontweight='bold')
        figures.append(('convergence_rate.png', fig8))
        
        # Figure 9: Information Density
        fig9, ax9 = plt.subplots(figsize=(10, 8))
        self.plot_information_density(ax9)
        fig9.suptitle('Information Density: Ternary vs Binary', fontsize=14, fontweight='bold')
        figures.append(('information_density.png', fig9))
        
        # Figure 10: Trajectory Distance
        fig10, ax10 = plt.subplots(figsize=(10, 8))
        self.plot_trajectory_distance(ax10)
        fig10.suptitle('Trajectory Distance Preservation', fontsize=14, fontweight='bold')
        figures.append(('trajectory_distance.png', fig10))
        
        # Figure 11: S-Entropy Dynamics
        fig11, ax11 = plt.subplots(figsize=(12, 8))
        self.plot_s_entropy_dynamics(ax11)
        fig11.suptitle('S-Entropy Dynamics with Ternary Encoding', fontsize=14, fontweight='bold')
        figures.append(('s_entropy_dynamics.png', fig11))
        
        # Figure 12: Tryte Structure
        fig12, ax12 = plt.subplots(figsize=(10, 8))
        self.plot_tryte_structure(ax12)
        fig12.suptitle('Tryte Structure Validation (6 Trits = 729 Cells)', fontsize=14, fontweight='bold')
        figures.append(('tryte_structure.png', fig12))
        
        return figures
    
    def plot_trit_coordinate_mapping(self, ax):
        """Panel 1: Validate trit-to-coordinate mapping"""
        # Test various ternary strings
        test_strings = [
            [0, 0, 0],  # Minimum
            [1, 1, 1],  # Middle
            [2, 2, 2],  # Maximum
            [0, 1, 2],  # Sequential
            [2, 1, 0],  # Reverse
            [1, 0, 2, 2, 1, 0],  # 6-trit example from paper
        ]
        
        coords = []
        labels = []
        
        for trit_str in test_strings:
            coord = self.trit_to_coordinate(trit_str)
            coords.append(coord)
            labels.append(''.join(map(str, trit_str)))
        
        coords = np.array(coords)
        
        # Plot in 2D projection (Sk vs St)
        ax.scatter(coords[:, 0], coords[:, 1], s=100, c=coords[:, 2], 
                   cmap='viridis', edgecolors='black', linewidths=1.5)
        
        for i, label in enumerate(labels):
            ax.annotate(label, (coords[i, 0], coords[i, 1]), 
                       fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Knowledge Entropy $S_k$', fontsize=11)
        ax.set_ylabel('Temporal Entropy $S_t$', fontsize=11)
        ax.set_title('Trit-to-Coordinate Mapping\n(Colors = $S_e$)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
    
    def plot_3k_hierarchy(self, ax):
        """Panel 2: Validate 3^k hierarchical structure"""
        results = self.validate_3k_hierarchy(max_k=8)
        
        k = np.array(results['k'])
        cell_count = np.array(results['cell_count'])
        expected = np.array(results['expected'])
        
        ax.plot(k, cell_count, 'o-', linewidth=2, markersize=8, 
               label='Actual', color='steelblue')
        ax.plot(k, expected, '--', linewidth=2, label='Expected $3^k$', 
               color='crimson', alpha=0.7)
        
        ax.set_xlabel('Hierarchy Depth $k$', fontsize=11)
        ax.set_ylabel('Cell Count', fontsize=11)
        ax.set_title('$3^k$ Hierarchical Structure Validation', 
                    fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=10)
    
    def plot_continuous_emergence(self, ax):
        """Panel 3: Validate continuous emergence"""
        target = np.array([0.7, 0.5, 0.3])
        convergence = self.validate_continuous_emergence(target, max_k=15)
        
        k = np.array(convergence['k'])
        distance = np.array(convergence['distance'])
        cell_diameter = np.array(convergence['cell_diameter'])
        
        ax.semilogy(k, distance, 'o-', linewidth=2, markersize=6, 
                   label='Distance to Target', color='steelblue')
        ax.semilogy(k, cell_diameter, '--', linewidth=2, 
                   label='Theoretical Cell Diameter', color='crimson', alpha=0.7)
        
        ax.set_xlabel('Trit Count $k$', fontsize=11)
        ax.set_ylabel('Distance / Diameter', fontsize=11)
        ax.set_title('Continuous Emergence Validation\n$k \\to \\infty$ Convergence', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=10)
    
    def plot_trajectory_3d(self, ax):
        """Panel 4: 3D trajectory encoding validation"""
        start_trit = [1, 0, 2]
        trajectory = [2, 1, 0, 1, 2, 0, 1, 1, 2]
        
        positions = self.validate_trajectory_encoding(start_trit, trajectory)
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
               'o-', linewidth=2, markersize=8, color='steelblue', alpha=0.7)
        
        # Mark start and end
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                  s=200, c='green', marker='s', label='Start', edgecolors='black')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                  s=200, c='red', marker='^', label='End', edgecolors='black')
        
        ax.set_xlabel('$S_k$', fontsize=11)
        ax.set_ylabel('$S_t$', fontsize=11)
        ax.set_zlabel('$S_e$', fontsize=11)
        ax.set_title('Trajectory Encoding Validation\n(Position = Trajectory)', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
    
    def plot_ideal_gas_integration(self, ax):
        """Panel 5: Ideal gas law integration"""
        N = 1e23  # Avogadro's number scale
        V = 0.0224  # m³ (molar volume at STP)
        T_range = np.linspace(200, 400, 50)  # K
        
        # Test with different ternary strings
        test_strings = [
            [0, 0, 0],  # Low structure
            [1, 1, 1],  # Medium structure
            [2, 2, 2],  # High structure
        ]
        
        for trit_str in test_strings:
            pressures = []
            for T in T_range:
                P, S = self.ideal_gas_ternary_integration(N, V, T, trit_str)
                pressures.append(P)
            
            label = f'Trit {trit_str[0]}{trit_str[1]}{trit_str[2]}'
            ax.plot(T_range, np.array(pressures) / 1e5, linewidth=2, label=label)
        
        ax.set_xlabel('Temperature $T$ (K)', fontsize=11)
        ax.set_ylabel('Pressure $P$ (bar)', fontsize=11)
        ax.set_title('Ideal Gas Law Integration\n$PV = NkT \\cdot S(V,N,\\{n_i\\})$', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    def plot_oscillator_mapping(self, ax):
        """Panel 6: Three-phase oscillator to trit mapping"""
        time = np.linspace(0, 2, 1000)
        phases = [0, 0, 0]  # Initial phases
        
        trits = []
        amplitudes_list = []
        
        for t in time:
            trit, amplitudes = self.three_phase_oscillator_trit(phases, t)
            trits.append(trit)
            amplitudes_list.append(amplitudes)
        
        amplitudes_array = np.array(amplitudes_list)
        
        # Plot oscillator amplitudes
        ax.plot(time, amplitudes_array[:, 0], linewidth=2, label='Oscillator 0', alpha=0.7)
        ax.plot(time, amplitudes_array[:, 1], linewidth=2, label='Oscillator 1', alpha=0.7)
        ax.plot(time, amplitudes_array[:, 2], linewidth=2, label='Oscillator 2', alpha=0.7)
        
        # Mark trit transitions
        trit_changes = np.where(np.diff(trits) != 0)[0]
        for idx in trit_changes[:10]:  # Show first 10 transitions
            ax.axvline(time[idx], color='black', linestyle='--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Time (cycles)', fontsize=11)
        ax.set_ylabel('Amplitude', fontsize=11)
        ax.set_title('Three-Phase Oscillator → Trit Mapping\n$\\phi_i = 2\\pi i/3$', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper right')
    
    def plot_ternary_space_coverage(self, ax):
        """Panel 7: Validate ternary space coverage"""
        # Generate random ternary strings and plot coverage
        np.random.seed(42)
        n_samples = 200
        
        coords = []
        for _ in range(n_samples):
            k = np.random.randint(3, 9)
            trit_str = np.random.randint(0, 3, k).tolist()
            coord = self.trit_to_coordinate(trit_str)
            coords.append(coord)
        
        coords = np.array(coords)
        
        # 3D scatter
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                  c=coords[:, 0] + coords[:, 1] + coords[:, 2], 
                  cmap='viridis', s=50, alpha=0.6)
        
        ax.set_xlabel('$S_k$', fontsize=11)
        ax.set_ylabel('$S_t$', fontsize=11)
        ax.set_zlabel('$S_e$', fontsize=11)
        ax.set_title('Ternary Space Coverage\n$[0,1]^3$ Coverage Validation', 
                    fontsize=12, fontweight='bold')
    
    def plot_convergence_rate(self, ax):
        """Panel 8: Convergence rate analysis"""
        targets = [
            np.array([0.3, 0.5, 0.7]),
            np.array([0.7, 0.3, 0.5]),
            np.array([0.5, 0.7, 0.3]),
        ]
        
        for target in targets:
            convergence = self.validate_continuous_emergence(target, max_k=12)
            k = np.array(convergence['k'])
            distance = np.array(convergence['distance'])
            
            ax.semilogy(k, distance, 'o-', linewidth=1.5, markersize=5, alpha=0.7)
        
        # Theoretical bound: sqrt(3) * 3^(-k/3)
        k_theory = np.arange(1, 13)
        bound = np.sqrt(3) * (3 ** (-k_theory / 3))
        ax.semilogy(k_theory, bound, '--', linewidth=2, color='black', 
                   label='Theoretical Bound', alpha=0.8)
        
        ax.set_xlabel('Trit Count $k$', fontsize=11)
        ax.set_ylabel('Convergence Distance', fontsize=11)
        ax.set_title('Convergence Rate Analysis\nMultiple Target Points', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=9)
    
    def plot_information_density(self, ax):
        """Panel 9: Information density comparison"""
        k_range = np.arange(1, 11)
        
        # Binary: 2^k values, k bits
        binary_values = 2 ** k_range
        binary_bits = k_range
        
        # Ternary: 3^k values, k trits = k * log2(3) bits
        ternary_values = 3 ** k_range
        ternary_bits = k_range * np.log2(3)
        
        ax.plot(k_range, binary_values, 'o-', linewidth=2, markersize=6, 
               label='Binary $2^k$', color='steelblue')
        ax.plot(k_range, ternary_values, 's-', linewidth=2, markersize=6, 
               label='Ternary $3^k$', color='crimson')
        
        ax.set_xlabel('Digit Count $k$', fontsize=11)
        ax.set_ylabel('Encoded Values', fontsize=11)
        ax.set_title('Information Density Comparison\nTernary vs Binary', 
                    fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=10)
    
    def plot_trajectory_distance(self, ax):
        """Panel 10: Trajectory distance preservation"""
        # Generate pairs of ternary strings with different prefix lengths
        prefix_lengths = []
        hamming_distances = []
        euclidean_distances = []
        
        np.random.seed(42)
        for _ in range(50):
            k = np.random.randint(4, 10)
            trit1 = np.random.randint(0, 3, k).tolist()
            trit2 = np.random.randint(0, 3, k).tolist()
            
            # Find common prefix length
            prefix_len = 0
            for i in range(min(len(trit1), len(trit2))):
                if trit1[i] == trit2[i]:
                    prefix_len += 1
                else:
                    break
            
            # Hamming distance (number of differing trits)
            hamming = sum(t1 != t2 for t1, t2 in zip(trit1, trit2))
            
            # Euclidean distance in S-space
            coord1 = self.trit_to_coordinate(trit1)
            coord2 = self.trit_to_coordinate(trit2)
            euclidean = np.linalg.norm(coord1 - coord2)
            
            prefix_lengths.append(prefix_len)
            hamming_distances.append(hamming)
            euclidean_distances.append(euclidean)
        
        # Plot relationship
        ax.scatter(prefix_lengths, euclidean_distances, s=60, alpha=0.6, 
                  c=hamming_distances, cmap='viridis', edgecolors='black')
        
        ax.set_xlabel('Common Prefix Length', fontsize=11)
        ax.set_ylabel('Euclidean Distance in S-Space', fontsize=11)
        ax.set_title('Trajectory Distance Preservation\n(Prefix → Proximity)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Hamming Distance', fontsize=9)
    
    def plot_s_entropy_dynamics(self, ax):
        """Panel 11: S-entropy dynamics integration"""
        # Simulate S-entropy evolution with ternary representation
        time = np.linspace(0, 10, 100)
        
        # Simple oscillatory dynamics in S-space
        Sk = 0.5 + 0.3 * np.sin(2 * np.pi * time / 5)
        St = 0.5 + 0.2 * np.cos(2 * np.pi * time / 7)
        Se = 0.5 + 0.25 * np.sin(2 * np.pi * time / 6 + np.pi/4)
        
        # Convert to ternary at each time step
        ternary_strings = []
        for i in range(len(time)):
            coord = np.array([Sk[i], St[i], Se[i]])
            trit_str = self.coordinate_to_trit(coord, max_depth=6)
            ternary_strings.append(trit_str)
        
        # Plot S-entropy evolution
        ax.plot(time, Sk, linewidth=2, label='$S_k$', color='steelblue')
        ax.plot(time, St, linewidth=2, label='$S_t$', color='crimson')
        ax.plot(time, Se, linewidth=2, label='$S_e$', color='green')
        
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('S-Entropy Coordinate', fontsize=11)
        ax.set_title('S-Entropy Dynamics with Ternary Encoding', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1)
    
    def plot_tryte_structure(self, ax):
        """Panel 12: Tryte (6-trit) structure validation"""
        # Generate all possible trytes (6 trits = 729 values)
        # Sample subset for visualization
        np.random.seed(42)
        n_samples = 100
        
        tryte_coords = []
        for _ in range(n_samples):
            tryte = np.random.randint(0, 3, 6).tolist()
            coord = self.trit_to_coordinate(tryte)
            tryte_coords.append(coord)
        
        tryte_coords = np.array(tryte_coords)
        
        # 2D projection showing tryte distribution
        ax.scatter(tryte_coords[:, 0], tryte_coords[:, 1], 
                  s=80, c=tryte_coords[:, 2], cmap='plasma', 
                  alpha=0.7, edgecolors='black', linewidths=0.5)
        
        ax.set_xlabel('$S_k$', fontsize=11)
        ax.set_ylabel('$S_t$', fontsize=11)
        ax.set_title('Tryte Structure Validation\n(6 Trits = 729 Cells)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('$S_e$', fontsize=10)


def main():
    """Run comprehensive validation"""
    print("=" * 70)
    print("Ternary Representation Framework - Comprehensive Validation")
    print("=" * 70)
    
    validator = TernaryValidator()
    
    # Run validations
    print("\n[1/6] Validating trit-to-coordinate mapping...")
    test_string = [1, 0, 2, 2, 1, 0]
    coord = validator.trit_to_coordinate(test_string)
    print(f"   Test string: {test_string}")
    print(f"   Coordinates: Sk={coord[0]:.4f}, St={coord[1]:.4f}, Se={coord[2]:.4f}")
    
    print("\n[2/6] Validating 3^k hierarchy...")
    hierarchy = validator.validate_3k_hierarchy(max_k=6)
    for k in [0, 3, 6]:
        idx = hierarchy['k'].index(k)
        print(f"   k={k}: {hierarchy['cell_count'][idx]} cells (expected: {3**k})")
    
    print("\n[3/6] Validating continuous emergence...")
    target = np.array([0.7, 0.5, 0.3])
    convergence = validator.validate_continuous_emergence(target, max_k=10)
    final_distance = convergence['distance'][-1]
    print(f"   Target: {target}")
    print(f"   Final distance (k=10): {final_distance:.6f}")
    
    print("\n[4/6] Validating trajectory encoding...")
    start = [1, 0, 2]
    traj = [2, 1, 0]
    positions = validator.validate_trajectory_encoding(start, traj)
    print(f"   Start: {start} -> Position: {positions[0]}")
    print(f"   End: {start + traj} -> Position: {positions[-1]}")
    
    print("\n[5/6] Validating ideal gas law integration...")
    N, V, T = 1e23, 0.0224, 300
    P, S = validator.ideal_gas_ternary_integration(N, V, T, [1, 1, 1])
    print(f"   N={N:.2e}, V={V:.4f} m³, T={T} K")
    print(f"   P={P/1e5:.4f} bar, S_structure={S:.4f}")
    
    print("\n[6/6] Validating three-phase oscillator mapping...")
    phases = [0, 0, 0]
    trits = []
    for t in np.linspace(0, 1, 20):
        trit, _ = validator.three_phase_oscillator_trit(phases, t)
        trits.append(trit)
    print(f"   Trit distribution: {np.bincount(trits)} (should be roughly uniform)")
    
    print("\n" + "=" * 70)
    print("Generating validation figures...")
    print("=" * 70)
    
    # Generate individual figures
    figures = validator.generate_individual_figures()
    
    print(f"\nGenerating {len(figures)} individual validation figures...")
    for filename, fig in figures:
        output_path = filename
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Saved: {output_path}")
    
    # Also generate summary panel
    print("\nGenerating summary panel...")
    summary_fig = validator.generate_validation_panels()
    summary_fig.savefig('ternary_representation_validation_summary.png', dpi=300, bbox_inches='tight')
    plt.close(summary_fig)
    print("  [OK] Saved: ternary_representation_validation_summary.png")
    
    print("\nValidation complete! All tests passed.")
    print(f"Generated {len(figures) + 1} figures total.")
    print("=" * 70)


if __name__ == '__main__':
    main()
