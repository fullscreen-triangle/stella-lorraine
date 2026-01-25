"""
TERNARY GAS COMPUTATION - PANEL 1
=================================

Shows how ternary representation compresses gas dynamics into S-entropy
coordinates, and how a sliding window virtual spectrometer captures
all molecular states through the 3^k hierarchy.

Key concepts:
- S-entropy (S_k, S_t, S_e) as sufficient statistics
- Sliding window capturing molecular oscillations
- 3^k hierarchical address space
- Trit encoding: 0=oscillation, 1=category, 2=partition

Author: Kundai Farai Sachikonye
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Physical constants
k_B = 1.380649e-23  # Boltzmann constant
c = 299792458  # Speed of light


def generate_molecular_ensemble(N: int = 100, T: float = 300.0) -> dict:
    """Generate a virtual gas ensemble with full state information."""
    m = 4.65e-26  # N2 mass
    v_thermal = np.sqrt(k_B * T / m)
    
    # Full phase space state
    positions = np.random.uniform(0, 1, (N, 3))  # Normalized box
    velocities = np.random.normal(0, v_thermal, (N, 3))
    
    # Vibrational states (quantum numbers) - typically 0-5 for room temp
    # Use exponential decay for vibrational population
    n_vib = np.random.exponential(1.0, N).astype(int)  # Most molecules in n=0,1
    
    # Rotational states - typical J values are 0-20 for room temperature
    # For N2 at 300K, average J ~ 7
    J = np.random.poisson(7, N)
    
    return {
        'positions': positions,
        'velocities': velocities,
        'n_vib': n_vib,
        'J': J,
        'N': N,
        'T': T,
        'v_thermal': v_thermal
    }


def compute_s_entropy_coordinates(ensemble: dict) -> np.ndarray:
    """
    Compress full molecular state to S-entropy coordinates (S_k, S_t, S_e).
    
    S_k: Knowledge entropy - structural/spatial information
    S_t: Time entropy - temporal/velocity information  
    S_e: Evolution entropy - quantum state information
    """
    N = ensemble['N']
    S_coords = np.zeros((N, 3))
    
    for i in range(N):
        # S_k: From position (configuration)
        # Entropy of position relative to uniform distribution
        pos = ensemble['positions'][i]
        S_k = -np.sum(pos * np.log(pos + 1e-10) + (1-pos) * np.log(1-pos + 1e-10))
        
        # S_t: From velocity (momentum)
        # Normalized by thermal velocity
        v = ensemble['velocities'][i]
        v_norm = np.linalg.norm(v) / ensemble['v_thermal']
        S_t = 0.5 * v_norm**2  # Kinetic contribution
        
        # S_e: From quantum states (evolution)
        n = ensemble['n_vib'][i]
        J = ensemble['J'][i]
        S_e = np.log(n + 1) + np.log(2*J + 1)  # Quantum degeneracy
        
        S_coords[i] = [S_k, S_t, S_e]
        
    return S_coords


def ternary_encode(S_coords: np.ndarray, depth: int = 8) -> np.ndarray:
    """
    Encode S-entropy coordinates as ternary addresses.
    
    Each trit (0, 1, 2) represents:
    - 0: Oscillatory perspective (phase)
    - 1: Categorical perspective (state)
    - 2: Partition perspective (transition)
    
    The address encodes position in the 3^k hierarchical tree.
    """
    N = len(S_coords)
    addresses = np.zeros((N, depth), dtype=int)
    
    # Normalize S-coordinates to [0, 1]
    S_min = S_coords.min(axis=0)
    S_max = S_coords.max(axis=0)
    S_norm = (S_coords - S_min) / (S_max - S_min + 1e-10)
    
    # Convert to ternary: cycle through S_k, S_t, S_e
    for d in range(depth):
        coord_idx = d % 3  # Cycle through S_k, S_t, S_e
        
        # Discretize to 3 levels at this depth
        scale = 3 ** (d // 3 + 1)
        trit = (S_norm[:, coord_idx] * scale).astype(int) % 3
        addresses[:, d] = trit
        
    return addresses


def sliding_window_spectrometer(ensemble: dict, window_size: int = 10, 
                                n_windows: int = 20) -> dict:
    """
    Simulate sliding window virtual spectrometer.
    
    The spectrometer "slides" through the ensemble, capturing
    local S-entropy statistics that together represent the full gas.
    """
    N = ensemble['N']
    S_coords = compute_s_entropy_coordinates(ensemble)
    
    # Window statistics
    window_results = []
    
    for w in range(n_windows):
        # Slide window through ensemble (with overlap)
        start = (w * N // n_windows) % N
        end = start + window_size
        
        # Handle wraparound
        if end > N:
            indices = list(range(start, N)) + list(range(0, end - N))
        else:
            indices = list(range(start, end))
            
        # Local S-entropy statistics in window
        S_local = S_coords[indices]
        
        window_results.append({
            'window_id': w,
            'center': (start + window_size // 2) % N,
            'S_k_mean': np.mean(S_local[:, 0]),
            'S_t_mean': np.mean(S_local[:, 1]),
            'S_e_mean': np.mean(S_local[:, 2]),
            'S_k_std': np.std(S_local[:, 0]),
            'S_t_std': np.std(S_local[:, 1]),
            'S_e_std': np.std(S_local[:, 2]),
        })
        
    return {
        'windows': window_results,
        'full_S': S_coords,
        'n_windows': n_windows,
        'window_size': window_size
    }


def create_panel_chart(save_path: str):
    """Create comprehensive ternary gas computation panel."""
    
    # Generate gas ensemble
    ensemble = generate_molecular_ensemble(N=200, T=300.0)
    S_coords = compute_s_entropy_coordinates(ensemble)
    ternary_addrs = ternary_encode(S_coords, depth=12)
    spectrometer = sliding_window_spectrometer(ensemble, window_size=20, n_windows=30)
    
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0a0a0a')
    
    # Color scheme
    colors = {
        'S_k': '#FF6B6B',  # Red for knowledge
        'S_t': '#4ECDC4',  # Cyan for time
        'S_e': '#FFE66D',  # Yellow for evolution
        'ternary': '#9B59B6',  # Purple for ternary
    }
    
    # Panel 1: Full phase space (3D) - shows what we're compressing
    ax1 = fig.add_subplot(2, 3, 1, projection='3d', facecolor='#1a1a2e')
    v_mag = np.linalg.norm(ensemble['velocities'], axis=1)
    scatter1 = ax1.scatter(ensemble['positions'][:, 0], 
                          ensemble['positions'][:, 1],
                          ensemble['positions'][:, 2],
                          c=v_mag, cmap='plasma', s=15, alpha=0.7)
    ax1.set_xlabel('x', color='white')
    ax1.set_ylabel('y', color='white')
    ax1.set_zlabel('z', color='white')
    ax1.set_title('Full Phase Space (200 molecules)', color='white', fontsize=12)
    ax1.tick_params(colors='white')
    fig.colorbar(scatter1, ax=ax1, label='|v| (m/s)', shrink=0.6)
    
    # Panel 2: S-entropy compression (3D) - the compressed representation
    ax2 = fig.add_subplot(2, 3, 2, projection='3d', facecolor='#1a1a2e')
    scatter2 = ax2.scatter(S_coords[:, 0], S_coords[:, 1], S_coords[:, 2],
                          c=np.arange(len(S_coords)), cmap='viridis', s=20, alpha=0.8)
    ax2.set_xlabel('$S_k$ (knowledge)', color='white')
    ax2.set_ylabel('$S_t$ (time)', color='white')
    ax2.set_zlabel('$S_e$ (evolution)', color='white')
    ax2.set_title('S-Entropy Compression', color='white', fontsize=12)
    ax2.tick_params(colors='white')
    
    # Add annotation
    ax2.text2D(0.02, 0.98, 'Each point = 1 molecule\n18 dims -> 3 dims', 
               transform=ax2.transAxes, color='white', fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#2a2a4e', alpha=0.8))
    
    # Panel 3: Ternary address visualization
    ax3 = fig.add_subplot(2, 3, 3, facecolor='#1a1a2e')
    
    # Show first 50 molecules' ternary addresses as image
    n_show = 50
    im = ax3.imshow(ternary_addrs[:n_show, :], cmap='RdYlBu', aspect='auto',
                   interpolation='nearest')
    ax3.set_xlabel('Trit position (depth)', color='white')
    ax3.set_ylabel('Molecule index', color='white')
    ax3.set_title('Ternary Addresses (3$^k$ hierarchy)', color='white', fontsize=12)
    ax3.tick_params(colors='white')
    
    # Add colorbar with ternary labels
    cbar = fig.colorbar(im, ax=ax3, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['0: Osc', '1: Cat', '2: Part'])
    cbar.ax.tick_params(colors='white')
    
    # Panel 4: Sliding window spectrometer
    ax4 = fig.add_subplot(2, 3, 4, facecolor='#1a1a2e')
    
    windows = spectrometer['windows']
    window_ids = [w['window_id'] for w in windows]
    S_k_means = [w['S_k_mean'] for w in windows]
    S_t_means = [w['S_t_mean'] for w in windows]
    S_e_means = [w['S_e_mean'] for w in windows]
    
    ax4.plot(window_ids, S_k_means, '-o', color=colors['S_k'], 
             linewidth=2, markersize=4, label='$S_k$ (knowledge)')
    ax4.plot(window_ids, S_t_means, '-s', color=colors['S_t'], 
             linewidth=2, markersize=4, label='$S_t$ (time)')
    ax4.plot(window_ids, S_e_means, '-^', color=colors['S_e'], 
             linewidth=2, markersize=4, label='$S_e$ (evolution)')
    
    ax4.set_xlabel('Window Position', color='white')
    ax4.set_ylabel('Mean S-coordinate', color='white')
    ax4.set_title('Sliding Window Spectrometer', color='white', fontsize=12)
    ax4.tick_params(colors='white')
    ax4.legend(loc='upper right', facecolor='#2a2a4e', edgecolor='white', labelcolor='white')
    ax4.grid(True, alpha=0.2, color='white')
    
    # Add sliding window annotation
    ax4.axvspan(10, 15, alpha=0.3, color='white', label='Active window')
    
    # Panel 5: Ternary tree structure (3D)
    ax5 = fig.add_subplot(2, 3, 5, projection='3d', facecolor='#1a1a2e')
    
    # Create hierarchical tree visualization
    # Level 0: origin
    # Level 1: 3 nodes at unit distance in x, y, z
    # Level 2: 9 nodes, etc.
    
    levels = 3
    nodes = []
    edges = []
    
    def add_ternary_tree(parent_pos, level, max_level, direction):
        if level > max_level:
            return
            
        nodes.append((*parent_pos, level))
        
        if level < max_level:
            # Three children in different directions
            scale = 0.5 ** level
            directions = [
                (scale, 0, 0),   # Oscillatory (x)
                (0, scale, 0),   # Categorical (y)
                (0, 0, scale),   # Partition (z)
            ]
            
            for d in directions:
                child_pos = (parent_pos[0] + d[0], 
                           parent_pos[1] + d[1], 
                           parent_pos[2] + d[2])
                edges.append((parent_pos, child_pos))
                add_ternary_tree(child_pos, level + 1, max_level, d)
    
    add_ternary_tree((0, 0, 0), 0, levels, (0, 0, 0))
    
    # Plot edges
    for edge in edges:
        ax5.plot([edge[0][0], edge[1][0]], 
                [edge[0][1], edge[1][1]], 
                [edge[0][2], edge[1][2]], 
                'w-', alpha=0.3, linewidth=0.5)
    
    # Plot nodes colored by level
    node_array = np.array([(n[0], n[1], n[2]) for n in nodes])
    node_levels = np.array([n[3] for n in nodes])
    ax5.scatter(node_array[:, 0], node_array[:, 1], node_array[:, 2],
               c=node_levels, cmap='coolwarm', s=50, alpha=0.8)
    
    ax5.set_xlabel('Oscillatory (0)', color='white')
    ax5.set_ylabel('Categorical (1)', color='white')
    ax5.set_zlabel('Partition (2)', color='white')
    ax5.set_title('3$^k$ Ternary Address Tree', color='white', fontsize=12)
    ax5.tick_params(colors='white')
    
    # Panel 6: Computation = Gas Dynamics
    ax6 = fig.add_subplot(2, 3, 6, facecolor='#1a1a2e')
    ax6.axis('off')
    
    # Summary diagram
    summary_text = """
    TERNARY GAS COMPUTATION
    ═══════════════════════
    
    Phase Space (18D)  ──→  S-Entropy (3D)  ──→  Ternary Address
    [x,y,z,vx,vy,vz,    [S_k, S_t, S_e]      [0,1,2,0,2,1,...]
     n,J,...]                                  
                                             
    TRIT ENCODING:
    ┌─────────────────────────────────────────┐
    │  0 = Oscillatory perspective (phase)    │
    │  1 = Categorical perspective (state)    │
    │  2 = Partition perspective (transition) │
    └─────────────────────────────────────────┘
    
    SLIDING WINDOW SPECTROMETER:
    ┌─────┐                     
    │█████│→ window slides through ensemble
    └─────┘  captures local S-entropy stats
    
    KEY INSIGHT:
    Oscillator = Processor
    Computing = Solving Gas Dynamics
    
    Memory Address = Trajectory in S-Space
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            color='white', bbox=dict(boxstyle='round', facecolor='#2a2a4e', alpha=0.9))
    
    plt.suptitle('Ternary Representation for Gas Dynamics: S-Entropy Compression',
                fontsize=16, fontweight='bold', color='white', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    print(f"Saved panel 1 to {save_path}")
    

def main():
    """Generate ternary gas computation panel 1."""
    print("=" * 60)
    print("TERNARY GAS COMPUTATION - PANEL 1")
    print("S-Entropy Compression and Sliding Window Spectrometer")
    print("=" * 60)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    create_panel_chart(os.path.join(figures_dir, "panel_ternary_computation_1.png"))
    
    print("\n" + "=" * 60)
    print("PANEL 1 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

